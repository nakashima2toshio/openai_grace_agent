# openai_grace_agent 移植仕様書（anthropic → openai / 2026-07-01）

本書は `openai_port_todo_july01.md` の各項目を**実装レベルで定義**する仕様書。
TODO が「何を・どの順で」を示すのに対し、本書は「**どのファイルの何行を・どう読み替えて・
どこへ配線し・どう検証するか**（IPO＋受入基準）」を定義する。

- 基準（正）: `anthropic_grace_agent_v2`（master）
- 対象: `openai_grace_agent` / ブランチ `claude/grace-agent-anthropic-to-openai-e6zmi0`
- 上位方針: 本仕様書は **ロジックのみ** anthropic に合わせ、provider 層は openai を維持する
  （`openai_port_todo_july01.md` §0 の読み替え表を必須遵守）

---

## 0. 全体アダプテーション規則（全項目に適用）

| 規則 | 内容 |
|---|---|
| **R1 LLM クライアント** | `from .llm_compat import create_chat_client` / `create_chat_client(self.config)` は **移植しない**。openai 既存の `from helper.helper_llm import create_llm_client` → `create_llm_client("openai", default_model=self.config....model)` に置換 |
| **R2 モデル既定** | `claude-sonnet-4-6` → `gpt-5-mini`。軽量用途も openai 側の既定に合わせる |
| **R3 トークン枠** | `max_tokens=` → `max_completion_tokens=`。**枠値は anthropic の数値を維持**（june14 H2 の教訓: 削り過ぎ厳禁。confidence=512/1024、relevance=256、空応答は適合フォールバック） |
| **R4 温度** | gpt-5 系は非デフォルト temperature 非対応。既存 `_drop_unsupported_temperature()` 経路を通す（新規に temperature を直接送らない） |
| **R5 Embedding/次元/コレクション** | `text-embedding-3-large` / 3072 / `*_openai` を維持。次元 3072 は anthropic と一致するため検証ロジックはそのまま流用可 |
| **R6 API キー** | `ANTHROPIC_API_KEY` → `OPENAI_API_KEY`（Embedding 文脈の `GOOGLE_API_KEY` は据え置き可） |
| **R7 コメント** | 移植/編集した箇所の docstring・コメントは openai 用語へ是正（§9.1） |

---

## 1. Phase P1 — GRACE 強化（本移植の中核）

### P1-1 `grace/calibration.py`（新規モジュール）

- **Input**: SOURCE `grace/calibration.py`（167行, provider 非依存）
- **Process**: 温度スケーリングによる信頼度の post-hoc 較正（`Calibrator.load()` / `.transform()` / ECE 計測）
- **Output**: TARGET `grace/calibration.py`（ほぼ verbatim）。LLM 呼び出しは無いため R1 の影響なし
- **配線**: P1-9 で executor が `Calibrator.load(config.confidence.calibration_path)` → 最終信頼度に `transform()`
- **受入基準**: `python -c "from grace.calibration import Calibrator"` 成功 / 較正ファイル不在時は恒等変換にフォールバック

### P1-2 `grace/memory.py`（新規モジュール）

- **Input**: SOURCE `grace/memory.py`（208行, provider 非依存）
- **Process**: `ExecutionMemory` がクエリ×コレクションの成功/信頼度を記録し、コレクション別事前分布を返す（`record()` / `best_collection()` / `create_execution_memory()`）
- **Output**: TARGET `grace/memory.py`（verbatim）。永続パスは `MemoryConfig.path`（P1-8）
- **配線**:
  - executor: `self._memory = create_execution_memory(config.memory)`、各ステップ後 `self._memory.record(query, collection, success, confidence)`
  - planner: `_prioritized_collection(query)` が `self._memory.best_collection(query)` を参照し rag_search の候補順を決定
- **受入基準**: memory 無効(config)時は no-op / 記録後に同一クエリで優先コレクションが返る単体テスト（P5）

### P1-4 `GroundednessVerifier`（`grace/confidence.py`）

- **Input**: SOURCE `grace/confidence.py` L649-836（`ClaimVerdict`/`GroundednessResponse`/`GroundednessResult`/`GroundednessVerifier`/`create_groundedness_verifier`）
- **Process**: 最終回答の各クレームを出典に対して LLM で含意判定 → support_rate を算出
- **Output**: TARGET `grace/confidence.py` に 4 クラス＋ファクトリを追加、`__all__` に追記。`ConfidenceFactors` に `groundedness: float = 0.0` を追加
- **アダプテーション**: R1（`create_chat_client`→`create_llm_client("openai")`）、R3（枠は anthropic 値維持）
- **配線**: P1-9 で executor が `groundedness_verifier.verify(query, final_answer, sources)` → `ConfidenceFactors.groundedness` に反映。重み `config.confidence.groundedness_weight`
- **受入基準**: 出典に支持されるクレームで support_rate↑ / 空応答・parse 失敗時は中立(0/0→中立、H2 契約)

### P1-5 曖昧クエリ判定（`grace/planner.py`）

- **Input**: SOURCE `grace/planner.py` L30-78（`is_ambiguous_query()` ＋ `_AMBIGUOUS_REFERENT_PATTERNS`/`_DEMONSTRATIVES`/`_COMPLEXITY_FACTORS`）
- **Process**: 指示語・曖昧照応・複雑度ヒューリスティックで「そのままでは検索不能」なクエリを検出
- **Output**: TARGET planner に関数＋定数を追加。`generate_plan()` で ambiguous 判定時は rag_search ではなく **ask_user/確認プラン**を生成
- **アダプテーション**: 正規表現・ヒューリスティックは provider 非依存でそのまま。`original_query` 連携は openai 既存(L205/370/420)を利用
- **受入基準**: 「あの件どうなった?」等で ask_user 経路 / 明確クエリは従来どおり検索経路

### P1-6 ReAct スキーマ＋ベンチフィールド（`grace/schemas.py`）

- **Input**: SOURCE `grace/schemas.py` L336-388（`ScratchpadEntry`/`Scratchpad`/`AgentThought`）、L257-271（`ExecutionResult.rag_max_score`/`rag_search_count`/`web_search_used`）
- **Process/Output**: 3 スキーマ＋3 フィールドを TARGET へ追加、`__all__` 追記
- **配線**: ベンチフィールドは executor が RAG 最大スコア・検索回数・web 使用を集計して充填（executor には局所 `rag_max_score` が既存 L277 → schema フィールドへ昇格）
- **受入基準**: `ExecutionResult` の import 互換維持 / 既存テスト緑

### P1-7 コレクション次元検証＋キャッシュ（`grace/tools.py`）

- **Input**: SOURCE `grace/tools.py` L254-342（`_VALID_COLLECTIONS_CACHE`/`_collection_dense_dim()`/次元・空コレクション除外）、L139-152（`restrict_to_collection` 分岐）
- **現状(TARGET)**: `_get_all_collections_dynamic()` の骨格のみ（L239）。次元検証・キャッシュ・restrict 分岐が欠落
- **Process**: Qdrant から各コレクションの dense 次元を取得し、埋め込み次元（3072）不一致・空コレクションを検索対象から除外。`(qdrant_url, dim)` をキーにプロセス内キャッシュ
- **Output**: TARGET tools に上記を移植。キャッシュキーの次元は **openai の 3072** を使用（R5）
- **受入基準**: 次元不一致コレクションが候補から除外され 400 回避 / `restrict_to_collection=ON` で単一検索

### P1-8 `grace/config.py` フィールド追加（足場・先行実施）

- **Input**: SOURCE `grace/config.py`（`MemoryConfig`、`ConfidenceConfig` の groundedness/calibration 系、`QdrantConfig.restrict_to_collection`、planner/executor 追加ノブ）
- **Output**: TARGET `grace/config.py` に対応クラス/フィールドを追加。既定値は anthropic を踏襲しつつ provider 値のみ openai
- **注意**: ReAct 系ノブ（`react_enabled` 等）は P1-6 を有効運用する場合のみ追加。`code_execute` 系は scope 外
- **受入基準**: 既存 `PlannerConfig`/`ExecutorConfig`（#58 既存）と衝突しない / `GraceConfig()` 構築成功

### P1-9 executor 配線（統合）

- **Input**: SOURCE `grace/executor.py`（`_calibrator`/`_memory`/`groundedness_verifier` 初期化 L157-167、`ExecutionState.used_collections` L85）、コミット `0e9c27f`（Groundedness 0/0 中立化＋トークン合算）、`4ee40b8`（`StepResult.token_usage` P3 結線）
- **Process/Output**: TARGET executor に以下を配線
  1. `__init__`: calibrator/memory/groundedness_verifier を config 条件付きで初期化（R1 適用）
  2. `ExecutionState.used_collections` フィールド復活・各ステップで追記
  3. ステップ後 `_memory.record(...)`
  4. 最終信頼度に `_calibrator.transform()` を適用
  5. Groundedness 0/0 を中立化、`StepResult.token_usage` を合算
- **受入基準**: `test_executor.py`/`test_confidence.py` 緑（H3 同様スタブで非密閉テストを決定化）/ 較正・groundedness 無効時は従来挙動

---

## 2. Phase P2 — provider 整合（要確認）

### P2-1 `services/qa_service.py:109`

- **現状**: `create_llm_client(provider="gemini")`（docstring「Gemini API使用」/ 既定 `gemini-2.0-flash`）
- **正(anthropic)**: `provider="anthropic"`
- **是正案**: `provider="openai"` ＋ 既定 `gpt-5-mini`、docstring/ログを OpenAI へ。※これは Q/A **生成 LLM** 用途（Embedding ではない）
- **判断**: モデル名/プロバイダ変更のため **CLAUDE.md 規約に従い着手前にユーザー確認**（本 PR では未変更／確認後に別コミット）

---

## 3. Phase P3 — ルートサンプル

### P3-1 `agent_example.py` / `agent_example_core8.py`

- anthropic のみ存在。移植時は R1/R2/R6/R7 を適用し用語翻訳。openai 既存エージェント（`agent_main.py`/`agent_rag.py`）と重複しない範囲で移植。core8 は設計ドキュメント `agent_example_core8.md`（P4-1）とセットで扱う

---

## 4. Phase P4 — ドキュメント／コメント

- **P4-1**: `grace/doc/*` の設計ドキュメントを `grace/docs/` へ翻訳移植（`calibration.md`/`benchmark.md`/`grace.md`/`grace_core.md`/`grace_core_flow.md`/`config.md`/`agent_example_core8.md`）。`llm_compat.md` は除外
- **P4-2**: `docs/agent.md`/`agent_a_b.md`(+html) を翻訳移植 or `readme_autonomous_agent.md` に統合
- **P4-3**: readme_* を監査し P1 機能を追記
- **P4-4**: 重点フォルダー＋root `*.py` のコメント/docstring 用語是正
- **スタイル**: Mermaid は §7（黒背景・白文字・PyCharm 互換）。用語は §9.1 表に統一

---

## 5. Phase P5 — テスト（機能実装後）

`test_calibration`（verbatim）/ `test_memory`・`test_groundedness`・`test_react`（値読み替え）。統合は env ゲート。
非密閉テストは june14 H3 と同様に `_evaluate_rag_relevance` 等をスタブして決定化。

---

## 6. リスク・非対象

| 事項 | 方針 |
|---|---|
| `llm_compat.py` | **移植しない**（openai は helper.helper_llm）。全参照を R1 で置換 |
| prompt caching（`cache_control`/`system`） | Anthropic 固有 API。移植しない（OpenAI は自動キャッシュ） |
| `code_execute` ツール／`CodeExecuteConfig` | 現状 scope 外（別途要判断） |
| provider/モデル名変更（P2-1・G3 系） | CLAUDE.md 規約によりユーザー確認を経てから |
| トークン枠 | H2 の教訓により anthropic の数値を維持（削り過ぎ禁止） |

---

## 7. 受入（DoD）

1. `ruff check .` ＋ `python -m compileall .` が緑
2. `uv run pytest`（統合は env ゲート）で P1 追加テスト含め緑
3. §0 provider 読み替え表からの逸脱ゼロ（`create_chat_client`/`claude-*`/`ANTHROPIC_API_KEY`/`*_anthropic` の混入なし）
4. 編集ファイルのコメント/docstring が §9.1 用語に整合

---

*作成日: 2026-07-01 / 対: openai_port_todo_july01.md / 検証: 実ファイル比較＋実コード grep*
