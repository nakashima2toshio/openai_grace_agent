# openai_grace_agent 再移植 TODO（anthropic → openai / 2026-07-01）

`anthropic_grace_agent_v2` を **ロジックの正**とし、`openai_grace_agent` へ横展開する
**2 回目の移植**の TODO。前回（`openai_refactor_todo_june14.md`）で Phase A〜H を完了済み。
本 TODO は **june14 以降に anthropic 側で進んだ差分**（主に 6/22〜6/28 の grace 強化）を対象とする。

- 作成日: 2026-07-01
- 基準（ロジックの正）: `anthropic_grace_agent_v2`（master, `9aec7ed`）
- 対象ブランチ: `claude/grace-agent-anthropic-to-openai-e6zmi0`
- 前提: 実ファイル比較 ＋ 実コード grep で **検証済み**（偽陽性を除外）

---

## §0 プロバイダー読み替え（確定値・維持する）

anthropic を正にするのは**ロジックのみ**。下表の値は **openai のものを維持**する。

| 項目 | anthropic（正にしない） | **openai（維持する値）** |
|---|---|---|
| LLM クライアント | `create_llm_client("anthropic")` | `create_llm_client("openai")` |
| Embedding クライアント | `create_embedding_client("gemini")` | `create_embedding_client("openai")` |
| デフォルト LLM | `claude-sonnet-4-6` | `gpt-5-mini` |
| Embedding モデル/次元 | `gemini-embedding-001` / 3072 | `text-embedding-3-large` / **3072** |
| max トークン引数 | `max_tokens` | `max_completion_tokens` |
| API キー | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` |
| Qdrant コレクション | `*_anthropic` | `*_openai` |

### ⚠️ 本移植の最重要アダプテーション（`llm_compat` の扱い）

anthropic の新規/更新モジュールは軒並み **`from .llm_compat import create_chat_client`** を使う
（executor / planner / confidence / tools）。openai には `grace/llm_compat.py` は**存在せず、移植もしない**。
openai 側の既存流儀 **`from helper.helper_llm import create_llm_client` → `create_llm_client("openai", default_model=...)`**
に**必ず読み替える**こと。ポートする全コードでこの置換を徹底する（新規 `create_chat_client` 呼び出しを持ち込まない）。

---

## §1 「移植不要」確定リスト（偽陽性除外・作業しないこと）

全ファイル diff は provider ノイズが支配的。以下は**検証済みで対応不要**。ここに時間を使わない。

| 対象 | 判定 | 根拠（検証済み） |
|---|---|---|
| `helper/helper_rag_qa.py`（3866行差） | ⚪ provider noise | anthropic=後方互換ラッパー / openai=LLM抽象再編。Q/A ロジックは双方 `qa_generation/` |
| `chunking/csv_text_to_chunks_text_csv.py` の `_split_document_into_blocks`/`_report_coverage`/doc_id 配線 | ⚪ openai が先行 | openai に実装済み・anthropic 未実装（逆方向） |
| `chunking/async_api_client.py` トークン使用量集計 | ⚪ openai が先行 | openai に `_accumulate_usage` 実装済み |
| `helper/helper_llm.py` トークンカウンタ（reset/get_token_counter/`_drop_unsupported_temperature`） | ⚪ 実装済み(F1/H1) | june14 で導入済み |
| `config.py`/`grace/config.py` の `PlannerConfig`/`ExecutorConfig` | ⚪ 実装済み(#58) | grep 確認 |
| `gpt-5.4-mini` 残存 | ⚪ 無し(G3済) | grep で 0 件 |
| `celery_tasks.collect_results`（ready()＋on_result） | ⚪ 実装済み(#67) | grep 確認 |
| `qdrant_client_wrapper`/`qdrant_service` の内容ハッシュ ID | ⚪ 実装済み(#51) | 登録経路は `stable_point_id` 使用 |
| planner の `original_query` reasoning 連携 | ⚪ 実装済み | grep 確認（planner L205/370/420） |
| `qa_generation/pipeline.py` provider 自動判定 | ⚪ openai が先行 | openai に model 先頭判定あり（逆方向） |
| `grace/replan.py` / `grace/intervention.py` | ⚪ 同一 | diff 0 |
| `run_benchmark.py` / `temp.txt` / `gemini_api_list.xlsx` / `eval/` | ⚪ SKIP | sh 版で代替 / 一時物 |

---

## Phase P1 — GRACE 強化の移植（最重要・anthropic が 6/22〜6/28 で先行）

> 依存順: **P1-1/P1-2（新規モジュール）→ P1-8（config 足場）→ P1-4/5/6/7 → P1-9（executor 配線）**。
> 全コードで `create_chat_client` → `create_llm_client("openai")` へ読み替える（§0）。

| # | 種別 | 状態 | 作業（SOURCE アンカー） |
|---|---|---|---|
| **P1-1** | feat(new) | ❌ MISSING | `grace/calibration.py`（167行）を移植。温度スケーリング／ECE の post-hoc 較正。**provider 非依存でほぼ verbatim**。executor が信頼度に適用 |
| **P1-2** | feat(new) | ❌ MISSING | `grace/memory.py`（208行）を移植。コレクション別成功事前分布を学習。**provider 非依存**。executor が `record()`、planner が `_prioritized_collection()` で参照 |
| **P1-3** | skip | ⛔ 移植しない | `grace/llm_compat.py` は移植しない。openai は `helper.helper_llm` を使用（§0 のアダプテーション） |
| **P1-4** | feat | ❌ MISSING | `grace/confidence.py` に `GroundednessVerifier` ＋ `ClaimVerdict`/`GroundednessResponse`/`GroundednessResult` ＋ `create_groundedness_verifier()` を移植（SOURCE L649-836）。`__all__` 追記。LLM 呼び出しを openai へ読み替え。`ConfidenceFactors.groundedness` フィールド追加 |
| **P1-5** | feat | ❌ MISSING | `grace/planner.py` に `is_ambiguous_query()` ＋ パターン定数（`_AMBIGUOUS_REFERENT_PATTERNS`/`_DEMONSTRATIVES`/`_COMPLEXITY_FACTORS`, SOURCE L30-78）を移植。曖昧クエリを ask_user/確認プランへ誘導（ollama #40/#44 由来） |
| **P1-6** | feat | ❌ MISSING | `grace/schemas.py` に ReAct スキーマ `ScratchpadEntry`/`Scratchpad`/`AgentThought`（SOURCE L336-388）を移植。`ExecutionResult` に `rag_max_score`/`rag_search_count`/`web_search_used`（SOURCE L257-271）を追加。`__all__` 追記 |
| **P1-7** | feat | ⚠️ PARTIAL | `grace/tools.py` の `RAGSearchTool` に次元検証＋キャッシュを移植: `_VALID_COLLECTIONS_CACHE`／`_collection_dense_dim()`／`_get_all_collections_dynamic()` の次元・空コレクション除外（SOURCE L254-342）＋ `restrict_to_collection` 分岐（L139-152）。openai は `_get_all_collections_dynamic` の骨格のみ存在 |
| **P1-8** | feat | ❌ MISSING | `grace/config.py` に不足フィールドを追加: `MemoryConfig`、`ConfidenceConfig` の groundedness/calibration 系（`groundedness_enabled`/`groundedness_weight`/`calibration_path` 等）、`QdrantConfig.restrict_to_collection`、planner/executor の追加ノブ（`step_timeout_seconds`/`llm_plan_max_attempts`/`complexity_*` 等）。※ReAct 系ノブは P1-6 を有効化する場合のみ |
| **P1-9** | feat/wire | ❌ MISSING | `grace/executor.py` 配線: `_calibrator`/`_memory`/`groundedness_verifier` 初期化、`ExecutionState.used_collections`、各ステップ後の `_memory.record()`、最終信頼度への較正適用、Groundedness 0/0 の中立化（SOURCE `0e9c27f`）、`StepResult.token_usage` の P3 結線（SOURCE `4ee40b8`） |

---

## Phase P2 — services / Q/A の provider 整合（⚠️ 要確認）

| # | 種別 | 状態 | 作業 |
|---|---|---|---|
| **P2-1** | fix(provider) | ⚠️ 要ユーザー確認 | `services/qa_service.py:109` が Q/A 生成 LLM に `create_llm_client(provider="gemini")`（docstring「Gemini API使用」・既定 `gemini-2.0-flash`）。anthropic は `provider="anthropic"`。openai 規約（§9.1: LLM は OpenAI）に照らすと `provider="openai"` ＋ 既定 `gpt-5-mini` が正。**モデル名/プロバイダ変更のため CLAUDE.md 規約に従い着手前に確認** |

> 備考: `qa_qdrant/*` の `provider="gemini"` は **Embedding 用途で正**（変更しない）。

---

## Phase P3 — ルートスクリプト / サンプル（低〜中優先）

| # | 種別 | 状態 | 作業 |
|---|---|---|---|
| **P3-1** | port | ❓ 判断 | `agent_example.py` / `agent_example_core8.py`（anthropic のみ）を移植。用語・APIキーを openai へ翻訳。openai の既存エージェント構成と重複しないか確認のうえ移植可否を決定 |

---

## Phase P4 — ドキュメント・コメント（ユーザー明示の重点）

CLAUDE.md **§7（Mermaid 黒背景）/§9（用語統一: OpenAI GPT / gpt-5-mini / text-embedding-3-large / OPENAI_API_KEY / *_openai）** に従う。anthropic 用語は**翻訳**（verbatim コピー禁止）。

| # | 種別 | 状態 | 作業 |
|---|---|---|---|
| **P4-1** | docs(new) | ❌ MISSING | `grace/doc/` の新設計ドキュメントを `grace/docs/` へ移植・翻訳: `calibration.md` / `memory` 相当 / `benchmark.md` / `grace.md` / `grace_core.md` / `grace_core_flow.md` / `config.md` / `agent_example_core8.md`（`llm_compat.md` は移植対象外モジュールのため除外） |
| **P4-2** | docs | ❓ 判断 | `docs/agent.md` / `docs/agent_a_b.md`（＋ `agent_a_b_infographic.html`）が anthropic のみ。openai へ翻訳移植 or 既存 `readme_autonomous_agent.md` に統合 |
| **P4-3** | docs(audit) | ⚠️ AUDIT | `readme_autonomous_agent.md` / `readme_rag.md` / `readme_react_reflection.md` の内容差を監査し、P1 で移植した機能（calibration/memory/groundedness/曖昧判定/ReAct）の記述を追記 |
| **P4-4** | comments | ❌ MISSING | 重点フォルダー（chunking/grace/services/helper/qa_generation/qa_qdrant）＋ root `*.py` のコメント/docstring を sweep。`Gemini API使用`・`[MIGRATION] gemini→anthropic`・`Anthropic Claude` 等の残存表記を openai 用語へ是正（ユーザー明示の「コメント文」対応） |

---

## Phase P5 — テスト移植（高価値のみ先行）

june14 で大半移植済み。anthropic が追加した高価値・provider 非依存テストを移植。

| テスト | 分類 | 作業 |
|---|---|---|
| `tests/grace/test_calibration.py` | PORT | ほぼ verbatim（P1-1 前提） |
| `tests/grace/test_memory.py` | PORT(sub) | コレクション名 `*_openai` へ読み替え（P1-2 前提） |
| `tests/grace/test_groundedness.py` | PORT(sub) | LLM モックを openai へ（P1-4 前提） |
| `tests/grace/test_react.py` | PORT(sub) | ReAct スキーマ（P1-6 前提）・コレクション名読み替え |
| `tests/grace/test_code_execute.py` | 判断 | code_execute を移植する場合のみ（現状 scope 外） |
| 実機依存（`test_collection`/`test_*register_qdrant*`/`test_metadata*`）・scratch（`verify_*`）・legacy | SKIP | 対象外 |

> P1 系の round-trip/挙動テストは **機能実装後**に移植。統合テストは `OPENAI_API_KEY` 等の env ゲート。

---

## Phase P6 — 検証

- **静的**: 各 PR 単位で `ruff check .` ＋ `python -m compileall` を緑に。
- **テスト**: `uv run pytest`（統合は env ゲート）。P1-9 の非密閉テストは june14 H3 と同様スタブで決定化。
- **実機（要サービス）**: 較正の効き・memory の事前分布・次元検証によるコレクション除外・groundedness は実 LLM/Qdrant で確認推奨。

---

## 進捗チェックリスト

### Phase P1 — grace 強化
- [ ] P1-1 `grace/calibration.py` 移植（verbatim + config）
- [ ] P1-2 `grace/memory.py` 移植 + executor/planner 配線
- [ ] P1-3 `llm_compat` 非移植（openai は helper.helper_llm）
- [ ] P1-4 `GroundednessVerifier` 一式 + ConfidenceFactors.groundedness
- [ ] P1-5 `is_ambiguous_query()` + パターン定数（ask_user 誘導）
- [ ] P1-6 ReAct スキーマ + ExecutionResult ベンチフィールド
- [ ] P1-7 tools 次元検証/キャッシュ/restrict_to_collection
- [ ] P1-8 grace/config フィールド追加（MemoryConfig/groundedness/calibration/restrict）
- [ ] P1-9 executor 配線（calibrator/memory/groundedness/0-0中立化/token_usage）

### Phase P2 — provider 整合
- [ ] P2-1 qa_service.py Q/A 生成 → openai（⚠️ 要確認）

### Phase P3 — ルートサンプル
- [ ] P3-1 agent_example(.py/_core8.py) 移植可否判断・実施

### Phase P4 — ドキュメント・コメント
- [ ] P4-1 grace 設計ドキュメント移植・翻訳
- [ ] P4-2 docs/agent*.md 翻訳移植 or 統合
- [ ] P4-3 readme_* 監査・追記
- [ ] P4-4 コメント/docstring 用語是正 sweep

### Phase P5 — テスト
- [ ] calibration/memory/groundedness/react テスト移植

### Phase P6 — 検証
- [ ] ruff + compileall 緑
- [ ] pytest（env ゲート）

---

*作成日: 2026-07-01 / 基準: 実ファイル比較 + 実コード grep 検証済み / 前回: openai_refactor_todo_june14.md*
