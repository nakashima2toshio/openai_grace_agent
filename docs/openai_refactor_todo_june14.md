# openai_grace_agent リファクタリング TODO（プレイブック横展開）

`ollama_grace_agent` で実施した改修（PR #51〜#89）を **openai_grace_agent へ横展開**するための
TODO。`anthropic_grace_agent` を「正（ロジックのみ）」とし、provider 層（LLM/Embedding
クライアント・モデル名・次元・コスト・コレクション名・APIキー）は **openai の構成を維持**する。

- 作成日: 2026-06-14
- 基準（ロジックの正）: `anthropic_grace_agent`（6/10 以降に最新化済み）
- 参照: `grace_agent_refactor_playbook.md`（ollama での総まとめ／横展開プレイブック）
- 対象ブランチ: `claude/upbeat-bell-rnr7ma`

> **偽陽性に注意**: 全ファイル diff は provider 差が支配的で「漏れ」と「provider 差」を
> 誤認しやすい。移植要否は **anthropic の git 履歴（コミット単位）× openai の現状** で判断する。
> 本 TODO は anthropic の grace/pipeline P0–P3 コミット（`98c4881`/`364eea5`/`53cbd43`/`4963e6d`、
> `0abcd7b`/`bdecd95`/`5ab2e66`/`c6e35db` 等）と openai 現状を照合済み。

---

## §0 プロバイダー読み替え（確定値）

anthropic を正にするのは**ロジックのみ**。下表の値は openai のものを維持する。

| 項目 | anthropic（正にしない） | **openai（維持する値）** |
|---|---|---|
| LLM クライアント | `create_llm_client("anthropic")` | `create_llm_client("openai")` |
| Embedding クライアント | `create_embedding_client("gemini")` | `create_embedding_client("openai")` |
| デフォルト LLM モデル | `claude-sonnet-4-6` | `gpt-5-mini` |
| Embedding モデル/次元 | `gemini-embedding-001` / 3072 | `text-embedding-3-large` / **3072** |
| コスト計算 | あり | **あり**（維持） |
| API キー | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` |
| Qdrant コレクション | `*_anthropic` | `*_openai` |

---

## Phase A — GRACE 自律エージェント（`grace/`）

推奨順（依存関係考慮）。**`#58` が `#57/#60/#61` の前提**。

| # | 種別 | 状態 | 作業 |
|---|---|---|---|
| **#56** | fix(security) | ❌ MISSING | `executor.py:485` の `eval(result.output)` を撤去。anthropic の `_handle_ask_user_response()`（`ast.literal_eval`）を移植。`import ast` 追加。※`executor.py:760` の `ast.literal_eval` は安全で対象外 |
| **#58** | feat | ❌ MISSING | `config.py` に `PlannerConfig`（`llm_plan_complexity_threshold` 等）/`ExecutorConfig`（`parallel_search`/`fallback_chain` 等）を追加。**最優先（他の足場）** |
| **#57** | feat | ❌ MISSING | `executor.py` に `_run_tool_with_timeout()`（ThreadPoolExecutor）を追加し、`timeout_seconds`（既存の 875/998 参照）を実際に強制 |
| **#60** | feat | ❌ MISSING | `_prefetch_parallel_searches()`（依存なし検索ステップの並列プリフェッチ）を追加し実行ループへ配線。`ExecutorConfig.parallel_search` を実利用化 |
| **#61** | feat | ❌ MISSING | `planner.py` に二層計画（複雑度ヒューリスティックで単純質問は LLM 省略 → `_create_rule_based_plan()`）。現状は全質問で LLM 経路 |
| **#64** | feat | ⚠️ PARTIAL | `executor.py` に `_should_trigger_replan()` を追加し、低信頼度リプランを**検索ステップ限定**に。現状 `replan.py:160-166` が全低信頼度で再発火しカスケードの懸念 |
| **#65** | feat | ❌ MISSING | `confidence.py` に `FinalEvaluationResult` + `evaluate_final()`（自己評価＋網羅度を 2 回→1 回に統合）を追加し executor から利用 |
| **#66** | refactor | ❌ MISSING | `executor.py` に `_build_confidence_factors()` を追加し信頼度ファクタ構築を共通化（出力型を `Optional[Any]` へ拡張） |
| #59 | refactor | ✅ DONE | 実行ループ統合済み（`execute_plan`→`execute_plan_generator` へ `yield from` 委譲）。対応不要 |

---

## Phase B — 登録・Q/A・パイプライン

| # | 種別 | 状態 | 作業 |
|---|---|---|---|
| **#51** | feat/fix | ❌ MISSING（3 部） | (a) `services/qdrant_service.py` の位置ベース point ID（`abs(hash(...))`）を**内容ハッシュ**（`_content_point_key()`/`stable_point_id()`）化＋payload デフォルト補完、(b) `register_to_qdrant.py` に**重複Q/A除去**ブロック、(c) **ThreadPoolExecutor 先読み並列化**（`--embed-workers`/`on_result` フック） |
| **#67** | feat | ❌ MISSING | `celery_tasks.collect_results` を blocking `task.get()` → **`ready()` 完了順ポーリング＋`on_result` フック**へ。`_GENERATOR_CACHE`/`_get_generator()` でワーカー内再利用、`usage_out` 集計追加（HOL ブロッキング解消） |
| **#53** | feat | ❌ MISSING | `pipeline.py` 同期経路に **JSONL 逐次永続化＋クラッシュ再開**（処理済みチャンク skip）、`_enforce_max_chunk_tokens`。※#67 と統合して配線 |
| **#52** | feat | ⚠️ PARTIAL | 単段化（構造化出力 1 回）は概ね済。死にフラグ `use_smart_generation` を `make_qa.py`/`pipeline.py`/`celery_tasks.py` から撤去 |
| **#54** | feat | ⚠️ PARTIAL | `make_qa.py` の `--use-smart-generation/--no-smart-generation` 引数群を撤去。既定モデル `gpt-5-mini`・`OPENAI_API_KEY` チェックは正 |
| **#55** | feat | ⚠️ PARTIAL | `make_qa_register_qdrant.py` の死にフラグ撤去＋`--combine-rows` 撤去（チャンキングは `csv_text_to_chunks_text_csv.py` へ誘導） |
| #82 | feat | ✅ DONE 相当 | `get_collection_embedding_params` は openai 既定（3072→`text-embedding-3-large`、1536→`text-embedding-3-small`）で存在。payload 優先読取は anthropic 側も未実装のため対応不要 |

**推奨順**: #55 → #54 → #52（軽いフラグ撤去）→ #51（最重・3 部）→ #67 → #53

---

## Phase C — テスト移植（`tests/`）

現状 openai は `tests/__init__.py` と `test_agent_4operations.py` のみ。anthropic（≈45 ファイル/≈468 関数）から移植する。

### provider 非依存（verbatim 移植可）
- `tests/grace/`: `test_schemas` / `test_confidence` / `test_intervention` / `test_executor` / `test_planner` / `test_replan`
- `tests/chunking/test_document_chunking`
- `tests/services/`: `test_token_service` / `test_json_service` / `test_cache_service` / `test_log_service` / `test_file_service`

### provider 結合（値読み替え：`OPENAI_API_KEY`・`provider="openai"`・3072 次元・`*_openai`・コストあり）
- `tests/grace/test_config`・`conftest`・`test_executor_integration`・`test_planner_integration`・`test_grace_integration`
- `tests/services/`: `test_qdrant_service`（3072 次元・コレクション名）・`test_config_service`・`test_dataset_service`・`test_qa_service`・`test_agent_service`
- `tests/qa_generation/`: `test_semantic`・`test_smart_qa_and_persistence`
- root: `test_register_qdrant_metadata`・`test_qdrant_service_metadata`・`test_metadata_and_full_process` 等

### helpers（**新規作成**：anthropic 版はコピー不可）
- `tests/helpers/test_helper_embedding`・`test_helper_llm` を openai 既定クライアント向けに書き起こし

### 移植しない（stale / 実機前提）
- `tests/legacy/`（対象モジュール非存在）
- `tests/qa_generation/`: `test_structure` / `test_generation` / `test_content` / `test_keyword_extraction`（対象モジュール非存在）
- 実機前提: `test_collection`（実 Qdrant）・`test_agent_4operations`（既存）

> **注意**: Phase B #51 系のメタデータ round-trip テストは、**機能実装後**に移植する。

---

## Phase D — 基盤・ドキュメント

| # | 種別 | 作業 |
|---|---|---|
| CI | ci | `.github/workflows/ci.yml` を新規作成（`ruff check .` + `python -m compileall`、Python 3.13、`claude/*` の auto-merge）。テスト整備後に **pytest ジョブ**を追加（統合テストは `OPENAI_API_KEY` 等の env ゲート） |
| #75 | docs | `CLAUDE.md` に **§7 Mermaid / §8 コーディング規約 / §9 技術スタック表記** を追加（§9 表は OpenAI 用に読み替え：`OpenAI GPT` / `text-embedding-3-large` 等） |
| #76 | chore | `requirements.txt` は `uv export --format requirements-txt -o requirements.txt` 生成を維持（**`pip freeze` 禁止**）。dev 依存に pytest/ruff |
| docs | docs | `tests/README.md`（テスト一覧）と、必要に応じ `docs/openai_refactor_playbook.md` を作成 |

---

## 検証

- **静的**: 各 PR 単位で `ruff check .` ＋ `python -m compileall` を緑に。
- **実機（要サービス）**: 並列検索のスレッド安全性、二層計画の閾値、Celery 完了順／逐次永続化、
  登録メタデータ round-trip は **実 LLM/Qdrant/Redis 環境での pytest** を推奨（env ゲートで実行）。

---

## 進捗チェックリスト

### Phase A — grace/
- [ ] #58 PlannerConfig/ExecutorConfig（最優先・足場）
- [ ] #56 eval 撤去 → ast.literal_eval
- [ ] #57 _run_tool_with_timeout
- [ ] #60 _prefetch_parallel_searches
- [ ] #61 二層計画（_create_rule_based_plan）
- [ ] #64 _should_trigger_replan（検索ステップ限定）
- [ ] #65 evaluate_final（FinalEvaluationResult）
- [ ] #66 _build_confidence_factors 共通化
- [x] #59 実行ループ統合（DONE）

### Phase B — 登録・Q/A・パイプライン
- [ ] #55 make_qa_register_qdrant 死にフラグ＋--combine-rows 撤去
- [ ] #54 make_qa.py スマート生成フラグ撤去
- [ ] #52 use_smart_generation フラグ撤去（残箇所）
- [ ] #51 内容ハッシュ ID ＋重複除去＋先読み並列化（3 部）
- [ ] #67 collect_results 完了順＋on_result
- [ ] #53 JSONL 逐次永続化＋クラッシュ再開
- [x] #82 get_collection_embedding_params（DONE 相当）

### Phase C — テスト
- [ ] provider 非依存テスト verbatim 移植
- [ ] provider 結合テスト 値読み替え移植
- [ ] helpers テスト 新規作成
- [ ] tests/README.md 作成

### Phase D — 基盤・ドキュメント
- [ ] .github/workflows/ci.yml 作成
- [ ] CLAUDE.md §7/§8/§9 追加
- [ ] requirements.txt（uv export）整理確認

---

*作成日: 2026-06-14*
