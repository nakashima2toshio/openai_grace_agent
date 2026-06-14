# openai ⇄ anthropic 実ファイル比較表（2026-06-14）

`openai_grace_agent` と `anthropic_grace_agent` の**全コードファイルを実ファイル同士で diff 比較**した結果表。
diff 行数の多い順。判定は provider 差（モデル名・APIキー・クライアント・次元）を除外した
**真の漏れ**の有無で行う。

判定凡例:
- ⛔ **GAP** … 移植すべき真のロジック漏れ（Phase/項番を併記）
- ⚪ **provider差/同等** … provider 値差・構造等価・既実装（移植不要）
- 🎨 **cosmetic** … docstring/typo/整形のみ（任意）
- 📦 **NEW** … anthropic のみ存在（新規移植）
- 🧪 **TEST** … テストファイル（Phase C）

---

## 1. 共通ファイル（diff あり）

| diff行 | ファイル | 判定 | 内容 |
|---:|---|---|---|
| 3865 | `helper/helper_rag_qa.py` | ⚪ provider noise | anthropic=後方互換ラッパー / openai=LLM抽象再編。Q/Aロジックは双方 `qa_generation/` に存在 |
| 1075 | `grace/executor.py` | ⛔ GAP | A:#56 eval撤去 / #57 timeout / #60 並列プリフェッチ / #64 _should_trigger_replan / #65 evaluate_final連携 / #66 _build_confidence_factors（#59 実行ループ統合は DONE） |
| 970 | `chunking/csv_text_to_chunks_text_csv.py` | ⛔ GAP | E1 Embedding上限連携 / E2 ルールベース継続 / E3 doc_id（timestamp固定名は DONE） |
| 604 | `qa_qdrant/make_qa_register_qdrant.py` | ⛔ GAP | B:#55 死にフラグ＋`--combine-rows` 撤去（他は provider差） |
| 417 | `qa_generation/smart_qa_generator.py` | ⛔ GAP(軽) | B:#52 残存 `use_smart_generation` フラグ撤去（単段化は DONE・他 provider差） |
| 348 | `helper/helper_llm.py` | ⛔ GAP | F1 トークンアキュムレータ（reset/get_token_counter）（F3 ToolUseResponse/TypeVar は任意・他 provider差） |
| 318 | `grace/planner.py` | ⛔ GAP | A:#61 二層計画（ルールベース計画）＋ G3 gpt-5.4-mini |
| 262 | `qa_generation/pipeline.py` | ⛔ GAP | B:#53 JSONL逐次永続化＋クラッシュ再開 / #52 フラグ撤去 |
| 258 | `tests/test_agent_4operations.py` | ⚪ provider差 | provider 値（"openai"）読み替えのみ。両 repo 存在 |
| 223 | `chunking/async_api_client.py` | ⛔ GAP | E4 トークン使用量集計（他 provider差） |
| 219 | `celery_tasks.py` | ⛔ GAP | B:#67 collect_results 完了順＋on_result / #52 フラグ撤去 |
| 171 | `grace/benchmark.py` | 🎨 cosmetic | docstring/typo/整形のみ。ロジック同一 |
| 160 | `services/qdrant_service.py` | ⛔ GAP | B:#51 内容ハッシュ point ID＋payloadデフォルト（#82 取得は DONE） |
| 154 | `qa_qdrant/register_to_qdrant.py` | ⛔ GAP | B:#51 重複Q/A除去＋先読み並列化 |
| 136 | `services/agent_service.py` | ⚪ provider差 | assistant_message 再構築は **OpenAI 形式が正**。gap ではない |
| 129 | `grace/__init__.py` | ⚪ 同等 | export 一覧差（#58 config 追加後に要更新） |
| 127 | `grace/confidence.py` | ⛔ GAP | A:#65 FinalEvaluationResult＋evaluate_final（他 provider差・G3） |
| 123 | `services/__init__.py` | ⚪ 同等 | import/export 順差のみ |
| 96 | `grace/tools.py` | ⚪ provider差 | 検索/ツール provider 差中心 |
| 89 | `agent_main.py` | ⚪ provider差 | legacy エージェント provider 差 |
| 77 | `helper/helper_api.py` | ⚪ provider差 | コスト/モデル設定 provider 差 |
| 70 | `down_load_non_qa_rag_data_from_huggingface.py` | ⚪ 同等 | `use_container_width` 修正済み・他 provider差 |
| 67 | `ui/pages/grace_chat_page.py` | ⚪ provider差 | 表示文言・モデル名 |
| 54 | `qa_qdrant/make_qa.py` | ⛔ GAP(軽) | B:#54 スマート生成フラグ撤去 |
| 49 | `agent_rag.py` | ⚪ provider差 | legacy provider 差 |
| 48 | `grace/config.py` | ⛔ GAP | A:#58 PlannerConfig/ExecutorConfig＋G3 |
| 45 | `celery_config.py` | ⚪ provider差 | キュー/モデル設定 |
| 39 | `qdrant_client_wrapper.py` | ⚪ provider差 | コレクション接尾辞 |
| 38 | `chunking/__init__.py` | ⚪ 同等 | export 差（E実装後に要更新） |
| 36 | `services/config_service.py` | ⛔ GAP(軽) | G3 gpt-5.4-mini → gpt-5-mini |
| 36 | `config.py` | ⛔ GAP(軽) | G3 gpt-5.4-mini → gpt-5-mini（L416/425） |
| 32 | `ui/pages/qdrant_search_page.py` | ⛔ GAP(軽) | G3 gpt-5.4-mini（L418） |
| 32 | `agent_tools.py` | ⚪ provider差 | legacy ツール |
| 30 | `ui/pages/qdrant_registration_page.py` | ⚪ provider差 | 表示・モデル名 |
| 24 | `qa_generation/semantic.py` | ⚪ provider差 | Embedding 次元/モデル |
| 23 | `ui/pages/agent_chat_page.py` | ⚪ provider差 | 表示・モデル名 |
| 22 | `helper/helper_embedding_fastembed.py` | ⚪ provider差 | Embedding provider |
| 20 | `helper/helper_rag.py` | ⛔ GAP(軽) | F2 save_files_to_output(output_name=) |
| 16 | `ui/pages/download_page.py` | ⚪ provider差 | 表示 |
| 15 | `ui/pages/qa_generation_page.py` | ⚪ provider差 | モデル名 |
| 15 | `tools/rag_data_preprocessor_app.py` | ⚪ provider差 | 表示 |
| 14 | `qa_generation/__init__.py` | ⚪ 同等 | export 差 |
| 13 | `helper/helper_rag_ui.py` | ⚪ provider差 | 表示 |
| 13 | `grace/intervention.py` | ⚪ 同等 | 文言/軽微 |
| 12 | `helper/helper_embedding.py` | ⚪ provider差 | Embedding provider |
| 12 | `grace/replan.py` | ⛔ GAP(軽) | A:#66 出力型 Optional[Any] 拡張 |
| 11 | `ui/components/grace_components.py` | ⚪ provider差 | 表示 |
| 10 | `services/qa_service.py` | ⚪ 同等 | 軽微 |
| 10 | `qa_generation/data_io.py` | ⚪ 同等 | 軽微 |
| 10 | `regex_mecab.py` / `agent_parallel_search.py` / `ui/pages/__init__.py` / `ui/components/rag_components.py` | ⚪ 同等/provider差 | 軽微 |
| 9 | `ui/pages/qdrant_show_page.py` | ⚪ provider差 | 表示 |
| 8 | `grace/schemas.py` | ⛔ GAP(軽) | A:#66 出力型 Optional[Any] 拡張 |
| 8 | `helper/helper_text.py` / `ui/components/__init__.py` | ⚪ 同等 | 軽微 |
| ≤7 | `services/log_service.py`(G4 未使用import) ほか多数 | ⚪ 同等/cosmetic | 軽微・provider差 |

---

## 2. anthropic のみ存在（NEW / 移植判定）

| ファイル | 判定 | 対応 |
|---|---|---|
| `qdrant_delete_collection.py` | 📦 NEW (G1) | コレクション削除 CLI を移植（接尾辞 `*_openai`） |
| `ui/pages/benchmark_page.py` | 📦 NEW (G2) | ベンチマーク UI ページを移植 |
| `run_benchmark.py` | ⚪ SKIP | openai は `run_benchmark.sh`/`_all.sh` で代替済み |
| `grace/check_code/test_planner*.py` | ⚪ SKIP | 開発用スクラッチ |
| `tests/**`（grace/services/qa_generation/chunking/helpers/root） | 🧪 TEST (Phase C) | provider 非依存=verbatim / 結合=値読み替え / helpers=新規作成 / legacy・stale=移植しない |

---

## 3. 改修対象サマリ（実ファイル比較 → 実施順）

1. **G3** gpt-5-mini 統一（`config.py`/`config_service.py`/`grace/config.py`/`ui/pages/qdrant_search_page.py` 他）— 安全・確定
2. **Phase A** grace: #58 → #56 → #57 → #66 → #64 → #65 → #61 → #60
3. **Phase B** 登録Q/Aパイプライン: #55 → #54 → #52 → #51 → #67 → #53
4. **Phase E** chunking: E1 → E2 → E4 → E3
5. **Phase F** helper: F1 → F2
6. **Phase G** 不足ファイル: G1 → G2 → G4
7. **Phase C/D** テスト・CI・CLAUDE.md §7/§8/§9

---

*作成日: 2026-06-14 / 基準: 実フォルダー・実ファイル比較*
