---
name: grace-agent-tests
description: >-
  Fix and maintain the pytest suite in the *_grace_agent repos. Use when
  `uv run pytest` reports collection errors, failures, or warnings, when tests
  break after the Gemini→Anthropic/OpenAI migration, or when adding/guarding
  tests. Encodes the common test-debt patterns (stale patch targets, default
  drift, removed behavior, integration-test skipif, import paths, fixtures) and
  how to verify with uv.
---

# grace_agent テスト保守スキル

`uv run pytest tests/` の失敗・収集エラー・警告を直すための知見。移行（Gemini→Anthropic/OpenAI）に伴う
**テスト負債**が大半で、原則 **テストのみ修正**（本番コードは現行を正とする。疑わしければ報告）。

## 実行・検証
- 依存は `uv run` で解決可能（`uv run pytest tests/ -q`）。`pyproject.toml` に `pythonpath=["."]`。
- 修正は**ファイル単位でサブエージェント並列**＋各自 `uv run pytest <file> -q` で 0 failed/0 error を確認。
- ruff はブロッキングCIゲート → 触ったファイルは `ruff check <file>` を必ず通す。

## よくある原因と対処
1. **収集エラー（ImportError）= import パス誤り**
   - パッケージ修飾を使う: `helper.helper_llm` / `helper.helper_embedding` / `services.qdrant_service` /
     `qa_qdrant.make_qa_register_qdrant`。bare import（`from helper_llm import`）はリポ直下しか path に無く失敗。
   - `build_points_for_qdrant`/`get_collection_embedding_params` は `services.qdrant_service`。
   - 削除済みモジュール（`qa_generation.{content,generation,keyword_extraction,structure}`、`register_qdrant`）参照のテストは**廃止＝削除**（要・削除確認）。
2. **旧 patch ターゲット（移行残骸）**
   - `patch("...genai")` / `google.generativeai`（旧SDK・未インストール）→ 新SDK `google.genai`。helper_llm はモジュール直下 `genai` を持つので `helper.helper_llm.genai` を patch。
   - helper_embedding の Gemini はメソッド内 `from google import genai`→ `google.genai.Client` を patch。
   - `services.agent_service.genai`/`.QdrantClient` は廃止。現行は `create_llm_client("anthropic")`（`agent.llm`）・`get_qdrant_client()`・tool は `search_rag_knowledge_base_cached`。LLM応答は `ToolUseResponse(text, tool_calls, stop_reason, assistant_message)`。
3. **既定値ドリフト（期待値を現行へ）**
   - モデル既定 `gemini-2.0-flash`→`claude-sonnet-4-6`。
   - OpenAI埋め込み次元 1536→3072。`SemanticCoverage.embedding_model` は `text-embedding-3-large`。
   - `config_service`: env override は `ANTHROPIC_API_KEY`→`api.anthropic_api_key`（`OPENAI_API_KEY` はマップしない）。
   - ValueError メッセージ `"ANTHROPIC_API_KEY is not set"`。
4. **削除された挙動**
   - `smart_qa_generator` の2段階フォールバック廃止 → 構造化失敗時は `success=False`/空。
   - `map_collection_to_csv` は完全一致のみ（`qa_` prefix strip 廃止 → 無ければ None）。
5. **モック不備**
   - `OpenAIClient` は `response.usage` を加算 → mock で `usage=None` か `prompt_tokens/completion_tokens` を int に。
   - config 由来の数値（`config.qdrant.rag_sufficient_score`、`vectors.size`）を MagicMock のままにすると `>`/`>=` で TypeError → 実 float/int を設定。
6. **Executor v5.0（実LLM呼び出しを mock）**
   - `_is_search_result_sufficient`→True で動的フォールバック連鎖（web_search/ask_user 挿入→partial化・step数増）を抑止。
   - `_llm_calculate_step_confidence` / `_calculate_overall_confidence`（`evaluate_final`）も patch。

## 統合テストは「未起動でskip」
- Qdrant: `socket` で `QDRANT_HOST`/`QDRANT_PORT`(既定 localhost:6333) に短timeout接続できなければ `pytest.mark.skipif` でモジュールごとskip。
- 実API: `skipif(not os.getenv("ANTHROPIC_API_KEY"/"GOOGLE_API_KEY"))`。ユニットは可能なら mock 化を優先。

## その他
- 欠落フィクスチャ → `tests/<dir>/conftest.py` を追加（複数タスクで同じ conftest を触るなら read-first で追記、clobber禁止）。
- `Test*` 命名のヘルパークラス（`__init__` あり）→ `__test__ = False` で `PytestCollectionWarning` 解消。
- **既知の本番バグ候補**: `services/qdrant_service.py::get_collection_embedding_params` は埋め込みモデルを**次元数だけ**で推定し payload(`embedding_model`/`embedding_provider`)を見ない。テストは現行挙動に合わせ、必要なら別途本番改修。
