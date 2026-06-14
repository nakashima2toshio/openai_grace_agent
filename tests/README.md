# テストスイート索引（openai_grace_agent）

日本語 RAG Q&A システムの pytest テスト群。LLM は **OpenAI GPT**（デフォルト
`gpt-5-mini`）、Embedding は **OpenAI text-embedding-3-large（3072次元）** を前提とする。

## 実行方法

```bash
# 全テスト
uv run pytest tests/

# ディレクトリ単位
uv run pytest tests/qa_generation/ -v
uv run pytest tests/helpers/ -v
uv run pytest tests/grace/ -v

# 単一ファイル / 単一テスト
uv run pytest tests/helpers/test_helper_llm.py -v
uv run pytest tests/qa_generation/test_semantic.py::TestSemanticCoverage::test_init -v
```

> 依存: `pytest`, `pydantic`, `numpy`, `pandas`, `tiktoken`, `openai`。
> 大半のテストは OpenAI SDK / Qdrant を `unittest.mock` でモックするため、
> API キーや稼働中の Qdrant なしで実行できる。

## ディレクトリ構成

| ディレクトリ | 内容 |
|---|---|
| `tests/grace/` | GRACE 自律エージェント（Plan + Executor、confidence、replan、intervention、schemas）の単体・統合テスト |
| `tests/services/` | サービス層（cache / config / dataset / file / json / log / qa / token）の単体テスト |
| `tests/qa_generation/` | Q/A 生成パイプライン（semantic / evaluation / SmartQAGenerator + 逐次永続化）のテスト |
| `tests/chunking/` | ドキュメントチャンキングのテスト |
| `tests/helpers/` | プロバイダー抽象化レイヤー（`helper/helper_llm.py` / `helper/helper_embedding.py`）の OpenAI クライアントテスト |
| `tests/*.py` | トップレベル: `test_agent_4operations.py`（エージェント4操作の結合テスト）など |
| `conftest.py` | テスト件数のカスタム出力フック |

### `tests/qa_generation/`

| ファイル | 対象 / 備考 |
|---|---|
| `test_semantic.py` | `qa_generation.semantic.SemanticCoverage`。埋め込み次元 3072・デフォルトモデル `text-embedding-3-large` を検証。クライアントは全モック |
| `test_evaluation.py` | `qa_generation.evaluation.analyze_coverage`。`SemanticCoverage` をモックするためプロバイダー非依存 |
| `test_smart_qa_and_persistence.py` | `SmartQAGenerator`（analyze → generate の2段階 `generate_content` 方式）と `QAPipeline` の逐次永続化・再開 |

### `tests/helpers/`

| ファイル | 対象 / 備考 |
|---|---|
| `test_helper_llm.py` | `create_llm_client("openai")` / `OpenAIClient`（`generate_content` / `generate_structured` / `count_tokens`）、トークン集計（`reset_token_counter` / `get_token_counter` / `_token_accumulator`）。OpenAI SDK はモック |
| `test_helper_embedding.py` | `create_embedding_client("openai")` / `OpenAIEmbedding`、次元 3072（`text-embedding-3-large`）。OpenAI SDK はモック |

## プロバイダー適応メモ（OpenAI）

このリポジトリは anthropic_grace_agent からの移植であり、テストは以下の OpenAI 値に
読み替えてある。

| 項目 | 値 |
|---|---|
| LLM プロバイダー / デフォルトモデル | `"openai"` / `gpt-5-mini` |
| LLM クライアント生成 | `create_llm_client("openai")` |
| 構造化出力 | `client.beta.chat.completions.parse(response_format=...)` |
| Embedding モデル / 次元 | `text-embedding-3-large` / 3072 |
| Embedding クライアント生成 | `create_embedding_client("openai")` |
| API キー | `OPENAI_API_KEY` |
| Qdrant コレクション | `*_openai` |
| コスト計算 | あり（`LLM_PRICING` / `EMBEDDING_PRICING`） |

### 移植時にスキップ / 書き換えたテスト

anthropic 版に存在したが、openai では対象モジュール / 実装が異なるため移植していない:

- `test_structure.py` / `test_generation.py` / `test_content.py` /
  `test_keyword_extraction.py`: 対象モジュールが openai の `qa_generation/` に存在しない（stale）。
- anthropic `test_smart_qa_and_persistence.py` の `TestSmartQAGenerator`（`generate_structured` /
  `SmartQAResult` / `SmartQAPair` による単一構造化呼び出し）と `TestManifestValidation`
  （`pipeline.load_data()` の manifest 検証）: openai 実装には該当機能がないため、
  前者は openai の2段階 `generate_content` 方式に書き換え、後者は移植せず。

## 環境変数でゲートされるテスト

| ゲート | 対象 | 挙動 |
|---|---|---|
| `OPENAI_API_KEY` 未設定 | `tests/helpers/test_helper_llm.py::TestOpenAIClientRealAPI` | `@pytest.mark.skipif` で実 API テストをスキップ |
| 実 Qdrant 稼働 | services / 結合系で実 Qdrant を参照するテスト | Qdrant 未稼働時はスキップ / モックで代替 |

> `tests/helpers/` と `tests/qa_generation/` の通常テストはすべて SDK をモックするため、
> API キー・Qdrant なしで完走する。
