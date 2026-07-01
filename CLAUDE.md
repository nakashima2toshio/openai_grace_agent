# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ ファイル書き込みポリシー

### GitHub ブランチ操作：全許可
- ブランチへのコミット・プッシュ・PR作成・master へのマージを確認なしで実行してよい。
- 指定ブランチ以外への push や force push は事前に確認すること。

### ローカルファイル操作：作業範囲内許可
- タスクに関連するファイルの新規作成・編集は確認なしで実行してよい。
- タスクと無関係なファイルへの書き込みは事前に確認すること。
- ファイルの削除など不可逆的な操作は事前に確認すること。

## Project Overview

This is a Japanese RAG (Retrieval-Augmented Generation) Question-Answering system that implements semantic coverage analysis for evaluating Q&A datasets against documents. The system uses OpenAI embeddings and Qdrant vector database for similarity search and coverage metrics calculation.

## Development Commands

### Environment Setup
```bash
# Initial setup (installs packages and configures environment)
python setup.py

# Install dependencies
pip install -r requirements.txt

# Start Qdrant vector database
docker-compose -f docker-compose/docker-compose.yml up -d

# Register data to Qdrant
python a30_qdrant_registration.py --recreate --limit 100
```

### Running the Application
```bash
# Start the Qdrant server management script
python server.py

# Run Streamlit search UI
streamlit run a50_rag_search_local_qdrant.py

# Run example semantic coverage analysis
python example.py
```

### Code Quality
```bash
# Run ruff linter (no configuration file exists yet)
ruff check .

# Format code with ruff
ruff format .
```

## Architecture

### Core Components

1. **SemanticCoverage** (`rag_qa.py`): Main class implementing document chunking and semantic coverage calculation
   - Creates semantic chunks from documents
   - Generates embeddings for documents and Q&A pairs
   - Calculates coverage metrics using cosine similarity
   - Supports Japanese text processing with sentence boundary detection

2. **Helper Modules**:
   - `helper_api.py`: OpenAI API integration, model configuration, and cost tracking
   - `helper_rag.py`: RAG data preprocessing, configuration management (AppConfig class)
   - `helper_st.py`: Streamlit utilities for customer support FAQ processing

3. **Data Management Scripts** (a-prefixed files):
   - `a01_load_set_rag_data.py`: Load and set RAG data
   - `a02_set_vector_store_vsid.py`: Configure vector store IDs
   - `a03_rag_search_cloud_vs.py`: Search cloud vector stores
   - `a30_qdrant_registration.py`: Register data to Qdrant
   - `a35_qdrant_truncate.py`: Truncate Qdrant collections
   - `a40_show_qdrant_data.py`: Display Qdrant data
   - `a50_rag_search_local_qdrant.py`: Streamlit UI for local Qdrant search

4. **Infrastructure**:
   - `server.py`: Qdrant server health checks and startup management
   - `docker-compose/docker-compose.yml`: Containerized Qdrant deployment

### Data Flow

1. Documents are split into semantic chunks preserving sentence boundaries
2. OpenAI embeddings are generated for chunks and Q&A pairs
3. Embeddings are stored in Qdrant vector database
4. Coverage analysis compares Q&A embeddings against document chunks
5. Results are presented via Streamlit UI or API endpoints

### Model Configuration

The system supports extensive OpenAI models (configured in `config.yml`):
- GPT-4o series (including mini and audio variants)
- GPT-4.1, GPT-5 series
- O-series models (o1, o3, o4 with mini variants)
- Embedding models (text-embedding-3-small/large)

## Environment Variables

Required in `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=http://localhost:6333  # Optional, defaults to localhost
PG_CONN_STR=postgresql://...       # Optional, for PostgreSQL integration
```

## Key Implementation Details

- **Japanese Text Processing**: Uses regex patterns for Japanese sentence splitting
- **Chunking Strategy**: Semantic chunking with 200 token limit per chunk
- **Embedding Model**: Default is "text-embedding-3-large" (3072 dims, OpenAI API)
- **Coverage Threshold**: 0.8 cosine similarity for matching Q&A to chunks
- **Token Counting**: Uses tiktoken with "cl100k_base" encoding

## Dependencies

Main packages:
- `openai>=1.100.2`: API client for embeddings and chat
- `qdrant-client>=1.15.1`: Vector database client
- `streamlit>=1.48.1`: Web UI framework
- `fastapi>=0.115.6`: API server framework
- `tiktoken`: Token counting for chunk size management
- `scikit-learn`: Cosine similarity calculations

## Important Notes

- No formal test suite exists - consider adding pytest when implementing new features
- The codebase uses Japanese variable names and comments in some places
- Old implementations are archived in the `old_code/` directory
- Qdrant must be running before using any data registration or search functionality

---

# プロジェクト固有規約

## 7. Mermaidダイアグラム スタイル規約

### 7.1 構文バージョン

**PyCharm Pro v9 互換構文**を使用する。

- ノードラベルにバッククォートや markdown文字列（`` `text` ``）を使用しない
- 特殊文字を含むノードラベルは必ずダブルクォート（`"..."` ）で囲む

### 7.2 カラーテーマ（黒背景・白文字）— **必須**

すべてのMermaidダイアグラムに以下のスタイルを適用すること。

| 要素 | 設定値 |
|---|---|
| ノード背景色 | `fill:#000` |
| ノードテキスト色 | `color:#fff` |
| ノード枠線色 | `stroke:#fff` |
| サブグラフ背景色 | `fill:#1a1a1a` |
| サブグラフテキスト色 | `color:#fff` |
| サブグラフ枠線色 | `stroke:#fff` |

### 7.3 flowchart / graph 図の実装パターン

```
flowchart TB
    subgraph Layer["レイヤー名"]
        NodeA["ノードA"]
        NodeB["ノードB"]
    end
    NodeA --> NodeB
classDef default fill:#000,stroke:#fff,color:#fff
classDef subgraphStyle fill:#1a1a1a,stroke:#fff,color:#fff
class NodeA,NodeB default
style Layer fill:#1a1a1a,stroke:#fff,color:#fff
```

**必須ルール:**

1. `classDef default fill:#000,stroke:#fff,color:#fff` を必ずブロック末尾に追加する
2. `classDef subgraphStyle fill:#1a1a1a,stroke:#fff,color:#fff` を追加する
3. 全ノードに `class <node_ids> default` を付与する
4. 全サブグラフに `style <subgraph_name> fill:#1a1a1a,stroke:#fff,color:#fff` を付与する
5. 既存の `style`/`classDef`/`class` 行は重複しないよう整理する

### 7.4 sequenceDiagram 図の実装パターン

```
%%{ init: { "theme": "base", "themeVariables": {
  "background": "#000000", "mainBkg": "#000000",
  "textColor": "#ffffff", "lineColor": "#ffffff",
  "actorBkg": "#000000", "actorTextColor": "#ffffff",
  "actorLineColor": "#ffffff", "noteBkg": "#1a1a1a",
  "noteTextColor": "#ffffff" } } }%%
sequenceDiagram
    participant A as "参加者A"
    A->>B: メッセージ
```

**必須ルール:**

- `sequenceDiagram` の前に必ず `%%{ init: ... }%%` ヘッダーを挿入する
- `classDef` / `class` 行は `sequenceDiagram` では使用しない（非対応）

---

## 8. コーディング規約

### 8.1 型ヒント

```python
# ❌ 誤り
def func(callback: Optional[callable] = None): ...

# ✅ 正しい
from typing import Optional, Callable
def func(callback: Optional[Callable] = None): ...
```

### 8.2 Streamlit DataFrame

```python
# ❌ 誤り（TypeError: 'str' cannot be interpreted as an integer）
st.dataframe(df, width='stretch')

# ✅ 正しい
st.dataframe(df, use_container_width=True)
```

### 8.3 出力ファイル命名（チャンク分割）

```bash
# ✅ デフォルト: 固定ファイル名（後続バッチとの連携のため）
cc_news_1per.csv  →  output_chunked/cc_news_1per_chunks.csv

# タイムスタンプが必要な場合は --timestamp オプションで明示指定
python -m chunking.csv_text_to_chunks_text_csv \
  --input-file OUTPUT/cc_news_1per.csv \
  --output output_chunked \
  --timestamp   # ← これがある場合のみ日時サフィックスを付与
```

---

## 9. ドキュメント規約

### 9.1 技術スタック表記の統一

コードサンプル・説明文・Mermaid図内の表記を以下に統一すること。

| 用途 | ✅ 正しい表記 | ❌ 禁止表記 |
|---|---|---|
| LLM全般 | `OpenAI GPT` / `GPT` | `Anthropic Claude`, `Gemini`, `claude-sonnet-4-6` |
| デフォルトモデル | `gpt-5-mini` | `claude-sonnet-4-6`, `gpt-5.4-mini`, `gemini-3-flash-preview` |
| Embedding | `OpenAI Embedding` / `text-embedding-3-large`（3072次元） | `gemini-embedding-001`, `claude` 系 |
| LLM/Embeddingクライアント | `create_llm_client("openai")` / `create_embedding_client("openai")` | `"anthropic"` / `"gemini"` |
| LLM用APIキー | `OPENAI_API_KEY` | `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`（LLM用途） |
| Qdrantコレクション | `*_openai` | `*_anthropic`, `*_gemini` |

### 9.2 ドキュメント一覧

| ファイル | 内容 |
|---|---|
| `README.md` | プロジェクト全体・GRACE自律エージェント詳細 |
| `readme_make_env.md` | Mac向け環境構築手順 |
| `readme_usage_tools.md` | チャンク作成・Q&A生成・Qdrant登録の操作手順 |
| `readme_rag.md` | RAGパイプライン設計・クラス・関数 IPO詳細 |
| `readme_react_reflection.md` | ReAct+Reflectionエージェントの設計と実装 |
| `readme_autonomous_agent.md` | GRACEアーキテクチャ（Plan+Executor）詳細 |
| `docs/openai_refactor_todo_june14.md` | リファクタリング横展開 TODO（進捗チェックリスト） |
| `docs/openai_anthropic_file_comparison_june14.md` | openai/anthropic 実ファイル比較表 |

---

# ⚠️ CRITICAL RULES - MUST READ BEFORE ANY MODIFICATION ⚠️

## 1. OpenAI Model Names - NEVER CREATE MAPPINGS

**ALL these models are REAL and VALID:**
- `gpt-5-nano`, `gpt-5-mini`, `gpt-5` ← Real GPT-5 series models
- `gpt-4.1`, `gpt-4.1-mini` ← Real GPT-4.1 models
- `o3`, `o3-mini`, `o4`, `o4-mini` ← Real O-series models

**❌ NEVER create model name mappings like:**
```python
MODEL_MAPPING = {"gpt-5-nano": "gpt-4o-mini"}  # ← WRONG! DO NOT DO THIS
```

**✅ Use model names directly as they are defined in `helper_rag.py` lines 28-87**

---

## 2. OpenAI API Methods - TWO CORRECT APIS

**Both APIs exist and are correct:**

### Structured Outputs API (Recommended for Q/A generation)
```python
response = client.responses.parse(
    input=combined_input,
    model=model,
    text_format=QAPairsResponse,  # Pydantic model for type-safe output
    max_output_tokens=1000
)
```
- **Purpose**: Type-safe output with Pydantic models
- **Used in**: `celery_tasks.py:202-207, 429-434`
- **Documentation**: `doc/helper_api.md` line 14

### Responses API (Standard text generation)
```python
response = client.responses.create(
    input=input_messages,
    model=model,
    max_output_tokens=1000
)
```
- **Purpose**: Standard text generation
- **Used in**: `helper_api.py:743`
- **Documentation**: `doc/helper_api.md` line 12

**⚠️ Both `.parse()` and `.create()` are CORRECT - use according to purpose**

---

## 3. Mandatory Verification Before Changes

**BEFORE modifying any OpenAI API code, you MUST:**

1. ✅ Read `doc/helper_api.md` (complete API documentation)
2. ✅ Read `helper_api.py` lines 715-758 (actual implementations)
3. ✅ Read `helper_rag.py` lines 28-87 (model list)
4. ✅ Ask yourself: "What does the documentation say?"
5. ✅ If uncertain → **ASK THE USER FIRST**

---

## 4. Common Mistakes to AVOID

**❌ Mistake 1: Assuming model names are wrong**
```
Wrong: "gpt-5-nano returns error, so it must not exist"
Truth: gpt-5-nano IS a real model - investigate the ACTUAL error cause
```

**❌ Mistake 2: Confusing parse() and create()**
```
Wrong: "responses.parse() doesn't exist, I should use create()"
Truth: BOTH exist - parse() is for structured output, create() for text
```

**❌ Mistake 3: "Helpful" mappings**
```
Wrong: "I'll create a mapping to translate old models to new ones"
Truth: Models are already correct - DO NOT create mappings
```

---

## 5. Emergency Checklist

**Before committing changes, verify:**

- [ ] Did I create a MODEL_MAPPING? (If YES → DELETE IT)
- [ ] Did I change `responses.parse()` to `responses.create()`? (If YES → REVERT)
- [ ] Did I read `doc/helper_api.md`? (If NO → READ IT NOW)
- [ ] Am I certain this is correct? (If NO → ASK USER)

---

## 6. When Errors Occur

**If you see "model not found" or "API error":**

1. ❌ The error is NOT because model name is wrong
2. ❌ The error is NOT because API method is wrong
3. ✅ Check: API key, network, Celery workers, Redis connection
4. ✅ Check: Actual error message and stack trace
5. ✅ **NEVER "fix" model names or API method names as a first response**