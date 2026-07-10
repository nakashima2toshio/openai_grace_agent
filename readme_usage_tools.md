# Agent RAG ツール使用ガイド

**Version 2.0** | 最終更新: 2026-07-10

チャンク作成 → Q/A生成 → Qdrant登録 → Agent検索 の一連の RAG パイプラインを操作するためのガイドです。
LLM は OpenAI GPT（既定 `gpt-5-mini`）、Embedding は `text-embedding-3-large`（3072次元）を使用します。

---

## 0. 環境の起動・設定

アプリケーションを使用する前に、以下のサービスを起動してください。

### 0.1 Docker コンテナの起動（Qdrant + Redis）

compose ファイルはリポジトリ直下ではなく `docker-compose/docker-compose.yml` にあります。

```bash
docker compose -f docker-compose/docker-compose.yml up -d
```

起動確認:

```bash
# Qdrant ヘルスチェック
curl http://localhost:6333/health

# Redis 接続確認
docker compose -f docker-compose/docker-compose.yml exec redis redis-cli ping
# → PONG が返れば OK
```

### 0.2 環境変数の確認

`.env` ファイルに API キーが設定されていることを確認:

```bash
# 必須（LLM・Embedding とも OpenAI API を使用）
OPENAI_API_KEY=your_openai_api_key

# オプション（Rerank 使用時。未設定でも動作する）
COHERE_API_KEY=your_cohere_api_key

# オプション（既定値のままなら設定不要）
REDIS_URL=redis://localhost:6379/0   # Celery ブローカー / 結果保存
LLM_PROVIDER=openai                  # helper/helper_llm.py の既定
EMBEDDING_PROVIDER=openai            # helper/helper_embedding.py の既定
```

Qdrant の接続先は `http://localhost:6333` 固定です（`config.py` の `QdrantConfig.URL`。
GRACE エージェント側は `grace/config.py` の `QdrantConfig` / `GRACE_*` 環境変数で上書き可能）。

### 0.3 Celery ワーカーの起動

Q/A 生成ツールで Celery 並列処理（`--use-celery`）を使う場合に必要です。
チャンク作成ツール（ツール1）は Celery を使用しません（asyncio 並列）。

```bash
# 起動（推奨: concurrency=8 + Flower モニタリング）
./start_celery.sh restart -c 8 --flower

# 状態確認
./start_celery.sh status

# 停止
./start_celery.sh stop
```

`start_celery.sh` の書式: `./start_celery.sh {start|stop|restart|status} [-c concurrency] [--flower] [--flower-port PORT]`
（`-w, --workers` は `-c` の後方互換エイリアス）

Flower（タスクモニタリング UI）: http://localhost:5555

※ `start_workers.sh` は優先度別キュー（high/normal/low）で 3 ワーカーを起動する旧方式のスクリプトです。通常は `start_celery.sh` を使用してください。

### 0.4 （任意）元データの準備

HuggingFace から非 Q&A 型データ（wikipedia_ja / cc_news 等）をダウンロードして `OUTPUT/*.csv` を作成する場合:

```bash
uv run streamlit run down_load_non_qa_rag_data_from_huggingface.py
```

（Streamlit UI 上でデータセットを選択してダウンロード・前処理を実行します）

---

## 1. クイックスタート（簡易版コマンド集）

3 つのツールを順に実行することで、テキストデータ → チャンク → Q/A ペア → Qdrant 登録 → Agent 検索 という一連の RAG パイプラインが完成します。

### ツール 1: チャンク作成

```bash
python -m chunking.csv_text_to_chunks_text_csv \
  --input-file OUTPUT/wikipedia_ja_1per.csv \
  --output output_chunked \
  --model gpt-5-mini \
  --workers 8
```

出力は既定で固定ファイル名（`output_chunked/wikipedia_ja_1per_chunks.csv`）です。

### ツール 2: Q/A ペア作成 + Qdrant 登録

```bash
# Celery 起動後に実行
./start_celery.sh restart -c 8 --flower

python qa_qdrant/make_qa_register_qdrant.py \
  --input-file output_chunked/wikipedia_ja_1per_chunks.csv \
  --collection wikipedia_ja_1per \
  --use-celery \
  --concurrency 8 \
  --recreate
```

### ツール 3: Agent 検索（Web UI）

```bash
uv run streamlit run agent_rag.py --server.port 8501
```

ブラウザで http://localhost:8501 にアクセス。

---

## 2. ツール詳細: チャンク作成

**スクリプト:** `chunking/csv_text_to_chunks_text_csv.py`（実行は `python -m chunking.csv_text_to_chunks_text_csv`）

LLM ベースのセマンティックチャンキングツールです。テキストを意味的なまとまりに 3 段階で分割します。Celery は使用しません（asyncio による並列処理）。

### 2.1 処理フロー

```
入力テキスト/CSV（doc_id 単位で文書境界を保持）
  ↓ Step 1: 階層構造化（段落分割）
  ↓ Step 2: 意味的チャンキング
  ↓ Step 3: 文脈連続性チェック（--continuity-mode: rule/llm/off）
チャンク CSV（メタデータ付き + シンプル版 + manifest の 3 ファイル出力）
```

### 2.2 実行コマンド

CSV ファイルからチャンク作成:

```bash
python -m chunking.csv_text_to_chunks_text_csv \
  --input-file OUTPUT/cc_news_1per.csv \
  --output output_chunked \
  --model gpt-5-mini \
  --workers 8 \
  --block-size 500
```

テキストファイルからチャンク作成:

```bash
python -m chunking.csv_text_to_chunks_text_csv \
  --input-file ./data/document.txt \
  --output output_chunked \
  --model gpt-5-mini \
  --workers 8
```

### 2.3 オプション一覧

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--input-file` | （必須） | 入力ファイル（.txt / .csv） |
| `--output` | `chunks_output` | 出力ディレクトリ |
| `--timestamp` | `false` | 出力ファイル名に日時サフィックスを付与（既定は固定ファイル名） |
| `--model` | `gpt-5-mini` | 使用する LLM モデル（OpenAI 以外のモデル名を指定するとエラー） |
| `--workers` | `8` | 並列ワーカー数（asyncio） |
| `--block-size` | `1000` | ブロックサイズ（文字数）。大きすぎると MAX_TOKENS エラー |
| `--max-chunk-tokens` | `512` | チャンクの最大トークン数（Step3 の結合上限かつ強制分割上限。Embedding 入力上限 2048 未満を推奨） |
| `--continuity-mode` | `rule` | Step3 のモード。`rule`: ルールベース判定（LLM 呼び出しなし・高速）/ `llm`: LLM ペア判定 / `off`: 結合しない |
| `--text-column` | 自動検出 | CSV のテキストカラム名 |
| `--max-rows` | 全行 | 最大処理行数（CSV 用） |
| `--combine-rows` | `false` | CSV 全行を 1 テキストに結合 |
| `--resume` | なし | チェックポイントから再開するジョブ ID |
| `--verbose` | `false` | 詳細ログ出力 |

### 2.4 出力ファイル

既定では**固定ファイル名**で 3 ファイルが生成されます（後続バッチとの連携のため）:

| ファイル | 内容 |
|---------|------|
| `{入力名}_chunks.csv` | メタデータ付き（chunk_id, text, tokens, sentence_count, doc_id 等） ← ツール 2 の入力 |
| `{入力名}_chunks_simple.csv` | シンプル版（Text カラムのみ） |
| `{入力名}_chunks.manifest.json` | 出力契約の manifest（後続ステージとの契約明示） |

`--timestamp` を指定した場合のみ `{入力名}_chunks_20260505_004819.csv` のように日時サフィックスが付与されます。

```bash
# 例: 固定ファイル名（デフォルト）
OUTPUT/cc_news_1per.csv → output_chunked/cc_news_1per_chunks.csv
```

### 2.5 CSV テキストカラムの自動検出

`--text-column` を省略した場合、以下の候補から自動検出します:
`text` / `Text` / `TEXT`, `content` / `Content` / `CONTENT`, `Combined_Text` / `combined_text`, `body` / `Body` / `BODY`, `document` / `Document`, `answer` / `Answer`

### 2.6 注意事項

OpenAI API のレート制限に引っかかる場合は、`--block-size` を小さく（例: 500）、`--workers` を減らして（例: 4）調整してください。

---

## 3. ツール詳細: Q/A ペア作成 + Qdrant 登録

**スクリプト:** `qa_qdrant/make_qa_register_qdrant.py`

チャンク済み CSV から Q/A ペアを LLM で自動生成し、Qdrant に登録する統合ツールです。
**チャンキング機能は本ツールから分離済み**です。テキスト（.txt）を直接入力することはできません。先にツール 1 でチャンク化してください。

### 3.1 処理フロー

```
入力（チャンク済み CSV / question・answer 付き CSV）
  ↓ Phase 1: Q/A ペア生成（SmartQAGenerator: 構造化出力 1 回/チャンク・Celery 並列対応）
  ↓ Phase 2: Qdrant 登録
  ↓   ・Embedding 生成（text-embedding-3-large, 3072 次元）
  ↓   ・コレクション作成 + バッチアップサート
Q/A ペア CSV + Qdrant 登録完了
```

### 3.2 実行コマンド

基本（Celery 使用 + コレクション再作成）:

```bash
./start_celery.sh restart -c 8 --flower

python qa_qdrant/make_qa_register_qdrant.py \
  --input-file output_chunked/cc_news_1per_chunks.csv \
  --collection cc_news_1per \
  --use-celery \
  --concurrency 8 \
  --recreate
```

Celery を使わない同期処理:

```bash
python qa_qdrant/make_qa_register_qdrant.py \
  --input-file output_chunked/cc_news_1per_chunks.csv \
  --collection cc_news_1per \
  --recreate
```

並列数を 4 に指定（軽量モード）:

```bash
python qa_qdrant/make_qa_register_qdrant.py \
  --input-file output_chunked/cc_news_5per_chunks.csv \
  --collection cc_news_5per \
  --use-celery \
  -c 4 \
  --recreate
```

### 3.3 オプション一覧

**入力ソース（いずれか 1 つ必須）:**

| オプション | 説明 |
|-----------|------|
| `--input-file` | チャンク済み CSV または question/answer 付き CSV（.csv のみ） |
| `--dataset` | 事前定義データセット名（`wikipedia_ja`, `wikipedia_ja_5per`, `japanese_text`, `fineweb_edu_ja`, `cc_news`, `livedoor`） |

**CSV 処理（`--input-file` が CSV の場合）:**

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--text-column` | `text` | テキストカラム名 |

**Q/A 生成:**

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--model` | `gpt-5-mini` | LLM モデル |
| `--use-celery` | `false` | Celery 並列処理を使用 |
| `-c`, `--concurrency` | `8` | 並列タスク数。`start_celery.sh -c` と同じ値を推奨 |
| `--celery-workers` | `1` | （非推奨）ワーカープロセス数チェック用。`--concurrency` を使用 |
| `--batch-chunks` | `3` | 1 API 呼び出しで処理するチャンク数 |
| `--max-docs` | 全件 | 処理する最大文書数 |

**Qdrant 登録:**

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--collection` | （必須） | Qdrant コレクション名 |
| `--recreate` | `false` | コレクションを再作成（既存データ削除） |
| `--batch-size` | `100` | Embedding バッチサイズ |
| `--provider` | `openai` | Embedding プロバイダー |

**出力:**

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--output` | `qa_output/pipeline` | Q/A ペア CSV 出力ディレクトリ |
| `--ui-output` | `qa_output` | UI 用正規化 CSV 出力ディレクトリ |

### 3.4 入力 CSV の自動判定

入力 CSV のカラム構成に応じて処理が自動切替されます:

| CSV のカラム | 動作 |
|------------|------|
| `question` + `answer` あり | Q/A 生成をスキップ → 直接 Qdrant 登録 |
| `--text-column`（既定 `text`）または `Combined_Text` あり | Q/A 生成 → Qdrant 登録 |
| 上記いずれもなし | エラー終了 |

### 3.5 出力ファイル

| 出力先 | 内容 |
|-------|------|
| `qa_output/pipeline/` | Q/A ペア CSV（question, answer カラム） |
| `qa_output/` | UI 用正規化 CSV（日時サフィックス除去済み） |

### 3.6 Celery 並列設定の推奨値

| マシン | `--concurrency` | `start_celery.sh -c` |
|-------|-----------------|----------------------|
| M2 MacBook Air (8 vCPU) | `8` | `8` |
| 軽量モード | `4` | `4` |

`--concurrency` と `start_celery.sh -c` は同じ値に揃えてください。

### 3.7 補助ツール: 登録のみ / 生成のみ

#### 登録のみ: `qa_qdrant/register_to_qdrant.py`

既に Q/A ペア CSV（または任意のテキスト CSV）がある場合、登録だけを実行できます。

```bash
python qa_qdrant/register_to_qdrant.py \
  --input-file qa_output/pipeline/cc_news_1per_qa.csv \
  --collection cc_news_1per \
  --recreate
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--input-file` | （必須） | 登録する CSV ファイル |
| `--collection` | （必須） | 登録先コレクション名 |
| `--recreate` | `false` | 既存の同名コレクションを削除して作り直す |
| `--batch-size` | `100` | 1 回の Embedding API 呼び出し/登録で扱う件数 |
| `--embed-workers` | `2` | Embedding 先読みの並列スレッド数（登録をパイプライン化） |
| `--text-col` | 自動検出 | ベクトル化対象のカラム名 |
| `--provider` | `openai` | Embedding プロバイダー（既定: `openai`） |
| `--domain` | コレクション名 | ペイロードの `domain` フィールド値 |
| `--max-docs` | 全件 | 登録する最大ドキュメント数（テスト用） |
| `--normalize-filename` / `--no-normalize-filename` | 有効 | ファイル名から日時サフィックスを除去 |
| `--create-ui-csv` / `--no-create-ui-csv` | 有効 | UI 用正規化 CSV を生成 |
| `--ui-output-dir` | `qa_output` | UI 用 CSV の出力ディレクトリ |

#### 生成のみ: `qa_qdrant/make_qa.py`

Qdrant 登録を行わず Q/A ペア CSV の生成だけを実行します（チャンク済み CSV 専用）。

```bash
python qa_qdrant/make_qa.py \
  --input-file output_chunked/cc_news_1per_chunks.csv \
  --use-celery \
  -c 8 \
  --analyze-coverage
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--input-file` / `--dataset` | （いずれか必須・排他） | チャンク済み CSV / 事前定義データセット名 |
| `--model` | `gpt-5-mini` | LLM モデル |
| `--output` | `qa_output/pipeline`（絶対パス） | 出力ディレクトリ |
| `--max-docs` | 全件 | 処理する最大チャンク数 |
| `--analyze-coverage` | `false` | カバレージ分析を実行 |
| `--coverage-threshold` | 設定値 | カバレージ判定の類似度閾値 |
| `--use-celery` | `false` | Celery 並列処理を使用 |
| `-c`, `--concurrency` | `8` | 並列タスク数 |
| `--celery-workers` | `1` | （非推奨）`--concurrency` を使用 |
| `--batch-chunks` | `3` | （非推奨・未使用）1 チャンク = 1 タスクで処理される |

---

## 4. ツール詳細: Agent 検索（Web UI）

**スクリプト:** `agent_rag.py`

Streamlit ベースの Web UI アプリケーションです。Qdrant に登録済みの Q/A データに対して、ReAct + Reflection パターンおよび GRACE（Plan+Executor）パターンの自律型 Agent 検索を実行できます。

### 4.1 起動コマンド

```bash
uv run streamlit run agent_rag.py --server.port 8501
```

ブラウザで http://localhost:8501 にアクセスします。

### 4.2 メニュー構成

| メニュー | 機能 |
|---------|------|
| 📖 説明 | システム全体の説明ページ |
| 🔎 Qdrant検索 | Qdrant に直接クエリを実行してベクトル検索 |
| 🤖 Agent(ReAct+Reflection) | ReAct + Reflection パターンのエージェント対話 |
| [最新] 自律型Agent(Plan+Executor) | GRACE アーキテクチャの自律型エージェント |
| 📊 未回答ログ | Agent が回答できなかった質問のログ閲覧 |
| 📄 RAGデータ作成 | RAG データ作成（実装予定） |
| 🗄️ QdrantのCRUD | Qdrant のデータ管理（実装予定） |

### 4.3 Agent（ReAct+Reflection）の動作

エージェントは以下の 2 フェーズで質問に回答します:

**Phase 1 — ReAct ループ:**
ユーザーの質問に対して Thought → Action（ツール呼び出し） → Observation のサイクルを繰り返し、回答案を作成します。エージェントは全 Qdrant コレクションを並列検索し、最適な結果を選択します。

**Phase 2 — Reflection:**
回答案を正確性・適切性・スタイルの観点で自己評価し、必要に応じて修正した最終回答を出力します。

### 4.4 検索の仕組み

エージェントは以下のツールを内部的に使用します（`agent_tools.py`）:

| ツール | 用途 |
|-------|------|
| `search_rag_knowledge_base` | 全コレクション並列検索 + コサイン類似度フィルタ |
| `list_rag_collections` | 利用可能なコレクション一覧取得 |

検索時には Hybrid Search（Dense + Sparse ベクトル）を使用し、コサイン類似度 ≥ 0.5（`COSINE_SIMILARITY_THRESHOLD`）の結果のみを採用します。`COHERE_API_KEY` が設定されている場合は Cohere Rerank による再ランキングも行われます（未設定時は RRF スコアのまま返却）。

### 4.5 前提条件

Agent 検索を利用するには、以下が必要です:

- Docker コンテナ（Qdrant + Redis）が起動していること
- `.env` に `OPENAI_API_KEY` が設定されていること
- Qdrant に 1 つ以上のコレクションが登録されていること（ツール 2 で登録）

---

## 5. パイプライン全体の実行例

Wikipedia 日本語データを例にした、データ準備から検索までの一連の流れです。

```bash
# === Step 0: 環境起動 ===
docker compose -f docker-compose/docker-compose.yml up -d
./start_celery.sh restart -c 8 --flower

# === Step 1: チャンク作成 ===
python -m chunking.csv_text_to_chunks_text_csv \
  --input-file OUTPUT/wikipedia_ja_1per.csv \
  --output output_chunked \
  --model gpt-5-mini \
  --workers 8
# → output_chunked/wikipedia_ja_1per_chunks.csv（固定ファイル名）

# === Step 2: Q/A 生成 + Qdrant 登録 ===
python qa_qdrant/make_qa_register_qdrant.py \
  --input-file output_chunked/wikipedia_ja_1per_chunks.csv \
  --collection wikipedia_ja_1per \
  --use-celery \
  --concurrency 8 \
  --recreate

# === Step 3: Agent 検索 ===
uv run streamlit run agent_rag.py --server.port 8501

# === 終了時 ===
# Ctrl+C で Streamlit 停止
./start_celery.sh stop
docker compose -f docker-compose/docker-compose.yml down
```

---

## 6. ポート一覧

| サービス | ポート | 用途 |
|---------|-------|------|
| Streamlit | 8501 | Agent RAG Web UI |
| Qdrant | 6333 | ベクトル DB REST API |
| Redis | 6379 | Celery ブローカー / 結果保存 |
| Flower | 5555 | Celery タスクモニタリング |

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 2.0 | 2026-07-10 | OpenAI API 移行・現行実装に合わせて全面改訂: LLM 既定を `gpt-5-mini`、Embedding を `text-embedding-3-large`（3072次元）、必須キーを `OPENAI_API_KEY` に統一（旧 Gemini 表記を全廃）。Docker コマンドを `docker compose -f docker-compose/docker-compose.yml` に是正。チャンク作成の出力を固定ファイル名（`--timestamp` 指定時のみ日時サフィックス）＋ `_simple.csv` ＋ manifest に更新し、`--max-chunk-tokens`（既定 512）・`--continuity-mode`（既定 rule）を追記。`make_qa_register_qdrant.py` から削除済みのオプション（`--combine-rows` / `--block-size` / スマート生成切替）を削除し、実 argparse（`--provider openai` 等）と突合。補助ツール `register_to_qdrant.py`（登録のみ）・`make_qa.py`（生成のみ）の節を新設。UI 起動を `uv run streamlit run agent_rag.py --server.port 8501` に統一し、メニュー構成・コサイン類似度閾値 0.5・Cohere Rerank の挙動を実コードに合わせて更新。データ DL（HuggingFace）手順と `start_celery.sh` 書式を追記。変更履歴を新設 |
| 1.0 | 2026-04-26 | 初版（Gemini ベース） |
