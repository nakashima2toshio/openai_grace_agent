# セットアップ・インストール手順書

**プロジェクト**: openai_grace_agent（GRACE Agent + RAG システム）  
**LLM**: OpenAI API（`gpt-4o` / `gpt-4o-mini`）  
**Embedding**: OpenAI Embedding API（`text-embedding-3-small`）  
**作成日**: 2026-05-26

---

## 目次

1. [システム構成・概要](#1-システム構成概要)
2. [動作環境・前提条件](#2-動作環境前提条件)
3. [必要ソフトウェアのインストール](#3-必要ソフトウェアのインストール)
   - [3.1 Homebrew（macOS）](#31-homebrewmacos)
   - [3.2 Python 3.13](#32-python-313)
   - [3.3 uv（パッケージマネージャー）](#33-uvパッケージマネージャー)
   - [3.4 Docker Desktop](#34-docker-desktop)
   - [3.5 MeCab（オプション）](#35-mecabオプション)
4. [プロジェクトのセットアップ](#4-プロジェクトのセットアップ)
   - [4.1 リポジトリのクローン](#41-リポジトリのクローン)
   - [4.2 Python 依存パッケージのインストール](#42-python-依存パッケージのインストール)
   - [4.3 環境変数の設定（.env）](#43-環境変数の設定env)
5. [インフラサービスの起動](#5-インフラサービスの起動)
   - [5.1 Docker サービス（Qdrant + Redis）](#51-docker-サービスqdrant--redis)
   - [5.2 Celery ワーカーの起動](#52-celery-ワーカーの起動)
6. [初回データ登録](#6-初回データ登録)
7. [アプリケーションの起動・停止](#7-アプリケーションの起動停止)
8. [ポート一覧](#8-ポート一覧)
9. [動作確認チェックリスト](#9-動作確認チェックリスト)
10. [トラブルシューティング](#10-トラブルシューティング)
11. [主要ファイル構成](#11-主要ファイル構成)
12. [参考ドキュメント](#12-参考ドキュメント)

---

## 1. システム構成・概要

### アーキテクチャ図

```mermaid
flowchart TB
    USER["ユーザー（ブラウザ）\nhttp://localhost:8501"]

    subgraph APP["Streamlit アプリ"]
        STREAMLIT["agent_rag.py"]
    end

    subgraph CLOUD["クラウド API"]
        OPENAI_LLM["OpenAI Chat API\ngpt-4o / gpt-4o-mini"]
        OPENAI_EMB["OpenAI Embedding API\ntext-embedding-3-small"]
    end

    subgraph LOCAL["ローカルサービス"]
        QDRANT["Qdrant\nVector DB\n:6333"]
        REDIS["Redis\nCelery Broker\n:6379"]
    end

    subgraph BG["バックグラウンド"]
        CELERY["Celery Workers\nQ/A生成・登録"]
    end

    USER --> STREAMLIT
    STREAMLIT --> OPENAI_LLM
    STREAMLIT --> OPENAI_EMB
    STREAMLIT --> QDRANT
    STREAMLIT --> REDIS
    REDIS --> CELERY
    CELERY --> OPENAI_LLM
    CELERY --> OPENAI_EMB
```

### コンポーネント概要

| コンポーネント | 役割 | 実行場所 |
|--------------|------|---------|
| Streamlit | Web UI（agent_rag.py） | Python プロセス |
| OpenAI Chat API | LLM（チャンク分割・Q&A生成・Agent応答） | クラウド |
| OpenAI Embedding API | Embedding（1536次元） | クラウド |
| Qdrant | ベクトル DB（RAG 検索） | Docker コンテナ |
| Redis | Celery タスクブローカー・結果保存 | Docker コンテナ |
| Celery | Q/A 生成などのバックグラウンドタスク | Python プロセス |

---

## 2. 動作環境・前提条件

### 対応 OS

| OS | 対応状況 |
|----|---------|
| macOS（Apple Silicon M1/M2/M3） | ✅ 推奨 |
| macOS（Intel） | ✅ 動作可 |
| Linux（Ubuntu 22.04+） | ✅ 動作可 |
| Windows（WSL2） | ⚠️ 未確認 |

### ハードウェア要件

| リソース | 最小 | 推奨 |
|---------|------|------|
| CPU | 4 コア | 8 コア以上 |
| RAM | 8 GB | 16 GB 以上 |
| ディスク空き | 5 GB | 10 GB 以上 |

> LLM・Embedding はクラウド API のため、ローカルへの大容量モデルダウンロードは不要です。

### 必須ソフトウェア一覧

| ソフトウェア | バージョン | 用途 |
|------------|----------|------|
| Python | **3.13.x**（必須） | アプリ実行環境 |
| uv | 最新版 | パッケージ管理 |
| Docker Desktop | 最新版 | Qdrant / Redis |
| Git | 2.x 以上 | リポジトリ操作 |

### 必須 API キー

| API | 用途 | 取得先 |
|-----|------|--------|
| `OPENAI_API_KEY` | LLM + Embedding | https://platform.openai.com/ |
| `COHERE_API_KEY` | Rerank（オプション） | https://dashboard.cohere.com/ |

---

## 3. 必要ソフトウェアのインストール

### 3.1 Homebrew（macOS）

macOS の場合、パッケージ管理に Homebrew を使います。

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

インストール確認:
```bash
brew --version   # → Homebrew 4.x.x
```

> Linux の場合は `apt` / `dnf` など OS 標準のパッケージマネージャーを使用してください。

---

### 3.2 Python 3.13

本プロジェクトは **Python 3.13 専用**（`pyproject.toml` の `requires-python = ">=3.13,<3.14"`）。

#### macOS（Homebrew）

```bash
brew install python@3.13
python3.13 --version   # → Python 3.13.x
```

#### pyenv を使う場合（macOS / Linux）

```bash
brew install pyenv            # macOS
# または: curl https://pyenv.run | bash   # Linux

pyenv install 3.13.3
pyenv local 3.13.3
python --version   # → Python 3.13.3
```

#### Linux（Ubuntu）

```bash
sudo apt update
sudo apt install -y python3.13 python3.13-venv python3.13-dev
```

---

### 3.3 uv（パッケージマネージャー）

`pip` の代わりに **`uv`** を使います。依存解決が高速で、`uv.lock` による完全再現が可能です。

```bash
# 公式インストーラー（macOS / Linux）
curl -LsSf https://astral.sh/uv/install.sh | sh

# または Homebrew（macOS）
brew install uv
```

インストール後、シェルを再起動（または `source ~/.zshrc`）してから確認:

```bash
uv --version   # → uv 0.x.x
```

---

### 3.4 Docker Desktop

Qdrant と Redis を Docker コンテナで起動します。

#### macOS

[Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/) からダウンロードしインストール。  
**Apple Silicon（M1/M2/M3）の場合は ARM 版**を選択すること。

Docker Desktop 起動後、**Settings → Resources** で推奨値を設定:

| リソース | 推奨値 |
|---------|--------|
| CPUs | 4 以上 |
| Memory | 8 GB 以上 |
| Swap | 1 GB |

#### Linux

```bash
# Ubuntu の場合
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

インストール確認:
```bash
docker --version          # → Docker version 27.x.x
docker compose version    # → Docker Compose version v2.x.x
```

---

### 3.5 MeCab（オプション）

日本語形態素解析（キーワード抽出）に使用。なくてもアプリは動作します。

```bash
# macOS
brew install mecab mecab-ipadic

# Linux（Ubuntu）
sudo apt install -y mecab libmecab-dev mecab-ipadic-utf8
```

Python バインディングは `uv sync` で自動インストールされます（`mecab-python3`）。

---

## 4. プロジェクトのセットアップ

### 4.1 リポジトリのクローン

```bash
git clone https://github.com/nakashima2toshio/openai_grace_agent.git
cd openai_grace_agent
```

---

### 4.2 Python 依存パッケージのインストール

```bash
# 本番依存のみ（推奨）
uv sync

# 開発用依存（ruff, pytest）も含める
uv sync --all-groups
```

> `uv sync` は `pyproject.toml` と `uv.lock` を読み込み、Python 仮想環境（`.venv/`）を自動作成してパッケージをインストールします。

インストール確認:
```bash
uv run python -c "import streamlit, qdrant_client, openai; print('OK')"
# → OK
```

主要パッケージバージョン（`pyproject.toml` より）:

| パッケージ | バージョン | 用途 |
|----------|----------|------|
| openai | >=1.100.2 | OpenAI API クライアント（LLM + Embedding） |
| streamlit | 1.48.1 | Web UI |
| qdrant-client | 1.15.1 | ベクトル DB クライアント |
| celery | 5.5.3 | タスクキュー |
| redis | 6.2.0 | Celery ブローカー |
| pydantic | 2.11.7 | データバリデーション |
| tiktoken | 最新版 | トークンカウント |

---

### 4.3 環境変数の設定（`.env`）

プロジェクトルートに `.env` ファイルを作成します。

```bash
touch .env
```

`.env` の設定例:

```dotenv
# ============================================================
# LLM + Embedding（必須）: OpenAI API
# ============================================================
OPENAI_API_KEY=sk-your-openai-api-key-here

# ============================================================
# Rerank（オプション）: Cohere
# ============================================================
# COHERE_API_KEY=your-cohere-api-key-here

# ============================================================
# Qdrant（Docker で起動する場合はデフォルトで動作）
# ============================================================
# QDRANT_HOST=localhost
# QDRANT_PORT=6333

# ============================================================
# Redis / Celery（Docker で起動する場合はデフォルトで動作）
# ============================================================
# CELERY_BROKER_URL=redis://localhost:6379/0
# CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

> **`OPENAI_API_KEY` は必須です。**

---

## 5. インフラサービスの起動

### 5.1 Docker サービス（Qdrant + Redis）

#### 起動

```bash
docker compose -f docker-compose/docker-compose.yml up -d
```

#### 状態確認

```bash
docker compose -f docker-compose/docker-compose.yml ps
```

期待される出力:
```
NAME      IMAGE                  STATUS
qdrant    qdrant/qdrant:latest   Up (healthy)
redis     redis:7-alpine         Up (healthy)
```

#### ヘルスチェック

```bash
# Qdrant
curl http://localhost:6333/health
# → {"title":"qdrant - vector search engine","version":"..."}

# Redis
docker compose -f docker-compose/docker-compose.yml exec redis redis-cli ping
# → PONG
```

#### ログ確認

```bash
docker compose -f docker-compose/docker-compose.yml logs -f qdrant
docker compose -f docker-compose/docker-compose.yml logs -f redis
```

#### 停止

```bash
# データを保持したまま停止
docker compose -f docker-compose/docker-compose.yml down

# データも削除してリセット
docker compose -f docker-compose/docker-compose.yml down -v
```

---

### 5.2 Celery ワーカーの起動

Q/A 自動生成などのバックグラウンドタスクを処理します。

```bash
# 実行権限付与（初回のみ）
chmod +x start_celery.sh

# 起動（M2 MacBook Air 推奨設定）
./start_celery.sh start -c 8 --flower

# 状態確認
./start_celery.sh status

# 再起動
./start_celery.sh restart -c 8 --flower

# 停止
./start_celery.sh stop
```

#### Flower タスクモニター

```
http://localhost:5555
```

#### 推奨パラメータ（M2 MacBook Air 8 コア）

| パラメータ | 推奨値 | 説明 |
|----------|--------|------|
| `-c`（concurrency） | 8 | CPU コア数に合わせる |
| `--flower` | 有効 | タスク状況のリアルタイム監視 |

---

## 6. 初回データ登録

Qdrant にベクトルデータを登録します（初回または再構築時）。

### 6.1 チャンク済みデータを Qdrant に登録

```bash
uv run python a30_qdrant_registration.py --recreate --limit 100
```

### 6.2 Celery 経由で並列 Q/A 生成 + 登録

```bash
# Celery ワーカーを先に起動してから実行
uv run python qa_qdrant/make_qa_register_qdrant.py \
    --input-file output_chunked/cc_news_5per_chunks.csv \
    --collection cc_news_5per \
    --use-celery \
    --recreate
```

### 6.3 登録データの確認

```bash
# Qdrant コレクション一覧
curl http://localhost:6333/collections | python3 -m json.tool
```

---

## 7. アプリケーションの起動・停止

### 7.1 全サービスの一括起動手順

```bash
# ── ターミナル 1: Docker (Qdrant + Redis) ────────────
docker compose -f docker-compose/docker-compose.yml up -d

# ── ターミナル 2: Celery ワーカー ────────────────────
./start_celery.sh start -c 8 --flower

# ── ターミナル 3: Streamlit アプリ ───────────────────
uv run streamlit run agent_rag.py --server.port 8501
```

ブラウザでアクセス:
```
http://localhost:8501
```

### 7.2 起動後のアクセス先一覧

| サービス | URL | 備考 |
|---------|-----|------|
| Streamlit Web UI | http://localhost:8501 | メインアプリ |
| Qdrant REST API | http://localhost:6333 | DB 管理 |
| Flower（Celery 監視） | http://localhost:5555 | タスクモニター |

### 7.3 全サービスの停止

```bash
# Streamlit: Ctrl+C

# Celery 停止
./start_celery.sh stop

# Docker 停止
docker compose -f docker-compose/docker-compose.yml down
```

---

## 8. ポート一覧

| サービス | ポート | プロトコル | 用途 |
|---------|--------|----------|------|
| Streamlit | **8501** | HTTP | Web UI |
| Qdrant | **6333** | HTTP | ベクトル DB REST API |
| Redis | **6379** | TCP | Celery ブローカー・結果保存 |
| Flower | **5555** | HTTP | Celery タスクモニタリング |

> OpenAI Chat API・Embedding API はクラウド経由のため、ローカルポートは不要です。

---

## 9. 動作確認チェックリスト

セットアップ完了後、以下をすべて確認してください。

### ソフトウェア

```
[ ] python3 --version  →  Python 3.13.x
[ ] uv --version       →  uv 0.x.x
[ ] docker --version   →  Docker version 27.x.x
```

### API キー

```
[ ] .env に OPENAI_API_KEY が設定されている
[ ] uv run python -c "import openai; openai.OpenAI()" がエラーなく動作
```

### サービス起動

```
[ ] docker compose ps で qdrant が Up (healthy)
[ ] docker compose ps で redis が Up (healthy)
[ ] curl http://localhost:6333/health が正常応答
[ ] ./start_celery.sh status でワーカーが起動中
```

### アプリ起動

```
[ ] uv run streamlit run agent_rag.py が正常起動
[ ] http://localhost:8501 にブラウザでアクセス可能
[ ] 左ペインのメニューが表示される
[ ] Agent(ReAct+Reflection) でエラーなく動作する
[ ] 自律型 Agent(Plan+Executor) でエラーなく動作する
```

---

## 10. トラブルシューティング

### `OPENAI_API_KEY が設定されていない` エラー

```bash
# .env を確認
grep OPENAI_API_KEY .env

# または環境変数として設定
export OPENAI_API_KEY=sk-your-key-here
```

### Qdrant に接続できない

```bash
# コンテナ状態確認
docker compose -f docker-compose/docker-compose.yml ps

# unhealthy の場合は再起動
docker compose -f docker-compose/docker-compose.yml restart qdrant

# ログで原因確認
docker compose -f docker-compose/docker-compose.yml logs qdrant
```

### Celery ワーカーが起動しない

```bash
# Redis が起動しているか確認
docker compose -f docker-compose/docker-compose.yml exec redis redis-cli ping
# → PONG でなければ Docker を再起動

# Celery ログ確認
tail -50 logs/celery_qa_worker.log
```

### `ModuleNotFoundError` が出る

```bash
# uv run 経由で実行する（自動で venv を解決）
uv run python agent_rag.py

# PYTHONPATH を明示する場合
export PYTHONPATH="$(pwd):$(pwd)/helper"
```

### `uv sync` が Python バージョンエラーで失敗する

```bash
# Python 3.13 を明示指定
uv sync --python 3.13

# .python-version ファイルで固定
echo "3.13" > .python-version
uv sync
```

### OpenAI API レート制限エラー

```bash
# エラー例: RateLimitError: 429 Too Many Requests
# 対処: Celery 並列数を下げる
./start_celery.sh restart -c 2 --flower
```

### Streamlit で「Q/A が生成されない」

Celery ワーカーが起動していない可能性があります:
```bash
./start_celery.sh status
# 起動していなければ
./start_celery.sh start -c 8 --flower
```

---

## 11. 主要ファイル構成

```
openai_grace_agent/
├── agent_rag.py              # Streamlit メインアプリ
├── agent_main.py             # エージェント共通ロジック
├── config.py                 # アプリ設定（モデル・DB・Celery など）
├── config.yml                # YAML 形式設定ファイル
├── pyproject.toml            # プロジェクト定義（uv 管理）
├── uv.lock                   # 依存ロックファイル（変更禁止）
├── .env                      # 環境変数（要作成・git 管理外）
│
├── docker-compose/
│   └── docker-compose.yml    # Qdrant + Redis コンテナ定義
│
├── start_celery.sh           # Celery ワーカー起動スクリプト
│
├── helper/
│   ├── helper_llm.py         # OpenAI LLM クライアント
│   └── helper_embedding.py   # OpenAI Embedding クライアント
│
├── grace/                    # GRACE Agent（Plan+Executor）
│   ├── confidence.py         # 信頼度スコア計算
│   ├── executor.py           # タスク実行エンジン
│   └── planner.py            # タスク計画生成
│
├── services/
│   └── agent_service.py      # Agent サービス層
│
├── qa_generation/
│   └── smart_qa_generator.py # Q/A 自動生成
│
├── qa_qdrant/
│   └── make_qa_register_qdrant.py  # Q/A 生成 + Qdrant 登録
│
├── chunking/                 # テキストチャンキング
├── output_chunked/           # チャンク済みデータ（CSV）
├── qa_output/                # 生成済み Q/A データ
├── logs/                     # Celery ログ
│
└── docs/
    ├── llm_api_comparison_v2.md          # LLM API 比較表（v2）
    ├── API_migration_gemini2anthropic.md # API 移行ガイド
    ├── llm_api_comparison.md             # LLM API 比較表（v1）
    ├── uv_install.md                     # pip → uv 移行手順
    └── setup_and_install.md              # 本ファイル
```

---

## 12. 参考ドキュメント

| ドキュメント | 内容 |
|------------|------|
| `docs/llm_api_comparison_v2.md` | OpenAI / Gemini / Anthropic API 比較表（v2） |
| `docs/API_migration_gemini2anthropic.md` | Gemini → Anthropic API 移行ガイド |
| `docs/llm_api_comparison.md` | LLM API 比較表（v1） |
| `docs/uv_install.md` | pip → uv 移行手順 |
| `readme_rag.md` | RAG システム概要 |
| `readme_autonomous_agent.md` | GRACE 自律エージェント概要 |
| `readme_react_reflection.md` | ReAct + Reflection エージェント概要 |
| `CLAUDE.md` | Claude Code 向けプロジェクトガイド |

---

*最終更新: 2026-05-26*
