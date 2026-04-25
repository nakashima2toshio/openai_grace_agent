# generation.py 完全ガイド

## 📋 目次

1. [概要](#概要)
2. [システムアーキテクチャ](#システムアーキテクチャ)
3. [データ処理フロー](#データ処理フロー)
4. [モジュール詳細](#モジュール詳細)
5. [使用方法](#使用方法)
6. [パラメータリファレンス](#パラメータリファレンス)
7. [実行例とワークフロー](#実行例とワークフロー)
8. [トラブルシューティング](#トラブルシューティング)
9. [ベストプラクティス](#ベストプラクティス)

---

## 📖 概要

`qa_generation/generation.py` は、**LLM（Gemini）を使用してテキストチャンクから高品質なQ/Aペアを自動生成するモジュール**です。単一チャンク処理とバッチ処理の両方に対応し、チャンクの特性に応じた最適なQ/A数の決定、構造化出力とテキストパースの併用によるロバストな生成を実現します。

### 主な特徴

✅ **インテリジェントなQ/A数決定**

- トークン数に基づく動的なQ/A数調整
- チャンク位置による補正（文書後半のバイアス対策）
- 最小2個～最大8個の範囲で最適化

✅ **多言語対応**

- 日本語（ja）と英語（en）の自動切り替え
- 言語別のプロンプト最適化
- 質問タイプの言語別分類（fact/reason/comparison/application）

✅ **ロバストな生成メカニズム**

- 構造化出力（Structured Output）による高精度生成
- テキストパースへの自動フォールバック
- バッチ処理と個別処理の切り替え

✅ **エラーハンドリングとリトライ**

- 最大3回のリトライ機能
- バッチ失敗時の個別処理フォールバック
- API制限対策（待機時間調整）

✅ **柔軟なバッチ処理**

- 複数チャンクの一括処理
- チャンクマージによる効率化
- API呼び出し数の最適化

---

## 🏗️ システムアーキテクチャ

### モジュール構成

```
qa_generation/generation.py
├── QAGenerator (クラス)
│   ├── __init__()                 # LLMクライアント初期化
│   ├── determine_qa_count()       # 最適Q/A数決定
│   ├── generate_for_chunk()       # 単一チャンクからQ/A生成
│   └── generate_for_batch()       # バッチ処理でQ/A生成
│
└── generate_qa_dataset() (関数)   # データセット全体のQ/A生成
    ├── チャンクマージ
    ├── バッチ分割
    ├── リトライ処理
    └── API制限対策
```

### 依存関係

```
generation.py
├── helper.helper_llm
│   ├── LLMClient                  # LLM操作の抽象化
│   └── create_llm_client()        # クライアント生成
│
├── models
│   └── QAPairsResponse            # Pydantic応答モデル
│
├── config
│   └── DATASET_CONFIGS            # データセット設定
│
└── qa_generation.structure
    └── merge_small_chunks()       # チャンク統合
```

### 処理フロー図

```
 ┌──────────────────────────────────────────┐
│        QAGenerator                       │
│                                          │
│  1. LLMクライアント初期化                   │
│     ↓                                    │
│  2. チャンク受け取り                        │
│     ↓                                    │
│  3. 最適Q/A数決定                          │
│     - トークン数分析                        │
│     - 位置バイアス補正                      │
│     ↓                                    │
│  4. プロンプト生成                         │
│     - 言語別システムプロンプト               │
│     - ユーザープロンプト（JSON形式）          │
│     ↓                                    │
│  5. LLM呼び出し                           │
│     ┌────────────────┐                   │
│     │ 構造化出力     │ ← 第1選択            │
│     └────────────────┘                   │
│            ↓ 失敗                        │
│     ┌────────────────┐                   │
│     │ テキスト生成   │ ← フォールバック       │
│     │ + JSON抽出     │                   │
│     └────────────────┘                   │
│     ↓                                    │
│  6. Q/Aペア構築                           │
│     - メタデータ付与                       │
│     - 検証とクリーンアップ                  │
│     ↓                                    │
│  7. 返却                                 │
└──────────────────────────────────────────┘
```

---

## 🔄 データ処理フロー

### 全体フロー

```
チャンクリスト
   │
   ├─ 単一チャンク → generate_for_chunk()
   └─ 複数チャンク → generate_for_batch()
          │
          v
   ┌──────────────────────────┐
   │  generate_qa_dataset()   │
   ├──────────────────────────┤
   │                          │
   │  1. チャンクマージ       │
   │     ↓                    │
   │  2. バッチ分割           │
   │     ↓                    │
   │  3. バッチごとに処理     │
   │     ├─ リトライ機能      │
   │     └─ フォールバック    │
   │     ↓                    │
   │  4. Q/Aペア収集          │
   │                          │
   └──────────────────────────┘
          │
          v
   Q/Aペアリスト
   [
     {
       "question": "...",
       "answer": "...",
       "question_type": "fact",
       "source_chunk_id": "...",
       "doc_id": "...",
       "dataset_type": "...",
       "chunk_idx": 0
     }
   ]
```

### 最適Q/A数決定ロジック

```python
def determine_qa_count(chunk, config) -> int:
    base_count = config["qa_per_chunk"]
    token_count = count_tokens(chunk['text'])
    chunk_position = chunk.get('chunk_idx', 0)

    # トークン数ベースの決定
    if token_count < 50:
        qa_count = 2
    elif token_count < 100:
        qa_count = 3
    elif token_count < 200:
        qa_count = base_count + 1
    elif token_count < 300:
        qa_count = base_count + 2
    else:
        qa_count = base_count + 3

    # 文書後半の補正（位置バイアス対策）
    if chunk_position >= 5:
        qa_count += 1

    # 最大8個に制限
    return min(qa_count, 8)
```

**処理例**:


| トークン数 | 位置 | base_count | Q/A数     |
| ---------- | ---- | ---------- | --------- |
| 40         | 0    | 3          | 2         |
| 80         | 0    | 3          | 3         |
| 150        | 0    | 3          | 4 (3+1)   |
| 250        | 0    | 3          | 5 (3+2)   |
| 350        | 0    | 3          | 6 (3+3)   |
| 150        | 5    | 3          | 5 (3+1+1) |
| 350        | 7    | 3          | 7 (3+3+1) |

---

## 📚 モジュール詳細

### QAGenerator クラス

**目的**: 単一のLLMクライアントを使用してQ/Aペアを生成する

#### `__init__(client, model)`

**シグネチャ**:

```python
def __init__(self,
             client: Optional[LLMClient] = None,
             model: str = "gemini-2.0-flash")
```

**パラメータ**:

- `client` (Optional[LLMClient]): LLMクライアント。Noneの場合は自動生成
- `model` (str): 使用するモデル名。デフォルトは"gemini-2.0-flash"

**処理**:

```python
self.client = client if client else create_llm_client(provider="gemini")
self.model = model
```

**使用例**:

```python
# デフォルトクライアント使用
generator = QAGenerator()

# カスタムクライアント使用
custom_client = create_llm_client(provider="gemini")
generator = QAGenerator(client=custom_client, model="gemini-2.0-flash")
```

---

#### `determine_qa_count(chunk, config)`

**シグネチャ**:

```python
def determine_qa_count(self, chunk: Dict, config: Dict) -> int
```

**目的**: チャンクの特性に基づいて最適なQ/A数を決定

**パラメータ**:

- `chunk` (Dict): チャンク情報
  - `text` (str): テキスト内容
  - `chunk_idx` (int): チャンク位置（オプション）
- `config` (Dict): データセット設定
  - `qa_per_chunk` (int): 基本Q/A数

**戻り値**:

- `int`: 生成するQ/A数（2～8の範囲）

**アルゴリズム**:

```python
1. 基本Q/A数取得（config["qa_per_chunk"]）
2. トークン数カウント（self.client.count_tokens()）
3. トークン数に基づく調整:
   - < 50トークン: 2個
   - 50-100トークン: 3個
   - 100-200トークン: base + 1
   - 200-300トークン: base + 2
   - 300+トークン: base + 3
4. 位置バイアス補正:
   - chunk_idx >= 5の場合: +1
5. 最大値制限: min(qa_count, 8)
```

**使用例**:

```python
chunk = {
    'text': "長いテキスト...",
    'chunk_idx': 6
}
config = {'qa_per_chunk': 3}

qa_count = generator.determine_qa_count(chunk, config)
# → 350トークン、位置6の場合: 3 + 3 + 1 = 7個
```

---

#### `generate_for_chunk(chunk, config)`

**シグネチャ**:

```python
def generate_for_chunk(self, chunk: Dict, config: Dict) -> List[Dict]
```

**目的**: 単一チャンクからQ/Aペアを生成

**パラメータ**:

- `chunk` (Dict): チャンク情報
  - `text` (str): テキスト内容 ※必須
  - `id` (str): チャンクID（オプション）
  - `doc_id` (str): 文書ID（オプション）
  - `dataset_type` (str): データセット種別（オプション）
  - `chunk_idx` (int): チャンク位置（オプション）
- `config` (Dict): データセット設定
  - `lang` (str): 言語コード（"ja" または "en"）
  - `qa_per_chunk` (int): 基本Q/A数

**戻り値**:

- `List[Dict]`: Q/Aペアのリスト
  ```python
  [
      {
          "question": str,
          "answer": str,
          "question_type": str,  # "fact" | "reason" | "comparison" | "application"
          "source_chunk_id": str,
          "doc_id": str,
          "dataset_type": str,
          "chunk_idx": int
      }
  ]
  ```

**処理フロー**:

```python
1. 最適Q/A数決定（determine_qa_count）
2. 言語別プロンプト生成
   - システムプロンプト（教育コンテンツ専門家）
   - ユーザープロンプト（JSON形式指定）
3. LLM呼び出し（2段階）
   ① 構造化出力（generate_structured）
      - QAPairsResponse Pydanticモデル使用
      - 高精度、型安全
   ② フォールバック（generate_content）
      - テキスト生成 + 正規表現でJSON抽出
      - ロバスト性向上
4. Q/Aペア構築（メタデータ付与）
5. 検証と返却
```

**プロンプト例（日本語）**:

```
【システムプロンプト】
あなたは教育コンテンツ作成の専門家です。
与えられた日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成

【ユーザープロンプト】
以下のテキストから3個のQ&Aペアを生成してください。

質問タイプ:
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）

テキスト:
{chunk_text}

JSON形式で出力:
{
  "qa_pairs": [
    {
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }
  ]
}
```

**使用例**:

```python
chunk = {
    'id': 'chunk_0',
    'text': '機械学習は、コンピュータがデータから学習するAIの一分野です。',
    'doc_id': 'doc_1',
    'dataset_type': 'wikipedia_ja',
    'chunk_idx': 0
}
config = {
    'lang': 'ja',
    'qa_per_chunk': 3
}

qa_pairs = generator.generate_for_chunk(chunk, config)

# 結果例:
# [
#     {
#         "question": "機械学習とは何ですか？",
#         "answer": "コンピュータがデータから学習するAIの一分野です。",
#         "question_type": "fact",
#         "source_chunk_id": "chunk_0",
#         "doc_id": "doc_1",
#         "dataset_type": "wikipedia_ja",
#         "chunk_idx": 0
#     }
# ]
```

**エラーハンドリング**:

```python
try:
    qa_pairs = generator.generate_for_chunk(chunk, config)
except ValueError as e:
    # "No parseable response from Gemini API"
    logger.error(f"Q/A生成失敗: {e}")
except Exception as e:
    # その他のエラー
    logger.error(f"予期しないエラー: {e}")
```

---

#### `generate_for_batch(chunks, config)`

**シグネチャ**:

```python
def generate_for_batch(self, chunks: List[Dict], config: Dict) -> List[Dict]
```

**目的**: 複数チャンクから一度にQ/Aペアを生成（API呼び出し数削減）

**パラメータ**:

- `chunks` (List[Dict]): チャンクリスト（最大5個推奨）
- `config` (Dict): データセット設定

**戻り値**:

- `List[Dict]`: すべてのチャンクのQ/Aペア

**処理フロー**:

```python
1. チャンク数確認
   - 0個: 空リスト返却
   - 1個: generate_for_chunk()に委譲
   - 2個以上: バッチ処理

2. 各チャンクのQ/A数決定

3. バッチプロンプト生成
   - 複数テキストを番号付きで提示
   - 各テキストごとのQ/A数指定

4. LLM呼び出し（構造化出力 → フォールバック）

5. 応答のチャンクへの紐付け
   - チャンクIDマッピング
   - メタデータ付与

6. フォールバック処理
   - バッチ失敗時: 個別処理に切り替え
   - for chunk in chunks:
         generate_for_chunk(chunk, config)
```

**バッチプロンプト例（日本語、3チャンク）**:

```
【システムプロンプト】
あなたは教育コンテンツ作成の専門家です。
複数の日本語テキストから、学習効果の高いQ&Aペアを生成してください。

【ユーザープロンプト】
以下の3つのテキストから、それぞれ指定された数のQ&Aペアを生成してください。

テキスト1（3個のQ&A）:
機械学習は...

テキスト2（4個のQ&A）:
深層学習は...

テキスト3（3個のQ&A）:
強化学習は...

JSON形式で出力:
{
  "batch_results": [
    {
      "text_index": 1,
      "qa_pairs": [...]
    },
    {
      "text_index": 2,
      "qa_pairs": [...]
    },
    {
      "text_index": 3,
      "qa_pairs": [...]
    }
  ]
}
```

**使用例**:

```python
chunks = [
    {'id': 'chunk_0', 'text': 'テキスト1...', 'chunk_idx': 0},
    {'id': 'chunk_1', 'text': 'テキスト2...', 'chunk_idx': 1},
    {'id': 'chunk_2', 'text': 'テキスト3...', 'chunk_idx': 2}
]
config = {'lang': 'ja', 'qa_per_chunk': 3}

all_qa_pairs = generator.generate_for_batch(chunks, config)
# → 約9個のQ/Aペア（各チャンク3個）
```

**バッチサイズの推奨**:

- **最小**: 2チャンク
- **推奨**: 3-5チャンク（API効率とエラーリスクのバランス）
- **最大**: 10チャンク（プロンプトサイズ制限）

---

### generate_qa_dataset() 関数

**シグネチャ**:

```python
def generate_qa_dataset(
    chunks: List[Dict],
    dataset_type: str,
    model: str = "gemini-2.0-flash",
    chunk_batch_size: int = 3,
    merge_chunks: bool = True,
    min_tokens: int = 150,
    max_tokens: int = 400,
    config: Optional[Dict] = None,
    client: Optional[LLMClient] = None
) -> List[Dict]
```

**目的**: データセット全体のQ/Aペアを生成（エンドツーエンド処理）

**パラメータ**:


| パラメータ         | 型                  | デフォルト         | 説明                                                    |
| ------------------ | ------------------- | ------------------ | ------------------------------------------------------- |
| `chunks`           | List[Dict]          | -                  | チャンクリスト（必須）                                  |
| `dataset_type`     | str                 | -                  | データセット種別（必須）                                |
| `model`            | str                 | "gemini-2.0-flash" | 使用するGeminiモデル                                    |
| `chunk_batch_size` | int                 | 3                  | バッチサイズ（1回のAPI呼び出しで処理するチャンク数）    |
| `merge_chunks`     | bool                | True               | 小さいチャンクを統合するか                              |
| `min_tokens`       | int                 | 150                | マージ時の最小トークン数                                |
| `max_tokens`       | int                 | 400                | マージ時の最大トークン数                                |
| `config`           | Optional[Dict]      | None               | データセット設定（Noneの場合はDATASET_CONFIGSから取得） |
| `client`           | Optional[LLMClient] | None               | LLMクライアント（Noneの場合は自動生成）                 |

**戻り値**:

- `List[Dict]`: 全チャンクのQ/Aペアリスト

**処理フロー**:

```python
1. 設定ロード
   - config指定あり → 使用
   - config未指定 → DATASET_CONFIGS[dataset_type]

2. クライアント初期化
   - client指定あり → 使用
   - client未指定 → create_llm_client("gemini")

3. QAGenerator初期化
   generator = QAGenerator(client, model)

4. チャンク前処理
   - merge_chunks=True → merge_small_chunks()
   - merge_chunks=False → そのまま使用

5. バッチ分割
   total_chunks = len(processed_chunks)
   api_calls = ceil(total_chunks / chunk_batch_size)

6. バッチループ処理
   for i in range(0, total_chunks, chunk_batch_size):
       batch = processed_chunks[i:i+chunk_batch_size]

       # リトライ機能（最大3回）
       for attempt in range(3):
           try:
               if chunk_batch_size == 1:
                   qa_pairs = generator.generate_for_chunk(batch[0], config)
               else:
                   qa_pairs = generator.generate_for_batch(batch, config)

               all_qa_pairs.extend(qa_pairs)
               break
           except Exception as e:
               if attempt == 2:  # 最後のリトライ失敗
                   # 個別処理にフォールバック
                   for chunk in batch:
                       qa_pairs = generator.generate_for_chunk(chunk, config)
                       all_qa_pairs.extend(qa_pairs)
               else:
                   # 指数バックオフ
                   time.sleep(2 ** attempt)

       # API制限対策
       time.sleep(0.2)

7. 結果返却
```

**使用例**:

```python
# 基本的な使用
chunks = [
    {'id': 'chunk_0', 'text': '...'},
    {'id': 'chunk_1', 'text': '...'},
    # ...
]

qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='wikipedia_ja'
)

# カスタム設定での使用
qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='custom',
    model='gemini-2.0-flash',
    chunk_batch_size=5,
    merge_chunks=True,
    min_tokens=100,
    max_tokens=500,
    config={
        'lang': 'ja',
        'qa_per_chunk': 4
    }
)
```

**ログ出力例**:

```
Q/Aペア生成開始:
- 元チャンク数: 100
- 処理チャンク数: 85  # マージにより削減
- バッチサイズ: 3
- API呼び出し予定: 29回
- モデル: gemini-2.0-flash

バッチ 1/29 処理中 (3チャンク)...
バッチ 2/29 処理中 (3チャンク)...
...
バッチ 29/29 処理中 (1チャンク)...

Q/Aペア生成完了:
- 生成されたQ/Aペア: 280個
- 実行されたAPI呼び出し: 約29回
```

**エラーハンドリング**:

```python
try:
    qa_pairs = generate_qa_dataset(chunks, 'wikipedia_ja')
except ValueError as e:
    # 未対応のデータセット
    logger.error(f"設定エラー: {e}")
except Exception as e:
    # その他のエラー
    logger.error(f"Q/A生成エラー: {e}")
```

---

## 🚀 使用方法

### パターン1: QAGeneratorクラスの直接使用

```python
from qa_generation.generation import QAGenerator
from helper.helper_llm import create_llm_client

# クライアント作成
client = create_llm_client(provider="gemini")

# QAGenerator初期化
generator = QAGenerator(client=client, model="gemini-2.0-flash")

# 単一チャンク処理
chunk = {
    'id': 'chunk_0',
    'text': '機械学習は、データから学習するAIの手法です。',
    'doc_id': 'doc_1',
    'dataset_type': 'wikipedia_ja',
    'chunk_idx': 0
}
config = {
    'lang': 'ja',
    'qa_per_chunk': 3
}

qa_pairs = generator.generate_for_chunk(chunk, config)
print(f"生成されたQ/A数: {len(qa_pairs)}")

# バッチ処理
chunks = [chunk1, chunk2, chunk3]
all_qa_pairs = generator.generate_for_batch(chunks, config)
```

---

### パターン2: generate_qa_dataset()関数の使用（推奨）

```python
from qa_generation.generation import generate_qa_dataset

# 大量のチャンクを効率的に処理
chunks = load_chunks_from_somewhere()  # 100チャンク

qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='wikipedia_ja',
    chunk_batch_size=3,  # 3チャンクずつ処理
    merge_chunks=True,   # 小さいチャンクを統合
    min_tokens=150,
    max_tokens=400
)

print(f"合計Q/A数: {len(qa_pairs)}")

# 結果を保存
import pandas as pd
df = pd.DataFrame(qa_pairs)
df.to_csv('qa_output.csv', index=False)
```

---

### パターン3: カスタム設定での使用

```python
# カスタムデータセット設定
custom_config = {
    'lang': 'en',
    'qa_per_chunk': 5,
    'name': 'Custom Dataset'
}

qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='custom',
    config=custom_config,
    model='gemini-2.0-flash',
    chunk_batch_size=5
)
```

---

## 📊 パラメータリファレンス

### QAGenerator.__init__()


| パラメータ | 型                  | デフォルト         | 説明            |
| ---------- | ------------------- | ------------------ | --------------- |
| `client`   | Optional[LLMClient] | None               | LLMクライアント |
| `model`    | str                 | "gemini-2.0-flash" | 使用モデル      |

---

### QAGenerator.determine_qa_count()


| パラメータ | 型   | 必須 | 説明                     |
| ---------- | ---- | ---- | ------------------------ |
| `chunk`    | Dict | ✅   | チャンク情報（text必須） |
| `config`   | Dict | ✅   | qa_per_chunk含む設定     |

**返却値**: `int` (2-8)

---

### QAGenerator.generate_for_chunk()


| パラメータ | 型   | 必須 | 説明                       |
| ---------- | ---- | ---- | -------------------------- |
| `chunk`    | Dict | ✅   | チャンク情報               |
| `config`   | Dict | ✅   | lang, qa_per_chunk含む設定 |

**chunk必須フィールド**:

- `text` (str): テキスト内容

**chunk推奨フィールド**:

- `id` (str): チャンクID
- `doc_id` (str): 文書ID
- `dataset_type` (str): データセット種別
- `chunk_idx` (int): チャンク位置

**config必須フィールド**:

- `lang` (str): "ja" | "en"
- `qa_per_chunk` (int): 基本Q/A数

**返却値**: `List[Dict]` (Q/Aペアリスト)

---

### QAGenerator.generate_for_batch()


| パラメータ | 型         | 必須 | 説明                         |
| ---------- | ---------- | ---- | ---------------------------- |
| `chunks`   | List[Dict] | ✅   | チャンクリスト（2-10個推奨） |
| `config`   | Dict       | ✅   | lang, qa_per_chunk含む設定   |

**返却値**: `List[Dict]` (すべてのチャンクのQ/Aペア)

---

### generate_qa_dataset()


| パラメータ         | 型                  | デフォルト         | 必須 | 説明                 |
| ------------------ | ------------------- | ------------------ | ---- | -------------------- |
| `chunks`           | List[Dict]          | -                  | ✅   | チャンクリスト       |
| `dataset_type`     | str                 | -                  | ✅   | データセット種別     |
| `model`            | str                 | "gemini-2.0-flash" | -    | 使用モデル           |
| `chunk_batch_size` | int                 | 3                  | -    | バッチサイズ         |
| `merge_chunks`     | bool                | True               | -    | チャンク統合フラグ   |
| `min_tokens`       | int                 | 150                | -    | マージ最小トークン数 |
| `max_tokens`       | int                 | 400                | -    | マージ最大トークン数 |
| `config`           | Optional[Dict]      | None               | -    | カスタム設定         |
| `client`           | Optional[LLMClient] | None               | -    | LLMクライアント      |

**返却値**: `List[Dict]` (全Q/Aペア)

---

## 💡 実行例とワークフロー

### ワークフロー1: 小規模データセット（10-100チャンク）

```python
from qa_generation.generation import generate_qa_dataset

# チャンク準備
chunks = [
    {'id': f'chunk_{i}', 'text': f'テキスト{i}...'}
    for i in range(50)
]

# Q/A生成
qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='wikipedia_ja',
    chunk_batch_size=3,
    merge_chunks=True
)

# 結果確認
print(f"生成Q/A数: {len(qa_pairs)}")
print(f"最初のQ/A: {qa_pairs[0]}")

# CSV保存
import pandas as pd
df = pd.DataFrame(qa_pairs)
df.to_csv('qa_output_small.csv', index=False)
```

**所要時間**: 約5-15分

---

### ワークフロー2: 中規模データセット（100-1,000チャンク）

```python
from qa_generation.generation import generate_qa_dataset
from helper.helper_llm import create_llm_client

# LLMクライアント作成（再利用）
client = create_llm_client(provider="gemini")

# チャンク準備
chunks = load_chunks_from_csv('chunks_medium.csv')  # 500チャンク

# Q/A生成（バッチサイズ大きめ）
qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='wikipedia_ja',
    chunk_batch_size=5,
    merge_chunks=True,
    min_tokens=100,
    max_tokens=500,
    client=client
)

# 統計情報
import pandas as pd
df = pd.DataFrame(qa_pairs)
print(f"合計Q/A数: {len(qa_pairs)}")
print(f"平均Q/A数/チャンク: {len(qa_pairs) / len(chunks):.2f}")
print(f"質問タイプ分布:\n{df['question_type'].value_counts()}")

# 保存
df.to_csv('qa_output_medium.csv', index=False)
```

**所要時間**: 約30-90分

---

### ワークフロー3: 大規模データセット（1,000+チャンク、Celery推奨）

```python
# Celeryを使った並列処理（別モジュールで実装）
from celery_tasks import submit_unified_qa_generation, collect_results

# チャンク準備
chunks = load_chunks_from_csv('chunks_large.csv')  # 5,000チャンク

# Celeryタスク投入
task_id = submit_unified_qa_generation(
    chunks=chunks,
    dataset_type='wikipedia_ja',
    batch_size=3,
    workers=24
)

# 結果収集
qa_pairs = collect_results(task_id)

print(f"合計Q/A数: {len(qa_pairs)}")
```

**所要時間**: 数時間（並列度による）

---

### ワークフロー4: エラーハンドリング付き実行

```python
from qa_generation.generation import generate_qa_dataset
import logging

logging.basicConfig(level=logging.INFO)

chunks = load_chunks()

try:
    qa_pairs = generate_qa_dataset(
        chunks=chunks,
        dataset_type='custom',
        config={
            'lang': 'ja',
            'qa_per_chunk': 3
        }
    )

    if len(qa_pairs) < len(chunks):
        logging.warning(f"一部チャンクのQ/A生成失敗（期待: {len(chunks)*3}, 実際: {len(qa_pairs)}）")

    # 保存
    save_qa_pairs(qa_pairs, 'output.csv')

except ValueError as e:
    logging.error(f"設定エラー: {e}")
except Exception as e:
    logging.error(f"予期しないエラー: {e}")
    # エラー時の処理
    save_partial_results(qa_pairs)
```

---

## 🔧 トラブルシューティング

### 問題1: "No parseable response from Gemini API"

**症状**:

```
ValueError: No parseable response from Gemini API
```

**原因**:

- Gemini APIからの応答がJSON形式でない
- 構造化出力とフォールバックの両方が失敗

**対処法**:

```python
# 1. ログレベルをDEBUGに設定して詳細確認
logging.basicConfig(level=logging.DEBUG)

# 2. チャンクテキストを確認（長すぎないか、特殊文字が含まれていないか）
print(f"チャンク長: {len(chunk['text'])}")

# 3. より小さいバッチサイズで試す
qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='wikipedia_ja',
    chunk_batch_size=1  # 1チャンクずつ処理
)

# 4. 異なるモデルを試す
qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='wikipedia_ja',
    model='gemini-1.5-pro'
)
```

---

### 問題2: API制限エラー

**症状**:

```
429 Resource Exhausted: Rate limit exceeded
```

**原因**:

- API呼び出しが多すぎる
- 待機時間が不足

**対処法**:

```python
# 1. バッチサイズを大きくしてAPI呼び出し数を削減
qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='wikipedia_ja',
    chunk_batch_size=5  # 3 → 5に増やす
)

# 2. 処理チャンク数を制限
limited_chunks = chunks[:100]  # 最初の100チャンクのみ
qa_pairs = generate_qa_dataset(limited_chunks, 'wikipedia_ja')

# 3. カスタム待機時間（generate_qa_dataset内で0.2秒固定）
# ソースコード修正が必要:
# time.sleep(0.2) → time.sleep(0.5)
```

---

### 問題3: チャンクマージが過剰

**症状**:

```
元チャンク数: 1000
処理チャンク数: 200  # 80%削減
```

**原因**:

- `min_tokens`が大きすぎる
- 元のチャンクが小さすぎる

**対処法**:

```python
# 1. マージを無効化
qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='wikipedia_ja',
    merge_chunks=False  # マージしない
)

# 2. min_tokensを調整
qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='wikipedia_ja',
    merge_chunks=True,
    min_tokens=50,   # 150 → 50に削減
    max_tokens=300   # 400 → 300に削減
)
```

---

### 問題4: Q/A数が期待と異なる

**症状**:

```
期待: 300個（100チャンク × 3）
実際: 520個
```

**原因**:

- `determine_qa_count()`がトークン数と位置で調整
- 大きいチャンクや後半のチャンクで増加

**対処法**:

```python
# 1. qa_per_chunkを調整
config = {
    'lang': 'ja',
    'qa_per_chunk': 2  # 3 → 2に削減
}

# 2. トークン数を確認
for chunk in chunks[:5]:
    token_count = client.count_tokens(chunk['text'])
    print(f"Chunk {chunk['id']}: {token_count} tokens")

# 3. determine_qa_count()のロジックを変更（ソース修正）
# より厳密な制御が必要な場合
```

---

### 問題5: メモリ不足

**症状**:

```
MemoryError: Unable to allocate array
```

**原因**:

- 大量のQ/Aペアをメモリに保持
- チャンクリストが大きすぎる

**対処法**:

```python
# 1. チャンクを分割して処理
chunk_groups = [chunks[i:i+100] for i in range(0, len(chunks), 100)]

all_qa_pairs = []
for group in chunk_groups:
    qa_pairs = generate_qa_dataset(group, 'wikipedia_ja')
    all_qa_pairs.extend(qa_pairs)

    # 途中結果を保存してメモリ解放
    save_partial_results(qa_pairs)
    del qa_pairs

# 2. Celeryを使用（別プロセスで処理）
# 推奨: 1,000チャンク以上
```

---

## 🎯 ベストプラクティス

### 1. バッチサイズの最適化

**推奨値**:

```python
# チャンク数に応じた推奨値
if len(chunks) < 50:
    chunk_batch_size = 1  # 個別処理
elif len(chunks) < 500:
    chunk_batch_size = 3  # 小バッチ
else:
    chunk_batch_size = 5  # 大バッチ
```

**理由**:

- バッチサイズが大きい → API呼び出し削減、エラーリスク増加
- バッチサイズが小さい → エラーに強い、API呼び出し増加

---

### 2. チャンクマージの活用

**推奨**:

```python
# 小さいチャンクが多い場合はマージ推奨
qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='wikipedia_ja',
    merge_chunks=True,
    min_tokens=150,  # 最小150トークン
    max_tokens=400   # 最大400トークン
)
```

**効果**:

- API呼び出し削減（チャンク数減少）
- Q/A品質向上（文脈の連続性）
- 処理時間短縮

---

### 3. エラーログの活用

**推奨**:

```python
import logging

# DEBUGレベルでログ出力
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

qa_pairs = generate_qa_dataset(chunks, 'wikipedia_ja')

# ログから以下を確認:
# - "Gemini構造化出力試行中..."
# - "構造化出力失敗、テキスト生成にフォールバック"
# - "バッチ X/Y 処理中"
```

---

### 4. 段階的なテスト

**推奨フロー**:

```python
# Step 1: 小規模テスト（10チャンク）
test_chunks = chunks[:10]
qa_pairs = generate_qa_dataset(test_chunks, 'wikipedia_ja')
print(f"テスト結果: {len(qa_pairs)}個のQ/A")

# Step 2: 中規模テスト（100チャンク）
test_chunks = chunks[:100]
qa_pairs = generate_qa_dataset(test_chunks, 'wikipedia_ja')

# Step 3: 本番実行（全チャンク）
qa_pairs = generate_qa_dataset(chunks, 'wikipedia_ja')
```

---

### 5. 結果の品質確認

**推奨**:

```python
import pandas as pd

df = pd.DataFrame(qa_pairs)

# 統計確認
print(f"合計Q/A数: {len(df)}")
print(f"質問タイプ分布:\n{df['question_type'].value_counts()}")
print(f"データセット分布:\n{df['dataset_type'].value_counts()}")

# サンプル確認
print("\n【サンプルQ/A】")
for i in range(min(5, len(df))):
    print(f"\nQ{i+1}: {df.iloc[i]['question']}")
    print(f"A{i+1}: {df.iloc[i]['answer']}")
    print(f"Type: {df.iloc[i]['question_type']}")

# 空文字列チェック
empty_questions = df[df['question'].str.strip() == '']
empty_answers = df[df['answer'].str.strip() == '']
print(f"\n空の質問: {len(empty_questions)}個")
print(f"空の回答: {len(empty_answers)}個")
```

---

### 6. カスタム設定の活用

**推奨**:

```python
# プロジェクト固有の設定を定義
CUSTOM_CONFIG = {
    'lang': 'ja',
    'qa_per_chunk': 4,  # プロジェクト要件
    'name': 'Project XYZ Dataset'
}

qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type='custom',
    config=CUSTOM_CONFIG,
    chunk_batch_size=5
)
```

---

### 7. パフォーマンス最適化

**推奨**:

```python
# LLMクライアントの再利用
from helper.helper_llm import create_llm_client

client = create_llm_client(provider="gemini")

# 複数のデータセットを処理
for dataset_name in ['dataset1', 'dataset2', 'dataset3']:
    chunks = load_chunks(dataset_name)
    qa_pairs = generate_qa_dataset(
        chunks=chunks,
        dataset_type=dataset_name,
        client=client  # クライアント再利用
    )
    save_results(qa_pairs, f'{dataset_name}_qa.csv')
```

---

### 8. 進捗モニタリング

**推奨**:

```python
# tqdmを使用した進捗表示
from tqdm import tqdm

# generate_qa_dataset内でのバッチ処理を可視化
# （ソースコード修正が必要）

# 代替案: チャンク分割して処理
chunk_groups = [chunks[i:i+100] for i in range(0, len(chunks), 100)]

all_qa_pairs = []
for group in tqdm(chunk_groups, desc="Processing chunk groups"):
    qa_pairs = generate_qa_dataset(group, 'wikipedia_ja')
    all_qa_pairs.extend(qa_pairs)
```

---

### 9. データセット別の最適化

**推奨**:

```python
# データセットの特性に応じた設定
DATASET_OPTIMIZATIONS = {
    'wikipedia_ja': {
        'chunk_batch_size': 5,
        'merge_chunks': True,
        'min_tokens': 200,
        'max_tokens': 500
    },
    'cc_news': {
        'chunk_batch_size': 3,
        'merge_chunks': True,
        'min_tokens': 150,
        'max_tokens': 400
    },
    'fineweb': {
        'chunk_batch_size': 3,
        'merge_chunks': False  # 既に最適化済み
    }
}

dataset = 'wikipedia_ja'
opt = DATASET_OPTIMIZATIONS[dataset]

qa_pairs = generate_qa_dataset(
    chunks=chunks,
    dataset_type=dataset,
    **opt
)
```

---

### 10. エラー時の部分保存

**推奨**:

```python
import json

all_qa_pairs = []

try:
    qa_pairs = generate_qa_dataset(chunks, 'wikipedia_ja')
    all_qa_pairs.extend(qa_pairs)

except Exception as e:
    logging.error(f"エラー発生: {e}")

    # 部分結果を保存
    with open('partial_qa_output.json', 'w') as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)

    logging.info(f"部分結果を保存: {len(all_qa_pairs)}個")

    raise  # エラーを再送出
```

---

## 📚 関連ドキュメント

- `qa_generation/pipeline.md` - パイプライン全体の制御
- `qa_generation/doc/qa_generation.md` - qa_generationモジュール全体ガイド
- `helper/helper_llm.py` - LLMクライアントの実装
- `models.py` - Pydanticモデル定義
- `config.py` - データセット設定

---

**作成日**: 2025-01-17
**対象ファイル**: `qa_generation/generation.py`
**バージョン**: 1.0.0
**総行数**: 442行
