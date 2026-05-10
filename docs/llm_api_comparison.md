# LLM API 3プロバイダー完全対比表

**対象プロジェクト**

| プロジェクト | LLM | Embedding |
|---|---|---|
| `anthropic_grace_agent` | Anthropic `claude-sonnet-4-6` | Gemini `gemini-embedding-001` |
| `openai_grace_agent` | OpenAI `gpt-4o` / `gpt-4o-mini` | **Gemini `gemini-embedding-001`** ※ |
| `gemini_grace_agent` | Gemini `gemini-2.5-flash` | Gemini `gemini-embedding-001` |

> **※ openai_grace_agent の Embedding について**: LLM は OpenAI を使用するが、Qdrant 登録・検索の Embedding は `services/qdrant_service.py` が `create_embedding_client(provider="gemini")` を呼び出すため **Gemini `gemini-embedding-001`（3072次元）** を使用する。OpenAI Embedding（`text-embedding-3-large`）は利用していない。

**参照実装**: `helper/helper_llm.py`（`AnthropicClient` / `OpenAIClient` / `GeminiClient`）
**作成日**: 2026-05-10
**更新日**: 2026-05-10（openai_grace_agent 実装検証・修正）

---

## 1. クライアント初期化

| 項目 | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| SDK パッケージ | `anthropic` | `openai` | `google-genai` |
| インポート | `import anthropic` | `from openai import OpenAI` | `from google import genai` |
| クライアント生成 | `anthropic.Anthropic(api_key=...)` | `OpenAI(api_key=...)` | `genai.Client(api_key=...)` |
| API キー環境変数 | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` | `GOOGLE_API_KEY` |
| チャットセッション | **なし**（ステートレス設計） | **なし**（ステートレス設計） | `client.chats.create(model, config)` |

```python
# Anthropic
import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# OpenAI
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Gemini
from google import genai
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
```

---

## 2. テキスト生成（シングルターン）

| 項目 | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| メソッド | `client.messages.create()` | `client.chat.completions.create()` | `client.models.generate_content()` |
| プロンプト引数 | `messages=[{"role":"user","content":prompt}]` | `messages=[{"role":"user","content":prompt}]` | `contents=prompt` |
| システムプロンプト | `system="..."` **（messages の外・トップレベル）** | `messages` 先頭に `{"role":"system","content":"..."}` を挿入 | `config=GenerateContentConfig(system_instruction="...")` |
| 出力トークン上限 | `max_tokens=...` **（必須）** | `max_completion_tokens=...`（gpt-5.4-mini 以降は `max_tokens` 廃止） | `config.max_output_tokens=...` |
| 温度パラメータ | `temperature=...`（直接パラメータ） | `temperature=...`（直接パラメータ） | `config=GenerateContentConfig(temperature=...)` |
| レスポンス取得 | `response.content[0].text` | `response.choices[0].message.content` | `response.text` |
| AFC 無効化 | **不要**（概念なし） | **不要**（概念なし） | `AutomaticFunctionCallingConfig(disable=True)` 必要 |

```python
# Anthropic
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,            # 必須
    system="あなたは...",        # messages の外
    temperature=0.7,
    messages=[{"role": "user", "content": prompt}]
)
answer = response.content[0].text

# OpenAI
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "あなたは..."},  # messages の中
        {"role": "user",   "content": prompt}
    ],
    max_completion_tokens=4096,
    temperature=0.7,
)
answer = response.choices[0].message.content

# Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=genai_types.GenerateContentConfig(
        system_instruction="あなたは...",
        max_output_tokens=4096,
        temperature=0.7,
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True)
    )
)
answer = response.text
```

---

## 3. 会話履歴の管理

| 項目 | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| 管理方式 | `messages` リストを**自前管理** | `messages` リストを**自前管理** | `chat` オブジェクトが**自動管理** |
| 初期化 | `messages = []` | `messages = []` | `client.chats.create(model, config)` |
| ユーザー追加 | 手動で `messages.append({"role":"user",...})` | 手動で `messages.append({"role":"user",...})` | `chat.send_message(input)` で自動追加 |
| アシスタント追加 | 手動で `messages.append({"role":"assistant",...})` | 手動で `messages.append({"role":"assistant",...})` | 自動（chat オブジェクトが保持） |
| ロール種別 | `"user"` / `"assistant"` | `"system"` / `"user"` / `"assistant"` / `"tool"` | `parts` 内で自動区別 |
| 再呼び出し | `client.messages.create(messages=全履歴)` | `client.chat.completions.create(messages=全履歴)` | `chat.send_message(次のメッセージ)` |

```python
# Anthropic（自前管理）
messages = []
messages.append({"role": "user", "content": "質問1"})
res1 = client.messages.create(model=..., messages=messages, max_tokens=4096)
messages.append({"role": "assistant", "content": res1.content[0].text})
messages.append({"role": "user", "content": "続き"})
res2 = client.messages.create(model=..., messages=messages, max_tokens=4096)

# OpenAI（自前管理 ※ system は最初に固定）
messages = [{"role": "system", "content": "あなたは..."}]
messages.append({"role": "user", "content": "質問1"})
res1 = client.chat.completions.create(model=..., messages=messages)
messages.append({"role": "assistant", "content": res1.choices[0].message.content})
messages.append({"role": "user", "content": "続き"})
res2 = client.chat.completions.create(model=..., messages=messages)

# Gemini（chat が自動管理）
chat = client.chats.create(model=model_name, config=config)
res1 = chat.send_message("質問1")
res2 = chat.send_message("続き")   # 履歴は chat オブジェクトが保持
```

---

## 4. 構造化出力（最大の差異）

| 項目 | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| 方式 | **Tool Use** で代替（`input_schema` にスキーマを渡す） | **Structured Outputs**（`beta.chat.completions.parse`） | `response_schema=PydanticClass` を直接渡す |
| スキーマ形式 | `"input_schema": PydanticClass.model_json_schema()` | `response_format=PydanticClass`（クラスをそのまま） | `response_schema=PydanticClass.model_json_schema()` |
| レスポンス取得 | `tool_block.input` → `model_validate(tool_block.input)` | `response.choices[0].message.parsed`（SDK が自動パース） | `response.text` → `model_validate_json(response.text)` |
| JSON 解析 | SDK が自動パース（`model_validate()` で型安全） | SDK が自動パース（`message.parsed` で型安全） | 手動パース（`JSONDecodeError` 発生リスクあり） |
| 終了検出 | `response.stop_reason == "tool_use"` | `response.choices[0].finish_reason == "stop"` | 通常の `generate_content()` と同様 |

```python
# Anthropic（Tool Use で代替）
tool_def = {
    "name"        : "structured_output",
    "description" : "Return structured data",
    "input_schema": ExecutionPlan.model_json_schema()
}
response = client.messages.create(
    model=model_name, max_tokens=4096,
    system="Always respond using the provided tool.",
    tools=[tool_def],
    tool_choice={"type": "tool", "name": "structured_output"},
    messages=[{"role": "user", "content": prompt}]
)
tool_block = next(b for b in response.content if b.type == "tool_use")
plan = ExecutionPlan.model_validate(tool_block.input)

# OpenAI（Structured Outputs）
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "..."},
        {"role": "user",   "content": prompt}
    ],
    response_format=ExecutionPlan,     # Pydantic クラスを直接渡す
    max_completion_tokens=4096,
)
plan = response.choices[0].message.parsed  # ExecutionPlan インスタンスが直接返る

# Gemini（response_schema 直渡し）
response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ExecutionPlan.model_json_schema(),
    )
)
plan = ExecutionPlan.model_validate_json(response.text)
```

---

## 5. Tool Use 定義形式

| 項目 | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| ツール定義形式 | `[{"name":..., "description":..., "input_schema":{...}}]` | `[{"type":"function","function":{"name":...,"description":...,"parameters":{...}}}]` | `types.Tool(function_declarations=[{"name":..., "parameters":{...}}])` |
| スキーマキー名 | **`"input_schema"`** | **`"parameters"`** | **`"parameters"`** |
| `description` | ツール直下に必須 | `function` 内に記述 | `function_declarations` 直下（任意） |

> **openai_grace_agent の注意点**: `grace/tools.py` のツール定義は Anthropic 形式（`"input_schema"` キー）のまま記述されているが、`helper/helper_llm.py` の `OpenAIClient.generate_with_tools()` が自動的に `parameters` キーへ変換する（`t.get("input_schema", t.get("parameters", {}))`）。ツール定義側の書き換えは不要。

```python
# Anthropic（"input_schema" キー）
tools = [
    {
        "name"        : "search_rag",
        "description" : "RAG 検索を実行する",
        "input_schema": {
            "type"      : "object",
            "properties": {"query": {"type": "string", "description": "検索クエリ"}},
            "required"  : ["query"]
        }
    }
]

# OpenAI（"parameters" キー + "type":"function" ラッパー）
tools = [
    {
        "type"    : "function",
        "function": {
            "name"       : "search_rag",
            "description": "RAG 検索を実行する",
            "parameters" : {
                "type"      : "object",
                "properties": {"query": {"type": "string", "description": "検索クエリ"}},
                "required"  : ["query"]
            }
        }
    }
]

# Gemini（types.Tool ラッパー + "parameters" キー）
tools = genai_types.Tool(function_declarations=[
    {
        "name"      : "search_rag",
        "parameters": {
            "type"      : "object",
            "properties": {"query": {"type": "string", "description": "検索クエリ"}},
        }
    }
])
```

---

## 6. ReAct ループ（Tool Use 検出・結果送信）

| 項目 | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| **ツール呼び出し検出** | `response.stop_reason == "tool_use"` | `response.choices[0].finish_reason == "tool_calls"` | `candidates[0].content.parts` を走査して `part.function_call` を探す |
| ツール名取得 | `b.name`（`b.type == "tool_use"` のブロック） | `tc.function.name` | `fn.name` |
| ツール引数取得 | `b.input`（`dict` が直接返る） | `json.loads(tc.function.arguments)` | `fn.args` |
| **ツール ID** | **`b.id`**（`tool_result` 返送時に必須） | `tc.id`（`tool_call_id` として必須） | なし |
| **ツール結果の送信** | **2件追記が必須**（① assistant + ② user/tool_result） | tool ごとに `{"role":"tool",...}` を **複数追記**（1件ずつ） | `Part.from_function_response()` + `chat.send_message(part)` **1回** |
| 複数ツール同時 | 全件を同一 `user` メッセージにまとめる | tool 1件ごとに独立した `"role":"tool"` メッセージ | 1件ずつ処理 |
| **終了判定** | `stop_reason == "end_turn"` | `finish_reason == "stop"` | `function_call` が見つからない |

```python
# ─────────────────────────────────────────────────────────
# Anthropic（2件追記が必須）
# ─────────────────────────────────────────────────────────
text, tool_calls, stop_reason = llm.generate_with_tools(
    messages=messages, tools=tools, system=system_prompt, max_tokens=4096
)

if stop_reason == "tool_use" and tool_calls:
    # ① assistant ターンを追記（response.content をそのまま）
    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for tc in tool_calls:
        result = execute_tool(tc["name"], tc["input"])
        tool_results.append({
            "type"       : "tool_result",
            "tool_use_id": tc["id"],    # ← b.id と必ず一致させる（必須）
            "content"    : str(result)
        })

    # ② 全ツール結果を同一 user メッセージにまとめて追記
    messages.append({"role": "user", "content": tool_results})
else:
    final_answer = text  # stop_reason == "end_turn" で終了

# ─────────────────────────────────────────────────────────
# OpenAI（tool ごとに独立した role:"tool" メッセージ）
# services/agent_service.py の ReActAgent 実装
# ─────────────────────────────────────────────────────────
text, tool_calls, finish_reason = self.llm.generate_with_tools(
    messages=self._messages, tools=tools, system=system_prompt
)

if finish_reason == "tool_calls" and tool_calls:
    # ① assistant ターンを追記（tool_calls フィールド必須）
    self._messages.append({
        "role"      : "assistant",
        "content"   : text or None,
        "tool_calls": [
            {
                "id"      : tc["id"],
                "type"    : "function",
                "function": {
                    "name"     : tc["name"],
                    "arguments": json.dumps(tc["input"], ensure_ascii=False),
                }
            }
            for tc in tool_calls
        ]
    })

    for tc in tool_calls:
        result = execute_tool(tc["name"], tc["input"])
        # ② tool 1件ごとに独立した role:"tool" メッセージ
        self._messages.append({
            "role"        : "tool",
            "tool_call_id": tc["id"],   # ← tc.id と必ず一致させる（必須）
            "content"     : str(result)
        })
else:
    final_answer = text  # finish_reason == "stop" で終了

# ─────────────────────────────────────────────────────────
# Gemini（chat 経由で 1 回送信）
# ─────────────────────────────────────────────────────────
response = chat.send_message(message=augmented_input)
for part in response.candidates[0].content.parts:
    if hasattr(part, "function_call") and part.function_call:
        fn = part.function_call
        result = execute_tool(fn.name, fn.args)
        part_response = genai_types.Part.from_function_response(
            name=fn.name, response={"result": result}
        )
        response = chat.send_message(message=part_response)  # 1回で送信
    else:
        final_answer = part.text   # function_call がなければ終了
```

---

## 7. トークンカウント

| 項目 | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| メソッド | `client.messages.count_tokens(model, messages)` | **ローカル計算**（`tiktoken` ライブラリ） | `client.models.count_tokens(model, contents)` |
| 戻り値 | `response.input_tokens` | `len(encoding.encode(text))` | `response.total_tokens` |
| API コール | **あり**（リモート） | **なし**（ローカル） | **あり**（リモート） |
| 備考 | メッセージ形式のまま渡せる | `cl100k_base` or モデル固有エンコーディング | シンプルなテキストを渡す |

```python
# Anthropic
response = client.messages.count_tokens(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": text}]
)
count = response.input_tokens

# OpenAI（ローカル・高速）
import tiktoken
try:
    encoding = tiktoken.encoding_for_model(model)
except KeyError:
    encoding = tiktoken.get_encoding("cl100k_base")
count = len(encoding.encode(text))

# Gemini
response = client.models.count_tokens(model="gemini-2.5-flash", contents=text)
count = response.total_tokens
```

---

## 8. Embedding

| 項目 | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Embedding API | **存在しない** | `client.embeddings.create(model, input, dimensions)` | `client.models.embed_content(model, contents, config)` |
| 代替手段 | **Gemini `gemini-embedding-001` を使用** | — | — |
| デフォルトモデル | `gemini-embedding-001`（代替） | `text-embedding-3-large` | `gemini-embedding-001` |
| 次元数 | **3072**（Gemini 経由） | 3072（`dimensions=3072`） | 3072 |
| `task_type` | Gemini 経由なので使用可 | **なし** | `retrieval_query` / `retrieval_document` 等 |
| API キー | `GOOGLE_API_KEY`（Gemini 経由） | `OPENAI_API_KEY` | `GOOGLE_API_KEY` |

> **openai_grace_agent の実態**: LLM は OpenAI を使用するが、Qdrant Embedding は `services/qdrant_service.py` で `create_embedding_client(provider="gemini")` を呼び出しており、**Gemini `gemini-embedding-001`** を使用する。`GOOGLE_API_KEY` が必須。

```python
# Anthropic プロジェクト（Gemini Embedding を使用）
from google import genai
embed_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
response = embed_client.models.embed_content(
    model="gemini-embedding-001",
    contents=text,
    config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
)
vector = response.embeddings[0].values  # 3072次元

# OpenAI Embedding
from openai import OpenAI
embed_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = embed_client.embeddings.create(
    model="text-embedding-3-large",
    input=text,
    dimensions=3072
)
vector = response.data[0].embedding  # 3072次元

# Gemini Embedding
from google import genai
embed_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
response = embed_client.models.embed_content(
    model="gemini-embedding-001",
    contents=text,
    config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
)
vector = response.embeddings[0].values  # 3072次元
```

---

## 9. モデル名・料金比較

### LLM モデル

| 用途目安 | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| 最高性能 | `claude-opus-4-7` | `gpt-4o` | `gemini-3-pro-preview` |
| **推奨デフォルト** | **`claude-sonnet-4-6`** | **`gpt-4o-mini`** | **`gemini-2.5-flash`** |
| Q/A生成・チャンキング | — | **`gpt-5.4-mini`** ★ | — |
| 高速・低コスト | `claude-haiku-4-5-20251001` | `gpt-4o-mini` | `gemini-2.0-flash` |

> **★ `gpt-5.4-mini`**: `chunking/csv_text_to_chunks_text_csv.py`（チャンキング）と `qa_generation/pipeline.py`（Q/A生成）のデフォルトモデル。このモデル以降は `max_tokens` パラメータが廃止され、**`max_completion_tokens` が必須**。

### 料金（USD / 1K tokens）

| モデル | Input | Output |
|---|---|---|
| `claude-opus-4-7` | $0.005 | $0.025 |
| `claude-sonnet-4-6` | $0.003 | $0.015 |
| `claude-haiku-4-5-20251001` | $0.0008 | $0.004 |
| `gpt-4o` | $0.005 | $0.015 |
| `gpt-4o-mini` | $0.00015 | $0.0006 |
| `gpt-5.4-mini` | 公式参照 | 公式参照 |
| `gemini-2.5-flash` | $0.0001 | $0.0004 |
| `gemini-3-pro-preview` | $0.00125 | $0.010 |

### Embedding モデル

| モデル | 次元数 | 料金（/1K tokens） | 備考 |
|---|---|---|---|
| `gemini-embedding-001` | **3072** | 無料枠あり | anthropic / gemini / openai プロジェクト採用 |
| `text-embedding-3-large` | **3072** | $0.00013 | OpenAI Embedding（openai_grace_agent では未使用） |
| `text-embedding-3-small` | 1536 | $0.00002 | 軽量・低コスト |

---

## 10. grace/ モジュール別 API 使用状況（openai_grace_agent）

| モジュール | クラス / 機能 | 使用プロバイダー | 主要 API |
|---|---|---|---|
| `grace/planner.py` | `QAPipeline`（Q/A生成パイプライン） ※1 | **OpenAI** | `SmartQAGenerator` 経由 |
| `grace/tools.py` | `ReasoningTool.execute()` | **OpenAI** | `generate_content()`（`create_llm_client("openai")`） |
| `grace/tools.py` | `AskUserTool` ツール定義 | — | `"input_schema"` キー形式（`helper_llm` が自動変換） |
| `grace/confidence.py` | `LLMSelfEvaluator.evaluate()` | **OpenAI** | `generate_content()`（`create_llm_client("openai")`） |
| `grace/confidence.py` | `LLMSelfEvaluator.evaluate_with_factors()` | **OpenAI** | `generate_structured()` → `beta.parse()` |
| `grace/confidence.py` | `QueryCoverageCalculator.calculate()` | **OpenAI** | `generate_content()` |
| `grace/confidence.py` | `SourceAgreementCalculator.calculate()` | **OpenAI Embedding** | `embed_text()` |
| `grace/executor.py` | `_check_rag_relevance_with_llm()` | **OpenAI** | `generate_content(temperature=0.0, max_tokens=5)` |
| `services/agent_service.py` | `ReActAgent` ReAct ループ | **OpenAI** | `generate_with_tools()`（`finish_reason=="tool_calls"`） |
| `helper/helper_embedding.py` | `create_embedding_client()` | **Gemini**（デフォルト） | `embed_text()` / `embed_texts()` |

> **※1 `grace/planner.py` について**: openai_grace_agent では `grace/planner.py` のファイル内容が `qa_generation/pipeline.py`（`QAPipeline` クラス）に置き換えられている。anthropic_grace_agent の `Planner` クラス（計画生成・複雑度推定）は存在しない。

---

## 11. プロバイダー切替方法

```python
# helper/helper_llm.py の create_llm_client() で切り替え可能
from helper.helper_llm import create_llm_client  # ← "helper.helper_llm"（モジュールパス注意）

llm = create_llm_client("anthropic")   # → AnthropicClient
llm = create_llm_client("openai")      # → OpenAIClient（デフォルト）
llm = create_llm_client("gemini")      # → GeminiClient

# 環境変数での切り替え
# export LLM_PROVIDER=anthropic   # anthropic_grace_agent
# export LLM_PROVIDER=openai      # openai_grace_agent
# export LLM_PROVIDER=gemini      # gemini_grace_agent

# Embedding の切り替え
from helper.helper_embedding import create_embedding_client

emb = create_embedding_client("gemini")   # → GeminiEmbedding（3072次元）
emb = create_embedding_client("openai")   # → OpenAIEmbedding（3072次元）
```

---

## 12. よくある移植ミスと対策

| ミス | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| システムプロンプトの場所 | `system=` トップレベル（messages の外） | `messages` 先頭に `{"role":"system",...}` | `config.system_instruction=...` |
| `max_tokens` 省略 | **エラー**（必須） | 省略可（gpt-5.4-mini以降は `max_completion_tokens` 必須） | `max_output_tokens` で指定（省略可） |
| ツール定義キー名 | `"input_schema"` | `"parameters"`（`helper_llm` が自動変換） | `"parameters"` |
| ツール結果の追記数 | **2件**（assistant + user） | **N件**（tool 1件ごと） | **1件**（`chat.send_message(part)`） |
| `tool_use_id` / `tool_call_id` | `b.id` を `tool_use_id` に設定（必須） | `tc.id` を `tool_call_id` に設定（必須） | **不要** |
| 終了判定 | `stop_reason == "end_turn"` | `finish_reason == "stop"` | `function_call` が見つからない |
| 構造化出力 | Tool Use で代替（`tool_block.input`） | `beta.parse()` + `message.parsed` | `response_schema=` 直渡し |
| AFC 無効化コード | **削除する**（概念なし） | **削除する**（概念なし） | `AutomaticFunctionCallingConfig(disable=True)` |
| インポートパス | `from helper.helper_llm import ...` | 同左 | 同左 |

---

## 13. セマンティックチャンキング（chunking/ モジュール）

**実装**: `chunking/csv_text_to_chunks_text_csv.py` + `chunking/async_api_client.py`

| 項目 | OpenAI（現行 openai_grace_agent） | Anthropic | Gemini |
|---|---|---|---|
| SDK | `openai`（直接使用） | `anthropic` | `google-genai` |
| クライアント生成 | `OpenAI(api_key=...)` ※直接 | `anthropic.Anthropic(api_key=...)` | `genai.Client(api_key=...)` |
| API メソッド | `client.beta.chat.completions.parse()` | `client.messages.create()` | `client.models.generate_content()` |
| 構造化出力方式 | **Structured Outputs**（`response_format=PydanticClass`） | **Tool Use 強制**（`tool_choice="tool"`） | `response_schema=` 直渡し |
| スキーマ渡し | `response_format=Schema`（クラスをそのまま） | `"input_schema": Schema.model_json_schema()` | `response_schema=Schema.model_json_schema()` |
| 結果取得 | `choice.message.parsed` → `json.dumps(parsed.model_dump())` | `block.input` → `json.dumps()` | `response.text` → `model_validate_json()` |
| 非同期化 | `asyncio.to_thread(client.beta.chat.completions.parse, ...)` | `asyncio.to_thread(client.messages.create, ...)` | `asyncio.to_thread(client.models.generate_content, ...)` |
| 並列制御 | `asyncio.Semaphore(max_workers)` | 同左 | 同左 |
| 出力トークン上限 | `max_completion_tokens=8192`（`max_tokens` は廃止） | `max_tokens=8192` | `max_output_tokens=4096` |
| デフォルトモデル | **`gpt-5.4-mini`** | `claude-sonnet-4-6` | `gemini-3-flash-preview` |
| system プロンプト | **なし**（user メッセージのみ） | `system=` パラメータ | `config.system_instruction=` |
| 正常終了検出 | `finish_reason == "stop"` | `stop_reason == "tool_use"` | 通常の `generate_content()` と同様 |
| 切断検出 | `finish_reason == "length"` → リトライ | `stop_reason == "max_tokens"` → リトライ | `finish_reason == MAX_TOKENS` → リトライ |
| レート制限検出 | `"429"` / `"rate"` / `"quota"` / `"insufficient_quota"` → 30秒待機 | 同様 | 同様 |

> **注意**: `chunking/async_api_client.py` は `create_llm_client()` を経由せず、直接 `OpenAI()` クライアントを生成して `client.beta.chat.completions.parse()` を呼び出す。`helper_llm.py` の `OpenAIClient.generate_structured()` とは異なる経路。

```python
# chunking/async_api_client.py（現行 OpenAI 実装）

class AsyncAPIClient:
    def __init__(self, api_key: str, max_workers: int = 8, max_output_tokens: int = 8192):
        # [MIGRATION] genai.Client() → OpenAI()（直接使用）
        self.client = OpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_workers)
        self.max_output_tokens = max_output_tokens

    def _is_truncated(self, finish_reason: Optional[str]) -> bool:
        # [MIGRATION] 複雑な Gemini finish_reason チェック → "length" のみ
        return finish_reason == "length"

    async def generate_content(self, model, contents, response_schema, task_id=None):
        async with self.semaphore:
            response = await asyncio.to_thread(
                self.client.beta.chat.completions.parse,
                model=model,
                max_completion_tokens=self.max_output_tokens,  # [FIX] max_tokens 廃止
                messages=[{"role": "user", "content": contents}],  # system なし
                response_format=response_schema,  # Pydantic クラスを直接渡す
            )
            choice = response.choices[0]

            if self._is_truncated(choice.finish_reason):
                raise ValueError("Response truncated")  # → リトライ

            # [MIGRATION] response.text (JSON文字列) → choice.message.parsed (Pydantic) → json.dumps()
            parsed = choice.message.parsed
            if parsed is None:
                raise ValueError(f"Refusal: {choice.message.refusal}")
            return json.dumps(parsed.model_dump(), ensure_ascii=False)
```

### 3段階チャンキングパイプライン

| Step | 処理内容 | 入力 | 出力 | API呼び出し |
|---|---|---|---|---|
| **Step 1** | 階層構造化（段落分割） | 生テキスト（ブロック単位） | 段落リスト | `StructuralResult` を Structured Outputs で取得 |
| **Step 2** | 意味的チャンキング | 段落リスト | チャンクリスト | `StructuralResult` を Structured Outputs で取得 |
| **Step 3** | 文脈連続性チェック | チャンクペア | 結合済みチャンクリスト | `ContinuityResult` を Structured Outputs で取得 |

---

## 14. Q/A自動生成（qa_generation/ + Celery）

**実装**: `qa_generation/smart_qa_generator.py` + `celery_tasks.py` + `qa_generation/pipeline.py`

| 項目 | OpenAI（現行 openai_grace_agent） | Anthropic | Gemini |
|---|---|---|---|
| SDK | `openai`（`helper_llm` 経由） | `anthropic`（`helper_llm` 経由） | `google-genai`（`helper_llm` 経由） |
| LLM クライアント生成 | `create_llm_client("openai")` | `create_llm_client("anthropic")` | `create_llm_client("gemini")` |
| `SmartQAGenerator` デフォルトモデル | **`gpt-4o-mini`** | `claude-sonnet-4-6` | `gemini-2.0-flash` |
| `QAPipeline` デフォルトモデル | **`gpt-5.4-mini`** | `claude-sonnet-4-6` | `gemini-2.0-flash` |
| API メソッド（テキスト生成） | `llm.generate_content(prompt, model, temperature, max_tokens)` | 同左（内部は `messages.create`） | 同左（内部は `models.generate_content`） |
| 処理フロー | `analyze_chunk()` → `generate_qa_pairs()` | 同左 | 同左 |
| 並列処理 | **Celery** + `apply_async(args=...)` | 同左 | 同左 |
| API キー | `OPENAI_API_KEY` | `ANTHROPIC_API_KEY` | `GOOGLE_API_KEY` |

```python
# qa_generation/smart_qa_generator.py（現行 OpenAI 実装）

class SmartQAGenerator:
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        # [MIGRATION] genai.Client() / AnthropicClient → create_llm_client("openai")
        self.llm = create_llm_client("openai", default_model=model)

    def _generate_content(self, prompt: str, temperature: float = 0.1) -> str:
        return self.llm.generate_content(
            prompt=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=4096,
        )

    def analyze_chunk(self, chunk_text: str) -> Dict:
        """Step 1: チャンク分析（重要度・トピック数・Q/A生成数を決定）"""
        prompt = ANALYZE_PROMPT.format(chunk_text=chunk_text)
        text = self._generate_content(prompt, temperature=0.1)
        return json.loads(text)   # → {"num_qa": 3, "importance": "high", ...}

    def generate_qa_pairs(self, chunk_text: str, analysis: Dict) -> List[Dict]:
        """Step 2: Q/Aペア生成"""
        prompt = QA_GENERATION_PROMPT.format(
            chunk_text=chunk_text,
            num_qa=analysis.get("num_qa", 3)
        )
        text = self._generate_content(prompt, temperature=0.3)
        return json.loads(text)   # → [{"question": ..., "answer": ...}, ...]


# celery_tasks.py（Celery 並列処理）
@celery_app.task(name='generate_qa_for_chunk')
def generate_qa_for_chunk_task(chunk, config, model, use_smart_generation=True):
    generator = SmartQAGenerator(model=model)
    return generator.process_chunk(chunk["text"])

# 並列投入
tasks = []
for chunk in chunks:
    task = generate_qa_for_chunk_task.apply_async(
        args=(chunk, config, model, use_smart_generation)
    )
    tasks.append(task)
```

---

## 15. Qdrant 登録・検索

**実装**: `services/qdrant_service.py` + `qdrant_client_wrapper.py` + `qa_qdrant/make_qa_register_qdrant.py`

### 15-1. Qdrant 登録フロー

| 項目 | openai_grace_agent（現行） | anthropic_grace_agent | gemini_grace_agent |
|---|---|---|---|
| Embedding プロバイダー | **Gemini `gemini-embedding-001`** ※ | Gemini `gemini-embedding-001` | Gemini `gemini-embedding-001` |
| Embedding クライアント | `create_embedding_client(provider="gemini")` | 同左 | 同左 |
| Embedding API | `embed_client.embed_texts(texts, batch_size)` | 同左 | 同左 |
| `task_type`（登録時） | `"RETRIEVAL_DOCUMENT"` | 同左 | 同左 |
| 次元数 | **3072** | 3072 | 3072 |
| Qdrant SDK | `qdrant_client.QdrantClient` | 同左 | 同左 |
| 登録メソッド | `client.upsert(collection_name, points)` | 同左 | 同左 |
| ベクトル構造 | `models.PointStruct(id, vector, payload)` | 同左 | 同左 |
| API キー（登録） | `OPENAI_API_KEY`（LLM）+ **`GOOGLE_API_KEY`（Embedding）** | `ANTHROPIC_API_KEY`（LLM）+ `GOOGLE_API_KEY`（Embedding） | `GOOGLE_API_KEY`（LLM + Embedding） |

> **※ openai_grace_agent の Embedding**: `services/qdrant_service.py` の `embed_texts_for_qdrant()` は `create_embedding_client(provider="gemini")` を呼び出す。OpenAI Embedding（`text-embedding-3-large`）は使用していない。Qdrant コレクションは全プロジェクト共通で **3072次元 / Gemini ベクトル空間**。

```python
# services/qdrant_service.py（openai_grace_agent 現行実装）

def embed_texts_for_qdrant(
    texts: List[str], model: str = "gemini-embedding-001", batch_size: int = 100
) -> List[List[float]]:
    # LLM は OpenAI だが Embedding は Gemini を使用
    embedding_client = create_embedding_client(provider="gemini")
    dims = get_embedding_dimensions("gemini")   # 3072
    valid_vecs = embedding_client.embed_texts(valid_texts, batch_size=batch_size)
    return valid_vecs
```

### 15-2. Qdrant 検索フロー

| 項目 | openai_grace_agent（現行） | anthropic_grace_agent | gemini_grace_agent |
|---|---|---|---|
| クエリ Embedding | **Gemini `gemini-embedding-001`**（デフォルト） | 同左 | 同左 |
| プロバイダー自動判定 | 次元数 1536 → OpenAI / それ以外 → Gemini | 同左 | 同左 |
| `task_type`（検索時） | `"retrieval_query"`（Gemini 時のみ） | 同左 | 同左 |
| Dense 検索 | `client.query_points(collection, query=vector, limit=N)` | 同左 | 同左 |
| Hybrid 検索 | `client.query_points(prefetch=[Dense+Sparse], query=FusionQuery(RRF))` | 同左 | 同左 |

```python
# services/qdrant_service.py（検索クエリ Embedding）

def embed_query_for_search(
    query: str, model: str = "gemini-embedding-001", dims: Optional[int] = None
) -> List[float]:
    provider = "gemini"           # デフォルト
    if dims == 1536:
        provider = "openai"
    elif "text-embedding-3" in (model or ""):
        provider = "openai"

    embedding_client = create_embedding_client(provider=provider, dims=dims)
    task_type = "retrieval_query" if provider == "gemini" else None
    vector = embedding_client.embed_text(query, task_type=task_type)
    return vector
```

### 15-3. エンドツーエンドのデータフロー（openai_grace_agent）

```
[テキスト/CSV]
    ↓ chunking/csv_text_to_chunks_text_csv.py
[セマンティックチャンク]  ← OpenAI API (beta.chat.completions.parse, gpt-5.4-mini)
    ↓ qa_qdrant/make_qa_register_qdrant.py (Phase 1)
[Q/Aペア CSV]           ← OpenAI API (generate_content, gpt-4o-mini) + Celery
    ↓ qa_qdrant/make_qa_register_qdrant.py (Phase 2)
[Qdrant 登録]           ← Gemini Embedding (embed_texts, gemini-embedding-001) + QdrantClient.upsert()
    ↓
[Qdrant コレクション]   ← 3072次元ベクトル（Gemini ベクトル空間）
    ↓ qdrant_client_wrapper.search_collection()
[検索結果]              ← Gemini Embedding (embed_text, retrieval_query) + QdrantClient.query_points()
```

### 15-4. 各フェーズの必要 API キー（openai_grace_agent）

| フェーズ | 必要 API キー | 用途 |
|---|---|---|
| チャンキング | `OPENAI_API_KEY` | OpenAI `beta.chat.completions.parse()` |
| Q/A生成 | `OPENAI_API_KEY` | OpenAI `chat.completions.create()` |
| Qdrant 登録 | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Gemini Embedding (`embed_texts`) |
| 検索クエリ | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Gemini Embedding (`embed_text`, `retrieval_query`) |

---

*本ドキュメントは `openai_grace_agent` の実装に基づき検証・修正済み。*
*anthropic_grace_agent との差分（Embedding が Gemini 統一、grace/planner.py の構成変更等）を反映。*
