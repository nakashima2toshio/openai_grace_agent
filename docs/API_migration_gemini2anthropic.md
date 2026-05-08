# API 移植完了報告書（Gemini → Anthropic → OpenAI）

**プロジェクト**: `openai_grace_agent`（旧: `anthropic_grace_agent`）
**移植元**: Gemini API (`google.genai`)
**中間**: Anthropic API (`anthropic`) + OpenAI Embedding API
**移植先（最終）**: OpenAI API (`openai`) + OpenAI Embedding API
**作成日**: 2026-04-20
**Anthropic 移植完了**: 2026-04-25
**OpenAI 移植完了**: 2026-05-05
**最終更新**: 2026-05-08

---

## 移植完了サマリー

### フェーズ① Gemini → Anthropic（2026-04-20〜25）

| 項目 | 内容 |
|---|---|
| 移植対象ファイル | **29 ファイル**（変更不要 5 ファイル含む） |
| 移植実施ファイル | **24 ファイル** |
| 変更不要ファイル | 5 ファイル（Qdrant UI 系・間接変更のみ） |
| Embedding | Anthropic に Embedding API なし → **OpenAI `text-embedding-3-large` (3072次元)** |
| Qdrant 互換性 | Gemini / OpenAI ともに **3072次元** → **コレクション再作成不要** |

### フェーズ② Anthropic → OpenAI（2026-04-25〜05-05）

| 項目 | 内容 |
|---|---|
| 移植実施ファイル | **主要 5 ファイル**（抽象化レイヤーの差し替えが中心） |
| LLM デフォルト | `claude-sonnet-4-6` → **`gpt-4o-mini`** |
| Embedding | 変更なし（OpenAI `text-embedding-3-large` を継続） |
| 後方互換性 | `AnthropicClient` / `GeminiClient` クラスはそのまま残存 |
| Qdrant 互換性 | 変更なし（コレクション再作成不要） |

---

## 第1部　Gemini API vs Anthropic API 完全対比表

---

### 1-1. クライアント初期化

| 項目 | Gemini（移植元） | Anthropic（移植先） |
|---|---|---|
| SDK | `google-genai` | `anthropic` |
| インポート | `from google import genai` | `import anthropic` |
| クライアント生成 | `genai.Client(api_key=...)` | `anthropic.Anthropic(api_key=...)` |
| API キー環境変数 | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | `ANTHROPIC_API_KEY` |
| チャットセッション | `client.chats.create(model, config)` | **なし（ステートレス設計）** |

```python
# Gemini
from google import genai
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Anthropic
import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

---

### 1-2. テキスト生成（シングルターン）

| 項目 | Gemini | Anthropic |
|---|---|---|
| メソッド | `client.models.generate_content()` | `client.messages.create()` |
| プロンプト引数名 | `contents=prompt` | `messages=[{"role":"user","content":prompt}]` |
| システムプロンプト | `config.system_instruction=...` | `system="..."` パラメータ（messages 外） |
| 温度パラメータ | `config=types.GenerateContentConfig(temperature=...)` | `temperature=...`（直接パラメータ） |
| 出力トークン上限 | `config.max_output_tokens=...` | `max_tokens=...`（**必須**） |
| レスポンス取得 | `response.text` | `response.content[0].text` |
| AFC 無効化 | `AutomaticFunctionCallingConfig(disable=True)` 必要 | **不要** |

```python
# Gemini
response = client.models.generate_content(
    model="gemini-3-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction="あなたは...",
        temperature=0.7,
        max_output_tokens=4096,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )
)
answer = response.text

# Anthropic
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,               # 必須
    system="あなたは...",           # messages の外
    temperature=0.7,
    messages=[{"role": "user", "content": prompt}]
)
answer = response.content[0].text
```

---

### 1-3. 会話履歴の管理

| 項目 | Gemini | Anthropic |
|---|---|---|
| 管理方式 | `chat` オブジェクトが**自動管理** | `messages` リストを**自前管理** |
| 初期化 | `client.chats.create(model, config)` | `messages = []` |
| メッセージ追加 | `chat.send_message(input)` で自動追加 | 手動で `messages.append(...)` |
| ロール種別 | `parts` 内で自動区別 | `{"role": "user" / "assistant", "content": ...}` で明示 |
| 再呼び出し | `chat.send_message(次のメッセージ)` | `client.messages.create(messages=全履歴)` |

```python
# Gemini（chat が履歴を自動管理）
chat = client.chats.create(model=model_name, config=config)
response1 = chat.send_message("質問1")
response2 = chat.send_message("続き")

# Anthropic（自前で管理）
messages = []
messages.append({"role": "user", "content": "質問1"})
response1 = client.messages.create(model=model_name, messages=messages, max_tokens=4096)
messages.append({"role": "assistant", "content": response1.content[0].text})
messages.append({"role": "user", "content": "続き"})
response2 = client.messages.create(model=model_name, messages=messages, max_tokens=4096)
```

---

### 1-4. 構造化出力（最大の差異）

| 項目 | Gemini | Anthropic |
|---|---|---|
| 方式 | `response_schema=PydanticClass` を直接渡す | **Tool Use** で代替 |
| スキーマ形式 | Pydantic クラスを直接渡す | `input_schema=PydanticClass.model_json_schema()` |
| レスポンス取得 | `response.text` → `model_validate_json()` | `tool_block.input` → `model_validate()` |
| JSON 解析 | 手動パース必要（エラー多発） | SDK が自動パース |

```python
# Gemini（Pydantic 直渡し）
response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ExecutionPlan,
    )
)
plan = ExecutionPlan.model_validate_json(response.text)

# Anthropic（Tool Use で代替）
tool_def = {
    "name"        : "structured_output",
    "description" : "Return structured data",
    "input_schema": ExecutionPlan.model_json_schema()
}
response = client.messages.create(
    model=model_name, max_tokens=4096,
    tools=[tool_def],
    tool_choice={"type": "tool", "name": "structured_output"},
    messages=[{"role": "user", "content": prompt}]
)
tool_block = next(b for b in response.content if b.type == "tool_use")
plan = ExecutionPlan.model_validate(tool_block.input)
```

---

### 1-5. Tool Use（ReAct ループ）

| 項目 | Gemini | Anthropic |
|---|---|---|
| **ツール定義形式** | `types.Tool(function_declarations=[...])` | `[{"name":..., "description":..., "input_schema":{...}}]` |
| スキーマキー名 | `"parameters"` | **`"input_schema"`** |
| **ツール検出方法** | `candidates[0].content.parts` を走査して `part.function_call` を探す | `response.stop_reason == "tool_use"` を確認後 `content` を走査 |
| ツール名取得 | `fn.name` | `b.name` |
| ツール引数取得 | `fn.args` | `b.input` |
| **ツール ID** | なし | **`b.id`**（`tool_result` 返送時に必須） |
| **ツール結果の送信** | `Part.from_function_response()` + `chat.send_message()` 1回 | `messages` に **2件追記**（assistant + tool_result） |
| 複数ツール同時 | 1件ずつ処理 | 全件を同一 `user` メッセージにまとめる |
| 終了判定 | `function_call` が見つからない | `stop_reason == "end_turn"` |

```python
# Gemini（ツール定義）
tools = types.Tool(function_declarations=[
    {"name": "search", "parameters": {"type": "object", "properties": {...}}}
])

# Anthropic（ツール定義）
tools = [
    {"name": "search", "description": "...",
     "input_schema": {"type": "object", "properties": {...}, "required": [...]}}
]

# ─────────────────────────────────
# Gemini（ツール結果の送信）
function_response_part = types.Part.from_function_response(
    name=tool_name, response={"result": tool_result}
)
response = chat.send_message(message=function_response_part)   # 1回

# Anthropic（ツール結果の送信）：2件追記が必須
messages.append({"role": "assistant", "content": response.content})  # ①
messages.append({                                                      # ②
    "role"   : "user",
    "content": [{
        "type"       : "tool_result",
        "tool_use_id": tool_id,           # b.id と一致（必須）
        "content"    : str(tool_result)
    }]
})
response = client.messages.create(model=..., messages=messages, ...)
```

---

### 1-6. トークンカウント

| 項目 | Gemini | Anthropic |
|---|---|---|
| メソッド | `client.models.count_tokens(model, contents)` | `client.messages.count_tokens(model, messages)` |
| 戻り値 | `response.total_tokens` | `response.input_tokens` |

---

### 1-7. Embedding

| 項目 | Gemini | OpenAI（Anthropic 代替）|
|---|---|---|
| API | `client.models.embed_content(model, contents, config)` | `client.embeddings.create(model, input, dimensions)` |
| デフォルトモデル | `gemini-embedding-001` | `text-embedding-3-large` |
| 次元数 | 3072 | **3072**（同じ → Qdrant 互換） |
| task_type | `retrieval_query` / `retrieval_document` 等 | **なし** |
| 理由 | — | **Anthropic に Embedding API がない** |

---

### 1-8. Gemini 固有機能で Anthropic に存在しないもの

| Gemini 固有機能 | Anthropic での代替手段 |
|---|---|
| `response_schema=PydanticClass` | Tool Use（`generate_structured()`） |
| AFC（Automatic Function Calling） | 不要・存在しない |
| `AutomaticFunctionCallingConfig(disable=True)` | 不要・削除する |
| `chats.create()` チャットセッション | `messages` リストを自前管理 |
| `types.Part.from_function_response()` | `messages` に `tool_result` を直接追記 |
| `response.candidates[0].content.parts` | `response.content`（フラットなリスト） |
| `response.usage_metadata.prompt_token_count` | `response.usage.input_tokens` |
| `task_type`（Embedding） | 存在しない |
| Embedding API | 存在しない → **OpenAI を使用** |

---

### 1-9. モデル名対比

| 用途 | Gemini（移植元） | Anthropic（移植先） |
|---|---|---|
| 最高性能 | `gemini-3-pro-preview` | `claude-opus-4-7` |
| バランス型（推奨） | `gemini-3-flash-preview` | **`claude-sonnet-4-6`** |
| 高速・低コスト | `gemini-2.0-flash` | `claude-haiku-4-5-20251001` |
| Embedding | `gemini-embedding-001` | `text-embedding-3-large`（OpenAI） |

---

## 第2部　移植コツ・ベストプラクティス

---

### コツ① 抽象化レイヤー（LLMClient）を先に作る

直接 Gemini を置き換えるのではなく、**`LLMClient` 抽象基底クラスを設計し `AnthropicClient` を追加する**。
これにより各ファイルの変更が `self.client → self.llm` の置き換えだけで済む。

```python
# 各ファイルの変更がこれだけになる
# 変更前
self.client = genai.Client()
response = self.client.models.generate_content(...)
answer = response.text

# 変更後
self.llm = create_llm_client("anthropic")
answer = self.llm.generate_content(prompt)   # str が直接返る
```

`generate_content()` / `generate_structured()` / `generate_with_tools()` の3メソッドを
抽象化することで、呼び出し側のコードを大幅に簡素化できる。

---

### コツ② `generate_structured()` で構造化出力を隠蔽する

```python
# helper_llm.py に一度だけ実装
def generate_structured(self, prompt, response_schema, **kwargs):
    tool_def = {
        "name"        : "structured_output",
        "description" : "Return structured data",
        "input_schema": response_schema.model_json_schema()
    }
    response = self.client.messages.create(
        tools=[tool_def],
        tool_choice={"type": "tool", "name": "structured_output"},
        messages=[{"role": "user", "content": prompt}], **kwargs
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return response_schema.model_validate(tool_block.input)

# 呼び出し側（planner.py, confidence.py 等）はこれだけ
plan = self.llm.generate_structured(prompt, ExecutionPlan)
```

---

### コツ③ ReAct ループは「2件追記」を必ず守る

Anthropic の ReAct ループで最も間違えやすい箇所。
Gemini は `chat.send_message(part)` の1回で済むが、Anthropic では **2件追記が必須**。

```python
# ❌ よくある間違い（1件しか追記しない）
messages.append({"role": "user", "content": [{"type": "tool_result", ...}]})

# ✅ 正しい（2件追記が必須）
messages.append({"role": "assistant", "content": response.content})  # ① 必須
messages.append({
    "role"   : "user",
    "content": [{"type": "tool_result", "tool_use_id": tc["id"], "content": str(result)}]
})  # ②
```

`tool_use_id` に `b.id` を正確に設定しないと **API エラー**になる。

---

### コツ④ `stop_reason` でループ制御する

```python
# Gemini（function_call を走査）
for part in response.candidates[0].content.parts:
    if hasattr(part, 'function_call') and part.function_call:
        function_call_found = True
if not function_call_found:
    break

# Anthropic（stop_reason で判定）
text, tool_calls, stop_reason = self.llm.generate_with_tools(...)
if stop_reason != "tool_use" or not tool_calls:
    final_text = text
    break
```

---

### コツ⑤ `max_tokens` は必須・`system` は `messages` の外

```python
# ❌ よくある間違い
response = client.messages.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "..."},  # ❌ system を messages に入れてはいけない
        {"role": "user",   "content": prompt}
    ]
    # max_tokens 未指定 → エラー
)

# ✅ 正しい
response = client.messages.create(
    model=model_name,
    max_tokens=4096,      # ✅ 必須
    system="...",         # ✅ messages の外
    messages=[{"role": "user", "content": prompt}]
)
```

---

### コツ⑥ GeminiClient は削除せず `try/except` で遅延インポートする

```python
# ✅ GeminiClient.__init__() 内で遅延インポート
try:
    from google import genai as _genai
except ImportError:
    raise ImportError("pip install google-genai")
self.client = _genai.Client(api_key=self.api_key)
```

`google-genai` が未インストールでも `GeminiClient` を使わない限りエラーにならない。

---

### コツ⑦ モジュールレベルの `from google import genai` を全て削除する

```bash
# 移植後の確認コマンド（全ファイルをチェック）
grep -rn "from google import genai" --include="*.py" .
grep -rn "from google.genai import" --include="*.py" .
# 何も出なければ合格（try/except 内の import は許容）
```

---

### コツ⑧ AFC 無効化コードを全て削除する

```python
# ❌ Gemini 固有コード（Anthropic では不要・削除する）
config=types.GenerateContentConfig(
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
)
```

Anthropic には AFC の概念が存在しない。

---

### コツ⑨ Reflection は `generate_with_tools(tools=[])` を使う

```python
# ❌ 会話履歴が引き継がれない
reflection_text = self.llm.generate_content(prompt=reflection_msg, ...)

# ✅ ReAct ループの検索結果・思考ログを引き継ぎ
self._messages.append({"role": "user", "content": reflection_msg})
reflection_text, _, _ = self.llm.generate_with_tools(
    messages=self._messages,
    tools=[],                      # Tool Use なし
    system=self.system_instruction,
)
```

---

### コツ⑩ 設定ファイルのデフォルト値の落とし穴

`get_config("models.default", "claude-sonnet-4-6")` の第2引数は「キーが存在しない場合のみ」使われる。
`_get_default_config()` 内に Gemini モデルが残っているとフォールバック値が使われない。

```python
# ❌ よくある落とし穴
def _get_default_config(self):
    return {"models": {"default": "gemini-2.0-flash"}}  # ← これが返り続ける

# ✅ 正しい修正
def _get_default_config(self):
    return {"models": {"default": "claude-sonnet-4-6"}}
```

---

### コツ⑪ YAML 設定ファイルも必ず更新する

Python コードを直しても YAML が古いと実行時に上書きされる。

```yaml
# grace_config.yml 変更後
llm:
  provider: "anthropic"
  model: "claude-sonnet-4-6"
embedding:
  provider: "openai"
  model: "text-embedding-3-large"
```

---

## 第3部　移植対象ファイル一覧（完了）

| Phase | ファイル | 変更種別 | 主な変更内容 | 状態 |
|---|---|---|---|---|
| **1** | `helper/helper_llm.py` | クラス追加 | `AnthropicClient` 追加、デフォルトプロバイダーを `anthropic` に変更 | ✅ |
| **1** | `helper_embedding.py` | 設定変更 | デフォルト `openai`、`text-embedding-3-large`（3072次元） | ✅ |
| **1** | `grace/config.py` | 設定変更 | `LLMConfig.model = claude-sonnet-4-6`、`EmbeddingConfig.provider = openai` | ✅ |
| **1** | `grace_config.yml` | 設定変更 | llm/embedding プロバイダー・モデルを更新 | ✅ |
| **1** | `services/config_service.py` | 設定変更 | `_get_default_config()` を Anthropic デフォルトに変更 | ✅ |
| **2** | `grace/planner.py` | API 置換 | `genai.Client()` → `AnthropicClient`、`generate_structured()` 使用 | ✅ |
| **2** | `grace/confidence.py` | API 置換 | `genai.Client()` → `AnthropicClient` / `create_embedding_client("openai")` | ✅ |
| **2** | `grace/tools.py` | API 置換 | `genai.Client()` → `AnthropicClient`、`"parameters"` → `"input_schema"` | ✅ |
| **2** | `grace/executor.py` | 間接変更 | 依存先の変更に追従 | ➖ |
| **2** | `grace/replan.py` | 間接変更 | 依存先の変更に追従 | ➖ |
| **2** | `grace/schemas.py` | 変更不要 | Pydantic 定義のみ・Gemini 依存なし | ➖ |
| **3** | `service/agent_service.py` | ループ書き直し | ReAct Tool Use を Anthropic 形式に完全書き直し | ✅ |
| **3** | `agent_main.py` | ループ書き直し | `UpgradedCLIAgent` を Anthropic 形式に書き直し | ✅ |
| **4** | `celery_config.py` | 設定追加 | `ANTHROPIC_CONFIG` を追加 | ✅ |
| **4** | `helper/helper_api.py` | import 追加 | `AnthropicClient` の再エクスポートを追加 | ✅ |
| **4** | `ui/pages/qa_generation_page.py` | UI 更新 | モデル選択リストを Claude モデルに更新 | ✅ |
| **4** | `ui/pages/qdrant_search_page.py` | 変更不要 | 次元数 3072 は変わらないため修正不要 | ➖ |
| **4** | `ui/pages/qdrant_registration_page.py` | 変更不要 | 次元数 3072 は変わらないため修正不要 | ➖ |
| **4** | `config.py`（ModelConfig） | 設定変更 | `AVAILABLE_MODELS` / `DEFAULT_MODEL` を Claude モデルに更新 | ✅ |
| **4** | `qa_generation/pipeline.py` | API 置換 | モデルデフォルトを `claude-sonnet-4-6` に変更 | ✅ |
| **4** | `qa_generation/smart_qa_generator.py` | API 置換 | `genai.Client()` → `create_llm_client("anthropic")` | ✅ |
| **4** | `qa_generation/semantic.py` | API 置換 | import パス修正、プロバイダーを Anthropic/OpenAI に変更 | ✅ |
| **4** | `celery_tasks.py` | 変更不要 | 移植完了済み・Gemini 依存なし | ➖ |
| **5** | Qdrant コレクション | 精度検証 | 次元数 3072 は互換だが、ベクトル空間の差異を並行テストで確認推奨 | ⏳ |

---

## 第4部　環境変数・設定

### .env ファイル

```bash
# Anthropic API（必須）
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI（Embedding 用・必須）
OPENAI_API_KEY=sk-...

# プロバイダー切替
LLM_PROVIDER=anthropic
EMBEDDING_PROVIDER=openai

# Gemini（後方互換・gemini_grace_agent 用）
GOOGLE_API_KEY=AIza...
```

### Anthropic モデル一覧（2026年4月時点）

| モデル文字列 | 用途目安 | RPM | TPM |
|---|---|---|---|
| `claude-opus-4-7` | 最高性能・複雑タスク | 50 | 200,000 |
| `claude-opus-4-6` | Opus 前世代（後方互換） | 50 | 200,000 |
| `claude-sonnet-4-6` | **推奨デフォルト**・バランス型 | 2,000 | 1,600,000 |
| `claude-sonnet-4-5` | Sonnet 前世代（後方互換） | 2,000 | 1,600,000 |
| `claude-haiku-4-5-20251001` | 高速・低コスト | 4,000 | 2,000,000 |

### OpenAI Embedding モデル比較

| モデル | 次元数 | 価格（/1K tokens） | 備考 |
|---|---|---|---|
| `text-embedding-3-large` | **3072** | $0.00013 | **本プロジェクト採用**。Gemini と同次元 |
| `text-embedding-3-small` | 1536 | $0.00002 | 軽量・低コスト |
| `text-embedding-ada-002` | 1536 | $0.00010 | 旧世代・非推奨 |

---

## 第5部　Qdrant コレクション互換性

`text-embedding-3-large`（OpenAI）と `gemini-embedding-001`（Gemini）は
**どちらも 3072 次元**のため、Qdrant コレクションの**再作成は不要**。

ただし、ベクトル空間の分布特性はモデルごとに異なるため、
既存コレクションに OpenAI Embedding でクエリすると精度が低下する可能性がある。

**推奨対応方針：**

1. **並行コレクション作成**（推奨）：サフィックス `_openai` で新コレクションを作成し精度を比較してから本番切り替えする。
2. **既存コレクション再登録**：データを `text-embedding-3-large` で再 Embedding して登録し直す（構造変更不要）。

```python
# dimensions パラメータで次元数を短縮可能
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=text,
    dimensions=1536    # 3072 → 1536（MTEB スコアはほぼ変わらない）
)
```

---

---

## 第6部　Anthropic → OpenAI 移植記録（2026-05追加）

---

### 6-1. クライアント初期化

| 項目 | Anthropic（移植元） | OpenAI（移植先） |
|---|---|---|
| SDK | `anthropic` | `openai` |
| インポート | `import anthropic` | `from openai import OpenAI` |
| クライアント生成 | `anthropic.Anthropic(api_key=...)` | `OpenAI(api_key=...)` |
| API キー環境変数 | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` |

```python
# Anthropic
import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# OpenAI
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

---

### 6-2. テキスト生成（シングルターン）

| 項目 | Anthropic | OpenAI |
|---|---|---|
| メソッド | `client.messages.create()` | `client.chat.completions.create()` |
| システムプロンプト | `system="..."` パラメータ（messages 外） | `messages` 先頭に `{"role":"system","content":"..."}` として挿入 |
| 出力トークン上限 | `max_tokens=...`（必須） | `max_completion_tokens=...`（新モデルは `max_tokens` 廃止） |
| レスポンス取得 | `response.content[0].text` | `response.choices[0].message.content` |
| 終了理由 | `response.stop_reason` | `response.choices[0].finish_reason` |

```python
# Anthropic
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    system="あなたは...",      # messages の外
    messages=[{"role": "user", "content": prompt}]
)
answer = response.content[0].text

# OpenAI
response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_completion_tokens=4096,
    messages=[
        {"role": "system", "content": "あなたは..."},  # messages の中に挿入
        {"role": "user",   "content": prompt}
    ]
)
answer = response.choices[0].message.content
```

---

### 6-3. 構造化出力

| 項目 | Anthropic | OpenAI |
|---|---|---|
| 方式 | Tool Use（`tool_choice` 強制） | `client.beta.chat.completions.parse()` |
| スキーマ渡し方 | `input_schema=Schema.model_json_schema()` | `response_format=PydanticClass` を直接渡す |
| レスポンス取得 | `tool_block.input` → `model_validate()` | `response.choices[0].message.parsed` |
| SDK 自動パース | ツールブロックの抽出は手動 | **SDK が自動パース** |

```python
# Anthropic（Tool Use で代替）
tool_def = {
    "name"        : "structured_output",
    "description" : "Return structured data",
    "input_schema": ExecutionPlan.model_json_schema()
}
response = client.messages.create(
    model="claude-sonnet-4-6", max_tokens=4096,
    tools=[tool_def],
    tool_choice={"type": "tool", "name": "structured_output"},
    messages=[{"role": "user", "content": prompt}]
)
tool_block = next(b for b in response.content if b.type == "tool_use")
plan = ExecutionPlan.model_validate(tool_block.input)

# OpenAI（Structured Outputs）
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    max_completion_tokens=4096,
    messages=[{"role": "user", "content": prompt}],
    response_format=ExecutionPlan,   # Pydantic クラスを直接渡す
)
plan = response.choices[0].message.parsed   # SDK が自動パース
```

---

### 6-4. Tool Use（ReAct ループ）

| 項目 | Anthropic | OpenAI |
|---|---|---|
| **ツール定義キー** | `"input_schema"` | **`"parameters"`**（OpenAI 標準） |
| **ツール定義ラッパー** | dict のリスト（`name/description/input_schema`） | `{"type":"function","function":{...}}` でラップ |
| **終了判定** | `stop_reason == "tool_use"` | **`finish_reason == "tool_calls"`** |
| **ツール呼び出し取得** | `response.content` から `type=="tool_use"` を抽出 | `message.tool_calls` リストを走査 |
| **ツール引数取得** | `b.input`（dict） | `json.loads(tc.function.arguments)` |
| **ツール ID** | `b.id` | `tc.id` |
| **assistant 追記形式** | `{"role":"assistant","content": response.content}` | `{"role":"assistant","content":...,"tool_calls":[...]}` |
| **ツール結果の追記** | `role:"user"` に **1件まとめて追記**（`tool_result` タイプ） | `role:"tool"` として **ツールごとに個別追記** |

```python
# Anthropic（ツール定義）
tools = [
    {"name": "search", "description": "...",
     "input_schema": {"type": "object", "properties": {...}, "required": [...]}}
]

# OpenAI（ツール定義）
tools = [
    {
        "type"    : "function",
        "function": {
            "name"       : "search",
            "description": "...",
            "parameters" : {"type": "object", "properties": {...}, "required": [...]}
        }
    }
]

# ─────────────────────────────────
# Anthropic（ツール結果の追記）：2件1セット
messages.append({"role": "assistant", "content": response.content})   # ①
messages.append({                                                       # ②
    "role"   : "user",
    "content": [{"type": "tool_result", "tool_use_id": tc_id, "content": result}]
})

# OpenAI（ツール結果の追記）：assistant + tool ×N
messages.append({                                                       # ①
    "role"      : "assistant",
    "content"   : text or None,
    "tool_calls": [
        {"id": tc["id"], "type": "function",
         "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])}}
        for tc in tool_calls
    ]
})
for tc, result in zip(tool_calls, results):                            # ② ×N
    messages.append({
        "role"        : "tool",
        "tool_call_id": tc["id"],
        "content"     : str(result),
    })
```

---

### 6-5. Anthropic 固有機能で OpenAI に存在しないもの・変わったもの

| Anthropic 機能 | OpenAI での代替手段 |
|---|---|
| `client.messages.create()` | `client.chat.completions.create()` |
| `system="..."` パラメータ | messages 先頭に `{"role":"system"}` を挿入 |
| `max_tokens`（必須） | `max_completion_tokens`（新モデルは `max_tokens` 廃止） |
| `stop_reason == "tool_use"` | `finish_reason == "tool_calls"` |
| ツール定義 `input_schema` | ツール定義 `parameters` |
| `response.content` ブロックリスト | `response.choices[0].message` |
| Tool Use による構造化出力 | `client.beta.chat.completions.parse()` |
| `response.usage.input_tokens` | `response.usage.prompt_tokens` |
| `response.usage.output_tokens` | `response.usage.completion_tokens` |
| `client.messages.count_tokens()` | tiktoken による手動カウント |

---

### 6-6. モデル名対比（Anthropic → OpenAI）

| 用途 | Anthropic（移植元） | OpenAI（移植先） |
|---|---|---|
| 最高性能 | `claude-opus-4-7` | `gpt-4o` |
| バランス型（推奨） | `claude-sonnet-4-6` | **`gpt-4o-mini`**（デフォルト） |
| 高速・低コスト | `claude-haiku-4-5-20251001` | `gpt-4o-mini` |
| Embedding | `text-embedding-3-large`（OpenAI）変更なし | `text-embedding-3-large`（継続） |

---

### 6-7. 移植コツ（Anthropic → OpenAI 固有）

#### コツ A：`generate_with_tools()` の抽象化で呼び出し側を無変更にする

```python
# helper_llm.py に一度だけ実装（呼び出し側は変更なし）
class OpenAIClient(LLMClient):
    def generate_with_tools(self, messages, tools, system="", model=None, **kwargs):
        # system を messages 先頭に挿入
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # ツール定義を OpenAI 形式に変換（"input_schema" → "parameters"）
        openai_tools = [
            {"type": "function", "function": {
                "name"       : t["name"],
                "description": t.get("description", ""),
                "parameters" : t.get("input_schema", t.get("parameters", {})),
            }}
            for t in tools
        ] if tools else None

        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=full_messages,
            tools=openai_tools or openai.NOT_GIVEN,
        )
        msg = response.choices[0].message
        tool_calls = [
            {"name": tc.function.name, "input": json.loads(tc.function.arguments), "id": tc.id}
            for tc in (msg.tool_calls or [])
        ]
        return msg.content or "", tool_calls, response.choices[0].finish_reason
```

#### コツ B：assistant メッセージに `tool_calls` フィールドを含める

```python
# ❌ Anthropic 形式（OpenAI では無効）
messages.append({"role": "assistant", "content": response.content})

# ✅ OpenAI 形式（tool_calls フィールド必須）
messages.append({
    "role"      : "assistant",
    "content"   : text or None,
    "tool_calls": [
        {"id": tc["id"], "type": "function",
         "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])}}
        for tc in tool_calls
    ]
})
```

#### コツ C：ツール結果は `role:"tool"` として個別追記

```python
# ❌ Anthropic 形式（OpenAI では無効）
messages.append({
    "role"   : "user",
    "content": [{"type": "tool_result", "tool_use_id": id, "content": result}]
})

# ✅ OpenAI 形式（ツールごとに個別追記）
for tc, result in zip(tool_calls, results):
    messages.append({
        "role"        : "tool",
        "tool_call_id": tc["id"],
        "content"     : str(result),
    })
```

#### コツ D：`max_tokens` → `max_completion_tokens`（新モデル対応）

```python
# gpt-5.4-mini 以降は max_tokens が廃止されている
# helper_llm.py の OpenAIClient で自動変換するのが安全
if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
```

---

### 6-8. 移植対象ファイル一覧（Anthropic → OpenAI）

| ファイル | 変更種別 | 主な変更内容 |
|---|---|---|
| `helper/helper_llm.py` | クラス追加・デフォルト変更 | `OpenAIClient` 追加、`create_llm_client()` デフォルトを `"openai"` に変更 |
| `grace/config.py` | 設定変更 | `LLMConfig.provider = "openai"`, `model = "gpt-4o-mini"` |
| `config.yml` | 設定変更 | `models.default: "gpt-4o-mini"`, `provider.default_llm: "openai"`, `openai` セクション追加 |
| `services/agent_service.py` | ループ書き直し | ReAct ループを OpenAI Tool Calls 形式に書き直し |
| `services/config_service.py` | 設定変更 | `_get_default_config()` を OpenAI デフォルトに変更 |

---

## 第7部　3段階移植の全体概観

```
Gemini API (google.genai)
        │
        │  2026-04-20〜25（第1部〜第5部参照）
        │  ・client.models.generate_content() → client.messages.create()
        │  ・response_schema → Tool Use
        │  ・chat（自動管理） → messages リスト（自前管理）
        │  ・Embedding: gemini-embedding-001 → text-embedding-3-large(OpenAI)
        ▼
Anthropic API (anthropic)
        │
        │  2026-05-04〜05（第6部参照）
        │  ・client.messages.create() → client.chat.completions.create()
        │  ・system= パラメータ → messages 先頭に role:"system"
        │  ・Tool Use(input_schema) → Tool Calls(parameters)
        │  ・stop_reason=="tool_use" → finish_reason=="tool_calls"
        │  ・tool_result(role:user 1件) → role:tool 個別追記
        │  ・構造化出力: Tool Use → beta.chat.completions.parse()
        ▼
OpenAI API (openai)  ← 現在の本番環境
        │
        └── Embedding: text-embedding-3-large（変更なし）
```

### 3段階移植の差異まとめ

| 操作 | Gemini | Anthropic | OpenAI（現在） |
|---|---|---|---|
| クライアント | `genai.Client()` | `anthropic.Anthropic()` | `OpenAI()` |
| テキスト生成 | `models.generate_content()` | `messages.create()` | `chat.completions.create()` |
| system prompt | `config.system_instruction` | `system=` パラメータ | `messages[0].role="system"` |
| 構造化出力 | `response_schema=PydanticClass` | Tool Use 強制 | `beta.parse(response_format=)` |
| ツール定義キー | `"parameters"` | `"input_schema"` | `"parameters"`（戻る） |
| ツール検出 | `part.function_call` 走査 | `stop_reason=="tool_use"` | `finish_reason=="tool_calls"` |
| ツール結果 | `Part.from_function_response()` | `role:"user"` にまとめ | `role:"tool"` 個別追記 |
| 終了判定 | function_call なし | `"end_turn"` | `"stop"` |
| レスポンス取得 | `response.text` | `response.content[0].text` | `response.choices[0].message.content` |
| トークン数 | `response.usage_metadata.prompt_token_count` | `response.usage.input_tokens` | `response.usage.prompt_tokens` |
| 会話管理 | `chat` オブジェクト（自動） | `messages` リスト（自前） | `messages` リスト（自前、同じ） |

---

*本ドキュメントは `openai_grace_agent` 移植作業の完了報告書兼技術参照資料として使用する。*
*Gemini → Anthropic → OpenAI の3段階移植の完全な記録として後続プロジェクトでも再利用可能。*
