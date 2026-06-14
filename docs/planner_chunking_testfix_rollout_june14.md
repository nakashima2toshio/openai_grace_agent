# planner ロジック移植漏れ(B)・chunking テスト不整合(C) 横展開ガイド

`openai_grace_agent` の `uv run pytest tests/` で顕在化したテスト失敗のうち、コード/テスト起因の
**B（planner 4件）/ C（chunking 2件）** を PR #21 で修正した。本書はこの修正を
`ollama_grace_agent` / `anthropic_grace_agent` へ展開すべきかの**判定と手順**をまとめる。

- 作成日: 2026-06-14
- 基準（ロジックの正）: `anthropic_grace_agent`
- 修正元: `openai_grace_agent` PR #21（`grace/planner.py`・`tests/chunking/test_document_chunking.py`）

---

## 0. 結論（横展開要否サマリ）

| 項目 | anthropic | ollama | openai |
|---|---|---|---|
| **B** planner: `_is_transient_error` / `_generate_plan_with_retry` / `refine_plan` 完全JSON化 | ✅ 既に実装済み（**正**）→ 対応不要 | ❌ **移植漏れあり → 要対応** | ✅ 修正済（#21） |
| **C** chunking テスト mock の `contents` parse | ✅ mock と実装が整合 → 対応不要 | ✅ 整合 → 対応不要 | ⚠️ 実装が anthropic/ollama と乖離（#21 は mock 側で解消） |

> **要点**: B は **ollama のみ要対応**。C は **openai 固有**（anthropic/ollama は波及しない）。

---

## 1. B — planner ロジック移植漏れ

### 1.1 症状（openai で観測されたテスト失敗）

```
FAILED tests/grace/test_planner.py::test_retry_on_transient_error
        - AttributeError: 'Planner' object has no attribute '_generate_plan_with_retry'
FAILED tests/grace/test_planner.py::test_no_retry_on_non_transient_error  - assert 0 == 1
FAILED tests/grace/test_planner.py::test_is_transient_error
        - AttributeError: type object 'Planner' has no attribute '_is_transient_error'
FAILED tests/grace/test_planner.py::test_refine_plan  - assert '検索クエリXYZ' in prompt
```

### 1.2 原因

`anthropic_grace_agent` で入った planner のロジック改善が移植されていなかった。

| 改善 | anthropic（正） | 未移植側（openai 旧 / ollama 現状） |
|---|---|---|
| リトライのメソッド抽出＋一時的エラー判定 | `_generate_plan_with_retry()` ＋ `_is_transient_error()`（429/timeout/5xx のみ再試行、認証エラー等は即送出） | `create_plan`/`_create_llm_plan` 内に**インライン** `for attempt in range(2)`。全エラーを無差別に再試行、抽出メソッド無し |
| `refine_plan` のプロンプト | 計画の**完全 JSON**（`plan.model_dump_json(exclude={"created_at","plan_id"})`）を埋め込み、query/依存/fallback を保持 | `ステップ: {[s.description for s in plan.steps]}`（**説明文のみ**）→ 修正後計画から query 等が欠落 |

### 1.3 修正内容（openai #21）

`grace/planner.py` に以下を追加・変更（`anthropic_grace_agent/grace/planner.py` L216-282, L389-403 を正に移植）:

1. `@staticmethod _is_transient_error(error)` — status_code（408/409/429/5xx）またはメッセージ
   マーカー（timeout/ratelimit/connection/429/5xx 等）で一時的エラーを判定。
2. `_generate_plan_with_retry(prompt)` — `config.error`（max_retries/retry_delay_base/
   retry_delay_max/exponential_backoff）に従い、**一時的エラーのみ**指数バックオフ再試行。
   非一時的エラーは即 `raise`。`_create_llm_plan` のインライン retry をこの呼び出しに置換。
3. `refine_plan` のプロンプトを完全 JSON 埋め込みに変更。

### 1.4 横展開判定

- **anthropic**: `grace/planner.py` に `_generate_plan_with_retry`(L232)・`_is_transient_error`(L216)・
  `refine_plan` 完全JSON(L395) が**既に存在**（これが移植元）。→ **対応不要**。
- **ollama**: `grace/planner.py` に両メソッドが**無く**、`_create_llm_plan` はインライン retry
  （`max_attempts = 2`）、`refine_plan` は `ステップ: {[s.description ...]}`。→ **要対応**。
  - 注: ollama の `tests/grace/test_planner.py` には B 関連テスト（`test_retry_on_transient_error` 等）が
    **未移植**のため、現状はテストが失敗しない。ロジック移植に加え、テストも移植すると回帰検知が効く。

### 1.5 ollama への移植手順（provider 値読み替え）

`anthropic_grace_agent/grace/planner.py` の該当ロジックを移植。**provider 値は ollama を維持**:

| 項目 | anthropic | **ollama（維持する値）** |
|---|---|---|
| LLM クライアント | `create_llm_client("anthropic")` | `create_llm_client("ollama")` |
| トークン上限引数 | `max_tokens=8192` | `max_tokens=8192`（**Ollama は `max_completion_tokens` 非対応**・現状維持） |
| temperature | `temperature=self.config.llm.temperature` | 同左（現状の planner に合わせる） |

手順:
1. `_is_transient_error`（static）と `_generate_plan_with_retry` を追加（ollama の `generate_structured`
   引数＝`max_tokens` を使用）。
2. `_create_llm_plan` のインライン retry ループを `self._generate_plan_with_retry(prompt)` 呼び出しに置換。
3. `refine_plan` のプロンプトを `plan.model_dump_json(indent=2, exclude={"created_at","plan_id"})` 埋め込みに変更。
4. （任意・推奨）anthropic の `tests/grace/test_planner.py` から
   `test_retry_on_transient_error`/`test_no_retry_on_non_transient_error`/`test_is_transient_error`/
   `test_refine_plan` を移植。
5. 検証: `uv run pytest tests/grace/test_planner.py`。

---

## 2. C — chunking テスト mock の `contents` 不整合（openai 固有）

### 2.1 症状（openai）

```
FAILED tests/chunking/test_document_chunking.py::TestEndToEnd::test_document_boundary_preserved
        - AssertionError: assert 7 == 3   （チャンクにプロンプト文が混入し増殖）
FAILED tests/chunking/test_document_chunking.py::TestEndToEnd::test_manifest_written_with_coverage
        - AssertionError: assert [0,0,0,1,2,2,...] == [0,1,2]
```

### 2.2 原因（実装の prompt 渡し方の乖離）

テストの `FakeAPIClient.generate_content` は `body = contents.split("\n", 1)[1]`（先頭1行除去）で
入力本文を取り出す前提。これは **`contents` が「`【入力テキスト】\n{text}`」形式**であることを想定している。

| repo | chunking 実装の `contents` | mock `split("\n",1)[1]` の結果 |
|---|---|---|
| **anthropic** | `f"【入力テキスト】\n{block}"`（プロンプトは `system=` で渡す） | `{block}` ✅ 正しい |
| **ollama** | `f"【入力テキスト】\n{block}"`（同上） | `{block}` ✅ 正しい |
| **openai** | `f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{block}"`（**プロンプトを contents に連結**） | プロンプト本文 ❌ → チャンクにプロンプトが混入 |

→ **真因は openai の Phase E 実装が、プロンプトを `system=` ではなく `contents` に連結**していたこと。
anthropic/ollama は `system=` でプロンプトを渡し `contents` は本文のみのため、同一 mock でも整合する。

### 2.3 修正内容（openai #21）

`tests/chunking/test_document_chunking.py` の `FakeAPIClient.generate_content` を、実装契約に合わせ
**`【入力テキスト】` 以降を抽出**するよう変更:

```python
# 修正前: body = contents.split("\n", 1)[1] if "\n" in contents else contents
body = contents.rsplit("【入力テキスト】", 1)[-1].strip()
```

### 2.4 横展開判定

- **anthropic / ollama**: 実装が `contents = 【入力テキスト】\n{text}` のため**現行 mock で整合**し、
  当該テストは失敗しない。→ **対応不要**。
- **openai**: #21 で mock 側を実装に合わせて解消済み。

### 2.5 推奨（openai のフォローアップ・任意）

openai の chunking 実装は anthropic/ollama と異なり**プロンプトを `contents` に連結**している。
将来的な統一の観点では、**openai の実装を anthropic/ollama パターン（プロンプトは `system=`、
`contents` は `【入力テキスト】\n{text}` のみ）に揃える**と、mock も他 repo と完全一致し保守性が上がる。
（#21 は mock 側で解消済みのため、これは任意のリファクタ。）

---

## 3. 横展開チェックリスト

### ollama_grace_agent（要対応: B）
- [ ] `grace/planner.py` に `_is_transient_error` 追加
- [ ] `grace/planner.py` に `_generate_plan_with_retry` 追加（`max_tokens` 使用）
- [ ] `_create_llm_plan` のインライン retry を `_generate_plan_with_retry` 呼び出しに置換
- [ ] `refine_plan` を完全 JSON プロンプトに変更
- [ ]（任意）`tests/grace/test_planner.py` に B 関連テストを移植
- [ ] `uv run pytest tests/grace/test_planner.py tests/chunking/` で検証
- [ ] C は対応不要（mock と実装が整合）

### anthropic_grace_agent
- [ ] B 対応不要（既に実装済み・移植元）
- [ ] C 対応不要（mock と実装が整合）
- [ ]（健全性確認）`uv run pytest tests/grace/test_planner.py tests/chunking/` が緑であること

### openai_grace_agent
- [x] B 修正済（#21）
- [x] C 修正済（#21・mock 側）
- [ ]（任意）chunking 実装を `system=` パターンへ揃えるフォローアップ

---

## 4. 再発防止

これらの不整合（移植漏れ・mock 不一致）は、CI が `ruff + compileall` のみで**実 pytest を回していない**
ため検出されなかった。各 repo の CI に **pytest ジョブ**を追加すると自動検知できる:

```yaml
# .github/workflows/ci.yml （build ジョブ後に追加する例）
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - run: pip install uv
      - run: uv sync
      # 実 LLM/Qdrant 依存テストは env ゲートで除外（API課金・サービス不要の単体のみ）
      - run: uv run pytest tests/ -m "not integration" -q
```

> 実 LLM/Qdrant/課金が前提の統合テスト（例: openai の `test_executor_integration`）は
> `OPENAI_API_KEY` 等の env ゲートで除外し、`workflow_dispatch` 等で別途実行する。

---

*作成日: 2026-06-14 / 参照: openai_grace_agent PR #21*
