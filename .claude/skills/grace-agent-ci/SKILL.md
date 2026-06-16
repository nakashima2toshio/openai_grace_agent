---
name: grace-agent-ci
description: >-
  Work with CI, branching, auto-merge, and PR workflow in the *_grace_agent
  repos. Use when editing .github/workflows/ci.yml, when ruff/lint blocks a PR,
  when configuring or relying on the claude/* auto-merge, or when creating
  branches/PRs via the GitHub MCP tools. Encodes the CI gate design, ruff
  config gotchas, and the remote-environment/PR conventions.
---

# grace_agent CI・自動マージ・PR運用スキル

## CI 構成（`.github/workflows/ci.yml`）
- **build（ブロッキング）** = `ruff check .`（全リポジトリ）＋ `python -m compileall -q -x '\.venv|/old_code/|/\.git/' .`。
- **unit-tests（advisory / `continue-on-error: true`）** = `pytest tests/ -q -rs`（実APIキー無し → 統合テストは skipif で自動skip）。安定したら blocking 化して auto-merge の `needs` に追加する余地あり。
- **auto-merge** = `build` 成功後、`head_ref` が `claude/*` の PR を master へマージ（`gh pr ready` → `gh pr merge --merge`）。`hold` ラベルが付いた PR は対象外。
- トリガ: `pull_request`(master宛, types に `ready_for_review`/`labeled` 含む) と `push`(master)。push 時は lint/compile のみ。

## ruff 設定の要点（環境差バグ回避）
- `pyproject.toml` `[tool.ruff.lint]` select=`E,F,I` / ignore=`E501`。
- **`[tool.ruff.lint.isort] known-first-party` を明示必須**。未設定だと「CI(未インストール)＝first-party」「ローカル(導入済)＝third-party」で isort 分類が割れ、I001 がローカル緑/CI赤になる。リポジトリのトップレベル module/package を列挙しておく。
- ローカル検証は `ruff check . --no-cache`。負債を一括解消する場合は安全fix（F401/I001/F541）を `ruff check . --fix`、残り（E402/E701/E722/E741/F841/F811）は手動。E402 は sys.path 後の意図的importなら `# noqa: E402`。

## ブランチ・PR 運用
- 開発は `claude/<topic>` ブランチ。**ドラフトPRで作成**（auto-merge が Ready 化してマージ）。bootstrap 用に自己マージさせたくない変更（ci.yml 自体など）は `ci/*` 等 `claude/` 以外の名前にする。
- master への直 push は許可（ブランチ保護なし）。CIが緑にならない＝マージ不可は build ジョブのみ。
- GitHub 操作は **`mcp__github__*` MCP ツール**（`gh` CLI 不可）。ToolSearch で都度ロード。スコープ外repoは `mcp__claude-code-remote__list_repos`/`add_repo`。
- **リモートGitプロキシは ref 削除を 403 で拒否**（`git push --delete` 不可、MCPにも削除系なし）→ ブランチ削除は GitHub UI かユーザのローカルで。
- commit メッセージ末尾・PR本文末尾に session リンクを付与（ハーネス規約）。PRは作成後ドラフトで、ユーザ確認不要。

## リモート実行環境
- コンテナは ephemeral・起動時に fresh clone。**コミット＆プッシュしないと消える**。
- `uv run` で依存解決可能（pytest 実走に利用）。`docker-compose/docker-compose.yml` が Qdrant、`start_celery.sh` が Celery+Redis。

## PRアクティビティ購読
- `subscribe_pr_activity` で CI失敗・レビューコメントを受信。CI成功・新push・コンフリクト遷移は webhook で来ないため、必要なら `send_later` で約1時間後の自己チェックインを再アーム（このサンドボックスでは `send_later` 不在のことが多い）。
