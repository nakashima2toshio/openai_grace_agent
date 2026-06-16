---
name: grace-agent-docs
description: >-
  Author or update Japanese module documentation for the *_grace_agent repos
  (anthropic/openai/gemini/ollama). Use when writing or modernizing docs under
  <package>/doc/*.md or the top-level readme_*.md / docs/*.md, when asked to
  follow `a_class_method_md_format.md`, or when adding Mermaid diagrams to these
  repos. Encodes the IPO doc format, the mandatory black-background Mermaid
  style, and the unified tech-stack terminology.
---

# grace_agent ドキュメント作成スキル

日本語RAG/GRACEプロジェクト群（`anthropic_grace_agent` ほか）のモジュールドキュメントを、
プロジェクト規約どおりに作成・最新化するための知見。

## 1. フォーマット仕様（必読）
- 仕様書はリポジトリ直下 `a_class_method_md_format.md`（IPO形式）。**先に読むこと**。
- タイトル: `# <module>.py - <説明> ドキュメント` → 次行 `**Version X.X** | 最終更新: YYYY-MM-DD`。
- 必須セクション順:
  1. 目次
  2. 概要（`### 主な責務` 箇条書き → `### 各責務対応のモジュール` 表 → `### 主要機能一覧` 表）
  3. アーキテクチャ構成図（Mermaid・3層）
  4. モジュール構成図（Mermaid）
  5. クラス・関数一覧表
  6. クラス・関数 IPO詳細：各要素に **概要 / シグネチャ / パラメータ表 / IPOテーブル(Input・Process・Output) / 戻り値例 / 使用例** を必ず付ける
  7. 設定・定数（あれば）
  8. 使用例（ワークフロー）
  9. エクスポート（`__all__`）
  10. 変更履歴（表。版を上げたら必ず追記）
  11. 付録: 依存関係図（Mermaid）
- 横断的な「まとめ」ドキュメントは IPO を各モジュール doc に委ね、本文はアーキテクチャ＋データフロー＋リンク集に徹してよい（例: `grace/doc/agent_rag_grace.md`）。

## 2. Mermaid 黒背景・白文字（CLAUDE.md §7 / 仕様書 §16.5）— 必須
- flowchart/graph はブロック末尾に必ず:
  - `classDef default fill:#000,stroke:#fff,color:#fff`
  - `classDef subgraphStyle fill:#1a1a1a,stroke:#fff,color:#fff`
  - 全ノード `class <id,...> default`
  - 各サブグラフ `style <Subgraph> fill:#1a1a1a,stroke:#fff,color:#fff`
- sequenceDiagram は先頭に `%%{ init: { "theme":"base", "themeVariables": { ...黒テーマ... } } }%%` を付け、`classDef`/`class` は使わない。
- ノードラベルの特殊文字はダブルクォートで囲む。バッククォート禁止。`<br>` は可。
- 検証（grep）: 各ファイルで `flowchart|graph` の数 == `classDef default fill:#000` の数、`sequenceDiagram` の数 == `%%{ init` の数。

## 3. 技術スタック表記の統一（CLAUDE.md §9.1）
- LLM = **Anthropic Claude**、既定 `claude-sonnet-4-6`（軽量 `claude-haiku-4-5-20251001`）。鍵 `ANTHROPIC_API_KEY`。
- Embedding = **Gemini** `gemini-embedding-001`（3072次元）。鍵 `GOOGLE_API_KEY`/`GEMINI_API_KEY`。
- LLM設定クラスは `ModelConfig`。`text-embedding-3-*` を LLM/本番Embedding用途で書かない。
- モデル名マッピングを作らない（CRITICAL RULES）。`responses.parse()`/`create()` は両方正。

## 4. 実装との整合（重要）
- 書く前に**対応ソースを実際に読む**。シグネチャ・既定値・`__all__` を突合。
- 廃止ファイルを参照しない: `setup.py` / `server.py` / a-prefixed scripts（`a30_qdrant_registration.py` 等）は**存在しない**。現行は
  - チャンク化: `python -m chunking.csv_text_to_chunks_text_csv`
  - Q/A生成+登録: `qa_qdrant/make_qa_register_qdrant.py`（登録のみ `register_to_qdrant.py`）
  - UI: `streamlit run agent_rag.py`
- 現行パイプラインは3段階（チャンキング→Q/A生成→Qdrant登録）。チャンキングは文書境界保証（`load_documents_from_csv`/`doc_id`）・`continuity_mode="rule"`・`max_chunk_tokens=512`・manifest出力。

## 5. ドキュメントの所在
- モジュール個別ドキュメント: `<package>/doc/<module>.md`（例 `chunking/doc/`, `qa_generation/doc/`, `grace/doc/`, `qa_qdrant/doc/`）。
- 横断/利用ガイド: リポジトリ直下 `readme_*.md`・`docs/*.md`。

## 6. 進め方のコツ
- 複数ファイルを最新化するときは **ファイルごとにサブエージェントを並列起動**（各に「フォーマット仕様パス＋対象ソース＋黒背景Mermaid規約＋スタック表記」を渡す）。
- 仕上げに mermaid 準拠を grep 検証し、版・最終更新日・変更履歴を更新。
