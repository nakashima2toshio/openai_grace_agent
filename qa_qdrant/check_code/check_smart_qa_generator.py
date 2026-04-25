#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_smart_qa_generator.py - SmartQAGenerator 学習用プログラム

🎯 目的:
    SmartQAGeneratorクラスの動作を理解するための学習プログラム。
    チャンク済みテキストからQ/Aペアが生成される過程を段階的に確認できます。

📚 学習内容:
    1. analyze_chunk() - チャンクを分析してQ/A数を決定
    2. generate_qa_pairs() - 分析結果に基づいてQ/Aペアを生成
    3. process_chunk() - 上記2つを一括実行

📋 前提条件:
    - GOOGLE_API_KEY 環境変数が設定されていること

使用例:
    python qa_qdrant/check_code/check_smart_qa_generator.py
"""

import os
import sys
import csv
from pathlib import Path

# プロジェクトルートをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qa_generation.smart_qa_generator import SmartQAGenerator


def print_separator(title: str, char: str = "=", width: int = 70):
    """セパレーターを出力"""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_subseparator(title: str, char: str = "-", width: int = 50):
    """サブセパレーターを出力"""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def load_sample_chunks(csv_path: str, num_chunks: int = 5) -> list[dict]:
    """
    チャンク済みCSVから指定数のサンプルを読み込む

    Args:
        csv_path: チャンク済みCSVファイルのパス
        num_chunks: 読み込むチャンク数

    Returns:
        チャンクのリスト
    """
    chunks = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_chunks:
                break
            chunks.append({
                'id': row.get('chunk_id', f'chunk_{i}'),
                'text': row.get('text', ''),
                'tokens': int(row.get('tokens', 0)),
            })
    return chunks


def demo_analyze_chunk(generator: SmartQAGenerator, chunk_text: str, chunk_id: str):
    """
    analyze_chunk() のデモ

    🔍 このメソッドの役割:
        - チャンクの内容を分析
        - 情報密度、重要度、複雑さを評価
        - 適切なQ/A数（0-5個）を決定
    """
    print_subseparator("Step 1: analyze_chunk() - チャンク分析")
    print(f"📝 チャンクID: {chunk_id}")
    print(f"📝 テキスト長: {len(chunk_text)} 文字")
    print(f"📝 テキスト（先頭100文字）:\n   {chunk_text[:100]}...")
    print()

    # 分析を実行
    print("🔄 分析を実行中...")
    analysis = generator.analyze_chunk(chunk_text)

    # 結果を表示
    print("\n📊 【分析結果】")
    print(f"   • Q/A数         : {analysis.get('qa_count', 0)} 個")
    print(f"   • 重要度スコア  : {analysis.get('importance_score', 0):.2f}")
    print(f"   • 複雑さ        : {analysis.get('complexity', 'N/A')}")
    print(f"   • 主要トピック  : {', '.join(analysis.get('key_topics', [])) or 'なし'}")
    print(f"   • 判断理由      : {analysis.get('reasoning', 'N/A')}")

    return analysis


def demo_generate_qa_pairs(generator: SmartQAGenerator, chunk_text: str, analysis: dict):
    """
    generate_qa_pairs() のデモ

    🔍 このメソッドの役割:
        - 分析結果に基づいてQ/Aペアを生成
        - qa_count=0 の場合は空リストを返す
        - 主要トピックを優先的にカバー
    """
    print_subseparator("Step 2: generate_qa_pairs() - Q/Aペア生成")

    qa_count = analysis.get('qa_count', 0)
    if qa_count == 0:
        print("⏭️  Q/A数が0のため、生成をスキップします")
        return []

    print(f"🔄 {qa_count}個のQ/Aペアを生成中...")

    # Q/Aペアを生成
    qa_pairs = generator.generate_qa_pairs(chunk_text, analysis)

    # 結果を表示
    print(f"\n📋 【生成されたQ/Aペア】 ({len(qa_pairs)}個)")
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n   Q{i} [{qa.get('topic', 'N/A')}]:")
        print(f"      質問: {qa.get('question', 'N/A')}")
        print(f"      回答: {qa.get('answer', 'N/A')}")

    return qa_pairs


def demo_process_chunk(generator: SmartQAGenerator, chunk_text: str, chunk_id: str):
    """
    process_chunk() のデモ

    🔍 このメソッドの役割:
        - analyze_chunk() と generate_qa_pairs() を一括実行
        - 実際のパイプラインではこのメソッドが使用される
    """
    print_subseparator("Step 3: process_chunk() - 一括処理")
    print(f"📝 チャンクID: {chunk_id}")
    print(f"📝 テキスト長: {len(chunk_text)} 文字")
    print()

    print("🔄 一括処理を実行中...")
    result = generator.process_chunk(chunk_text)

    # 結果を表示
    if result['success']:
        print("\n✅ 処理成功!")
        print(f"\n📊 【分析結果】")
        analysis = result['analysis']
        print(f"   • Q/A数: {analysis.get('qa_count', 0)} 個")
        print(f"   • 重要度: {analysis.get('importance_score', 0):.2f}")

        print(f"\n📋 【生成Q/A数】: {len(result['qa_pairs'])} 個")
        for i, qa in enumerate(result['qa_pairs'], 1):
            print(f"   Q{i}: {qa.get('question', 'N/A')[:50]}...")
    else:
        print("\n❌ 処理失敗")

    return result


def main():
    """メイン関数"""

    print_separator("SmartQAGenerator 学習プログラム", "=", 70)

    # ================================================================
    # APIキー確認
    # ================================================================
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ エラー: GOOGLE_API_KEY が設定されていません")
        print("   export GOOGLE_API_KEY='your-api-key'")
        sys.exit(1)

    # ================================================================
    # サンプルデータの準備
    # ================================================================
    print_separator("1. サンプルデータの準備", "-", 50)

    # チャンク済みCSVのパス（自動検出）
    chunked_dir = PROJECT_ROOT / "output_chunked"
    csv_files = list(chunked_dir.glob("*.csv"))

    if csv_files:
        csv_path = csv_files[0]
        print(f"📁 CSVファイル: {csv_path}")
        chunks = load_sample_chunks(str(csv_path), num_chunks=3)
        print(f"📊 読み込んだチャンク数: {len(chunks)}")
    else:
        print(f"⚠️ CSVファイルが見つかりません: {chunked_dir}")
        print("   ハードコードされたサンプルデータを使用します")
        # フォールバック: ハードコードされたサンプル
        chunks = [
            {
                'id': 'sample_chunk_1',
                'text': """AES-256暗号化アルゴリズムは、対称鍵暗号方式の一種で、
                256ビットの鍵長を持ちます。NIST（米国国立標準技術研究所）
                により承認されており、機密情報の保護に広く使用されています。
                ブロック暗号として動作し、128ビットのブロックサイズで
                データを処理します。""",
                'tokens': 120
            },
            {
                'id': 'sample_chunk_2',
                'text': "この製品は赤色です。",
                'tokens': 10
            },
            {
                'id': 'sample_chunk_3',
                'text': "詳細については付録Aを参照してください。",
                'tokens': 15
            }
        ]

    # ================================================================
    # SmartQAGenerator の初期化
    # ================================================================
    print_separator("2. SmartQAGenerator の初期化", "-", 50)

    model = "gemini-3.0-flash"
    print(f"🤖 モデル: {model}")

    generator = SmartQAGenerator(model=model)
    print("✅ 初期化完了")

    # ================================================================
    # デモ1: 詳細なステップバイステップ処理（1つ目のチャンク）
    # ================================================================
    print_separator("3. ステップバイステップ処理デモ", "=", 70)

    chunk = chunks[0]
    print(f"🎯 対象チャンク: {chunk['id']}")
    print(f"   トークン数: {chunk['tokens']}")

    # Step 1: 分析
    analysis = demo_analyze_chunk(generator, chunk['text'], chunk['id'])

    # Step 2: Q/Aペア生成
    qa_pairs = demo_generate_qa_pairs(generator, chunk['text'], analysis)

    # ================================================================
    # デモ2: 一括処理（残りのチャンク）
    # ================================================================
    print_separator("4. 一括処理デモ (process_chunk)", "=", 70)

    for chunk in chunks[1:]:
        demo_process_chunk(generator, chunk['text'], chunk['id'])

    # ================================================================
    # サマリー
    # ================================================================
    print_separator("5. 学習のポイント", "=", 70)

    print("""
📚 SmartQAGenerator の処理フロー:

    ┌─────────────────┐
    │  チャンクテキスト  │
    └────────┬────────┘
             │
             ▼
    ┌───────────────────┐
    │  analyze_chunk()  │  ← LLMがチャンクを分析
    │  ・情報密度評価     │    Q/A数（0-5）を決定
    │  ・重要度スコア     │
    │  ・主要トピック抽出 │
    └────────┬─────────┘
             │
             ▼
    ┌─────────────────┐
    │generate_qa_pairs()│  ← 分析結果に基づきQ/A生成
    │  ・質問生成       │    qa_count=0なら生成しない
    │  ・回答生成       │
    │  ・トピック付与   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Q/Aペアリスト   │
    └─────────────────┘

💡 ポイント:
    1. analyze_chunk() で「いくつQ/Aを作るべきか」を先に判断
    2. メタ情報のみのチャンクは qa_count=0 でスキップ
    3. 重要度が高いチャンクは詳細なQ/Aを生成
    4. process_chunk() は上記2つを一括実行（実運用向け）
""")

    print_separator("✅ 学習プログラム完了", "=", 70)


if __name__ == "__main__":
    main()
