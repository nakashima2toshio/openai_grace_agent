#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_qapipeline.py - QAPipeline 学習用プログラム

🎯 目的:
    QAPipelineクラスの動作を理解するための学習プログラム。
    パイプライン全体の処理フロー（CSV読み込み→チャンク変換→Q/A生成→保存）を
    段階的に確認できます。

📚 学習内容:
    1. QAPipeline の初期化
    2. load_data() - CSVデータの読み込み
    3. _load_chunks_from_csv() - チャンクリストへの変換
    4. generate_qa() - Q/Aペアの生成（SmartQAGenerator使用）
    5. save() - 結果の保存

📋 前提条件:
    - GOOGLE_API_KEY 環境変数が設定されていること
    - チャンク済みCSVファイルが存在すること

⚠️ 注意:
    - このプログラムは学習用のため、処理チャンク数を制限しています（デフォルト: 3チャンク）
    - APIコストを抑えるため、--max-chunks オプションで制御できます

使用例:
    # デフォルト（3チャンク処理）
    python qa_qdrant/check_code/check_qapipeline.py

    # チャンク数を指定
    python qa_qdrant/check_code/check_qapipeline.py --max-chunks 5

    # 保存をスキップ（表示のみ）
    python qa_qdrant/check_code/check_qapipeline.py --no-save
"""

import os
import sys
import argparse
from pathlib import Path

# プロジェクトルートをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qa_generation.pipeline import QAPipeline


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


def demo_initialization(input_file: str, max_docs: int, output_dir: str):
    """
    QAPipeline の初期化デモ

    🔍 初期化時に行われること:
        - 入力パラメータの検証 (_validate_inputs)
        - 設定のロード (_load_config)
        - SmartQAGenerator の初期化
    """
    print_subseparator("Step 1: QAPipeline 初期化")

    print(f"📁 入力ファイル: {input_file}")
    print(f"📊 最大チャンク数: {max_docs}")
    print(f"📂 出力ディレクトリ: {output_dir}")
    print(f"🤖 モデル: gemini-2.0-flash")
    print()

    print("🔄 初期化中...")
    print("   → _validate_inputs(): 入力パラメータを検証")
    print("   → _load_config(): 設定をロード")
    print("   → SmartQAGenerator(): Q/Aジェネレーターを初期化")

    pipeline = QAPipeline(
        input_file=input_file,
        model="gemini-3.0-flash",
        output_dir=output_dir,
        max_docs=max_docs
    )

    print("\n✅ 初期化完了!")
    print(f"   • 設定名: {pipeline.config.get('name', 'N/A')}")
    print(f"   • テキストカラム: {pipeline.config.get('text_column', 'N/A')}")
    print(f"   • 言語: {pipeline.config.get('lang', 'N/A')}")
    print(f"   • Q/A数/チャンク: {pipeline.config.get('qa_per_chunk', 'N/A')}")

    return pipeline


def demo_load_data(pipeline: QAPipeline):
    """
    load_data() のデモ

    🔍 このメソッドの役割:
        - CSVファイルを読み込む (load_uploaded_file)
        - max_docs で指定された数に制限
        - DataFrameとして返す
    """
    print_subseparator("Step 2: load_data() - データ読み込み")

    print("🔄 データを読み込み中...")
    print("   → load_uploaded_file(): CSVファイルを読み込み")
    df = pipeline.load_data()

    print(f"\n✅ 読み込み完了!")
    print(f"   • 行数: {len(df)}")
    print(f"   • カラム: {list(df.columns)}")
    print(f"\n📋 先頭3行のプレビュー:")

    # 先頭3行を表示（textカラムは短縮）
    for i, row in df.head(3).iterrows():
        text = str(row.get('text', row.get('Combined_Text', '')))[:50]
        chunk_id = row.get('chunk_id', f'row_{i}')
        print(f"   [{chunk_id}] {text}...")

    return df


def demo_load_chunks(pipeline: QAPipeline, df):
    """
    _load_chunks_from_csv() のデモ

    🔍 このメソッドの役割:
        - DataFrameをチャンクリスト（List[Dict]）に変換
        - テキストカラムの自動検出 ('text', 'Combined_Text', 'content', 'chunk_text')
        - IDカラムの自動検出 ('chunk_id', 'id', 'chunk_idx')
    """
    print_subseparator("Step 3: _load_chunks_from_csv() - チャンク変換")

    print("🔄 チャンクリストに変換中...")
    print("   → テキストカラムを自動検出")
    print("   → IDカラムを自動検出")
    chunks = pipeline._load_chunks_from_csv(df)

    print(f"\n✅ 変換完了!")
    print(f"   • チャンク数: {len(chunks)}")
    print(f"\n📋 チャンクの構造（最初の1件）:")

    if chunks:
        chunk = chunks[0]
        print(f"   {{")
        print(f"       'id': '{chunk.get('id', 'N/A')}',")
        print(f"       'text': '{chunk.get('text', 'N/A')[:40]}...',")
        print(f"       'tokens': {chunk.get('tokens', 'N/A')},")
        print(f"       'type': '{chunk.get('type', 'N/A')}',")
        print(f"       'dataset_type': '{chunk.get('dataset_type', 'N/A')}'")
        print(f"   }}")

    return chunks


def demo_generate_qa(pipeline: QAPipeline, chunks: list):
    """
    generate_qa() のデモ（同期処理モード）

    🔍 このメソッドの役割:
        - SmartQAGenerator を使用してQ/Aペアを生成
        - 各チャンクを順次処理 (_generate_sync)
        - 結果をリストとして返す

    🔍 内部で呼ばれるメソッド:
        - _generate_sync(): 同期処理（Celery不使用時）
        - _generate_with_celery(): 非同期並列処理（Celery使用時）
    """
    print_subseparator("Step 4: generate_qa() - Q/A生成")

    print(f"🔄 {len(chunks)} チャンクからQ/Aを生成中...")
    print("   → _generate_sync(): 同期処理モードで実行")
    print("   → SmartQAGenerator.process_chunk(): 各チャンクを処理")
    print()

    # 同期処理モードで実行
    qa_pairs = pipeline.generate_qa(
        chunks,
        use_celery=False,  # Celeryは使用しない（学習用）
        batch_chunks=3,
        use_smart_generation=True
    )

    print(f"\n✅ 生成完了!")
    print(f"   • 生成Q/A数: {len(qa_pairs)}")

    print(f"\n📋 生成されたQ/Aペア:")
    for i, qa in enumerate(qa_pairs[:5], 1):  # 最大5件表示
        print(f"\n   [{i}] チャンク: {qa.get('chunk_id', 'N/A')}")
        print(f"       トピック: {qa.get('topic', 'N/A')}")
        print(f"       Q: {qa.get('question', 'N/A')[:60]}...")
        print(f"       A: {qa.get('answer', 'N/A')[:60]}...")

    if len(qa_pairs) > 5:
        print(f"\n   ... 他 {len(qa_pairs) - 5} 件")

    return qa_pairs


def demo_save(pipeline: QAPipeline, qa_pairs: list, do_save: bool):
    """
    save() のデモ

    🔍 このメソッドの役割:
        - Q/Aペアをファイルに保存 (save_results)
        - CSVファイルとサマリーJSONを生成
    """
    print_subseparator("Step 5: save() - 結果保存")

    if not do_save:
        print("⏭️  --no-save オプションが指定されたため、保存をスキップします")
        return None

    print("🔄 結果を保存中...")
    print("   → save_results(): CSV と サマリーJSON を生成")

    # カバレージ結果（ダミー）
    coverage_results = {
        "coverage_rate": 0.0,
        "covered_chunks": 0,
        "total_chunks": len(qa_pairs),
        "uncovered_chunks": []
    }

    saved_files = pipeline.save(qa_pairs, coverage_results)

    print(f"\n✅ 保存完了!")
    print(f"   • サマリー: {saved_files.get('summary', 'N/A')}")
    print(f"   • Q/A CSV: {saved_files.get('qa_csv', 'N/A')}")

    return saved_files


def main():
    """メイン関数"""

    # ================================================================
    # 引数解析
    # ================================================================
    parser = argparse.ArgumentParser(
        description="QAPipeline 学習用プログラム",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=3,
        help="処理する最大チャンク数（デフォルト: 3、APIコスト節約のため）"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="結果の保存をスキップ（表示のみ）"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="入力CSVファイルのパス（省略時は自動検出）"
    )

    args = parser.parse_args()

    print_separator("QAPipeline 学習プログラム", "=", 70)

    # ================================================================
    # APIキー確認
    # ================================================================
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ エラー: GOOGLE_API_KEY が設定されていません")
        print("   export GOOGLE_API_KEY='your-api-key'")
        sys.exit(1)

    # ================================================================
    # 入力ファイルの決定
    # ================================================================
    print_separator("0. 入力ファイルの確認", "-", 50)

    if args.input_file:
        input_file = args.input_file
    else:
        # 自動検出
        chunked_dir = PROJECT_ROOT / "output_chunked"
        csv_files = list(chunked_dir.glob("*.csv"))

        if csv_files:
            input_file = str(csv_files[0])
            print(f"📁 自動検出されたCSVファイル: {input_file}")
        else:
            print(f"❌ エラー: {chunked_dir} にCSVファイルが見つかりません")
            sys.exit(1)

    if not Path(input_file).exists():
        print(f"❌ エラー: ファイルが見つかりません: {input_file}")
        sys.exit(1)

    print(f"✅ 入力ファイル: {input_file}")
    print(f"📊 処理チャンク数: {args.max_chunks}")

    # 出力ディレクトリ
    output_dir = str(PROJECT_ROOT / "qa_output" / "check_pipeline")

    # ================================================================
    # パイプライン処理のデモ
    # ================================================================

    print_separator("パイプライン処理開始", "=", 70)

    # Step 1: 初期化
    pipeline = demo_initialization(input_file, args.max_chunks, output_dir)

    # Step 2: データ読み込み
    df = demo_load_data(pipeline)

    # Step 3: チャンク変換
    chunks = demo_load_chunks(pipeline, df)

    # Step 4: Q/A生成
    qa_pairs = demo_generate_qa(pipeline, chunks)

    # Step 5: 保存
    saved_files = demo_save(pipeline, qa_pairs, not args.no_save)

    # ================================================================
    # サマリー
    # ================================================================
    print_separator("学習のポイント", "=", 70)

    print("""
📚 QAPipeline の処理フロー:

    ┌─────────────────────────────────────────────────────────────┐
    │                    QAPipeline                               │
    │                                                             │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
    │  │ load_data() │ →  │_load_chunks │ →  │generate_qa()│     │
    │  │             │    │_from_csv()  │    │             │     │
    │  │ CSVを読込   │    │チャンク変換 │    │Q/A生成      │     │
    │  └─────────────┘    └─────────────┘    └──────┬──────┘     │
    │                                                │            │
    │                                    ┌───────────▼──────────┐ │
    │                                    │ SmartQAGenerator     │ │
    │                                    │ ・analyze_chunk()    │ │
    │                                    │ ・generate_qa_pairs()│ │
    │                                    └───────────┬──────────┘ │
    │                                                │            │
    │                                    ┌───────────▼──────────┐ │
    │                                    │     save()          │ │
    │                                    │  結果をファイル保存  │ │
    │                                    └──────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘

💡 ポイント:
    1. QAPipeline は全体のオーケストレーションを担当
    2. 実際のQ/A生成は SmartQAGenerator に委譲
    3. use_celery=True で並列処理が可能（本番向け）
    4. 学習時は use_celery=False（同期処理）で動作確認
""")

    # 結果サマリー
    print_separator("実行結果サマリー", "-", 50)
    print(f"   • 入力チャンク数: {len(chunks)}")
    print(f"   • 生成Q/A数: {len(qa_pairs)}")
    if saved_files:
        print(f"   • 出力ファイル: {saved_files.get('qa_csv', 'N/A')}")

    print_separator("✅ 学習プログラム完了", "=", 70)


if __name__ == "__main__":
    main()
