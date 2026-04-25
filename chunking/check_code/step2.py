# check_step2.py
"""
Step2: 意味的分割（Semantic Chunking）の同期版・簡易確認プログラム

【目的】
段落を意味的な類似度に基づいて再構成する。
話題の転換点で分割し、形式的な改行ではなく意味のまとまりで分割する。

【処理の流れ】
1. Step1の出力（段落リスト）を入力として受け取る
2. 各段落をGemini APIに送信し、意味的なチャンクに分割
3. 分割されたチャンクのリストを返す

【Step3との連携】
このStep2の出力は、Step3の入力として使用される。
以下のパターンを検証できるよう設計：
- 前方依存: 「この」「それ」等の指示語で前を参照
- 後方依存: 専門用語が未定義のまま使用される
- 独立判定: 話題は同じでも単独で理解可能
- 章構造: 章が変わった場合の独立性
"""

import os
from google import genai
from google.genai import types

# chunking モジュールからインポート
from chunking.models import StructuralResult
from chunking.prompts import SEMANTIC_CHUNKING_PROMPT


def step2_semantic_chunking(paragraphs: list[str], api_key: str) -> list[str]:
    """
    段落を意味的なチャンクに分割する（Step2のコア機能）
    Args:
        paragraphs: 段落のリスト（Step1の出力）
        api_key: Gemini API キー
    Returns:
        意味的に分割されたチャンクのリスト
    """
    client = genai.Client(api_key=api_key)

    print(f"入力: {len(paragraphs)}段落")

    chunks = []

    for i, para in enumerate(paragraphs):
        print(f"段落 {i + 1}/{len(paragraphs)} 処理中...")

        # プロンプト作成
        prompt = f"{SEMANTIC_CHUNKING_PROMPT}\n\n【入力テキスト】\n{para}"

        # Gemini API 呼び出し（同期）
        # gemini-2.5-flash: 最新の安定版、高いレート制限
        # # URL: https://ai.google.dev/gemini-api/docs/text-generation?lang=python
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuralResult
            )
        )

        # レスポンスをパース
        result = StructuralResult.model_validate_json(response.text)

        # チャンクを抽出
        for chunk_para in result.paragraphs:
            chunks.append(chunk_para.full_text)

        print(f"  → {len(result.paragraphs)}個のチャンクに分割")

    return chunks


def main():
    """メイン処理"""

    # APIキー取得
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY 環境変数を設定してください")
        print("  export GOOGLE_API_KEY='your-api-key'")
        return

    # ============================================================
    # テスト用段落（Step1の出力を想定）
    # Step3で前方依存・後方依存・完全独立を検証するための入力
    # ============================================================
    test_paragraphs = [
        # ============================================================
        # 段落1: RAGの説明（前方依存のテスト用）
        # Step2で分割されることを想定：定義部分 vs 利点部分
        # ============================================================
        """RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。
外部知識ベースから関連情報を取得し、それをLLMのコンテキストとして渡します。
2020年にFacebookが発表し、現在では多くのシステムで採用されています。
この手法の最大の利点は、最新情報を反映できることです。
それにより、LLM単体では対応できない時事的な質問にも回答可能になります。
また、ハルシネーションを軽減する効果も報告されています。""",

        # ============================================================
        # 段落2: セマンティックチャンキングの説明（後方依存のテスト用）
        # Step2で分割されることを想定：用語定義 vs 用語使用
        # ============================================================
        """セマンティックチャンキングは、テキストを意味単位で分割する技術です。
「チャンク」とは、分割されたテキストの各ブロックを指します。
「埋め込み」（Embedding）は、テキストを数値ベクトルに変換したものです。
チャンクサイズは検索精度に大きく影響します。
小さすぎると文脈が失われ、埋め込みの品質が低下します。
大きすぎると検索ノイズが増加し、関連性の低い情報が混入します。""",

        # ============================================================
        # 段落3: 観光情報（独立判定のテスト用）
        # Step2で分割されることを想定：京都 vs 沖縄（同じ「観光」だが独立）
        # ============================================================
        """京都の紅葉は11月中旬から下旬が見頃です。
清水寺や嵐山が特に人気のスポットとして知られています。
混雑を避けるなら平日の早朝がおすすめです。
沖縄の海は透明度が高く、シュノーケリングに最適です。
那覇から車で約1時間の恩納村には美しいビーチが点在しています。
夏季は台風に注意が必要ですが、それ以外の季節も温暖で過ごしやすいです。""",

        # ============================================================
        # 段落4: ベクトルDBの説明（後方依存のテスト用）
        # Step2で分割されることを想定：定義 vs 活用
        # ============================================================
        """ベクトルデータベースは、高次元ベクトルを効率的に格納・検索するシステムです。
代表的な製品にPinecone、Weaviate、Chromaなどがあります。
ANN（Approximate Nearest Neighbor）アルゴリズムにより高速な類似検索を実現します。
ANNの精度とスピードはトレードオフの関係にあります。
HNSWやIVFなどのインデックス手法を選択することで、このバランスを調整できます。
ベクトルDBの選定では、スケーラビリティとコストも重要な判断基準となります。""",

        # ============================================================
        # 段落5: 章構造（章による独立のテスト用）
        # Step2で分割されることを想定：第1章 vs 第2章
        # ============================================================
        """第1章 機械学習入門
機械学習は、データからパターンを学習するアルゴリズムの総称です。
教師あり学習、教師なし学習、強化学習の3つに大別されます。
本章では、これらの基本概念を解説しました。
第2章 深層学習の基礎
深層学習は、多層のニューラルネットワークを用いる機械学習の一手法です。
画像認識や自然言語処理で革命的な成果を上げています。
本章では、CNNとRNNの基本アーキテクチャを説明します。"""
    ]

    print("=" * 50)
    print("【入力段落（Step1の出力）】")
    print("=" * 50)
    for i, para in enumerate(test_paragraphs, 1):
        print(f"\n--- 段落{i} ---")
        print(para)
    print()

    print("【期待される分割結果】")
    print("  段落1 (RAG説明) → 2チャンクに分割")
    print("    チャンク1: RAGの定義（検索と生成の組み合わせ、2020年発表）")
    print("    チャンク2: RAGの利点（「この手法」「それ」で前を参照）")
    print("         ※ 前方依存: 指示語が前のチャンクを参照")
    print()
    print("  段落2 (チャンキング説明) → 2チャンクに分割")
    print("    チャンク3: 用語定義（チャンク、埋め込みの説明）")
    print("    チャンク4: 用語使用（チャンクサイズ、埋め込みの品質）")
    print("         ※ 後方依存: 専門用語が未定義のまま使用")
    print()
    print("  段落3 (観光情報) → 2チャンクに分割")
    print("    チャンク5: 京都観光（紅葉、清水寺、嵐山）")
    print("    チャンク6: 沖縄観光（海、シュノーケリング、恩納村）")
    print("         ※ 独立: 同じ「観光」話題だが単独で理解可能")
    print()
    print("  段落4 (ベクトルDB説明) → 2チャンクに分割")
    print("    チャンク7: ベクトルDBの定義（Pinecone等、ANNアルゴリズム）")
    print("    チャンク8: ベクトルDBの活用（ANNのトレードオフ、HNSW/IVF）")
    print("         ※ 後方依存: ANNが未定義のまま使用")
    print()
    print("  段落5 (章構造) → 2チャンクに分割")
    print("    チャンク9: 第1章 機械学習入門")
    print("    チャンク10: 第2章 深層学習の基礎")
    print("         ※ 章構造: 章が変わり独立した内容")
    print()
    print("【期待される最終結果】5段落 → 10チャンク")
    print()

    # Step2 実行
    print("=" * 50)
    print("【Step2 実行】")
    print("=" * 50)
    chunks = step2_semantic_chunking(test_paragraphs, api_key)

    # 結果表示
    print()
    print("=" * 50)
    print(f"【結果】{len(test_paragraphs)}段落 → {len(chunks)}チャンク")
    print("=" * 50)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- チャンク{i} ({len(chunk)}文字) ---")
        print(chunk)

    # 結果検証
    print()
    print("=" * 50)
    print("【結果検証】")
    print("=" * 50)
    expected_chunks = 10
    if len(chunks) == expected_chunks:
        print(f"✅ 期待通り {expected_chunks} チャンクに分割されました")
    else:
        print(f"⚠️  期待: {expected_chunks} チャンク, 実際: {len(chunks)} チャンク")
        print("   分割が期待と異なる場合、プロンプトの調整が必要な可能性があります")

    # 検証ポイント
    print()
    print("=" * 50)
    print("【検証ポイント】")
    print("=" * 50)
    print("✓ 段落1: RAGの「定義」と「利点」が分割されるか")
    print("  → 前方依存テスト用（「この手法」「それ」の参照）")
    print("✓ 段落2: 「用語定義」と「用語使用」が分割されるか")
    print("  → 後方依存テスト用（専門用語の前提知識）")
    print("✓ 段落3: 「京都観光」と「沖縄観光」が分割されるか")
    print("  → 独立判定テスト用（同じ話題だが独立）")
    print("✓ 段落4: ベクトルDBの「定義」と「活用」が分割されるか")
    print("  → 後方依存テスト用（ANN等の専門用語）")
    print("✓ 段落5: 「第1章」と「第2章」が分割されるか")
    print("  → 章構造による独立テスト用")

    # Step3との連携情報
    print()
    print("=" * 50)
    print("【Step3との連携】")
    print("=" * 50)
    print("Step2の出力がStep3の入力となり、以下の判定を検証:")
    print()
    print("  チャンク1→2: 前方依存 → 結合（True）")
    print("  チャンク2→3: 話題転換 → 分離（False）")
    print("  チャンク3→4: 後方依存 → 結合（True）")
    print("  チャンク4→5: 話題転換 → 分離（False）")
    print("  チャンク5→6: 独立     → 分離（False）")
    print("  チャンク6→7: 話題転換 → 分離（False）")
    print("  チャンク7→8: 後方依存 → 結合（True）")
    print("  チャンク8→9: 話題転換 → 分離（False）")
    print("  チャンク9→10: 章構造  → 分離（False）")
    print()
    print("Step3実行後の期待結果: 10チャンク → 7チャンク")
    print("  最終チャンク1: チャンク1+2（前方依存で結合）")
    print("  最終チャンク2: チャンク3+4（後方依存で結合）")
    print("  最終チャンク3: チャンク5（独立）")
    print("  最終チャンク4: チャンク6（独立）")
    print("  最終チャンク5: チャンク7+8（後方依存で結合）")
    print("  最終チャンク6: チャンク9（独立）")
    print("  最終チャンク7: チャンク10（独立）")


if __name__ == "__main__":
    main()
