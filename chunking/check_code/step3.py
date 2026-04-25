# check_step3.py
"""
Step3: 文脈連続性チェックの同期版・簡易確認プログラム

【目的】
隣接するチャンク間の文脈連続性を判定し、
連続している場合は結合、非連続の場合は分離する。

【処理の流れ】
1. Step2の出力（チャンクリスト）を入力として受け取る
2. 隣接ペアごとにGemini APIで連続性を判定
3. 判定結果に基づいてチャンクを結合/分離
4. 最終的なチャンクリストを返す

【検証パターン】
- 前方依存: 「この」「それ」等の指示語で前を参照 → 結合
- 後方依存: 専門用語が未定義のまま使用 → 結合
- 話題転換: 完全に別のトピック → 分離
- 独立判定: 話題は同じでも単独で理解可能 → 分離
- 章構造: 章が変わった場合 → 分離
"""

import os
from google import genai
from google.genai import types

# chunking モジュールからインポート
from chunking.models import ContinuityResult
from chunking.prompts import CONTINUITY_CHECK_PROMPT


def step3_continuity_check(chunks: list[str], api_key: str) -> list[str]:
    """
    隣接チャンク間の連続性をチェックし結合/分離する（Step3のコア機能）
    Args:
        chunks: チャンクのリスト（Step2の出力）
        api_key: Gemini API キー
    Returns:
        連続性に基づいて結合/分離された最終チャンクリスト
    """
    client = genai.Client(api_key=api_key)

    print(f"入力: {len(chunks)}チャンク")

    if len(chunks) <= 1:
        print("チャンクが1つ以下のため、そのまま返します")
        return chunks

    # 隣接ペアの連続性を判定
    continuity_flags = []

    for i in range(len(chunks) - 1):
        print(f"ペア {i + 1}/{len(chunks) - 1} 判定中...")

        # プロンプト作成
        prompt = f"{CONTINUITY_CHECK_PROMPT}\n\n【前のテキスト】\n{chunks[i]}\n\n【次のテキスト】\n{chunks[i + 1]}"

        # Gemini API 呼び出し（同期）
        # gemini-2.5-flash: 最新の安定版、高いレート制限
        # # URL: https://ai.google.dev/gemini-api/docs/text-generation?lang=python
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ContinuityResult
            )
        )

        # レスポンスをパース
        result = ContinuityResult.model_validate_json(response.text)
        continuity_flags.append(result.is_connected)

        status = "連続 → 結合" if result.is_connected else "非連続 → 分離"
        print(f"  → {status}")

    # マージ処理
    print()
    print("マージ処理...")
    final_chunks = [chunks[0]]

    for i, is_connected in enumerate(continuity_flags):
        if is_connected:
            # 結合
            final_chunks[-1] += "\n\n" + chunks[i + 1]
            print(f"  チャンク{i} + チャンク{i + 1} → 結合")
        else:
            # 分離
            final_chunks.append(chunks[i + 1])
            print(f"  チャンク{i + 1} → 新規追加")

    return final_chunks


def main():
    """メイン処理"""

    # APIキー取得
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY 環境変数を設定してください")
        print("  export GOOGLE_API_KEY='your-api-key'")
        return

    # ============================================================
    # テスト用チャンク（Step2の出力を想定）
    # 前方依存・後方依存・完全独立の各パターンを含む
    # ============================================================
    test_chunks = [
        # ============================================================
        # パターン1: 前方依存（明示的参照）
        # ============================================================

        # チャンク1: RAGの定義
        """RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。
外部知識ベースから関連情報を取得し、それをLLMのコンテキストとして渡します。
2020年にFacebookが発表し、現在では多くのシステムで採用されています。""",

        # チャンク2: 前方依存あり（「この手法」「それ」で前を参照）
        # → 結合すべき（True）
        """この手法の最大の利点は、最新情報を反映できることです。
それにより、LLM単体では対応できない時事的な質問にも回答可能になります。
また、ハルシネーションを軽減する効果も報告されています。""",

        # ============================================================
        # パターン2: 後方依存（暗黙の前提知識）- 指示語なし
        # ============================================================

        # チャンク3: 用語定義を含む
        """セマンティックチャンキングは、テキストを意味単位で分割する技術です。
「チャンク」とは、分割されたテキストの各ブロックを指します。
「埋め込み」（Embedding）は、テキストを数値ベクトルに変換したものです。""",

        # チャンク4: 後方依存あり（「チャンク」「埋め込み」を定義なしで使用）
        # → 結合すべき（True）：単独では「チャンク」「埋め込み」の意味が不明
        """チャンクサイズは検索精度に大きく影響します。
小さすぎると文脈が失われ、埋め込みの品質が低下します。
大きすぎると検索ノイズが増加し、関連性の低い情報が混入します。""",

        # ============================================================
        # パターン3: 完全独立（話題転換）
        # ============================================================

        # チャンク5: 完全に異なる話題
        # → 分離すべき（False）
        """京都の紅葉は11月中旬から下旬が見頃です。
清水寺や嵐山が特に人気のスポットとして知られています。
混雑を避けるなら平日の早朝がおすすめです。""",

        # ============================================================
        # パターン4: 話題は同じだが独立して理解可能
        # ============================================================

        # チャンク6: 別の観光地（話題は「観光」だが独立）
        # → 分離すべき（False）：チャンク5なしでも完全に理解可能
        """沖縄の海は透明度が高く、シュノーケリングに最適です。
那覇から車で約1時間の恩納村には美しいビーチが点在しています。
夏季は台風に注意が必要ですが、それ以外の季節も温暖で過ごしやすいです。""",

        # ============================================================
        # パターン5: 後方依存（専門用語の連鎖）
        # ============================================================

        # チャンク7: 専門用語の定義
        """ベクトルデータベースは、高次元ベクトルを効率的に格納・検索するシステムです。
代表的な製品にPinecone、Weaviate、Chromaなどがあります。
ANN（Approximate Nearest Neighbor）アルゴリズムにより高速な類似検索を実現します。""",

        # チャンク8: 後方依存あり（ANN、ベクトルDBを前提として使用）
        # → 結合すべき（True）：「ANN」「ベクトルDB」の説明がないと意味不明
        """ANNの精度とスピードはトレードオフの関係にあります。
HNSWやIVFなどのインデックス手法を選択することで、このバランスを調整できます。
ベクトルDBの選定では、スケーラビリティとコストも重要な判断基準となります。""",

        # ============================================================
        # パターン6: 章構造による独立
        # ============================================================

        # チャンク9: 第1章（完結）
        """第1章 機械学習入門
機械学習は、データからパターンを学習するアルゴリズムの総称です。
教師あり学習、教師なし学習、強化学習の3つに大別されます。
本章では、これらの基本概念を解説しました。""",

        # チャンク10: 第2章（独立して理解可能）
        # → 分離すべき（False）：章が変わり、独立した内容
        """第2章 深層学習の基礎
深層学習は、多層のニューラルネットワークを用いる機械学習の一手法です。
画像認識や自然言語処理で革命的な成果を上げています。
本章では、CNNとRNNの基本アーキテクチャを説明します。"""
    ]

    print("=" * 50)
    print("【入力チャンク（Step2の出力）】")
    print("=" * 50)
    for i, chunk in enumerate(test_chunks, 1):
        print(f"\n--- チャンク{i} ---")
        print(chunk)
    print()

    print("【期待される判定】")
    print("  ペア1→2 (RAG定義 → RAGの利点): 前方依存 → 結合（True）")
    print("         理由: 「この手法」「それ」が前のチャンクを参照")
    print("  ペア2→3 (RAGの利点 → チャンキング定義): 話題転換 → 分離（False）")
    print("  ペア3→4 (用語定義 → 用語使用): 後方依存 → 結合（True）")
    print("         理由: 「チャンク」「埋め込み」が未定義のまま使用")
    print("  ペア4→5 (チャンキング → 京都観光): 話題転換 → 分離（False）")
    print("  ペア5→6 (京都観光 → 沖縄観光): 独立 → 分離（False）")
    print("         理由: 話題は「観光」だが、単独で完全に理解可能")
    print("  ペア6→7 (沖縄観光 → ベクトルDB定義): 話題転換 → 分離（False）")
    print("  ペア7→8 (ベクトルDB定義 → ベクトルDB活用): 後方依存 → 結合（True）")
    print("         理由: 「ANN」「ベクトルDB」を説明なしで使用")
    print("  ペア8→9 (ベクトルDB → 第1章機械学習): 話題転換 → 分離（False）")
    print("  ペア9→10 (第1章 → 第2章): 章構造 → 分離（False）")
    print("         理由: 章が変わり、単独で理解可能")
    print()
    print("【期待される最終結果】10チャンク → 7チャンク")
    print("  最終チャンク1: チャンク1+2（前方依存で結合）")
    print("  最終チャンク2: チャンク3+4（後方依存で結合）")
    print("  最終チャンク3: チャンク5（独立）")
    print("  最終チャンク4: チャンク6（独立）")
    print("  最終チャンク5: チャンク7+8（後方依存で結合）")
    print("  最終チャンク6: チャンク9（独立）")
    print("  最終チャンク7: チャンク10（独立）")
    print()

    # Step3 実行
    print("=" * 50)
    print("【Step3 実行】")
    print("=" * 50)
    final_chunks = step3_continuity_check(test_chunks, api_key)

    # 結果表示
    print()
    print("=" * 50)
    print(f"【結果】{len(test_chunks)}チャンク → {len(final_chunks)}チャンク")
    print("=" * 50)
    for i, chunk in enumerate(final_chunks, 1):
        print(f"\n--- 最終チャンク{i} ({len(chunk)}文字) ---")
        print(chunk)

    # 結果検証
    print()
    print("=" * 50)
    print("【結果検証】")
    print("=" * 50)
    expected_chunks = 7
    if len(final_chunks) == expected_chunks:
        print(f"✅ 期待通り {expected_chunks} チャンクに結合されました")
    else:
        print(f"⚠️  期待: {expected_chunks} チャンク, 実際: {len(final_chunks)} チャンク")
        print("   結合/分離が期待と異なる場合、プロンプトの調整が必要な可能性があります")

    # 検証ポイント
    print()
    print("=" * 50)
    print("【検証ポイント】")
    print("=" * 50)
    print("✓ 前方依存: 「この」「それ」等の指示語で前を参照 → 結合されるか")
    print("✓ 後方依存: 専門用語が未定義のまま使用 → 結合されるか")
    print("✓ 話題転換: 完全に別のトピック → 分離されるか")
    print("✓ 独立判定: 話題は同じでも単独で理解可能 → 分離されるか")
    print("✓ 章構造: 章が変わった場合 → 分離されるか")
    print("✓ 結合後のテキストが正しく保持されているか")


if __name__ == "__main__":
    main()
