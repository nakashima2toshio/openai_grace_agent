# step1_2.py
"""
Step1 + Step2 統合実行スクリプト

【処理の流れ】
1. Step1: テキストを段落単位に分割（階層構造化）
2. Step2: 段落を意味的なチャンクに分割（Semantic Chunking）
"""

import os
from google import genai
from google.genai import types

from chunking.models import StructuralResult
from chunking.prompts import PARAGRAPH_SEPARATION_PROMPT, SEMANTIC_CHUNKING_PROMPT
from chunking.regex_string import chunk_text


# ============================================================
# Step1: 階層構造化（段落分割）
# ============================================================

def preprocess_text(text: str) -> str:
    """テキストの前処理：長い1行を適切に分割する"""
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append('')
            continue
        chunks = chunk_text(line, keep_delimiter=True)
        processed_lines.extend(chunks) if len(chunks) > 1 else processed_lines.append(line)
    return '\n'.join(processed_lines)


def postprocess_paragraph(paragraph: str) -> str:
    """段落の後処理：句読点で文を分割し、改行で区切る"""
    lines = paragraph.split('\n') if '\n' in paragraph else [paragraph]
    processed = []
    for line in lines:
        line = line.strip()
        if line:
            processed.extend(chunk_text(line, keep_delimiter=True))
    return '\n'.join(processed)


def step1_hierarchical_split(text: str, client: genai.Client, block_size: int = 2000) -> list[str]:
    """Step1: テキストを段落単位に分割"""
    preprocessed = preprocess_text(text)
    blocks = [preprocessed[i:i + block_size] for i in range(0, len(preprocessed), block_size)]

    paragraphs = []
    for block in blocks:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{block}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuralResult
            )
        )
        result = StructuralResult.model_validate_json(response.text)
        paragraphs.extend(para.full_text for para in result.paragraphs)

    return [postprocess_paragraph(p) for p in paragraphs]


# ============================================================
# Step2: 意味的分割（Semantic Chunking）
# ============================================================

def step2_semantic_chunking(paragraphs: list[str], client: genai.Client) -> list[str]:
    """Step2: 段落を意味的なチャンクに分割"""
    chunks = []
    for para in paragraphs:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"{SEMANTIC_CHUNKING_PROMPT}\n\n【入力テキスト】\n{para}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuralResult
            )
        )
        result = StructuralResult.model_validate_json(response.text)
        chunks.extend(chunk.full_text for chunk in result.paragraphs)
    return chunks


# ============================================================
# 統合実行
# ============================================================

def run_pipeline(text: str, api_key: str, verbose: bool = True) -> list[str]:
    """Step1 → Step2 を連続実行"""
    client = genai.Client(api_key=api_key)

    # Step1
    if verbose:
        print("=" * 50)
        print("【Step1: 段落分割】")
        print("=" * 50)
    paragraphs = step1_hierarchical_split(text, client)
    if verbose:
        print(f"→ {len(paragraphs)}個の段落に分割")

    # Step2
    if verbose:
        print("\n" + "=" * 50)
        print("【Step2: 意味的チャンク分割】")
        print("=" * 50)
    chunks = step2_semantic_chunking(paragraphs, client)
    if verbose:
        print(f"→ {len(chunks)}個のチャンクに分割")

    return chunks


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY 環境変数を設定してください")
        return

    # テストテキスト
    test_text = """RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。
外部知識ベースから関連情報を取得し、それをLLMのコンテキストとして渡します。
2020年にFacebookが発表し、現在では多くのシステムで採用されています。
この手法の最大の利点は、最新情報を反映できることです。
それにより、LLM単体では対応できない時事的な質問にも回答可能になります。
また、ハルシネーションを軽減する効果も報告されています。

セマンティックチャンキングは、テキストを意味単位で分割する技術です。
「チャンク」とは、分割されたテキストの各ブロックを指します。
「埋め込み」（Embedding）は、テキストを数値ベクトルに変換したものです。
チャンクサイズは検索精度に大きく影響します。
小さすぎると文脈が失われ、埋め込みの品質が低下します。
大きすぎると検索ノイズが増加し、関連性の低い情報が混入します。

京都の紅葉は11月中旬から下旬が見頃です。
清水寺や嵐山が特に人気のスポットとして知られています。
混雑を避けるなら平日の早朝がおすすめです。
沖縄の海は透明度が高く、シュノーケリングに最適です。
那覇から車で約1時間の恩納村には美しいビーチが点在しています。
夏季は台風に注意が必要ですが、それ以外の季節も温暖で過ごしやすいです。

ベクトルデータベースは、高次元ベクトルを効率的に格納・検索するシステムです。
代表的な製品にPinecone、Weaviate、Chromaなどがあります。
ANN（Approximate Nearest Neighbor）アルゴリズムにより高速な類似検索を実現します。
ANNの精度とスピードはトレードオフの関係にあります。
HNSWやIVFなどのインデックス手法を選択することで、このバランスを調整できます。
ベクトルDBの選定では、スケーラビリティとコストも重要な判断基準となります。

第1章 機械学習入門
機械学習は、データからパターンを学習するアルゴリズムの総称です。
教師あり学習、教師なし学習、強化学習の3つに大別されます。
本章では、これらの基本概念を解説しました。
第2章 深層学習の基礎
深層学習は、多層のニューラルネットワークを用いる機械学習の一手法です。
画像認識や自然言語処理で革命的な成果を上げています。
本章では、CNNとRNNの基本アーキテクチャを説明します。"""

    # パイプライン実行
    chunks = run_pipeline(test_text, api_key)

    # 結果表示
    print("\n" + "=" * 50)
    print(f"【最終結果】{len(chunks)}チャンク")
    print("=" * 50)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- チャンク{i} ({len(chunk)}文字) ---")
        print(chunk)


if __name__ == "__main__":
    main()
