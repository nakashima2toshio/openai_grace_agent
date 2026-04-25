# step1_2_3.py
"""
Step1 + Step2 + Step3 統合実行スクリプト

【処理の流れ】
1. Step1: テキストを段落単位に分割（階層構造化）
2. Step2: 段落を意味的なチャンクに分割（Semantic Chunking）
3. Step3: 隣接チャンク間の連続性をチェックし結合/分離（Continuity Check）
"""

import os
from google import genai
from google.genai import types

from chunking.models import StructuralResult, ContinuityResult
from chunking.prompts import (
    PARAGRAPH_SEPARATION_PROMPT,
    SEMANTIC_CHUNKING_PROMPT,
    CONTINUITY_CHECK_PROMPT
)
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
# Step3: 連続性チェック（Continuity Check）
# ============================================================

def step3_continuity_check(chunks: list[str], client: genai.Client) -> list[str]:
    """Step3: 隣接チャンク間の連続性をチェックし結合/分離"""
    if len(chunks) <= 1:
        return chunks

    continuity_flags = []
    for i in range(len(chunks) - 1):
        prompt = f"{CONTINUITY_CHECK_PROMPT}\n\n【前のテキスト】\n{chunks[i]}\n\n【次のテキスト】\n{chunks[i + 1]}"
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ContinuityResult
            )
        )
        result = ContinuityResult.model_validate_json(response.text)
        continuity_flags.append(result.is_connected)

    final_chunks = [chunks[0]]
    for i, is_connected in enumerate(continuity_flags):
        if is_connected:
            final_chunks[-1] += "\n\n" + chunks[i + 1]
        else:
            final_chunks.append(chunks[i + 1])

    return final_chunks


# ============================================================
# 統合実行
# ============================================================

def run_pipeline(text: str, api_key: str, verbose: bool = True) -> list[str]:
    """Step1 → Step2 → Step3 を連続実行"""
    client = genai.Client(api_key=api_key)

    print('step 1 (テキストを段落単位に階層構造化) ---------------------------')
    paragraphs = step1_hierarchical_split(text, client)
    if verbose:
        print(f"Step1: {len(paragraphs)}段落")
    for i, para in enumerate(paragraphs, 1):
        print(f"[{i}]= {para}")
    print()

    print('step 2 (段落を意味的なチャンクに分割) ------------------------------')
    chunks = step2_semantic_chunking(paragraphs, client)
    if verbose:
        print(f"Step2: {len(chunks)}チャンク")
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}]= {chunk}")
    print()

    print('step 3 (隣接チャンク間の連続性チェックと結合/分離) -------------------')
    final_chunks = step3_continuity_check(chunks, client)
    if verbose:
        print(f"Step3: {len(final_chunks)}チャンク")
    for i, final_chunk in enumerate(final_chunks, 1):
        print(f"[{i}]= {final_chunk}")
    print()

    return final_chunks


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY 環境変数を設定してください")
        return

    text_jp = """RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。
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

    # テストケース2: 改行なしの長い1行
    text_jp2 = """RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。外部知識ベースから関連情報を取得し、それをLLMのコンテキストとして渡します。2020年にFacebookが発表し、現在では多くのシステムで採用されています。この手法の最大の利点は、最新情報を反映できることです。それにより、LLM単体では対応できない時事的な質問にも回答可能になります。また、ハルシネーションを軽減する効果も報告されています。セマンティックチャンキングは、テキストを意味単位で分割する技術です。「チャンク」とは、分割されたテキストの各ブロックを指します。「埋め込み」（Embedding）は、テキストを数値ベクトルに変換したものです。チャンクサイズは検索精度に大きく影響します。小さすぎると文脈が失われ、埋め込みの品質が低下します。大きすぎると検索ノイズが増加し、関連性の低い情報が混入します。京都の紅葉は11月中旬から下旬が見頃です。清水寺や嵐山が特に人気のスポットとして知られています。混雑を避けるなら平日の早朝がおすすめです。沖縄の海は透明度が高く、シュノーケリングに最適です。那覇から車で約1時間の恩納村には美しいビーチが点在しています。夏季は台風に注意が必要ですが、それ以外の季節も温暖で過ごしやすいです。ベクトルデータベースは、高次元ベクトルを効率的に格納・検索するシステムです。代表的な製品にPinecone、Weaviate、Chromaなどがあります。ANN（Approximate Nearest Neighbor）アルゴリズムにより高速な類似検索を実現します。ANNの精度とスピードはトレードオフの関係にあります。HNSWやIVFなどのインデックス手法を選択することで、このバランスを調整できます。ベクトルDBの選定では、スケーラビリティとコストも重要な判断基準となります。第1章 機械学習入門 機械学習は、データからパターンを学習するアルゴリズムの総称です。教師あり学習、教師なし学習、強化学習の3つに大別されます。本章では、これらの基本概念を解説しました。第2章 深層学習の基礎　深層学習は、多層のニューラルネットワークを用いる機械学習の一手法です。画像認識や自然言語処理で革命的な成果を上げています。本章では、CNNとRNNの基本アーキテクチャを説明します。"""

    # テストケース3: 英語（改行なし）
    text_en = """Artificial intelligence (AI) is rapidly advancing based on machine learning and deep learning. In the field of natural language processing (NLP) in particular, transformer models have achieved revolutionary results. Large language models like BERT and GPT have significantly enhanced contextual understanding capabilities. AI applications span widely from medical diagnosis to autonomous driving, profoundly impacting society."""

    print('\n日本語・改行あり ====================================')
    final_chunks = run_pipeline(text_jp, api_key)
    print(f"\n最終結果: {len(final_chunks)}チャンク")
    for i, chunk in enumerate(final_chunks, 1):
        print(f"\n--- チャンク{i} ({len(chunk)}文字) ---")
        print(chunk)

    print('\n日本語・改行なし ====================================')
    final_chunks2 = run_pipeline(text_jp2, api_key)
    print(f"\n最終結果: {len(final_chunks2)}チャンク")
    for i, chunk in enumerate(final_chunks2, 1):
        print(f"\n--- チャンク{i} ({len(chunk)}文字) ---")
        print(chunk)

    # print('\n英語・改行なし ======================================')
    # final_chunks3 = run_pipeline(text_en, api_key)
    #
    # print(f"\n最終結果: {len(final_chunks3)}チャンク")
    # for i, chunk in enumerate(final_chunks3, 1):
    #     print(f"\n--- チャンク{i} ({len(chunk)}文字) ---")
    #     print(chunk)

if __name__ == "__main__":
    main()

