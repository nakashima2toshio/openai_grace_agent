# step1.py
"""
Step1: 階層構造化（段落分割）の同期版・簡易確認プログラム

【目的】
テキストを段落単位に分割する。
見出し（第X章など）と本文は分離せず、1つの段落としてまとめる。

【処理の流れ】
1. 入力テキストの各行を chunk_text で前処理（長い行の分割）
2. 前処理済みテキストをブロック（2000文字単位）に分割
3. 各ブロックをGemini APIに送信し、段落構造を抽出
4. 出力の各段落を chunk_text で後処理（句読点で文を分割）
5. 段落のリストを返す（各段落は改行区切りの文の集合）

【Step2との連携】
Step2は「隣り合う文同士の意味的な距離を分析」するため、
このStep1の出力は「句読点で区切られた文の集合」である必要がある。
"""

import os
from google import genai
from google.genai import types

from chunking.models import StructuralResult
from chunking.prompts import PARAGRAPH_SEPARATION_PROMPT
from chunking.regex_string import chunk_text


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
        if len(chunks) > 1:
            processed_lines.extend(chunks)
        else:
            processed_lines.append(line)

    return '\n'.join(processed_lines)


def postprocess_paragraph(paragraph: str) -> str:
    """段落の後処理：句読点で文を分割し、改行で区切る"""
    if '\n' in paragraph:
        lines = paragraph.split('\n')
        processed_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            chunks = chunk_text(line, keep_delimiter=True)
            processed_lines.extend(chunks)
        return '\n'.join(processed_lines)

    chunks = chunk_text(paragraph, keep_delimiter=True)
    return '\n'.join(chunks)


def step1_hierarchical_split(text: str, api_key: str, block_size: int = 2000) -> list[str]:
    """テキストを段落単位に分割する（Step1のコア機能）"""
    client = genai.Client(api_key=api_key)

    preprocessed_text = preprocess_text(text)
    blocks = [preprocessed_text[i:i + block_size] for i in range(0, len(preprocessed_text), block_size)]

    paragraphs = []
    for i, block in enumerate(blocks):
        prompt = f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{block}"

        response = client.models.generate_content(
            model="gemini-3-flash-preview",  # ← これが現時点での最新・正式なモデル名
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuralResult
            )
        )

        result = StructuralResult.model_validate_json(response.text)
        for para in result.paragraphs:
            paragraphs.append(para.full_text)

    processed_paragraphs = []
    for para in paragraphs:
        processed_para = postprocess_paragraph(para)
        processed_paragraphs.append(processed_para)

    return processed_paragraphs


def run_test(title: str, text: str, api_key: str):
    """テスト実行"""
    print(f"\n【{title}】")
    print(f"入力:\n{text}\n")

    chunks = chunk_text(text, keep_delimiter=True)
    print(f"chunk_text結果: {len(chunks)}個")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. {chunk}")

    print('---')
    paragraphs = step1_hierarchical_split(text, api_key)

    print(f"\nStep1結果: {len(paragraphs)}個の段落")
    for i, para in enumerate(paragraphs, 1):
        lines = [line for line in para.split('\n') if line.strip()]
        print(f"\n--- 段落{i} ({len(lines)}文) ---")
        for j, line in enumerate(lines, 1):
            print(f"  {j}. {line}")
    print('----------------------------------------')


def main():
    """メイン処理"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY 環境変数を設定してください")
        return

    # テストケース1: 改行ありの通常テキスト
    text_jp1 = """RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。
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
    text_jp2 = """RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。外部知識ベースから関連情報を取得し、それをLLMのコンテキストとして渡します。2020年にFacebookが発表し、現在では多くのシステムで採用されています。この手法の最大の利点は、最新情報を反映できることです。それにより、LLM単体では対応できない時事的な質問にも回答可能になります。また、ハルシネーションを軽減する効果も報告されています。"""

    # テストケース3: 英語（改行なし）
    text_en = """Artificial intelligence (AI) is rapidly advancing based on machine learning and deep learning. In the field of natural language processing (NLP) in particular, transformer models have achieved revolutionary results. Large language models like BERT and GPT have significantly enhanced contextual understanding capabilities. AI applications span widely from medical diagnosis to autonomous driving, profoundly impacting society."""

    run_test("日本語（改行あり・複数段落）", text_jp1, api_key)
    run_test("日本語（改行なし・句読点複数）", text_jp2, api_key)
    run_test("英語（改行なし・ピリオド複数）", text_en, api_key)


if __name__ == "__main__":
    main()
