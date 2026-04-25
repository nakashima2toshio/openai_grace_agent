# check_async_improved.py
"""
非同期・並列処理の学習用プログラム（改善版）

【改善点】
1. AsyncAPIClient による真の非同期API使用（asyncio.to_thread()）
2. コマンドラインオプションで逐次/並列処理を選択可能
3. エラーハンドリング＆リトライ（AsyncAPIClient内蔵）
4. 進捗表示＆処理時間計測
5. 設定の外部化（dataclass）
6. セマフォによる同時実行数制限（AsyncAPIClient内蔵）

【使用方法】
# デフォルト（逐次処理：学習用）
python check_async_improved.py

# 並列処理モード
python check_async_improved.py --mode parallel

# 並列処理 + ワーカー数指定
python check_async_improved.py --mode parallel --workers 5

# 空行なしテキストでテスト
python check_async_improved.py --text test2

# ヘルプ表示
python check_async_improved.py --help

【処理フロー】
Step1: テキスト → 段落リスト（階層構造化）
Step2: 段落リスト → チャンクリスト（意味的分割）
Step3: チャンクリスト → 最終チャンクリスト（連続性チェック・結合）

【検証パターン】
- 前方依存: 「この」「それ」等の指示語で前を参照
- 後方依存: 専門用語が未定義のまま使用される
- 独立判定: 話題は同じでも単独で理解可能
- 章構造: 章が変わった場合の独立性
"""

import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional

# chunking モジュールからインポート
from chunking.async_api_client import AsyncAPIClient
from chunking.models import StructuralResult, ContinuityResult
from chunking.prompts import (
    PARAGRAPH_SEPARATION_PROMPT,
    SEMANTIC_CHUNKING_PROMPT,
    CONTINUITY_CHECK_PROMPT
)


# ================================================================
# 設定クラス（改善案5: 設定の外部化）
# ================================================================
@dataclass
class ChunkingConfig:
    """チャンキング処理の設定"""
    # 処理モード
    mode: str = "sequential"  # "sequential" or "parallel"

    # モデル設定
    model: str = "gemini-2.5-flash"

    # Step1設定
    block_size: int = 2000  # ブロックサイズ（文字数）

    # 並列処理設定
    max_workers: int = 8  # 最大並列数
    max_retries: int = 3  # 最大リトライ回数
    max_output_tokens: int = 8192  # 出力トークン制限

    # テストテキスト選択
    text_variant: str = "test1"  # "test1" or "test2"


# グローバル設定インスタンス
config = ChunkingConfig()


# ================================================================
# ユーティリティ関数
# ================================================================
def format_elapsed_time(seconds: float) -> str:
    """経過時間をフォーマット"""
    if seconds < 60:
        return f"{seconds:.2f}秒"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.2f}秒"


def print_section(title: str, char: str = "=", width: int = 60):
    """セクションヘッダーを表示"""
    print(f"\n{char * width}")
    print(f"【{title}】")
    print(f"{char * width}")


def print_progress(current: int, total: int, prefix: str = "", suffix: str = ""):
    """プログレス表示"""
    percent = (current / total) * 100 if total > 0 else 0
    bar_length = 30
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\r  {prefix} [{bar}] {current}/{total} ({percent:.1f}%) {suffix}", end="", flush=True)


# ================================================================
# Step1: 階層構造化（段落分割）
# ================================================================
async def step1_hierarchical_split_sequential(
        text: str,
        api_client: AsyncAPIClient
) -> list[str]:
    """
    【逐次処理版】テキストを段落単位に分割する

    学習ポイント:
    - forループで1つずつ処理
    - awaitで各API呼び出しの完了を待つ
    - 処理順序が保証される

    Args:
        text: 入力テキスト
        api_client: 非同期APIクライアント

    Returns:
        段落のリスト
    """
    print_section("Step1: 階層構造化（段落分割）- 逐次処理")
    start_time = time.time()

    # テキストをブロックに分割
    blocks = [text[i:i + config.block_size] for i in range(0, len(text), config.block_size)]
    print(f"入力: {len(text)}文字 → {len(blocks)}ブロック")

    paragraphs = []

    for i, block in enumerate(blocks):
        print(f"\n  ブロック {i + 1}/{len(blocks)} 処理中...")

        # プロンプト作成
        prompt = f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{block}"

        # AsyncAPIClient でAPI呼び出し（改善案1: 真の非同期API）
        response_text = await api_client.generate_content(
            model=config.model,
            contents=prompt,
            response_schema=StructuralResult,
            task_id=f"step1_block_{i}"
        )

        if response_text:
            result = StructuralResult.model_validate_json(response_text)
            for para in result.paragraphs:
                paragraphs.append(para.full_text)
            print(f"    → {len(result.paragraphs)}個の段落を抽出")
        else:
            print(f"    ⚠️ ブロック{i + 1}の処理に失敗（スキップ）")

    elapsed = time.time() - start_time
    print(f"\n  ⏱️ Step1 処理時間: {format_elapsed_time(elapsed)}")
    print(f"  📊 結果: {len(blocks)}ブロック → {len(paragraphs)}段落")

    return paragraphs


async def step1_hierarchical_split_parallel(
        text: str,
        api_client: AsyncAPIClient
) -> list[str]:
    """
    【並列処理版】テキストを段落単位に分割する

    学習ポイント:
    - asyncio.gather() で複数タスクを同時実行
    - セマフォ（AsyncAPIClient内）で並列数を制御
    - 結果を順序通りに並べ替え

    Args:
        text: 入力テキスト
        api_client: 非同期APIクライアント

    Returns:
        段落のリスト
    """
    print_section("Step1: 階層構造化（段落分割）- 並列処理")
    start_time = time.time()

    # テキストをブロックに分割
    blocks = [text[i:i + config.block_size] for i in range(0, len(text), config.block_size)]
    print(f"入力: {len(text)}文字 → {len(blocks)}ブロック（並列処理）")

    async def process_block(index: int, block: str) -> tuple[int, Optional[StructuralResult]]:
        """1ブロックを処理"""
        prompt = f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{block}"
        response_text = await api_client.generate_content(
            model=config.model,
            contents=prompt,
            response_schema=StructuralResult,
            task_id=f"step1_block_{index}"
        )
        if response_text:
            return index, StructuralResult.model_validate_json(response_text)
        return index, None

    # 【改善案2】asyncio.gather() で全ブロックを並列処理
    tasks = [process_block(i, block) for i, block in enumerate(blocks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 結果を順序通りに並べ替え
    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  ⚠️ エラー: {r}")
        elif r[1] is not None:
            valid_results.append(r)

    valid_results.sort(key=lambda x: x[0])

    # 段落を抽出
    paragraphs = []
    for index, result in valid_results:
        for para in result.paragraphs:
            paragraphs.append(para.full_text)
        print(f"  ブロック{index + 1} → {len(result.paragraphs)}段落")

    elapsed = time.time() - start_time
    print(f"\n  ⏱️ Step1 処理時間: {format_elapsed_time(elapsed)}")
    print(f"  📊 結果: {len(blocks)}ブロック → {len(paragraphs)}段落")

    return paragraphs


# ================================================================
# Step2: 意味的分割
# ================================================================
async def step2_semantic_chunking_sequential(
        paragraphs: list[str],
        api_client: AsyncAPIClient
) -> list[str]:
    """
    【逐次処理版】段落を意味的なチャンクに分割する

    学習ポイント:
    - 1段落ずつ処理することで、処理の流れが理解しやすい
    - デバッグ時に問題箇所を特定しやすい

    Args:
        paragraphs: 段落のリスト
        api_client: 非同期APIクライアント

    Returns:
        チャンクのリスト
    """
    print_section("Step2: 意味的分割 - 逐次処理")
    start_time = time.time()

    print(f"入力: {len(paragraphs)}段落")

    chunks = []

    for i, para in enumerate(paragraphs):
        print(f"\n  段落 {i + 1}/{len(paragraphs)} 処理中...")

        # プロンプト作成
        prompt = f"{SEMANTIC_CHUNKING_PROMPT}\n\n【入力テキスト】\n{para}"

        # AsyncAPIClient でAPI呼び出し
        response_text = await api_client.generate_content(
            model=config.model,
            contents=prompt,
            response_schema=StructuralResult,
            task_id=f"step2_para_{i}"
        )

        if response_text:
            result = StructuralResult.model_validate_json(response_text)
            for chunk_para in result.paragraphs:
                chunks.append(chunk_para.full_text)
            print(f"    → {len(result.paragraphs)}個のチャンクに分割")
        else:
            # フォールバック: 元の段落をそのまま使用
            chunks.append(para)
            print(f"    ⚠️ 処理失敗、元の段落を使用")

    elapsed = time.time() - start_time
    print(f"\n  ⏱️ Step2 処理時間: {format_elapsed_time(elapsed)}")
    print(f"  📊 結果: {len(paragraphs)}段落 → {len(chunks)}チャンク")

    return chunks


async def step2_semantic_chunking_parallel(
        paragraphs: list[str],
        api_client: AsyncAPIClient
) -> list[str]:
    """
    【並列処理版】段落を意味的なチャンクに分割する

    学習ポイント:
    - 段落間に依存関係がないため、並列処理が効果的
    - N段落を同時に処理 → 処理時間が約1/Nに短縮

    Args:
        paragraphs: 段落のリスト
        api_client: 非同期APIクライアント

    Returns:
        チャンクのリスト
    """
    print_section("Step2: 意味的分割 - 並列処理")
    start_time = time.time()

    print(f"入力: {len(paragraphs)}段落（並列処理）")

    completed = 0
    total = len(paragraphs)

    async def process_paragraph(index: int, para: str) -> tuple[int, list[str]]:
        """1段落を処理"""
        nonlocal completed

        prompt = f"{SEMANTIC_CHUNKING_PROMPT}\n\n【入力テキスト】\n{para}"
        response_text = await api_client.generate_content(
            model=config.model,
            contents=prompt,
            response_schema=StructuralResult,
            task_id=f"step2_para_{index}"
        )

        completed += 1
        print_progress(completed, total, prefix="処理中")

        if response_text:
            result = StructuralResult.model_validate_json(response_text)
            return index, [p.full_text for p in result.paragraphs]
        return index, [para]  # フォールバック

    # 【改善案2】全段落を並列処理
    tasks = [process_paragraph(i, para) for i, para in enumerate(paragraphs)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print()  # プログレスバーの後に改行

    # 結果を順序通りに並べ替え
    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  ⚠️ エラー: {r}")
        else:
            valid_results.append(r)

    valid_results.sort(key=lambda x: x[0])

    # チャンクを抽出
    chunks = []
    for index, chunk_list in valid_results:
        chunks.extend(chunk_list)
        print(f"  段落{index + 1} → {len(chunk_list)}チャンク")

    elapsed = time.time() - start_time
    print(f"\n  ⏱️ Step2 処理時間: {format_elapsed_time(elapsed)}")
    print(f"  📊 結果: {len(paragraphs)}段落 → {len(chunks)}チャンク")

    return chunks


# ================================================================
# Step3: 文脈連続性チェック
# ================================================================
async def step3_continuity_check_sequential(
        chunks: list[str],
        api_client: AsyncAPIClient
) -> list[str]:
    """
    【逐次処理版】隣接チャンク間の連続性をチェックし結合/分離する

    学習ポイント:
    - 判定結果を確認しながら処理を追跡できる
    - マージ処理は逐次（前の結果に依存）

    Args:
        chunks: チャンクのリスト
        api_client: 非同期APIクライアント

    Returns:
        最終チャンクリスト
    """
    print_section("Step3: 文脈連続性チェック - 逐次処理")
    start_time = time.time()

    if len(chunks) <= 1:
        print("  チャンク数が1以下のため、スキップ")
        return chunks

    print(f"入力: {len(chunks)}チャンク（{len(chunks) - 1}ペアを判定）")

    # 隣接ペアの連続性を判定
    continuity_flags = []

    for i in range(len(chunks) - 1):
        print(f"\n  ペア {i + 1}/{len(chunks) - 1} 判定中...")

        # プロンプト作成
        prompt = f"{CONTINUITY_CHECK_PROMPT}\n\n【前のテキスト】\n{chunks[i]}\n\n【次のテキスト】\n{chunks[i + 1]}"

        # AsyncAPIClient でAPI呼び出し
        response_text = await api_client.generate_content(
            model=config.model,
            contents=prompt,
            response_schema=ContinuityResult,
            task_id=f"step3_pair_{i}"
        )

        if response_text:
            result = ContinuityResult.model_validate_json(response_text)
            continuity_flags.append(result.is_connected)
            status = "連続 → 結合" if result.is_connected else "非連続 → 分離"
            print(f"    → {status}")
        else:
            # フォールバック: 安全側（分離）を選択
            continuity_flags.append(False)
            print(f"    ⚠️ 処理失敗、分離（False）として処理")

    # マージ処理
    print("\n  マージ処理...")
    final_chunks = _merge_chunks(chunks, continuity_flags)

    elapsed = time.time() - start_time
    print(f"\n  ⏱️ Step3 処理時間: {format_elapsed_time(elapsed)}")
    print(f"  📊 結果: {len(chunks)}チャンク → {len(final_chunks)}チャンク")

    return final_chunks


async def step3_continuity_check_parallel(
        chunks: list[str],
        api_client: AsyncAPIClient
) -> list[str]:
    """
    【並列処理版】隣接チャンク間の連続性をチェックし結合/分離する

    学習ポイント:
    - 判定処理は並列化可能（各ペアは独立）
    - マージ処理は逐次（前の結果に依存するため並列化不可）

    Args:
        chunks: チャンクのリスト
        api_client: 非同期APIクライアント

    Returns:
        最終チャンクリスト
    """
    print_section("Step3: 文脈連続性チェック - 並列処理（判定のみ）")
    start_time = time.time()

    if len(chunks) <= 1:
        print("  チャンク数が1以下のため、スキップ")
        return chunks

    num_pairs = len(chunks) - 1
    print(f"入力: {len(chunks)}チャンク（{num_pairs}ペアを並列判定）")

    completed = 0

    async def judge_pair(index: int, prev: str, next_: str) -> tuple[int, bool]:
        """1ペアを判定"""
        nonlocal completed

        prompt = f"{CONTINUITY_CHECK_PROMPT}\n\n【前のテキスト】\n{prev}\n\n【次のテキスト】\n{next_}"
        response_text = await api_client.generate_content(
            model=config.model,
            contents=prompt,
            response_schema=ContinuityResult,
            task_id=f"step3_pair_{index}"
        )

        completed += 1
        print_progress(completed, num_pairs, prefix="判定中")

        if response_text:
            result = ContinuityResult.model_validate_json(response_text)
            return index, result.is_connected
        return index, False  # フォールバック: 分離

    # 【改善案2】全ペアを並列判定
    tasks = [judge_pair(i, chunks[i], chunks[i + 1]) for i in range(num_pairs)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print()  # プログレスバーの後に改行

    # 結果を順序通りに並べ替え
    valid_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  ⚠️ ペア{i + 1}でエラー: {r}")
            valid_results.append((i, False))  # フォールバック: 分離
        else:
            valid_results.append(r)

    valid_results.sort(key=lambda x: x[0])

    # 連続性フラグを抽出
    continuity_flags = [r[1] for r in valid_results]

    # 判定結果を表示
    for i, is_connected in enumerate(continuity_flags):
        status = "連続→結合" if is_connected else "非連続→分離"
        print(f"  ペア{i + 1}: {status}")

    # マージ処理（逐次：前の結果に依存）
    print("\n  マージ処理...")
    final_chunks = _merge_chunks(chunks, continuity_flags)

    elapsed = time.time() - start_time
    print(f"\n  ⏱️ Step3 処理時間: {format_elapsed_time(elapsed)}")
    print(f"  📊 結果: {len(chunks)}チャンク → {len(final_chunks)}チャンク")

    return final_chunks


def _merge_chunks(chunks: list[str], continuity_flags: list[bool]) -> list[str]:
    """
    連続性フラグに基づいてチャンクをマージする

    注意: この処理は逐次実行（前の結果に依存するため並列化不可）

    Args:
        chunks: チャンクのリスト
        continuity_flags: 連続性フラグのリスト（True=結合、False=分離）

    Returns:
        マージされたチャンクリスト
    """
    final_chunks = [chunks[0]]

    for i, is_connected in enumerate(continuity_flags):
        if is_connected:
            # 結合: 空行（\n\n）で連結し、段落構造を保持
            final_chunks[-1] += "\n\n" + chunks[i + 1]
            print(f"    チャンク{i + 1} + チャンク{i + 2} → 結合")
        else:
            # 分離: 新しいチャンクとして追加
            final_chunks.append(chunks[i + 1])
            print(f"    チャンク{i + 2} → 新規追加")

    return final_chunks


# ================================================================
# メイン処理（全Stepを実行）
# ================================================================
async def process_text(text: str, api_key: str) -> list[str]:
    """
    テキストを処理する全体フロー

    Args:
        text: 入力テキスト
        api_key: Gemini API キー

    Returns:
        最終チャンクリスト
    """
    total_start = time.time()

    # 【改善案1】AsyncAPIClient を初期化
    # - 真の非同期API（asyncio.to_thread()）
    # - セマフォで並列数制御（改善案6）
    # - リトライロジック内蔵（改善案3）
    api_client = AsyncAPIClient(
        api_key=api_key,
        max_workers=config.max_workers,
        max_retries=config.max_retries,
        max_output_tokens=config.max_output_tokens
    )

    print(f"\n📌 処理モード: {config.mode.upper()}")
    print(f"📌 並列数: {config.max_workers}")
    print(f"📌 モデル: {config.model}")

    # Step1: テキスト → 段落
    if config.mode == "parallel":
        paragraphs = await step1_hierarchical_split_parallel(text, api_client)
    else:
        paragraphs = await step1_hierarchical_split_sequential(text, api_client)

    # Step2: 段落 → チャンク（意味的分割）
    if config.mode == "parallel":
        chunks = await step2_semantic_chunking_parallel(paragraphs, api_client)
    else:
        chunks = await step2_semantic_chunking_sequential(paragraphs, api_client)

    # Step3: チャンク → 最終チャンク（連続性チェック）
    if config.mode == "parallel":
        final_chunks = await step3_continuity_check_parallel(chunks, api_client)
    else:
        final_chunks = await step3_continuity_check_sequential(chunks, api_client)

    # 【改善案4】統計情報表示
    total_elapsed = time.time() - total_start
    stats = api_client.get_stats()

    print_section("処理統計")
    print(f"  ⏱️ 総処理時間: {format_elapsed_time(total_elapsed)}")
    print(f"  📡 API呼び出し回数: {stats['total_requests']}")
    print(f"  ❌ 失敗回数: {stats['failed_requests']}")
    print(f"  ⚠️ 切断レスポンス: {stats['truncated_responses']}")
    print(f"  ✅ 成功率: {stats['success_rate']:.1f}%")

    return final_chunks


# ================================================================
# テストテキスト
# ================================================================
TEST_TEXT_1 = """RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。
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

TEST_TEXT_2 = """RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせた手法です。
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


# ================================================================
# コマンドライン引数のパース
# ================================================================
def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description="非同期・並列処理の学習用プログラム（改善版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト（逐次処理：学習用）
  python check_async_improved.py

  # 並列処理モード
  python check_async_improved.py --mode parallel

  # 並列処理 + ワーカー数指定
  python check_async_improved.py --mode parallel --workers 5

  # 空行なしテキストでテスト
  python check_async_improved.py --text test2

学習ポイント:
  逐次処理 (sequential):
    - forループで1つずつ処理
    - 処理の流れが理解しやすい
    - デバッグ時に問題箇所を特定しやすい

  並列処理 (parallel):
    - asyncio.gather()で複数タスクを同時実行
    - セマフォで並列数を制御
    - 処理時間が約1/Nに短縮
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["sequential", "parallel"],
        default="sequential",
        help="処理モード: sequential（逐次処理）または parallel（並列処理）。デフォルト: sequential"
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="並列処理の最大ワーカー数。デフォルト: 8"
    )

    parser.add_argument(
        "--text", "-t",
        choices=["test1", "test2"],
        default="test1",
        help="テストテキスト: test1（空行あり）または test2（空行なし）。デフォルト: test1"
    )

    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="使用するGeminiモデル。デフォルト: gemini-2.5-flash"
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=2000,
        help="Step1のブロックサイズ（文字数）。デフォルト: 2000"
    )

    return parser.parse_args()


# ================================================================
# メイン関数
# ================================================================
async def main():
    """
    エントリーポイント

    【非同期処理の基本構造】
    async def main():      # 非同期関数として定義
        result = await xxx()  # 非同期関数を呼び出し、完了を待つ

    asyncio.run(main())    # イベントループを起動して実行
    """
    # コマンドライン引数をパース
    args = parse_args()

    # 【改善案5】設定を更新
    config.mode = args.mode
    config.max_workers = args.workers
    config.text_variant = args.text
    config.model = args.model
    config.block_size = args.block_size

    # APIキー取得
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ エラー: GOOGLE_API_KEY 環境変数を設定してください")
        print("   export GOOGLE_API_KEY='your-api-key'")
        return

    # テストテキスト選択
    test_text = TEST_TEXT_1 if config.text_variant == "test1" else TEST_TEXT_2
    text_desc = "空行あり（5段落）" if config.text_variant == "test1" else "空行なし（1段落）"

    print_section("入力テキスト")
    print(f"テキストバリアント: {config.text_variant} ({text_desc})")
    print(f"文字数: {len(test_text)}文字")
    print()
    print(test_text)

    print_section("期待される処理結果")
    if config.text_variant == "test1":
        print("Step1実行後: 1テキスト → 5段落")
        print("  段落1: RAGの説明（定義 + 利点）")
        print("  段落2: セマンティックチャンキングの説明（用語定義 + 用語使用）")
        print("  段落3: 観光情報（京都 + 沖縄）")
        print("  段落4: ベクトルDBの説明（定義 + 活用）")
        print("  段落5: 章構造（第1章 + 第2章）")
        print()
        print("Step2実行後: 5段落 → 10チャンク")
        print("  チャンク1-2: RAG（定義 / 利点）")
        print("  チャンク3-4: チャンキング（用語定義 / 用語使用）")
        print("  チャンク5-6: 観光（京都 / 沖縄）")
        print("  チャンク7-8: ベクトルDB（定義 / 活用）")
        print("  チャンク9-10: 章構造（第1章 / 第2章）")
        print()
        print("Step3実行後: 10チャンク → 7チャンク")
        print("  最終チャンク1: チャンク1+2（前方依存で結合）")
        print("  最終チャンク2: チャンク3+4（後方依存で結合）")
        print("  最終チャンク3: チャンク5（独立）")
        print("  最終チャンク4: チャンク6（独立）")
        print("  最終チャンク5: チャンク7+8（後方依存で結合）")
        print("  最終チャンク6: チャンク9（独立）")
        print("  最終チャンク7: チャンク10（独立）")
    else:
        print("Step1実行後: 1テキスト → 1段落（空行がないため）")
        print("Step2実行後: 1段落 → 複数チャンク（意味的分割）")
        print("Step3実行後: チャンク数減少（連続性による結合）")

    # 全Stepを実行
    final_chunks = await process_text(test_text, api_key)

    # 最終結果表示
    print_section("最終結果")
    for i, chunk in enumerate(final_chunks, 1):
        print(f"\n--- 最終チャンク{i} ({len(chunk)}文字) ---")
        print(chunk)

    print_section("結果検証")
    if config.text_variant == "test1":
        expected_chunks = 7
        if len(final_chunks) == expected_chunks:
            print(f"✅ 期待通り {expected_chunks} チャンクに結合されました")
        else:
            print(f"⚠️  期待: {expected_chunks} チャンク, 実際: {len(final_chunks)} チャンク")
            print("   結合/分離が期待と異なる場合、プロンプトの調整が必要な可能性があります")
    else:
        print(f"📊 最終チャンク数: {len(final_chunks)}")
        print("   ※ 空行なしテキストのため、Step2・Step3の結果に依存")

    print_section("検証ポイント")
    print("✓ 前方依存: 「この」「それ」等の指示語で前を参照 → 結合されるか")
    print("✓ 後方依存: 専門用語が未定義のまま使用 → 結合されるか")
    print("✓ 話題転換: 完全に別のトピック → 分離されるか")
    print("✓ 独立判定: 話題は同じでも単独で理解可能 → 分離されるか")
    print("✓ 章構造: 章が変わった場合 → 分離されるか")
    print("✓ 結合後のテキストが正しく保持されているか")

    print_section("処理完了")


# ================================================================
# プログラム実行
# ================================================================
if __name__ == "__main__":
    # asyncio.run() で非同期処理を開始
    # これがイベントループを作成し、main()を実行する
    asyncio.run(main())
