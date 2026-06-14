# csv_text_to_chunks_text_csv.py
"""
csv_text_to_chunks_text_csv.py - LLMベースセマンティックチャンキング（統一版）

主要機能:
- chunks_all_async(): テキストからチャンクを作成（LLMベース、asyncio並列処理）
- load_text_from_csv(): CSVファイルからテキストを読み込み
- save_chunks_as_csv(): チャンクをCSV形式で保存（改行正規化対応 + シンプルCSV追加出力）
- save_chunks_as_simple_csv(): チャンクをシンプルCSV形式で保存（Textカラムのみ）
- generate_output_filename(): 出力ファイル名の自動生成

テキストまたはCSVファイルを意味的なチャンクに分割するパイプライン。
非同期・並列処理により高速化。CSV出力時に改行を削除してクリーンなCSVを作成。

改修内容（v2.1）:
- シンプルCSV保存機能を追加（Textカラムのみ）
- save_chunks_as_csv()にsave_simple_csvパラメータを追加
- デフォルトで2つのCSVファイルを出力:
  1. メタデータ付きCSV（chunk_id, text, tokens, ...）
  2. シンプルCSV（Text カラムのみ）
- ファイル名の重複を回避（_simple サフィックスを使用）

Usage:　chunking.csv_text_to_chunks_text_csv　はceleryを利用していない。
　　　　次のmake_qa_register_qdrant.pyで、利用する。
# Worker起動
# ./start_celery.sh stop
# ./start_celery.sh status
# ./start_celery.sh restart -w 4 --flower

# CSVファイル → チャンクCSV
# [MIGRATION] Gemini → OpenAI。--model は gpt-5-mini を使用してください。
uv run python -m chunking.csv_text_to_chunks_text_csv \
  --input-file OUTPUT/cc_news_1per.csv \
  --output chunks_output \
  --model gpt-5-mini \
  --workers 8 \
  --block-size 500

python -m chunking.csv_text_to_chunks_text_csv \
  --input-file OUTPUT/wikipedia_ja_1per.csv \
  --output chunks_output \
  --model gpt-5-mini \
  --workers 8 \

# 出力例:
# chunks_output/wikipedia_ja_5per_chunks_20260207_123456.csv （メタデータ付き）
# chunks_output/wikipedia_ja_5per_chunks_20260207_123456_simple.csv （シンプル版、Textのみ）

# ----------------------------------------------
# Step1: テキストファイル → チャンク分割、CSV
# ----------------------------------------------
uv run python -m chunking.csv_text_to_chunks_text_csv \
  --input-file OUTPUT/cc_news_1per.csv \
  --output output_chunked \
  --model gpt-5-mini \
  --workers 8

# ----------------------------------------------
# tep2: Q/A生成 + Qdrant登録
# ----------------------------------------------
uv run python qa_qdrant/make_qa_register_qdrant.py \
  --input-file output_chunked/cc_news_1per_chunks.csv \
  --collection cc_news_1per \
  --model gpt-5-mini \
  --concurrency 8 \
  --recreate

  # → chunks_output/document_chunks_20250118_123456.csv が生成される
  # → chunks_output/document_chunks_20250118_123456_simple.csv （シンプル版）も同時生成
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import tiktoken
import re
from tqdm.asyncio import tqdm as async_tqdm

# 既存のインポート
from chunking.async_api_client import AsyncAPIClient
from chunking.checkpoint_manager import CheckpointManager
from chunking.models import StructuralResult, ParagraphUnit, ContinuityResult
from chunking.prompts import (
    PARAGRAPH_SEPARATION_PROMPT,
    SEMANTIC_CHUNKING_PROMPT,
    CONTINUITY_CHECK_PROMPT
)
from chunking.utils import (
    setup_logging,
    format_time,
    format_size,
    estimate_api_calls
)
from chunking.regex_string import chunk_text

logger = logging.getLogger(__name__)

# チャンクの最大トークン数（tiktoken cl100k_base 換算）。
# - Step3 の結合上限と、最終チャンクの強制分割上限の両方に使う
# - Embedding（text-embedding-3-large）の入力上限を超えると超過分が
#   無言で切り捨てられるため、上限は必ずそれ未満にすること
MAX_CHUNK_TOKENS = 512

# Embedding モデルの入力トークン上限（プロバイダー中立な安全弁）。
# max_chunk_tokens がこれ以上の設定は警告する。
# anthropic の数値（2048）を踏襲する。text-embedding-3-large の実上限は
# ~8191 だが、本定数は「これ未満を推奨する」保守的なガード値である。
EMBEDDING_INPUT_TOKEN_LIMIT = 2048

_TOKENIZER = None
_TOKENIZER_FAILED = False


def _count_tokens(text: str) -> int:
    """トークン数を数える。

    tiktoken の BPE 取得に失敗する環境（オフライン等）では
    文字数/4 の概算にフォールバックする。
    """
    global _TOKENIZER, _TOKENIZER_FAILED
    if _TOKENIZER is None and not _TOKENIZER_FAILED:
        try:
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            _TOKENIZER_FAILED = True
            logger.warning(f"tiktoken 初期化失敗、文字数/4 で概算します: {e}")
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))
    # 概算: ASCII は約4文字≈1トークン、非ASCII（日本語等）は約1文字≈1トークン
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return max(1, ascii_chars // 4 + (len(text) - ascii_chars))


# ================================================================
# ✅ 新規追加: テキスト正規化関数
# ================================================================

def _normalize_whitespace(text: str) -> str:
    """
    テキストの改行・空白を正規化

    - 改行(\n)を半角スペースに置換
    - 連続する空白を1つに正規化
    - 先頭・末尾の空白を削除
    Args:
        text: 正規化対象テキスト
    Returns:
        正規化されたテキスト
    Examples:
        _normalize_whitespace("行1\\n\\n行2")
        '行1 行2'
        _normalize_whitespace("  複数    空白  ")
        '複数 空白'
    """
    # 改行を半角スペースに置換
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')

    # タブを半角スペースに置換
    text = text.replace('\t', ' ')

    # 連続する空白を1つに正規化
    text = re.sub(r'\s+', ' ', text)

    # 先頭・末尾の空白を削除
    text = text.strip()

    return text


# ================================================================
# ✅ 新規追加: Step1用 前処理・後処理関数（step1_2_3.pyより移植）
# ================================================================

def _preprocess_text(text: str) -> str:
    """
    テキストの前処理：長い1行を適切に分割する

    改行のない長いテキストを句読点（日本語: 。、英語: . ）で
    適切に分割し、LLMへの入力を整形する。

    Args:
        text: 前処理対象のテキスト

    Returns:
        前処理されたテキスト（句読点で改行区切り）
    """
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append('')
            continue
        # chunk_text: 日本語・英語対応の文分割
        chunks = chunk_text(line, keep_delimiter=True)
        if len(chunks) > 1:
            processed_lines.extend(chunks)
        else:
            processed_lines.append(line)
    return '\n'.join(processed_lines)


def _postprocess_paragraph(paragraph: str) -> str:
    """
    段落の後処理：句読点で文を分割し、改行で区切る

    Step1の出力（段落）を後処理し、各文を改行で区切ることで、
    Step2・Step3での処理精度を向上させる。

    Args:
        paragraph: 後処理対象の段落テキスト

    Returns:
        後処理された段落テキスト（文ごとに改行区切り）
    """
    lines = paragraph.split('\n') if '\n' in paragraph else [paragraph]
    processed = []
    for line in lines:
        line = line.strip()
        if line:
            processed.extend(chunk_text(line, keep_delimiter=True))
    return '\n'.join(processed)


# ================================================================
# CSV読み込み機能
# ================================================================

def _detect_text_column(df: pd.DataFrame, text_column: Optional[str] = None) -> str:
    """テキストカラムを特定する（指定があれば検証、なければ自動検出）"""
    if text_column:
        if text_column not in df.columns:
            raise ValueError(
                f"指定されたカラム '{text_column}' が見つかりません。\n"
                f"利用可能なカラム: {list(df.columns)}"
            )
        return text_column

    text_candidates = [
        'text', 'Text', 'TEXT',
        'content', 'Content', 'CONTENT',
        'Combined_Text', 'combined_text',
        'body', 'Body', 'BODY',
        'document', 'Document',
        'answer', 'Answer'
    ]
    for candidate in text_candidates:
        if candidate in df.columns:
            return candidate

    col = df.columns[0]
    logger.warning(
        f"テキストカラムを自動検出できませんでした。\n"
        f"  最初のカラム '{col}' を使用します。"
    )
    return col


def load_documents_from_csv(
        csv_path: str,
        text_column: Optional[str] = None,
        max_rows: Optional[int] = None,
) -> List[Dict]:
    """CSVファイルから文書リストを読み込む（1行=1文書）。

    旧実装（load_text_from_csv）は全行を1つの巨大テキストに結合しており、
    文書（記事）境界が破壊されていた。本関数は行（文書）単位の構造を
    保持して返す。doc_id により元文書へのトレーサビリティを確保する。

    Returns:
        [{'doc_id': 行番号, 'text': テキスト}, ...]（空行は除外）
    """
    logger.info("=" * 60)
    logger.info("CSV読み込み処理（文書単位）")
    logger.info("=" * 60)

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"  📁 読み込み: {len(df)} 行")
    except Exception as e:
        logger.error(f"CSV読み込みエラー: {e}")
        raise

    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        logger.info(f"  ✂️  制限: {len(df)} 行に制限")

    col = _detect_text_column(df, text_column)
    logger.info(f"  📝 テキストカラム: '{col}'")

    documents: List[Dict] = []
    for row_idx, value in df[col].fillna('').astype(str).items():
        doc_text = value.strip()
        if doc_text:
            documents.append({"doc_id": int(row_idx), "text": doc_text})

    total_size = sum(len(d["text"]) for d in documents)
    logger.info(f"  ✅ 抽出: {len(documents)} 件の文書（非空行）")
    logger.info(f"  📊 総サイズ: {format_size(total_size)}")
    return documents


def load_text_from_csv(
        csv_path: str,
        text_column: Optional[str] = None,
        max_rows: Optional[int] = None,
        combine_rows: bool = False
) -> str:
    """CSVファイルからテキストを読み込む（後方互換: 全行結合）。

    Note:
        文書境界を保持するには load_documents_from_csv() を使用すること。
        combine_rows は後方互換のため受け取るが、いずれのモードでも
        全行を改行区切りで結合した文字列を返す（旧挙動）。
    """
    documents = load_documents_from_csv(csv_path, text_column, max_rows)
    return "\n\n".join(d["text"] for d in documents)


# ================================================================
# ✅ 新規追加: シンプルCSV保存機能（Textカラムのみ）
# ================================================================

def _as_chunk_dicts(chunks: List) -> List[Dict]:
    """チャンクリストを辞書形式に正規化する（str / dict 混在を許容）"""
    normalized = []
    for c in chunks:
        if isinstance(c, dict):
            normalized.append({"text": c.get("text", ""), "doc_id": c.get("doc_id")})
        else:
            normalized.append({"text": str(c), "doc_id": None})
    return normalized

def save_chunks_as_simple_csv(
        chunks: List[str],
        output_file: str,
        normalize_whitespace: bool = True
) -> str:
    """
    チャンクをシンプルなCSV形式で保存（Textカラムのみ）

    Args:
        chunks: チャンクのリスト
        output_file: 出力ファイルパス
        normalize_whitespace: 改行・空白を正規化するか（デフォルト: True）

    Returns:
        保存したCSVファイルパス

    Example:
        >>> output_file = "chunks_output/wikipedia_ja_5per_chunks_simple.csv"
        >>> save_chunks_as_simple_csv(chunks, output_file)

        出力CSV:
        Text
        "チャンク1のテキスト..."
        "チャンク2のテキスト..."
    """
    data = []
    for chunk in _as_chunk_dicts(chunks):
        chunk_str = chunk["text"]
        # 改行・空白を正規化
        if normalize_whitespace:
            chunk_cleaned = _normalize_whitespace(chunk_str)
        else:
            chunk_cleaned = chunk_str

        data.append({'Text': chunk_cleaned})

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8')

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ シンプルCSV保存完了（Textカラムのみ）")
    logger.info("=" * 60)
    logger.info(f"  ファイル: {output_file}")
    logger.info(f"  チャンク数: {len(df)}")
    logger.info(f"  カラム: Text のみ")
    logger.info("=" * 60)

    return output_file


# ================================================================
# ✅ 改修: CSV保存機能（改行削除対応 + シンプルCSV追加出力）
# ================================================================

def save_chunks_as_csv(
        chunks: List[str],
        output_file: str,
        dataset_type: str = "custom",
        source_file: Optional[str] = None,
        normalize_whitespace: bool = True,  # ✅ 新規パラメータ
        save_simple_csv: bool = True  # ✅ 新規パラメータ
) -> str:
    """
    チャンクをCSV形式で保存（メタデータ付き + シンプルCSV）

    Args:
        chunks: チャンクのリスト
        output_file: 出力ファイルパス（メタデータ付きCSV）
        dataset_type: データセット種別
        source_file: 元ファイル名
        normalize_whitespace: 改行・空白を正規化するか（デフォルト: True）
        save_simple_csv: シンプルCSV（Textのみ）も保存するか（デフォルト: True）

    Returns:
        保存したCSVファイルパス（メタデータ付き）

    Note:
        save_simple_csv=True の場合、以下の2つのファイルが生成されます:
        1. {output_file}: メタデータ付きCSV（chunk_id, text, tokens, ...）
        2. {output_file_stem}_simple.csv: シンプルCSV（Text カラムのみ）

        例:
        - wikipedia_ja_5per_chunks_20260207_123456.csv （メタデータ付き）
        - wikipedia_ja_5per_chunks_20260207_123456_simple.csv （シンプル版）
    """
    data = []
    for i, chunk in enumerate(_as_chunk_dicts(chunks)):
        chunk_str = chunk["text"]
        # ✅ 改行・空白を正規化（CSV出力をクリーンにする）
        if normalize_whitespace:
            chunk_cleaned = _normalize_whitespace(chunk_str)
        else:
            chunk_cleaned = chunk_str

        # センテンス分割（正規化前のテキストで実施）
        sentences = _split_sentences_simple(chunk_str)

        data.append({
            'chunk_id'      : f"{dataset_type}_chunk_{i}",
            'text'          : chunk_cleaned,  # ✅ 正規化されたテキスト
            'tokens'        : _count_tokens(chunk_cleaned),
            'chunk_idx'     : i,
            'doc_id'        : chunk["doc_id"],  # 元文書（CSV行）へのトレーサビリティ
            'dataset_type'  : dataset_type,
            'type'          : 'llm_chunk',
            'sentence_count': len(sentences),
            'source_file'   : source_file or ''
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8')

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ CSV保存完了（メタデータ付き）")
    logger.info("=" * 60)
    logger.info(f"  ファイル: {output_file}")
    logger.info(f"  チャンク数: {len(df)}")
    if 'tokens' in df.columns and len(df) > 0:
        logger.info(f"  総トークン数: {df['tokens'].sum()}")
        logger.info(f"  平均トークン数: {df['tokens'].mean():.1f}")
    else:
        logger.warning("  ⚠️ チャンクが0件です。APIキーを確認してください。")
    logger.info(f"  改行正規化: {'有効' if normalize_whitespace else '無効'}")
    logger.info("=" * 60)

    # ================================================================
    # ✅ 新規追加: シンプルCSV（Textのみ）も保存
    # ================================================================
    if save_simple_csv:
        output_path = Path(output_file)

        # ファイル名生成: メタデータ付きCSVと同じ名前で "_simple" を追加
        # 例: wikipedia_ja_5per_chunks_20260207_123456.csv
        #  → wikipedia_ja_5per_chunks_20260207_123456_simple.csv
        simple_csv_name = output_path.stem + "_simple.csv"
        simple_csv_path = output_path.parent / simple_csv_name

        # シンプルCSVを保存
        save_chunks_as_simple_csv(
            chunks=chunks,
            output_file=str(simple_csv_path),
            normalize_whitespace=normalize_whitespace
        )

    return output_file


def save_chunks_as_text(chunks: List[str], output_file: str) -> str:
    """テキスト形式で保存（既存形式・後方互換性）"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n---\n')

    logger.info(f"テキストファイル保存: {output_file} ({len(chunks)}チャンク)")
    return output_file


# ================================================================
# ✅ 新規追加: 出力ファイル名自動生成機能
# ================================================================

def generate_output_filename(
        input_file: str,
        output_dir: str,
        dataset_type: str = "custom",
        use_timestamp: bool = False
) -> str:
    """
    入力ファイル名から出力ファイル名を自動生成

    Args:
        input_file:    入力ファイルパス
        output_dir:    出力ディレクトリ
        dataset_type:  データセット種別（後方互換のため残す）
        use_timestamp: True の場合タイムスタンプを付与（デフォルト: False）
                       後続バッチとの連携のため、デフォルトは固定ファイル名

    Returns:
        出力ファイルの絶対パス

    Examples:
        # デフォルト（固定ファイル名）← バッチ連携用
        generate_output_filename("OUTPUT/cc_news_1per.csv", "output_chunked")
        → 'output_chunked/cc_news_1per_chunks.csv'

        # タイムスタンプあり（--timestamp オプション指定時）
        generate_output_filename("OUTPUT/cc_news_1per.csv", "output_chunked", use_timestamp=True)
        → 'output_chunked/cc_news_1per_chunks_20260505_004819.csv'
    """
    import os

    input_path = Path(input_file)
    base_name = input_path.stem  # 例: cc_news_1per

    if use_timestamp:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_chunks_{timestamp}.csv"
    else:
        # 固定ファイル名（後続バッチとの連携用）
        output_filename = f"{base_name}_chunks.csv"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    return output_path


def _split_sentences_simple(text: str) -> List[str]:
    """簡易的な文分割（日本語対応）"""
    sentences = re.findall(r'[^。．.！？!?]+[。．.！？!?]\s*', text)

    if not sentences:
        sentences = [text.strip()] if text.strip() else []
    else:
        last_pos = text.rfind(sentences[-1]) + len(sentences[-1])
        if last_pos < len(text):
            remaining = text[last_pos:].strip()
            if remaining:
                sentences.append(remaining)

    return [s.strip() for s in sentences if s.strip()]


# ================================================================
# 文書単位チャンキング（コア実装）
# ================================================================

def _split_document_into_blocks(text: str, block_size: int) -> List[str]:
    """文書を文境界を保ってブロックに分割する。

    旧実装は全文書を結合した文字列を block_size 文字で機械的に切断しており、
    文・文書の途中で分断されていた。本実装は _preprocess_text() で文ごとに
    改行区切りへ整形した後、文単位でブロックへ詰める（文の途中では切らない）。
    1文が block_size を超える場合のみ、その文単独で1ブロックとする。
    """
    preprocessed = _preprocess_text(text)
    lines = [line for line in preprocessed.split('\n') if line.strip()]

    blocks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in lines:
        if current and current_len + len(line) > block_size:
            blocks.append('\n'.join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        blocks.append('\n'.join(current))
    return blocks


def _report_coverage(
        total_input_chars: int,
        final_chunks: List[Dict],
        stats: Dict,
        api_stats: Optional[Dict] = None,
        coverage_threshold: float = 0.95
) -> Dict:
    """入力に対する出力のカバレッジを検証してレポートする。

    各ステップのフォールバックでテキスト自体は保全されるが、LLMが本文を
    省略するケースの検知のため正規化後の文字数比を必ず突合する。
    """
    output_chars = sum(len(_normalize_whitespace(c["text"])) for c in final_chunks)
    ratio = output_chars / total_input_chars if total_input_chars else 1.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("📐 入力カバレッジ検証")
    logger.info("=" * 60)
    logger.info(f"  入力文字数（正規化後）: {total_input_chars:,}")
    logger.info(f"  出力文字数（正規化後）: {output_chars:,}")
    logger.info(f"  カバレッジ: {ratio:.1%}")
    logger.info(
        f"  フォールバック: Step1={stats.get('step1_fallbacks', 0)}, "
        f"Step2={stats.get('step2_fallbacks', 0)}, Step3={stats.get('step3_fallbacks', 0)}"
    )
    if api_stats:
        logger.info(
            f"  API: 総リクエスト={api_stats.get('total_requests', 0)}, "
            f"失敗={api_stats.get('failed_requests', 0)}, "
            f"切断={api_stats.get('truncated_responses', 0)}"
        )
        usage = api_stats.get("usage") or {}
        if any(usage.values()):
            logger.info(
                f"  トークン: 入力={usage.get('input_tokens', 0):,}, "
                f"出力={usage.get('output_tokens', 0):,}, "
                f"キャッシュ読取={usage.get('cached_input_tokens', 0):,}"
            )
    if ratio < coverage_threshold:
        logger.warning(
            f"⚠️ 入力カバレッジが {coverage_threshold:.0%} を下回っています（{ratio:.1%}）。"
            f"LLMが本文を省略・要約した可能性があります。"
        )
    logger.info("=" * 60)

    return {"input_chars": total_input_chars, "output_chars": output_chars, "ratio": ratio}


# ================================================================
# chunks_all_async関数
# ================================================================

async def chunks_all_async(
        text: Optional[str] = None,
        model: str = "gpt-5-mini",  # [MIGRATION] "gemini-3-flash-preview" → "gpt-5-mini"
        max_workers: int = 8,
        block_size: int = 1000,  # ✅ 2000→1000に変更（MAX_TOKENS対策）
        checkpoint_manager: Optional[CheckpointManager] = None,
        output_file: Optional[str] = None,
        dataset_type: str = "custom",
        source_file: Optional[str] = None,
        documents: Optional[List[Dict]] = None,
        continuity_mode: str = "rule",
        max_chunk_tokens: int = MAX_CHUNK_TOKENS,
) -> List[str]:
    """テキストまたは文書リストを3段階で意味的にチャンク化する。

    Args:
        text: 入力テキスト（単一文書として扱う。documents と排他）
        documents: 文書リスト [{'doc_id': ..., 'text': ...}, ...]
            CSV入力では1行=1文書として渡すこと。チャンクが文書を
            またいで結合されることはない（文書境界の保証）。
        continuity_mode: Step3（連続性チェック）の動作モード
            - "rule": ルールベース判定（LLM呼び出しなし・デフォルト）
            - "llm" : LLMによるペア判定（旧動作。チャンク数-1回のLLM呼び出し）
            - "off" : 結合を行わない
        max_chunk_tokens: チャンクの最大トークン数。Step3の結合上限かつ
            最終チャンクの強制分割上限。Embedding（text-embedding-3-large）の
            入力上限未満であること
        その他: モデル・並列数・ブロックサイズ・チェックポイント・出力先

    Returns:
        チャンクテキストのリスト（後方互換のため文字列リスト。
        doc_id 等のメタデータは出力CSVに含まれる）
    """
    import os

    if documents is None:
        if text is None:
            raise ValueError("text または documents のいずれかを指定してください")
        documents = [{"doc_id": 0, "text": text}]

    # [MIGRATION] GOOGLE_API_KEY → OPENAI_API_KEY
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEYが設定されていません")

    client = AsyncAPIClient(
        api_key=api_key,
        max_workers=max_workers,
        max_retries=3,
        max_output_tokens=16384  # 出力切断（finish_reason=length）対策のため大きめに設定
    )

    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager()

    total_chars = sum(len(d["text"]) for d in documents)
    total_input_chars = sum(len(_normalize_whitespace(d["text"])) for d in documents)
    stats: Dict[str, int] = {"step1_fallbacks": 0, "step2_fallbacks": 0, "step3_fallbacks": 0}

    logger.info("=" * 60)
    logger.info("チャンク化処理開始 (3段階・文書単位)")
    logger.info("=" * 60)
    logger.info(f"入力文書数: {len(documents)}")
    logger.info(f"入力テキスト: {format_size(total_chars)}")
    logger.info(f"モデル: {model}")
    logger.info(f"並列ワーカー数: {max_workers}")

    paragraphs = await _step1_hierarchical_split(
        documents, client, model, block_size, checkpoint_manager, stats
    )

    step2_chunks = await _step2_semantic_chunking(
        paragraphs, client, model, checkpoint_manager, stats
    )

    if max_chunk_tokens >= EMBEDDING_INPUT_TOKEN_LIMIT:
        logger.warning(
            f"⚠️ max_chunk_tokens={max_chunk_tokens} は Embedding 入力上限"
            f"（{EMBEDDING_INPUT_TOKEN_LIMIT}）以上です。Embedding 時に超過分が"
            f"無言で切り捨てられるため、上限未満の値を推奨します。"
        )

    final_chunks = await _step3_continuity_check(
        step2_chunks, client, model, checkpoint_manager, stats,
        continuity_mode=continuity_mode,
        max_chunk_tokens=max_chunk_tokens,
    )

    # 上限強制分割: Step2出力・フォールバック由来の巨大チャンクも上限内に収める
    final_chunks = _enforce_max_chunk_tokens(final_chunks, max_chunk_tokens)

    # 入力カバレッジ検証（無言のデータ欠落の検知）
    coverage = _report_coverage(
        total_input_chars, final_chunks, stats, api_stats=client.get_stats()
    )

    if output_file:
        output_path = Path(output_file)

        if output_path.suffix.lower() == '.csv':
            save_chunks_as_csv(
                chunks=final_chunks,
                output_file=output_file,
                dataset_type=dataset_type,
                source_file=source_file,
                normalize_whitespace=True,  # ✅ 改行正規化を有効化
                save_simple_csv=True  # ✅ シンプルCSVも保存
            )
            _write_manifest(
                output_file=output_file,
                documents=documents,
                final_chunks=final_chunks,
                coverage=coverage,
                stats=stats,
                model=model,
                block_size=block_size,
                source_file=source_file,
                continuity_mode=continuity_mode,
                max_chunk_tokens=max_chunk_tokens,
            )
        else:
            save_chunks_as_text(
                chunks=[c["text"] for c in final_chunks],
                output_file=output_file
            )

    return [c["text"] for c in final_chunks]


def _write_manifest(
        output_file: str,
        documents: List[Dict],
        final_chunks: List[Dict],
        coverage: Dict,
        stats: Dict,
        model: str,
        block_size: int,
        source_file: Optional[str],
        continuity_mode: str = "rule",
        max_chunk_tokens: int = MAX_CHUNK_TOKENS
) -> str:
    """チャンクCSVと対になる manifest を出力する（後続ステージとの契約明示）。"""
    import json
    from datetime import datetime

    manifest = {
        "schema_version": "chunks:v2",
        "created_at": datetime.now().isoformat(),
        "source_file": source_file or "",
        "output_file": str(output_file),
        "params": {
            "model": model,
            "block_size": block_size,
            "continuity_mode": continuity_mode,
            "max_chunk_tokens": max_chunk_tokens,
            "embedding_input_token_limit": EMBEDDING_INPUT_TOKEN_LIMIT,
        },
        "counts": {
            "documents": len(documents),
            "chunks": len(final_chunks),
        },
        "coverage": coverage,
        "fallbacks": dict(stats),
        "columns": ["chunk_id", "text", "tokens", "chunk_idx", "doc_id",
                    "dataset_type", "type", "sentence_count", "source_file"],
        "text_column": "text",
    }

    manifest_path = str(Path(output_file).with_suffix("")) + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"📜 manifest 出力: {manifest_path}")
    return manifest_path


async def _step1_hierarchical_split(
        documents: List[Dict],
        client: AsyncAPIClient,
        model: str,
        block_size: int,
        checkpoint_manager: CheckpointManager,
        stats: Dict
) -> List[Dict]:
    """
    Step 1: 階層構造化（段落分割）

    文書ごとに文境界を保ってブロック分割し、各ブロックをLLMで段落に分割する。
    文書をまたいだブロックは作られない。

    【カバレッジ保証】
    LLM呼び出しの失敗・パース失敗時は、そのブロックをそのまま1段落として
    引き継ぐ（旧実装はブロックを警告ログのみで破棄していた）。

    Returns:
        段落のリスト [{'doc_id': ..., 'text': ...}, ...]
    """
    if checkpoint_manager.exists("step1"):
        logger.info("Step1: チェックポイントから再開")
        return checkpoint_manager.load("step1")

    logger.info("\n[Step 1/3] 階層構造化（段落分割・文書単位）")

    # 文書ごとに文境界でブロック分割
    block_entries: List[Dict] = []
    for doc in documents:
        for block in _split_document_into_blocks(doc["text"], block_size):
            block_entries.append({"doc_id": doc["doc_id"], "block": block})

    logger.info(f"  入力: {len(documents)} 文書 → {len(block_entries)} ブロック（文境界で分割）")

    tasks = []
    for i, entry in enumerate(block_entries):
        # [MIGRATION] OpenAI はプロンプトを user メッセージにインライン結合する
        # （Anthropic の system + cache_control は使用しない）
        prompt = f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{entry['block']}"
        task = client.generate_content(
            model=model,
            contents=prompt,
            response_schema=StructuralResult,
            task_id=f"step1_block_{i}",
        )
        tasks.append(task)

    results = await async_tqdm.gather(
        *tasks,
        desc="Step1: 段落分割",
        total=len(tasks)
    )

    paragraphs: List[Dict] = []
    for entry, result_json in zip(block_entries, results):
        parsed = None
        if result_json:
            try:
                parsed = StructuralResult.model_validate_json(result_json)
            except Exception as e:
                logger.warning(f"Step1 パース失敗（フォールバックで保全）: {e}")

        if parsed and parsed.paragraphs:
            for para in parsed.paragraphs:
                para_text = _postprocess_paragraph(para.full_text)
                if para_text.strip():
                    paragraphs.append({"doc_id": entry["doc_id"], "text": para_text})
        else:
            # 【カバレッジ保証】失敗ブロックは捨てずにそのまま1段落として引き継ぐ
            stats["step1_fallbacks"] += 1
            paragraphs.append({
                "doc_id": entry["doc_id"],
                "text"  : _postprocess_paragraph(entry["block"])
            })

    logger.info(f"  出力: {len(paragraphs)} 段落（フォールバック: {stats['step1_fallbacks']}）")
    checkpoint_manager.save("step1", paragraphs)

    return paragraphs


async def _step2_semantic_chunking(
        paragraphs: List[Dict],
        client: AsyncAPIClient,
        model: str,
        checkpoint_manager: CheckpointManager,
        stats: Dict
) -> List[Dict]:
    """
    Step 2: 意味的チャンキング

    Step1の段落を意味的なチャンクに分割する。

    【カバレッジ保証】
    LLM呼び出しの失敗・パース失敗時は、その段落をそのまま1チャンクとして
    引き継ぐ（旧実装は段落を警告ログのみで破棄していた）。

    Returns:
        チャンクのリスト [{'doc_id': ..., 'text': ...}, ...]
    """
    if checkpoint_manager.exists("step2"):
        logger.info("Step2: チェックポイントから再開")
        return checkpoint_manager.load("step2")

    logger.info("\n[Step 2/3] 意味的チャンキング")
    logger.info(f"  入力: {len(paragraphs)} 段落")

    tasks = []
    for i, para in enumerate(paragraphs):
        # [MIGRATION] OpenAI はプロンプトを user メッセージにインライン結合する
        prompt = f"{SEMANTIC_CHUNKING_PROMPT}\n\n【入力テキスト】\n{para['text']}"
        task = client.generate_content(
            model=model,
            contents=prompt,
            response_schema=StructuralResult,
            task_id=f"step2_para_{i}",
        )
        tasks.append(task)

    results = await async_tqdm.gather(
        *tasks,
        desc="Step2: 意味的分割",
        total=len(tasks)
    )

    chunks: List[Dict] = []
    for para, result_json in zip(paragraphs, results):
        parsed = None
        if result_json:
            try:
                parsed = StructuralResult.model_validate_json(result_json)
            except Exception as e:
                logger.warning(f"Step2 パース失敗（フォールバックで保全）: {e}")

        if parsed and parsed.paragraphs:
            for p in parsed.paragraphs:
                if p.full_text.strip():
                    chunks.append({"doc_id": para["doc_id"], "text": p.full_text})
        else:
            # 【カバレッジ保証】失敗段落は捨てずにそのまま1チャンクとして引き継ぐ
            stats["step2_fallbacks"] += 1
            chunks.append({"doc_id": para["doc_id"], "text": para["text"]})

    logger.info(f"  出力: {len(chunks)} チャンク（フォールバック: {stats['step2_fallbacks']}）")
    checkpoint_manager.save("step2", chunks)

    return chunks


# ルールベース連続性判定: 次チャンクがこれらで始まる場合は前チャンクへの
# 依存（指示語・接続語）とみなして結合候補とする
_CONTINUITY_MARKERS = (
    "この", "その", "それ", "これ", "また", "しかし", "さらに", "一方",
    "なお", "ただし", "そして", "だが", "したがって", "つまり",
    "このため", "そのため", "他にも", "加えて", "同様に",
)

# 次チャンクがこのトークン数未満なら単独チャンクとして短すぎるため結合候補とする
_MIN_STANDALONE_TOKENS = 50


def _split_oversized_text(text: str, max_tokens: int) -> List[str]:
    """max_tokens を超えるテキストを文境界で複数ピースに分割する。

    1文単独で max_tokens を超える場合はその文をそのまま1ピースとする
    （文の途中では切らない。Embedding 側の切り捨ては警告で可視化）。
    """
    sentences = _split_sentences_simple(text)
    if not sentences:
        return [text]

    pieces: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        if current and current_tokens + sent_tokens > max_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sent)
        current_tokens += sent_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _enforce_max_chunk_tokens(chunks: List[Dict], max_tokens: int) -> List[Dict]:
    """全チャンクに最大トークン数を強制する（超過分は文境界で分割）。

    Step3 は「結合時」のみ上限を見ており、Step2 が出力する単一チャンクや
    フォールバックで保全されたブロックには上限がなかった。Embedding
    （text-embedding-3-large）の無言切り捨てを防ぐため、最終チャンク全件に
    対して上限を強制する。doc_id は分割後も引き継ぐ。
    """
    enforced: List[Dict] = []
    split_count = 0
    for chunk in chunks:
        tokens = _count_tokens(chunk["text"])
        if tokens <= max_tokens:
            enforced.append(chunk)
            continue
        pieces = _split_oversized_text(chunk["text"], max_tokens)
        if len(pieces) == 1:
            logger.warning(
                f"  1文で {tokens} トークン（上限 {max_tokens} 超）のチャンクは分割不可のため保持。"
                f"Embedding 入力上限（{EMBEDDING_INPUT_TOKEN_LIMIT}）超過時は切り捨てに注意"
            )
            enforced.append(chunk)
            continue
        split_count += 1
        for piece in pieces:
            enforced.append({"doc_id": chunk["doc_id"], "text": piece})

    if split_count:
        logger.info(
            f"  📏 上限強制分割: {split_count} チャンクが {max_tokens} トークン超のため"
            f"文境界で分割（{len(chunks)} → {len(enforced)} チャンク）"
        )
    return enforced


def _rule_based_continuity(prev_text: str, next_text: str) -> bool:
    """ルールベースの連続性判定（LLM呼び出しなし）。

    以下のいずれかなら「連続」と判定する:
    - 次チャンクが指示語・接続語で始まる（前チャンクへの文脈依存）
    - 次チャンクが単独チャンクとして短すぎる（過分割の修正）

    トークン上限チェックは呼び出し側（マージ処理）で共通に行う。
    """
    next_head = next_text.lstrip()
    if any(next_head.startswith(m) for m in _CONTINUITY_MARKERS):
        return True
    if _count_tokens(next_text) < _MIN_STANDALONE_TOKENS:
        return True
    return False


async def _step3_continuity_check(
        chunks: List[Dict],
        client: AsyncAPIClient,
        model: str,
        checkpoint_manager: CheckpointManager,
        stats: Dict,
        continuity_mode: str = "rule",
        max_chunk_tokens: int = MAX_CHUNK_TOKENS
) -> List[Dict]:
    """
    Step 3: 文脈連続性チェック（同一文書内のみ）

    隣接するチャンク間の文脈連続性を判定し、連続している場合は結合する。

    - 文書（doc_id）をまたぐ結合は行わない（文書境界の保証）
    - 結合後のチャンクが max_chunk_tokens を超える場合は結合しない
      （旧実装は無条件結合でチャンクが際限なく肥大化していた）
    - 判定失敗時は結合しない（テキストは保全される）

    continuity_mode:
        - "rule": 指示語・接続語と短チャンク判定によるルールベース（LLM 0回）。
          旧実装のチャンク数-1回のLLM呼び出しを置き換えるデフォルト
        - "llm" : LLMによるペア判定（旧動作）
        - "off" : 結合しない
    """
    if checkpoint_manager.exists("step3"):
        logger.info("Step3: チェックポイントから再開")
        return checkpoint_manager.load("step3")

    logger.info(f"\n[Step 3/3] 文脈連続性チェック（mode={continuity_mode}, 同一文書内のみ）")
    logger.info(f"  入力: {len(chunks)} チャンク")

    if continuity_mode == "off" or len(chunks) <= 1:
        checkpoint_manager.save("step3", chunks)
        return chunks

    # 同一文書内の隣接ペアのみ判定対象とする
    pair_indices = [
        i for i in range(len(chunks) - 1)
        if chunks[i]["doc_id"] == chunks[i + 1]["doc_id"]
    ]
    logger.info(
        f"  判定ペア: {len(pair_indices)} / {len(chunks) - 1} "
        f"（文書をまたぐペアは判定せず分離）"
    )

    results_map: Dict[int, Optional[str]] = {}
    if continuity_mode == "llm":
        # LLM判定（旧動作）: 同一文書内ペアごとに1回のLLM呼び出し
        tasks = []
        for i in pair_indices:
            # [MIGRATION] OpenAI はプロンプトを user メッセージにインライン結合する
            prompt = (
                f"{CONTINUITY_CHECK_PROMPT}\n\n"
                f"【前のテキスト】\n{chunks[i]['text']}\n\n"
                f"【次のテキスト】\n{chunks[i + 1]['text']}"
            )
            task = client.generate_content(
                model=model,
                contents=prompt,
                response_schema=ContinuityResult,
                task_id=f"step3_pair_{i}",
            )
            tasks.append(task)

        results = await async_tqdm.gather(
            *tasks,
            desc="Step3: 連続性チェック",
            total=len(tasks)
        )
        results_map = dict(zip(pair_indices, results))

    pair_index_set = set(pair_indices)

    # マージ処理
    final_chunks: List[Dict] = [dict(chunks[0])]
    for i in range(len(chunks) - 1):
        nxt = chunks[i + 1]
        is_connected = False

        if i in pair_index_set:
            if continuity_mode == "rule":
                is_connected = _rule_based_continuity(chunks[i]["text"], nxt["text"])
            else:  # "llm"
                result_json = results_map.get(i)
                if result_json:
                    try:
                        result = ContinuityResult.model_validate_json(result_json)
                        is_connected = result.is_connected
                    except Exception as e:
                        logger.warning(f"Step3 パース失敗（分離扱い）: {e}")
                        stats["step3_fallbacks"] += 1
                else:
                    stats["step3_fallbacks"] += 1

        if is_connected:
            # トークン上限チェック: 超える場合は結合しない
            merged_text = final_chunks[-1]["text"] + "\n\n" + nxt["text"]
            if _count_tokens(merged_text) <= max_chunk_tokens:
                final_chunks[-1]["text"] = merged_text
                continue
            logger.debug(
                f"  チャンク{i + 1}+{i + 2}: 連続判定だが結合後 {max_chunk_tokens} トークン超のため分離"
            )

        final_chunks.append(dict(nxt))

    logger.info(f"  出力: {len(final_chunks)} チャンク（マージ後）")
    checkpoint_manager.save("step3", final_chunks)

    return final_chunks


# ================================================================
# メイン関数
# ================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="LLMベースセマンティックチャンキング（統一版 - make_qa形式互換）"
    )

    # ================================================================
    # INPUT オプション（✅ 統一版）
    # ================================================================
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="入力ファイル (.txt, .csv)"
    )

    # ================================================================
    # OUTPUT オプション（✅ 統一版）
    # ================================================================
    parser.add_argument(
        "--output",
        type=str,
        default="chunks_output",
        help="出力ディレクトリ（デフォルト: chunks_output）"
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        default=False,
        help="出力ファイル名にタイムスタンプを付与する（デフォルト: 固定ファイル名）\n"
             "  固定(デフォルト): cc_news_1per_chunks.csv\n"
             "  タイムスタンプ:   cc_news_1per_chunks_20260505_004819.csv"
    )

    # ================================================================
    # モデル・処理パラメータ（✅ 短縮形削除）
    # ================================================================
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",  # [MIGRATION] "gemini-3-flash-preview" → "gpt-5-mini"
        help="使用するLLMモデル"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="並列ワーカー数"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1000,  # ✅ 2000→1000に変更（MAX_TOKENS対策）
        help="ブロックサイズ（文字数）。大きすぎるとMAX_TOKENSエラーが発生"
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=MAX_CHUNK_TOKENS,
        help=f"チャンクの最大トークン数（デフォルト: {MAX_CHUNK_TOKENS}）。\n"
             f"Step3の結合上限かつ最終チャンクの強制分割上限。\n"
             f"Embedding(text-embedding-3-large)の入力上限 {EMBEDDING_INPUT_TOKEN_LIMIT} 未満を推奨"
    )
    parser.add_argument(
        "--continuity-mode",
        type=str,
        choices=["rule", "llm", "off"],
        default="rule",
        help="Step3（連続性チェック）のモード（デフォルト: rule）\n"
             "  rule: ルールベース判定（LLM呼び出しなし・高速）\n"
             "  llm : LLMによるペア判定（旧動作・チャンク数-1回のLLM呼び出し）\n"
             "  off : 結合しない"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログ出力"
    )

    # ================================================================
    # その他のオプション（変更なし）
    # ================================================================
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="再開するジョブID"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="CSVのテキストカラム名"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="最大処理行数（CSV用）"
    )
    parser.add_argument(
        "--combine-rows",
        action="store_true",
        help="CSV全行を結合"
    )

    args = parser.parse_args()

    # ロギング設定
    setup_logging(verbose=args.verbose)

    # ================================================================
    # モデル名検証: このプロジェクトは OpenAI API を使用します
    # Claude モデル名（claude-*）は OpenAI API では使用できません
    # ================================================================
    if args.model.lower().startswith("claude-"):
        logger.error(
            f"❌ モデル名エラー: '{args.model}' は Anthropic/Claude のモデルです。\n"
            f"   このプロジェクト (openai_grace_agent) は OpenAI API を使用しています。\n"
            f"   OpenAI モデル名を指定してください。例:\n"
            f"     --model gpt-5-mini   (推奨・デフォルト)\n"
            f"     --model gpt-4o-mini\n"
            f"     --model gpt-4o"
        )
        return

    # ================================================================
    # 入力ファイル読み込み（✅ args.input → args.input_file）
    # ================================================================
    input_path = Path(args.input_file)  # ✅ 変更
    if not input_path.exists():
        logger.error(f"入力ファイルが見つかりません: {args.input_file}")
        return

    file_extension = input_path.suffix.lower()

    # テキスト読み込み（CSVは1行=1文書として文書境界を保持）
    if file_extension == '.csv':
        documents = load_documents_from_csv(
            csv_path=args.input_file,
            text_column=args.text_column,
            max_rows=args.max_rows,
        )
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            documents = [{"doc_id": 0, "text": f.read()}]

    logger.info("")
    logger.info("=" * 60)
    logger.info("チャンキング処理開始")
    logger.info("=" * 60)
    logger.info(f"📁 入力ファイル: {args.input_file}")
    logger.info(f"📄 文書数: {len(documents)}")
    logger.info(f"📊 テキストサイズ: {format_size(sum(len(d['text']) for d in documents))}")
    logger.info(f"🤖 モデル: {args.model}")
    logger.info(f"👥 並列ワーカー数: {args.workers}")
    logger.info("=" * 60)

    # ================================================================
    # 出力ファイル名の自動生成（✅ 新規機能）
    # ================================================================
    dataset_type = input_path.stem
    output_file = generate_output_filename(
        args.input_file,
        args.output,
        dataset_type,
        use_timestamp=args.timestamp
    )

    logger.info(f"📝 出力ファイル: {output_file}")
    logger.info("=" * 60)
    logger.info("")

    # ================================================================
    # チャンク作成（既存処理）
    # ================================================================
    checkpoint_manager = CheckpointManager(job_id=args.resume) if args.resume else CheckpointManager()

    final_chunks = await chunks_all_async(
        documents=documents,
        model=args.model,
        max_workers=args.workers,
        block_size=args.block_size,
        continuity_mode=args.continuity_mode,
        max_chunk_tokens=args.max_chunk_tokens,
        checkpoint_manager=checkpoint_manager,
        output_file=output_file,  # ✅ 自動生成されたファイル名
        dataset_type=dataset_type,
        source_file=input_path.name
    )

    # ================================================================
    # 完了ログ
    # ================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ チャンク作成完了")
    logger.info("=" * 60)
    logger.info(f"📊 生成チャンク数: {len(final_chunks)}")
    logger.info(f"📁 出力ファイル: {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
