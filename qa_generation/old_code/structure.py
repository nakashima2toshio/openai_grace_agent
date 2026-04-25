#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/structure.py - チャンク作成・統合モジュール（v2.0 - semantic.py依存削除版）

修正内容（v2.0）:
- ★重要★ semantic.py (SemanticCoverage) への依存を完全に削除
- create_semantic_chunks() をシンプルな段落/文ベースの分割に置き換え
- create_document_chunks() をシンプルな実装に置き換え
- merge_small_chunks() は変更なし

注意:
- 高度なセマンティック分割が必要な場合は、事前に csv_text_to_chunks_text_csv.py を使用してください
- このモジュールはシンプルなチャンク分割のみを提供します（LLMベースの分割は行いません）
"""

import re
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import tiktoken
from config import DATASET_CONFIGS

logger = logging.getLogger(__name__)

# トークナイザーの初期化（モジュールレベルで1回のみ）
_tokenizer = None


def _get_tokenizer():
    """トークナイザーのシングルトン取得"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    """テキストのトークン数をカウント"""
    return len(_get_tokenizer().encode(text))


def _split_into_paragraphs(text: str) -> List[str]:
    """テキストを段落に分割

    Args:
        text: 分割対象テキスト

    Returns:
        段落のリスト
    """
    # 複数の改行で段落を分割
    paragraphs = re.split(r'\n\s*\n', text)
    # 空の段落を除去
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def _split_into_sentences(text: str) -> List[str]:
    """テキストを文に分割（日本語/英語対応）

    Args:
        text: 分割対象テキスト

    Returns:
        文のリスト
    """
    # 日本語の句点、英語のピリオド等で分割
    sentences = re.findall(r'[^。．.！？!?]+[。．.！？!?]?\s*', text)
    if not sentences:
        # 句点がない場合は全体を1つの文として扱う
        sentences = [text.strip()] if text.strip() else []
    else:
        # 最後の文の後に句点がないテキストが残っている場合
        last_pos = text.rfind(sentences[-1]) + len(sentences[-1])
        if last_pos < len(text):
            remaining = text[last_pos:].strip()
            if remaining:
                sentences.append(remaining)

    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _force_split_by_tokens(text: str, max_tokens: int) -> List[str]:
    """トークン数で強制分割（最終手段）

    Args:
        text: 分割対象テキスト
        max_tokens: 最大トークン数

    Returns:
        分割されたテキストのリスト
    """
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


def create_simple_chunks(text: str, max_tokens: int = 200, min_tokens: int = 50,
                         chunk_id_prefix: str = "chunk") -> List[Dict]:
    """
    シンプルなチャンク分割（段落/文ベース）

    SemanticCoverageを使用せず、シンプルな規則ベースでチャンクを作成。

    分割ロジック:
    1. 段落（\n\n）で分割を試みる
    2. 段落がmax_tokensを超える場合は文（。）で分割
    3. 単一文がmax_tokensを超える場合はトークン単位で強制分割

    Args:
        text: 分割対象テキスト
        max_tokens: チャンクの最大トークン数
        min_tokens: チャンクの最小トークン数（これ未満は次と結合を検討）
        chunk_id_prefix: チャンクIDのプレフィックス

    Returns:
        チャンクのリスト
    """
    if not text or not text.strip():
        return []

    chunks = []

    # Step 1: 段落で分割
    paragraphs = _split_into_paragraphs(text)

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        if para_tokens <= max_tokens:
            # 段落がそのままチャンクとして適切
            chunks.append({
                'text'  : para,
                'tokens': para_tokens,
                'type'  : 'paragraph'
            })
        else:
            # 段落が大きすぎる → 文単位で分割
            sentences = _split_into_sentences(para)
            current_chunk_texts = []
            current_tokens = 0

            for sent in sentences:
                sent_tokens = _count_tokens(sent)

                if sent_tokens > max_tokens:
                    # 単一文が上限超過 → 強制分割
                    if current_chunk_texts:
                        chunks.append({
                            'text'  : ''.join(current_chunk_texts),
                            'tokens': current_tokens,
                            'type'  : 'sentence_group'
                        })
                        current_chunk_texts = []
                        current_tokens = 0

                    # 強制分割を実施
                    forced_parts = _force_split_by_tokens(sent, max_tokens)
                    for part in forced_parts:
                        chunks.append({
                            'text'  : part,
                            'tokens': _count_tokens(part),
                            'type'  : 'forced_split'
                        })

                elif current_tokens + sent_tokens > max_tokens:
                    # 追加すると上限超過 → 現在のチャンクを確定
                    if current_chunk_texts:
                        chunks.append({
                            'text'  : ''.join(current_chunk_texts),
                            'tokens': current_tokens,
                            'type'  : 'sentence_group'
                        })
                    current_chunk_texts = [sent]
                    current_tokens = sent_tokens

                else:
                    # 追加可能
                    current_chunk_texts.append(sent)
                    current_tokens += sent_tokens

            # 残りを確定
            if current_chunk_texts:
                chunks.append({
                    'text'  : ''.join(current_chunk_texts),
                    'tokens': current_tokens,
                    'type'  : 'sentence_group'
                })

    # Step 2: 小さすぎるチャンクを結合（オプション）
    if min_tokens > 0:
        chunks = _merge_tiny_chunks(chunks, min_tokens, max_tokens)

    # Step 3: チャンクIDを付与
    for i, chunk in enumerate(chunks):
        chunk['id'] = f"{chunk_id_prefix}_{i}"
        chunk['sentences'] = _split_into_sentences(chunk['text'])

    return chunks


def _merge_tiny_chunks(chunks: List[Dict], min_tokens: int, max_tokens: int) -> List[Dict]:
    """小さすぎるチャンクを前後と結合

    Args:
        chunks: チャンクのリスト
        min_tokens: 最小トークン数
        max_tokens: 最大トークン数

    Returns:
        結合後のチャンクリスト
    """
    if not chunks:
        return chunks

    merged = []
    current = None

    for chunk in chunks:
        if current is None:
            current = chunk.copy()
        elif chunk['tokens'] < min_tokens:
            # 小さいチャンク → 結合を試みる
            combined_tokens = current['tokens'] + chunk['tokens']
            if combined_tokens <= max_tokens:
                current['text'] += '\n\n' + chunk['text']
                current['tokens'] = combined_tokens
                current['type'] = 'merged'
            else:
                merged.append(current)
                current = chunk.copy()
        else:
            # 通常サイズのチャンク
            if current['tokens'] < min_tokens:
                # 前のチャンクが小さい場合は結合を試みる
                combined_tokens = current['tokens'] + chunk['tokens']
                if combined_tokens <= max_tokens:
                    chunk['text'] = current['text'] + '\n\n' + chunk['text']
                    chunk['tokens'] = combined_tokens
                    chunk['type'] = 'merged'
                    current = chunk
                else:
                    merged.append(current)
                    current = chunk.copy()
            else:
                merged.append(current)
                current = chunk.copy()

    if current:
        merged.append(current)

    return merged


# ================================================================
# 後方互換性のためのエイリアス
# ================================================================

def create_semantic_chunks(text: str, lang: str = "ja", max_tokens: int = 200,
                           chunk_id_prefix: str = "chunk",
                           semantic_analyzer=None,  # 無視（後方互換性のため残す）
                           overlap_tokens: int = 0,  # 現在は未使用
                           use_similarity: bool = False,  # 無視
                           similarity_threshold: float = 0.7  # 無視
                           ) -> List[Dict]:
    """
    セマンティック分割によるチャンク作成（後方互換性のためのエイリアス）

    注意:
    - この関数はSemanticCoverageを使用しません
    - 高度なセマンティック分割が必要な場合は csv_text_to_chunks_text_csv.py を使用してください
    - overlap_tokens, use_similarity, similarity_threshold は現在無視されます

    Args:
        text: 分割対象テキスト
        lang: 言語（現在は自動判定、この引数は無視）
        max_tokens: チャンクの最大トークン数
        chunk_id_prefix: チャンクIDのプレフィックス
        semantic_analyzer: 無視（後方互換性のため残す）
        overlap_tokens: 無視（将来の実装用）
        use_similarity: 無視（SemanticCoverage不使用のため）
        similarity_threshold: 無視（SemanticCoverage不使用のため）

    Returns:
        チャンクのリスト
    """
    if semantic_analyzer is not None:
        logger.warning(
            "semantic_analyzer引数は無視されます。"
            "高度なセマンティック分割が必要な場合は csv_text_to_chunks_text_csv.py を使用してください。"
        )

    if use_similarity:
        logger.warning(
            "use_similarity=True は無視されます。"
            "ベクトル類似度分割が必要な場合は csv_text_to_chunks_text_csv.py を使用してください。"
        )

    return create_simple_chunks(
        text=text,
        max_tokens=max_tokens,
        min_tokens=50,
        chunk_id_prefix=chunk_id_prefix
    )


def _process_single_document(idx, row, dataset_type, text_col, title_col, lang, chunk_size,
                             semantic_analyzer=None,  # 無視
                             overlap_tokens=0, use_similarity=False,
                             similarity_threshold=0.7) -> List[Dict]:
    """単一文書の処理（並列実行用）"""
    # row[text_col]はSeriesやオブジェクトの可能性があるため、明示的にstrに変換
    text = str(row[text_col]) if pd.notna(row[text_col]) else ""

    if not text.strip():
        return []

    # タイトルがある場合は含める
    if title_col and title_col in row and pd.notna(row[title_col]):
        doc_id = f"{dataset_type}_{idx}_{str(row[title_col])[:30]}"
    else:
        doc_id = f"{dataset_type}_{idx}"

    try:
        chunk_id_prefix = f"{doc_id}_chunk"

        # ✅ 修正: create_simple_chunks を使用（SemanticCoverage不使用）
        chunks = create_simple_chunks(
            text=text,
            max_tokens=chunk_size,
            min_tokens=50,
            chunk_id_prefix=chunk_id_prefix
        )

        # 各チャンクにメタデータを追加
        for i, chunk in enumerate(chunks):
            chunk['doc_id'] = doc_id
            chunk['doc_idx'] = idx
            chunk['chunk_idx'] = i
            chunk['dataset_type'] = dataset_type

        return chunks
    except Exception as e:
        logger.warning(f"チャンク作成エラー (doc {idx}): {e}")
        return []


def create_document_chunks(df: pd.DataFrame, dataset_type: str, max_docs: Optional[int] = None,
                           config: Optional[Dict] = None,
                           overlap_tokens: int = 0,  # 現在は未使用
                           use_similarity: bool = False,  # 無視
                           similarity_threshold: float = 0.7,  # 無視
                           max_workers: int = 8) -> List[Dict]:
    """DataFrameから文書チャンクを作成（シンプル分割・並列処理）

    注意:
    - この関数はSemanticCoverageを使用しません
    - 高度なセマンティック分割が必要な場合は csv_text_to_chunks_text_csv.py を事前に使用してください
    - overlap_tokens, use_similarity, similarity_threshold は現在無視されます

    Args:
        df: データフレーム
        dataset_type: データセットタイプ
        max_docs: 処理する最大文書数
        config: データセット設定
        overlap_tokens: 無視（将来の実装用）
        use_similarity: 無視（SemanticCoverage不使用のため）
        similarity_threshold: 無視（SemanticCoverage不使用のため）
        max_workers: 並列処理のワーカー数

    Returns:
        チャンクのリスト
    """
    if use_similarity:
        logger.warning(
            "use_similarity=True は無視されます。"
            "ベクトル類似度分割が必要な場合は csv_text_to_chunks_text_csv.py を使用してください。"
        )

    if config is None:
        config = DATASET_CONFIGS.get(dataset_type)
        if not config:
            raise ValueError(f"未対応のデータセット: {dataset_type}")

    text_col = config["text_column"]
    title_col = config.get("title_column")
    chunk_size = config["chunk_size"]
    lang = config["lang"]

    all_chunks = []

    # 処理する文書数を制限
    docs_to_process = df.head(max_docs) if max_docs else df

    logger.info(f"チャンク作成開始: {len(docs_to_process)}件の文書（シンプル分割・{max_workers}並列）")

    total_docs = len(docs_to_process)
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # タスクのサブミット
        future_to_idx = {
            executor.submit(
                _process_single_document,
                idx, row, dataset_type, text_col, title_col, lang, chunk_size,
                None,  # semantic_analyzer（無視）
                overlap_tokens, use_similarity, similarity_threshold
            ): idx
            for idx, row in docs_to_process.iterrows()
        }

        # 結果の収集
        for future in as_completed(future_to_idx):
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_docs:
                logger.info(f"  チャンク作成進捗: {completed_count}/{total_docs} 文書完了")

            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"予期せぬエラー: {e}")

    logger.info(f"チャンク作成完了: {len(all_chunks)}個のチャンク（シンプル分割）")
    return all_chunks


def merge_small_chunks(chunks: List[Dict], min_tokens: int = 150, max_tokens: int = 400) -> List[Dict]:
    """小さいチャンクを統合して適切なサイズにする

    Args:
        chunks: チャンクのリスト
        min_tokens: このトークン数未満のチャンクは統合対象
        max_tokens: 統合後の最大トークン数

    Returns:
        統合されたチャンクのリスト
    """
    tokenizer = _get_tokenizer()
    merged_chunks = []
    current_merge = None

    for chunk in chunks:
        chunk_tokens = len(tokenizer.encode(chunk['text']))

        # 大きいチャンクはそのまま追加
        if chunk_tokens >= min_tokens:
            if current_merge:
                merged_chunks.append(current_merge)
                current_merge = None
            merged_chunks.append(chunk)
        else:
            # 小さいチャンクは統合候補
            if current_merge is None:
                current_merge = chunk.copy()
                current_merge['merged'] = True
                current_merge['original_chunks'] = [chunk['id']]
            else:
                # 統合可能かチェック
                merge_tokens = len(tokenizer.encode(current_merge['text']))
                if merge_tokens + chunk_tokens <= max_tokens:
                    # 同じ文書からのチャンクのみ統合
                    if current_merge.get('doc_id') == chunk.get('doc_id'):
                        current_merge['text'] += "\n\n" + chunk['text']
                        current_merge['original_chunks'].append(chunk['id'])
                        if 'chunk_idx' in current_merge:
                            current_merge['chunk_idx'] = f"{current_merge['chunk_idx']}-{chunk['chunk_idx']}"
                    else:
                        # 異なる文書の場合は別々に
                        merged_chunks.append(current_merge)
                        current_merge = chunk.copy()
                        current_merge['merged'] = True
                        current_merge['original_chunks'] = [chunk['id']]
                else:
                    # サイズオーバーの場合は現在の統合を追加して新規開始
                    merged_chunks.append(current_merge)
                    current_merge = chunk.copy()
                    current_merge['merged'] = True
                    current_merge['original_chunks'] = [chunk['id']]

    # 最後の統合チャンクを追加
    if current_merge:
        merged_chunks.append(current_merge)

    logger.info(
        f"チャンク統合: {len(chunks)}個 → {len(merged_chunks)}個 ({100 * (1 - len(merged_chunks) / len(chunks)):.1f}%削減)")
    return merged_chunks


# ================================================================
# エクスポート
# ================================================================

__all__ = [
    # 主要関数
    "create_simple_chunks",
    "create_document_chunks",
    "merge_small_chunks",
    # 後方互換性
    "create_semantic_chunks",
]
