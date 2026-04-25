#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/pipeline.py - Q/A生成パイプライン制御モジュール（リファクタリング版）

改修内容:
- input_chunksパラメータを削除（チャンク処理の統一）
- テキストファイル（.txt）対応を追加
- load_chunks_from_csv()メソッドを削除
- コードの簡素化（約149行削減）
- ✅ concurrencyパラメータを追加（並列タスク数の指定）

使用例:
  # make_qa_register_qdrant.py から呼び出し
  result = pipeline.run(
      use_celery=True,
      celery_workers=1,      # ワーカープロセス数チェック用
      concurrency=8,         # ✅ 並列タスク数
      batch_chunks=3,
      merge_chunks=True,
      use_smart_generation=True
  )
"""

import sys
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
from pathlib import Path

from config import DATASET_CONFIGS
from helper.helper_llm import LLMClient
from qa_generation.config import LOCAL_DATASET_EXTENSIONS
from qa_generation.structure import create_document_chunks, merge_small_chunks
from qa_generation.generation import QAGenerator, generate_qa_dataset
from qa_generation.evaluation import analyze_coverage
from celery_tasks import submit_unified_qa_generation, collect_results, check_celery_workers

logger = logging.getLogger(__name__)


class QAPipeline:
    """Q/A生成パイプライン"""

    def __init__(self,
                 dataset_name: Optional[str] = None,
                 input_file: Optional[str] = None,
                 model: str = "gemini-2.0-flash",
                 output_dir: str = "qa_output/pipeline",
                 max_docs: Optional[int] = None,
                 client: Optional[LLMClient] = None):
        """
        Args:
            dataset_name: データセット名 (cc_news, wikipedia_ja, etc.)
            input_file: ローカル入力ファイルパス（.txt, .csv）
            model: 使用するモデル
            output_dir: 出力ディレクトリ
            max_docs: 最大処理文書数
            client: LLMクライアント（DI用）
        """
        self.dataset_name = dataset_name
        self.input_file = input_file
        self.model = model
        self.output_dir = output_dir
        self.max_docs = max_docs
        self.client = client

        # 引数の排他制御
        self._validate_inputs()

        self.config = self._load_config()

    def _validate_inputs(self):
        """入力パラメータの検証"""
        inputs = [self.dataset_name, self.input_file]
        non_none_count = sum(1 for x in inputs if x is not None)

        if non_none_count == 0:
            raise ValueError(
                "dataset_name, input_file のいずれか1つを指定してください"
            )

        if non_none_count > 1:
            raise ValueError(
                "dataset_name, input_file は同時に指定できません"
            )

    def _load_config(self) -> Dict[str, Any]:
        """設定をロード"""
        if self.input_file:
            # ローカルファイル用の動的設定
            file_basename = Path(self.input_file).stem
            lang = "ja"  # デフォルト
            return {
                "name"        : f"ローカルファイル ({file_basename})",
                "text_column" : "Combined_Text",
                "title_column": None,
                "lang"        : lang,
                "chunk_size"  : 300,
                "qa_per_chunk": 3,
                "type"        : file_basename
            }

        elif self.dataset_name:
            # 事前定義データセット
            if self.dataset_name not in DATASET_CONFIGS:
                raise ValueError(f"未知のデータセット: {self.dataset_name}")

            config = DATASET_CONFIGS[self.dataset_name].copy()
            logger.info(f"データセット設定をロード: {self.dataset_name}")
            return config

        else:
            raise ValueError("設定の読み込みに失敗しました")

    def load_data(self) -> pd.DataFrame:
        """データを読み込む"""
        from qa_generation.data_io import load_uploaded_file, load_preprocessed_data

        logger.info("\n[1/4] データ読み込み...")

        if self.input_file:
            file_path = Path(self.input_file)

            # ✅ テキストファイル対応
            if file_path.suffix == '.txt':
                logger.info(f"  📄 テキストファイル: {self.input_file}")

                # テキストファイルを読み込み
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # DataFrameに変換
                df = pd.DataFrame([{
                    'Combined_Text': text,
                    'title'        : file_path.stem
                }])

                logger.info(f"  ✅ 読み込み完了: テキスト長 {len(text):,} 文字")

            # ✅ CSVファイル対応
            elif file_path.suffix == '.csv':
                logger.info(f"  📊 CSVファイル: {self.input_file}")
                df = load_uploaded_file(self.input_file)

            else:
                raise ValueError(f"未対応のファイル形式: {file_path.suffix}\n対応形式: .txt, .csv")

            # 最大文書数制限
            if self.max_docs and len(df) > self.max_docs:
                df = df.head(self.max_docs)
                logger.info(f"  📊 最大文書数制限: {len(df)} 件に制限")

            return df
        else:
            return load_preprocessed_data(self.dataset_name)

    def create_chunks(self, df: pd.DataFrame,
                      overlap_tokens: int = 0,
                      use_similarity: bool = False,
                      similarity_threshold: float = 0.7,
                      max_workers: int = 8) -> List[Dict]:
        """チャンクを作成する

        Args:
            df: データフレーム
            overlap_tokens: チャンク間の重複トークン数
            use_similarity: ベクトル類似度分割を使用するか
            similarity_threshold: 類似度閾値
            max_workers: 並列処理のワーカー数（チャンク作成時）
        """
        logger.info("\n[2/4] チャンク作成...")
        dataset_type = self.config.get("type", "unknown")
        max_docs_for_chunks = None if self.input_file else self.max_docs

        chunks = create_document_chunks(
            df, dataset_type, max_docs_for_chunks, config=self.config,
            overlap_tokens=overlap_tokens,
            use_similarity=use_similarity,
            similarity_threshold=similarity_threshold,
            max_workers=max_workers
        )

        if not chunks:
            logger.error("チャンクが作成されませんでした")
            raise RuntimeError("Chunk creation failed")

        return chunks

    def generate_qa(self, chunks: List[Dict],
                    use_celery: bool = False,
                    celery_workers: int = 1,
                    concurrency: int = 8,  # ✅ 改修: concurrency パラメータ追加
                    batch_chunks: int = 3,
                    merge_chunks: bool = True,
                    min_tokens: int = 150,
                    max_tokens: int = 400,
                    use_smart_generation: bool = True) -> List[Dict]:
        """Q/Aペアを生成する

        Args:
            chunks: チャンクのリスト
            use_celery: Celery並列処理を使用するか
            celery_workers: Celeryワーカープロセス数チェック用（デフォルト: 1）
            concurrency: 並列タスク数（デフォルト: 8）★新規追加
            batch_chunks: 1回のAPIで処理するチャンク数
            merge_chunks: 小さいチャンクを統合するか
            min_tokens: 統合対象の最小トークン数
            max_tokens: 統合後の最大トークン数
            use_smart_generation: スマートQ/A生成を使用するか
        """
        logger.info("\n[3/4] Q/Aペア生成...")

        # ✨ スマート生成モードのログ出力
        mode_name = "スマート生成" if use_smart_generation else "従来方式"
        logger.info(f"  生成モード: {mode_name}")

        if use_celery:
            return self._generate_with_celery(
                chunks, celery_workers, concurrency, batch_chunks,
                merge_chunks, min_tokens, max_tokens, use_smart_generation
            )
        else:
            return self._generate_sync(
                chunks, batch_chunks, merge_chunks, min_tokens, max_tokens, use_smart_generation
            )

    def _generate_with_celery(self, chunks: List[Dict],
                              workers: int,
                              concurrency: int,  # ✅ 改修: concurrency 追加
                              batch_size: int,
                              merge: bool, min_tokens: int, max_tokens: int,
                              use_smart_generation: bool) -> List[Dict]:
        """Celeryを使用した非同期生成

        Args:
            chunks: チャンクのリスト
            workers: ワーカープロセス数チェック用
            concurrency: 並列タスク数★新規追加
            batch_size: バッチサイズ
            merge: チャンク統合を行うか
            min_tokens: 統合対象の最小トークン数
            max_tokens: 統合後の最大トークン数
            use_smart_generation: スマートQ/A生成を使用するか
        """
        # ✅ 改修: 並列数をログ出力
        logger.info(f"Celery並列処理モード:")
        logger.info(f"  - ワーカープロセス数チェック: {workers}")
        logger.info(f"  - 並列タスク数 (concurrency): {concurrency}")

        logger.info("Celeryワーカーの状態を確認中...")
        if not check_celery_workers(workers):
            raise RuntimeError("Celery workers are not running")

        if merge:
            processed_chunks = merge_small_chunks(chunks, min_tokens, max_tokens)
        else:
            processed_chunks = chunks

        # ✨ use_smart_generationをCeleryタスクに渡す
        tasks = submit_unified_qa_generation(
            processed_chunks, self.config, self.model, provider="gemini",
            use_smart_generation=use_smart_generation
        )

        # ✅ 改修: タイムアウトをconcurrencyベースで計算
        timeout_seconds = min(max(len(tasks) * 10, 600), 1800)
        logger.info(f"結果収集タイムアウト: {timeout_seconds}秒（{len(tasks)}タスク）")
        return collect_results(tasks, timeout=timeout_seconds)

    def _generate_sync(self, chunks: List[Dict], batch_size: int,
                       merge: bool, min_tokens: int, max_tokens: int,
                       use_smart_generation: bool) -> List[Dict]:
        """同期生成"""
        logger.info("通常処理モード")
        dataset_type = self.config.get("type", "unknown")

        return generate_qa_dataset(
            chunks,
            dataset_type,
            self.model,
            chunk_batch_size=batch_size,
            merge_chunks=merge,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            config=self.config,
            client=self.client,
            use_smart_generation=use_smart_generation
        )

    def evaluate_coverage(self, chunks: List[Dict], qa_pairs: List[Dict],
                          threshold: Optional[float] = None) -> Dict:
        """カバレッジを評価する"""
        logger.info("\n[4/4] カバレージ分析...")
        dataset_type = self.config.get("type", "unknown")
        return analyze_coverage(chunks, qa_pairs, dataset_type, custom_threshold=threshold)

    def save(self, qa_pairs: List[Dict], coverage_results: Dict) -> Dict[str, str]:
        """結果を保存する"""
        from qa_generation.data_io import save_results

        logger.info("\n結果を保存中...")
        dataset_type = self.config.get("type", "unknown")
        return save_results(qa_pairs, coverage_results, dataset_type, self.output_dir)

    def run(
            self,
            use_celery: bool = False,
            celery_workers: int = 1,  # ✅ 改修: デフォルトを1に変更
            concurrency: int = 8,  # ✅ 改修: concurrency パラメータ追加
            batch_chunks: int = 3,
            merge_chunks: bool = True,
            min_tokens: int = 150,
            max_tokens: int = 400,
            analyze_coverage: bool = True,
            coverage_threshold: Optional[float] = None,
            overlap_tokens: int = 0,
            use_similarity: bool = False,
            similarity_threshold: float = 0.7,
            use_smart_generation: bool = True):
        """
        パイプライン実行

        Args:
            use_celery: Celery並列処理を使用するか
            celery_workers: Celeryワーカープロセス数チェック用（デフォルト: 1）
            concurrency: 並列タスク数（デフォルト: 8）★新規追加
            batch_chunks: 1回のAPIで処理するチャンク数
            merge_chunks: 小さいチャンクを統合するか
            min_tokens: 統合対象の最小トークン数
            max_tokens: 統合後の最大トークン数
            analyze_coverage: カバレージ分析を実行するか
            coverage_threshold: カバレージ判定の類似度閾値
            overlap_tokens: チャンク間の重複トークン数
            use_similarity: ベクトル類似度分割を使用するか
            similarity_threshold: 類似度分割の閾値
            use_smart_generation: スマートQ/A生成を使用するか（デフォルト: True）

        Returns:
            Dict: 実行結果
                - saved_files: 保存されたファイルパス
                - qa_count: 生成されたQ/Aペア数
                - coverage_results: カバレージ分析結果
                - success: 成功フラグ
        """
        try:
            # ================================================================
            # データ読み込み + チャンク作成
            # ================================================================
            logger.info("=" * 60)
            logger.info("モード: チャンク作成 + Q/A生成")
            logger.info("=" * 60)

            # データ読み込み
            df = self.load_data()

            # ✅ 改修: チャンク作成時のmax_workersにconcurrencyを使用
            chunks = self.create_chunks(
                df,
                overlap_tokens=overlap_tokens,
                use_similarity=use_similarity,
                similarity_threshold=similarity_threshold,
                max_workers=concurrency  # ✅ 改修: celery_workers → concurrency
            )

            # ================================================================
            # Q/A生成
            # ================================================================
            qa_pairs = self.generate_qa(
                chunks,
                use_celery,
                celery_workers,
                concurrency,  # ✅ 改修: concurrency を渡す
                batch_chunks,
                merge_chunks,
                min_tokens,
                max_tokens,
                use_smart_generation
            )

            if not qa_pairs:
                logger.warning("Q/Aペアが生成されませんでした")

            # ================================================================
            # カバレージ分析
            # ================================================================
            coverage_results = {}
            if analyze_coverage and qa_pairs:
                coverage_results = self.evaluate_coverage(chunks, qa_pairs, coverage_threshold)
            else:
                coverage_results = {
                    "coverage_rate"   : 0,
                    "covered_chunks"  : 0,
                    "total_chunks"    : len(chunks),
                    "uncovered_chunks": []
                }

            # ================================================================
            # 結果保存
            # ================================================================
            saved_files = self.save(qa_pairs, coverage_results)

            # 返り値
            return {
                "saved_files"     : saved_files,
                "qa_count"        : len(qa_pairs),
                "coverage_results": coverage_results,
                "success"         : True
            }

        except Exception as e:
            logger.error(f"パイプライン実行エラー: {e}")
            raise
