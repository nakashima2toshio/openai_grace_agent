#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/pipeline.py - Q/A生成パイプライン制御モジュール（リファクタリング版 v2）

改修内容 (v2):
- generation.py への依存を削除
- SmartQAGenerator を直接使用
- skip_chunking オプションを追加（既にチャンク済みCSVの場合）
- create_document_chunks() を動的インポートに変更
- concurrencyパラメータを追加（並列タスク数の指定）

使用例:
  # 既にチャンク済みCSVを使用（チャンク作成スキップ）
  pipeline = QAPipeline(
      input_file="output_chunked/cc_news_5per_chunks_20260124_005716.csv",
      skip_chunking=True　　　　# チャンク作成をスキップ
  )
  result = pipeline.run(
      use_celery=True,
      concurrency=8,
      use_smart_generation=True
  )

  # 従来通りのチャンク作成 + Q/A生成
  pipeline = QAPipeline(
      input_file="raw_data.csv",
      skip_chunking=False
  )
  result = pipeline.run(...)
"""

import sys
import logging
from typing import List, Dict, Optional, Any
import pandas as pd
from pathlib import Path

from config import DATASET_CONFIGS
from helper.helper_llm import LLMClient
from qa_generation.config import LOCAL_DATASET_EXTENSIONS
from qa_generation.structure import merge_small_chunks  # ✅ create_document_chunks を削除
from qa_generation.smart_qa_generator import SmartQAGenerator  # ✅ generation.py → smart_qa_generator.py
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
                 client: Optional[LLMClient] = None,
                 skip_chunking: bool = False):  # ✅ 新規追加
        """
        Args:
            dataset_name: データセット名 (cc_news, wikipedia_ja, etc.)
            input_file: ローカル入力ファイルパス（.txt, .csv）
            model: 使用するモデル
            output_dir: 出力ディレクトリ
            max_docs: 最大処理文書数
            client: LLMクライアント（DI用）
            skip_chunking: チャンク作成をスキップするか（既にチャンク済みCSVの場合True）
        """
        self.dataset_name = dataset_name
        self.input_file = input_file
        self.model = model
        self.output_dir = output_dir
        self.max_docs = max_docs
        self.client = client
        self.skip_chunking = skip_chunking  # ✅ 新規追加

        # 引数の排他制御
        self._validate_inputs()

        self.config = self._load_config()

        # ✅ 新規追加: SmartQAGeneratorの初期化
        self.smart_generator = SmartQAGenerator(model=model)
        logger.info(f"SmartQAGenerator初期化完了 (model={model})")

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
        """チャンクを作成する（またはスキップ）

        Args:
            df: データフレーム
            overlap_tokens: チャンク間の重複トークン数
            use_similarity: ベクトル類似度分割を使用するか
            similarity_threshold: 類似度閾値
            max_workers: 並列処理のワーカー数（チャンク作成時）

        Returns:
            チャンクのリスト
        """
        # ✅ 新規追加: チャンク作成スキップモード
        if self.skip_chunking:
            logger.info("\n[2/4] チャンク作成スキップ（既にチャンク済み）")
            return self._convert_df_to_chunks(df)

        # 従来のチャンク作成処理
        logger.info("\n[2/4] チャンク作成...")

        # ✅ 動的インポート（skip_chunking=Falseの場合のみ）
        from qa_generation.structure import create_document_chunks

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

    def _convert_df_to_chunks(self, df: pd.DataFrame) -> List[Dict]:
        """DataFrameをチャンク形式に変換（スキップモード用）
        既にチャンク済みのCSVをチャンク形式のリストに変換する。
        Args:
            df: チャンク済みデータを含むDataFrame
        Returns:
            チャンクのリスト
        """
        # テキストカラムの検出
        text_col = None
        for col in ['text', 'Combined_Text', 'content', 'chunk_text']:
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            text_col = df.columns[0]
            logger.warning(f"テキストカラムを自動検出できません。最初のカラム '{text_col}' を使用")

        # チャンクIDカラムの検出
        id_col = None
        for col in ['chunk_id', 'id', 'chunk_idx']:
            if col in df.columns:
                id_col = col
                break

        chunks = []
        dataset_type = self.config.get("type", "unknown")

        for idx, row in df.iterrows():
            chunk_id = row[id_col] if id_col else f"{dataset_type}_chunk_{idx}"
            chunk_text = str(row[text_col]).strip()

            if not chunk_text:
                continue

            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'type': row.get('type', 'pre_chunked'),
                'tokens': row.get('tokens', len(chunk_text) // 4),  # 概算
                'dataset_type': dataset_type
            })

        logger.info(f"  ✅ チャンク変換完了: {len(chunks)} チャンク")
        return chunks

    def generate_qa(self, chunks: List[Dict],
                    use_celery: bool = False,
                    celery_workers: int = 1,
                    concurrency: int = 8,
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
            concurrency: 並列タスク数（デフォルト: 8）
            batch_chunks: 1回のAPIで処理するチャンク数
            merge_chunks: 小さいチャンクを統合するか
            min_tokens: 統合対象の最小トークン数
            max_tokens: 統合後の最大トークン数
            use_smart_generation: スマートQ/A生成を使用するか（常にTrue推奨）
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
                              concurrency: int,
                              batch_size: int,
                              merge: bool, min_tokens: int, max_tokens: int,
                              use_smart_generation: bool) -> List[Dict]:
        """Celeryを使用した非同期生成

        Args:
            chunks: チャンクのリスト
            workers: ワーカープロセス数チェック用
            concurrency: 並列タスク数
            batch_size: バッチサイズ
            merge: チャンク統合を行うか
            min_tokens: 統合対象の最小トークン数
            max_tokens: 統合後の最大トークン数
            use_smart_generation: スマートQ/A生成を使用するか
        """
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

        timeout_seconds = min(max(len(tasks) * 10, 600), 1800)
        logger.info(f"結果収集タイムアウト: {timeout_seconds}秒（{len(tasks)}タスク）")
        return collect_results(tasks, timeout=timeout_seconds)

    def _generate_sync(self, chunks: List[Dict], batch_size: int,
                       merge: bool, min_tokens: int, max_tokens: int,
                       use_smart_generation: bool) -> List[Dict]:
        """同期生成（SmartQAGenerator使用）
        Args:
            chunks: チャンクのリスト
            batch_size: バッチサイズ（現在は未使用、将来の拡張用）
            merge: チャンク統合を行うか
            min_tokens: 統合対象の最小トークン数
            max_tokens: 統合後の最大トークン数
            use_smart_generation: スマートQ/A生成を使用するか（常にTrue推奨）
        Returns:
            Q/Aペアのリスト
        """
        logger.info("通常処理モード（SmartQAGenerator使用）")

        # チャンク統合（オプション）
        if merge:
            processed_chunks = merge_small_chunks(chunks, min_tokens, max_tokens)
            logger.info(f"  チャンク統合: {len(chunks)} → {len(processed_chunks)}")
        else:
            processed_chunks = chunks

        all_qa_pairs = []
        total = len(processed_chunks)

        for i, chunk in enumerate(processed_chunks, 1):
            chunk_text = chunk.get('text', '')
            chunk_id = chunk.get('id', f'chunk_{i}')

            if not chunk_text.strip():
                logger.warning(f"  [{i}/{total}] 空のチャンクをスキップ: {chunk_id}")
                continue

            logger.info(f"  [{i}/{total}] 処理中: {chunk_id}")

            try:
                # SmartQAGeneratorでQ/A生成
                result = self.smart_generator.process_chunk(chunk_text)

                if result['success'] and result['qa_pairs']:
                    for qa in result['qa_pairs']:
                        all_qa_pairs.append({
                            'question': qa['question'],
                            'answer': qa['answer'],
                            'chunk_id': chunk_id,
                            'topic': qa.get('topic', ''),
                            'dataset_type': chunk.get('dataset_type', 'unknown')
                        })
                    logger.info(f"    → {len(result['qa_pairs'])} Q/A生成")
                else:
                    logger.warning(f"    → Q/A生成なし（qa_count=0 または失敗）")

            except Exception as e:
                logger.error(f"    → エラー: {e}")
                continue

        logger.info(f"  ✅ 同期生成完了: {len(all_qa_pairs)} Q/Aペア")
        return all_qa_pairs

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
            celery_workers: int = 1,
            concurrency: int = 8,
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
            concurrency: 並列タスク数（デフォルト: 8）
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
            if self.skip_chunking:
                logger.info("モード: Q/A生成のみ（チャンク済みデータ使用）")
            else:
                logger.info("モード: チャンク作成 + Q/A生成")
            logger.info("=" * 60)

            # データ読み込み
            df = self.load_data()

            # チャンク作成（またはスキップ）
            chunks = self.create_chunks(
                df,
                overlap_tokens=overlap_tokens,
                use_similarity=use_similarity,
                similarity_threshold=similarity_threshold,
                max_workers=concurrency
            )

            # ================================================================
            # Q/A生成
            # ================================================================
            qa_pairs = self.generate_qa(
                chunks,
                use_celery,
                celery_workers,
                concurrency,
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

