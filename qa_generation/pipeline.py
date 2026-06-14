#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/pipeline.py - Q/A生成パイプライン制御モジュール（v3.0 - チャンク処理削除版）

改修内容 (v3.0):
- ★重要★ チャンク分割処理を完全に削除（前段のchunkingで完了済み）
- create_chunks() メソッドを削除
- _convert_df_to_chunks() メソッドを削除
- skip_chunking パラメータを削除
- merge_chunks, min_tokens, max_tokens パラメータを削除
- overlap_tokens, use_similarity, similarity_threshold パラメータを削除
- structure.py への依存を削除
- SmartQAGenerator を直接使用

前提条件:
- 入力CSVは既にチャンク済み（csv_text_to_chunks_text_csv.py で処理済み）
- チャンクCSVには 'text' または 'Combined_Text' カラムが必要

使用例:
  # チャンク済みCSVからQ/A生成
  pipeline = QAPipeline(
      input_file="output_chunked/data_chunks.csv",
      model="gpt-5-mini",  # [MIGRATION] "claude-sonnet-4-6" → "gpt-5-mini"
      output_dir="qa_output/pipeline"
  )
  result = pipeline.run(
      use_celery=True,
      concurrency=8
  )
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from celery_tasks import check_celery_workers, collect_results, submit_unified_qa_generation
from config import DATASET_CONFIGS
from helper.helper_llm import LLMClient
from qa_generation.evaluation import analyze_coverage
from qa_generation.smart_qa_generator import SmartQAGenerator

# トークン集計（同期実行時のみ。Celery経路はワーカープロセス側で消費されるため対象外）
try:
    from helper.helper_llm import (
        LLM_PRICING as _LLM_PRICING,
    )
    from helper.helper_llm import (
        get_token_counter as _get_token_counter,
    )
    from helper.helper_llm import (
        reset_token_counter as _reset_token_counter,
    )
    _TOKEN_TRACKING_AVAILABLE = True
except ImportError:
    _TOKEN_TRACKING_AVAILABLE = False

logger = logging.getLogger(__name__)


class QAPipeline:
    """Q/A生成パイプライン（チャンク済みCSV専用）"""

    def __init__(self,
                 dataset_name: Optional[str] = None,
                 input_file: Optional[str] = None,
                 model: str = "gpt-5-mini",  # [MIGRATION] "claude-sonnet-4-6" → "gpt-5-mini"
                 output_dir: str = "qa_output/pipeline",
                 max_docs: Optional[int] = None,
                 client: Optional[LLMClient] = None):
        """
        Args:
            dataset_name: データセット名 (cc_news, wikipedia_ja, etc.)
            input_file: ローカル入力ファイルパス（チャンク済み.csv）
            model: 使用するモデル
            output_dir: 出力ディレクトリ
            max_docs: 最大処理チャンク数
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

        # SmartQAGeneratorの初期化
        self.smart_generator = SmartQAGenerator(model=model)
        logger.info(f"SmartQAGenerator初期化完了 (model={model})")

        # Celeryワーカー側トークン使用量の集計先（collect_results が加算する）
        self._celery_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

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
                "text_column" : "text",  # チャンク済みCSVのデフォルトカラム
                "title_column": None,
                "lang"        : lang,
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
        from qa_generation.data_io import load_preprocessed_data, load_uploaded_file

        logger.info("\n[1/3] データ読み込み...")

        if self.input_file:
            file_path = Path(self.input_file)

            # CSVファイルのみ対応（チャンク済み前提）
            if file_path.suffix == '.csv':
                logger.info(f"  📊 チャンク済みCSVファイル: {self.input_file}")
                df = load_uploaded_file(self.input_file)
                logger.info(f"  ✅ 読み込み完了: {len(df)} 行")

            else:
                raise ValueError(
                    f"未対応のファイル形式: {file_path.suffix}\n"
                    f"チャンク済みCSVファイル（.csv）を指定してください。\n"
                    f"テキストファイルの場合は、先に csv_text_to_chunks_text_csv.py でチャンク化してください。"
                )

            # 最大チャンク数制限
            if self.max_docs and len(df) > self.max_docs:
                df = df.head(self.max_docs)
                logger.info(f"  📊 最大チャンク数制限: {len(df)} 件に制限")

            return df
        else:
            return load_preprocessed_data(self.dataset_name)

    def _load_chunks_from_csv(self, df: pd.DataFrame) -> List[Dict]:
        """チャンク済みCSVをチャンクリストに変換

        Args:
            df: チャンク済みデータを含むDataFrame

        Returns:
            チャンクのリスト
        """
        logger.info("\n[2/3] チャンクデータ変換...")

        # テキストカラムの検出
        text_col = None
        for col in ['text', 'Combined_Text', 'content', 'chunk_text']:
            if col in df.columns:
                text_col = col
                break

        if text_col is None:
            available_cols = list(df.columns)
            raise ValueError(
                f"テキストカラムが見つかりません。\n"
                f"利用可能なカラム: {available_cols}\n"
                f"必要なカラム: 'text', 'Combined_Text', 'content', 'chunk_text' のいずれか"
            )

        logger.info(f"  テキストカラム: {text_col}")

        # チャンクIDカラムの検出
        id_col = None
        for col in ['chunk_id', 'id', 'chunk_idx']:
            if col in df.columns:
                id_col = col
                break

        if id_col:
            logger.info(f"  IDカラム: {id_col}")

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

    # ================================================================
    # 逐次永続化（クラッシュ時の全損防止・再開用）
    # ================================================================

    def _progress_path(self) -> Path:
        """チャンク単位の処理結果を逐次追記する JSONL のパス"""
        dataset_type = self.config.get("type", "unknown")
        return Path(self.output_dir) / f"qa_progress_{dataset_type}.jsonl"

    def _load_progress(self) -> Dict[str, List[Dict]]:
        """逐次保存ファイルから処理済みチャンクの結果を読み込む"""
        path = self._progress_path()
        if not path.exists():
            return {}

        progress: Dict[str, List[Dict]] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        progress[str(record["chunk_id"])] = record.get("qa_pairs", [])
                    except (json.JSONDecodeError, KeyError):
                        # 途中クラッシュで壊れた行はスキップ（その行のチャンクは再処理される）
                        continue
        except OSError as e:
            logger.warning(f"逐次保存ファイルの読み込みに失敗: {e}")
            return {}

        return progress

    def _append_progress(self, chunk_id: Any, qa_pairs: List[Dict]) -> None:
        """チャンク1件分の結果を JSONL に追記する（qa_count=0 も記録して再処理を防ぐ）"""
        path = self._progress_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"chunk_id": str(chunk_id), "qa_pairs": qa_pairs}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _clear_progress(self) -> None:
        """最終保存の完了後に逐次保存ファイルを削除する"""
        path = self._progress_path()
        try:
            if path.exists():
                path.unlink()
                logger.info(f"  逐次保存ファイルを削除: {path}")
        except OSError as e:
            logger.warning(f"逐次保存ファイルの削除に失敗: {e}")

    def generate_qa(self, chunks: List[Dict],
                    use_celery: bool = False,
                    celery_workers: int = 1,
                    concurrency: int = 8,
                    batch_chunks: int = 3) -> List[Dict]:
        """Q/Aペアを生成する

        チャンクごとの結果は qa_progress_*.jsonl に逐次追記され、
        プロセスが途中で落ちても再実行時に処理済みチャンクをスキップして再開できる。

        Q/A生成は SmartQAGenerator（構造化出力1回）に一本化されている。

        Args:
            chunks: チャンクのリスト
            use_celery: Celery並列処理を使用するか
            celery_workers: Celeryワーカープロセス数チェック用（デフォルト: 1）
            concurrency: 並列タスク数（デフォルト: 8）
            batch_chunks: (非推奨・未使用) 旧バッチ処理用。1チャンク=1タスクで処理される
        """
        logger.info("\n[3/3] Q/Aペア生成...")
        logger.info("  生成モード: SmartQAGenerator（構造化出力1回/チャンク）")
        logger.info(f"  処理チャンク数: {len(chunks)}")

        # 逐次保存ファイルからの再開
        progress = self._load_progress()
        prior_pairs: List[Dict] = []
        if progress:
            done_ids = set(progress.keys())
            pending = [c for c in chunks if str(c.get('id')) not in done_ids]
            skipped = len(chunks) - len(pending)
            if skipped:
                prior_pairs = [qa for pairs in progress.values() for qa in pairs]
                logger.info(
                    f"  📂 逐次保存ファイルから再開: 処理済み {skipped} チャンクをスキップ "
                    f"（復元Q/A: {len(prior_pairs)}件, 残り: {len(pending)}チャンク）"
                )
            chunks = pending

        if not chunks:
            logger.info("  全チャンク処理済み（逐次保存ファイルから復元）")
            return prior_pairs

        if use_celery:
            new_pairs = self._generate_with_celery(
                chunks, celery_workers, concurrency, batch_chunks
            )
        else:
            new_pairs = self._generate_sync(chunks, batch_chunks)

        return prior_pairs + new_pairs

    def _generate_with_celery(self, chunks: List[Dict],
                              workers: int,
                              concurrency: int,
                              batch_size: int) -> List[Dict]:
        """Celeryを使用した非同期生成

        Args:
            chunks: チャンクのリスト
            workers: ワーカープロセス数チェック用
            concurrency: 並列タスク数
            batch_size: バッチサイズ
        """
        logger.info("  Celery並列処理モード:")
        logger.info(f"    - ワーカープロセス数チェック: {workers}")
        logger.info(f"    - 並列タスク数 (concurrency): {concurrency}")

        logger.info("  Celeryワーカーの状態を確認中...")
        if not check_celery_workers(workers):
            raise RuntimeError("Celery workers are not running")

        # モデル名からプロバイダーを自動判定
        if self.model.startswith("claude"):
            _provider = "anthropic"
        elif self.model.startswith("gemini"):
            _provider = "gemini"
        else:
            _provider = "openai"  # [MIGRATION] gpt-* は openai
        tasks = submit_unified_qa_generation(
            chunks, self.config, self.model, provider=_provider
        )

        # 逐次永続化: タスク完了ごとにチャンク結果を JSONL へ追記
        def _persist(task_index: int, qa_pairs: List[Dict]) -> None:
            chunk_id = chunks[task_index].get('id', f'chunk_{task_index}')
            self._append_progress(chunk_id, qa_pairs)

        timeout_seconds = min(max(len(tasks) * 10, 600), 1800)
        logger.info(f"  結果収集タイムアウト: {timeout_seconds}秒（{len(tasks)}タスク）")
        # ワーカー側のトークン使用量を集約（run() のコストサマリーで使用）
        return collect_results(
            tasks, timeout=timeout_seconds, on_result=_persist,
            usage_out=self._celery_usage,
        )

    def _generate_sync(self, chunks: List[Dict], batch_size: int) -> List[Dict]:
        """同期生成（SmartQAGenerator使用）

        Args:
            chunks: チャンクのリスト
            batch_size: バッチサイズ（現在は未使用、将来の拡張用）

        Returns:
            Q/Aペアのリスト
        """
        logger.info("  通常処理モード（SmartQAGenerator使用）")

        all_qa_pairs = []
        total = len(chunks)

        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get('text', '')
            chunk_id = chunk.get('id', f'chunk_{i}')

            if not chunk_text.strip():
                logger.warning(f"    [{i}/{total}] 空のチャンクをスキップ: {chunk_id}")
                continue

            logger.info(f"    [{i}/{total}] 処理中: {chunk_id}")

            try:
                # SmartQAGeneratorでQ/A生成
                result = self.smart_generator.process_chunk(chunk_text)

                chunk_pairs = []
                if result['success'] and result['qa_pairs']:
                    for qa in result['qa_pairs']:
                        chunk_pairs.append({
                            'question': qa['question'],
                            'answer': qa['answer'],
                            'chunk_id': chunk_id,
                            'topic': qa.get('topic', ''),
                            'dataset_type': chunk.get('dataset_type', 'unknown')
                        })
                    all_qa_pairs.extend(chunk_pairs)
                    logger.info(f"      → {len(chunk_pairs)} Q/A生成")
                else:
                    logger.warning("      → Q/A生成なし（qa_count=0 または失敗）")

                # 逐次永続化（qa_count=0 も記録して再処理を防ぐ）
                if result['success']:
                    self._append_progress(chunk_id, chunk_pairs)

            except Exception as e:
                logger.error(f"      → エラー: {e}")
                continue

        logger.info(f"  ✅ 同期生成完了: {len(all_qa_pairs)} Q/Aペア")
        return all_qa_pairs

    def evaluate_coverage(self, chunks: List[Dict], qa_pairs: List[Dict],
                          threshold: Optional[float] = None) -> Dict:
        """カバレッジを評価する"""
        logger.info("\n[追加] カバレージ分析...")
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
            analyze_coverage: bool = True,
            coverage_threshold: Optional[float] = None):
        """
        パイプライン実行

        Args:
            use_celery: Celery並列処理を使用するか
            celery_workers: Celeryワーカープロセス数チェック用（デフォルト: 1）
            concurrency: 並列タスク数（デフォルト: 8）
            batch_chunks: (非推奨・未使用) 旧バッチ処理用。1チャンク=1タスクで処理される
            analyze_coverage: カバレージ分析を実行するか
            coverage_threshold: カバレージ判定の類似度閾値

        Returns:
            Dict: 実行結果
                - saved_files: 保存されたファイルパス
                - qa_count: 生成されたQ/Aペア数
                - coverage_results: カバレージ分析結果
                - success: 成功フラグ
        """
        try:
            # ================================================================
            # パイプライン開始
            # ================================================================
            logger.info("=" * 60)
            logger.info("Q/A生成パイプライン（チャンク済みCSV専用）")
            logger.info("=" * 60)

            if _TOKEN_TRACKING_AVAILABLE and not use_celery:
                _reset_token_counter()

            # ================================================================
            # データ読み込み
            # ================================================================
            df = self.load_data()

            # ================================================================
            # チャンクリストに変換
            # ================================================================
            chunks = self._load_chunks_from_csv(df)

            if not chunks:
                raise RuntimeError("有効なチャンクがありません")

            # ================================================================
            # Q/A生成
            # ================================================================
            qa_pairs = self.generate_qa(
                chunks,
                use_celery,
                celery_workers,
                concurrency,
                batch_chunks
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

            # 最終保存が完了したため逐次保存ファイル（再開用）を削除
            self._clear_progress()

            # ================================================================
            # 完了サマリー
            # ================================================================
            logger.info("")
            logger.info("=" * 60)
            logger.info("✅ パイプライン完了")
            logger.info("=" * 60)
            logger.info(f"  入力チャンク数: {len(chunks)}")
            logger.info(f"  生成Q/A数: {len(qa_pairs)}")
            if analyze_coverage:
                logger.info(f"  カバレージ率: {coverage_results.get('coverage_rate', 0):.1%}")

            # トークン/コストサマリー
            # - 同期実行: helper_llm のプロセス内カウンタから取得
            # - Celery実行: 各ワーカータスクが返す usage を collect_results が
            #   self._celery_usage に集約したものを使用（旧実装はCelery経路で
            #   コストが常に不明だった）
            in_tok = out_tok = 0
            source_label = ""
            if use_celery:
                in_tok = self._celery_usage.get("input_tokens", 0)
                out_tok = self._celery_usage.get("output_tokens", 0)
                source_label = "（ワーカー集計）"
            elif _TOKEN_TRACKING_AVAILABLE:
                counter = _get_token_counter()
                in_tok = counter.get("input_tokens", 0)
                out_tok = counter.get("output_tokens", 0)

            if (in_tok or out_tok) and _TOKEN_TRACKING_AVAILABLE:
                pricing = _LLM_PRICING.get(self.model, {"input": 0.0, "output": 0.0})
                cost = in_tok * pricing["input"] / 1000 + out_tok * pricing["output"] / 1000
                logger.info(f"  トークン{source_label}: 入力={in_tok:,}, 出力={out_tok:,}")
                logger.info(f"  概算コスト: ${cost:.4f} (model={self.model})")
            logger.info("=" * 60)

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
