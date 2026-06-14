#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
celery_tasks.py - Celeryタスク定義（修正版 v3.0）

修正内容（v3.0）:
- ★重要★ generation.py への依存を削除
- SmartQAGenerator を直接使用
- generate_qa_dataset() の呼び出しを廃止

修正内容（v2.7）:
- 並列数（concurrency）指定への対応
- check_celery_workers() の戻り値を拡張（Dict形式）
- get_total_concurrency() ヘルパー関数の追加
- get_worker_info() 詳細情報取得関数の追加
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable

# ================================================================
# 重要: プロジェクトルートをsys.pathに追加
# ================================================================
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# helper/ディレクトリもsys.pathに追加
helper_path = project_root / 'helper'
if helper_path.exists() and str(helper_path) not in sys.path:
    sys.path.insert(0, str(helper_path))

from celery_config import app

logger = logging.getLogger(__name__)

# ワーカープロセス内で SmartQAGenerator を再利用するためのキャッシュ。
# 旧実装はタスク（チャンク）ごとに生成器＝LLMクライアントを新規生成しており、
# 数千チャンクの処理で無駄な初期化コストが発生していた。prefork プールでは
# 子プロセスごとにモジュール状態が独立するため、プロセス内キャッシュで安全。
_GENERATOR_CACHE: Dict[str, object] = {}


def _get_generator(model: str):
    """モデル名ごとに SmartQAGenerator をプロセス内キャッシュから取得する"""
    generator = _GENERATOR_CACHE.get(model)
    if generator is None:
        from qa_generation.smart_qa_generator import SmartQAGenerator
        generator = SmartQAGenerator(model=model)
        _GENERATOR_CACHE[model] = generator
    return generator


# ================================================================
# Q/A生成タスク
# ================================================================

def submit_unified_qa_generation(
        chunks: List[Dict],
        config: Dict,
        model: str,
        provider: str = "openai",  # 互換性のために残すが使用しない [MIGRATION] gemini → openai
) -> List:
    """
    チャンクのQ/A生成タスクを並列実行

    Q/A生成は SmartQAGenerator（構造化出力1回）に一本化されている。

    Args:
        chunks: チャンクのリスト
        config: データセット設定
        model: 使用するモデル（例: "gpt-5-mini"）
        provider: 互換性のために残すが使用しない

    Returns:
        Celeryタスクのリスト（AsyncResultオブジェクト）
    """
    logger.info(f"Celeryタスクを投入: {len(chunks)}チャンク")
    logger.info(f"モデル: {model}")

    task_list = []
    for chunk in chunks:
        task = generate_qa_for_chunk_task.apply_async(
            args=(chunk, config, model)
        )
        task_list.append(task)

    logger.info(f"タスク投入完了: {len(task_list)}個")
    return task_list


@app.task(
    name='generate_qa_for_chunk',
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True
)
def generate_qa_for_chunk_task(
        self,
        chunk: Dict,
        config: Dict,
        model: str
) -> Dict:
    """
    単一チャンクのQ/A生成タスク（SmartQAGenerator使用版）

    Args:
        self: Celeryタスクインスタンス
        chunk: チャンク
        config: データセット設定
        model: 使用するモデル

    Returns:
        {"qa_pairs": [...], "usage": {...}} 形式の dict
    """
    chunk_id = chunk.get('id', 'unknown')
    chunk_text = chunk.get('text', '')
    dataset_type = config.get("type", "unknown")

    logger.info("=" * 60)
    logger.info("[ワーカー] タスク開始")
    logger.info("=" * 60)
    logger.info(f"  chunk_id: {chunk_id}")
    logger.info(f"  model: {model}")
    logger.info(f"  text_length: {len(chunk_text)} 文字")

    try:
        from helper.helper_llm import reset_token_counter, get_token_counter

        # 空テキストチェック
        if not chunk_text.strip():
            logger.warning(f"[ワーカー] ⚠️ 空のチャンク: {chunk_id}")
            return {"qa_pairs": [], "usage": {"input_tokens": 0, "output_tokens": 0}}

        logger.info(f"[ワーカー] Q/A生成開始: chunk={chunk_id}")

        # トークン使用量の計測。helper_llm のカウンタはモジュールレベルだが、
        # Celery の prefork プールでは子プロセスがタスクを直列実行するため
        # reset→実行→get で当該タスク分のみを切り出せる
        reset_token_counter()

        # 生成器はプロセス内キャッシュから再利用（タスクごとの再初期化を回避）
        generator = _get_generator(model)
        result = generator.process_chunk(chunk_text)
        usage = get_token_counter()

        qa_pairs = []
        if result['success'] and result['qa_pairs']:
            for qa in result['qa_pairs']:
                qa_pairs.append({
                    'question'    : qa['question'],
                    'answer'      : qa['answer'],
                    'chunk_id'    : chunk_id,
                    'topic'       : qa.get('topic', ''),
                    'dataset_type': dataset_type
                })

        logger.info(
            f"[ワーカー] ✅ タスク完了: chunk={chunk_id}, Q/A数={len(qa_pairs)}, "
            f"tokens(in/out)={usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)}"
        )
        logger.info("=" * 60)
        # 戻り値はワーカーのトークン使用量を同梱した dict 形式。
        # collect_results() が新旧両形式（dict / 旧list）を吸収する
        return {"qa_pairs": qa_pairs, "usage": usage}

    except ImportError as exc:
        logger.error("=" * 60)
        logger.error("[ワーカー] ❌ モジュールインポートエラー")
        logger.error("=" * 60)
        logger.error(f"  chunk_id: {chunk_id}")
        logger.error(f"  エラー: {exc}")
        logger.error(f"  sys.path: {sys.path[:3]}")
        logger.error("=" * 60)

        raise ImportError(
            f"qa_generation.smart_qa_generatorのインポート失敗: {exc}\n"
            f"sys.path[0]={sys.path[0]}"
        )

    except ValueError as exc:
        logger.error(f"[ワーカー] ❌ データ形式エラー: chunk={chunk_id}, error={exc}")
        raise ValueError(f"データ形式エラー: {exc}")

    except Exception as exc:
        logger.error(f"[ワーカー] ❌ タスクエラー: chunk={chunk_id}")
        logger.error("[ワーカー] エラー詳細:", exc_info=True)

        # リトライ
        if self.request.retries < self.max_retries:
            retry_count = self.request.retries + 1
            logger.warning(f"[ワーカー] リトライ {retry_count}/{self.max_retries}: chunk={chunk_id}")
            raise self.retry(exc=exc, countdown=60)
        else:
            logger.error(f"[ワーカー] ❌ 最大リトライ回数超過: chunk={chunk_id}")
            raise RuntimeError(f"最大リトライ回数超過: {exc}")


# ================================================================
# 結果収集
# ================================================================

def collect_results(
        tasks: List,
        timeout: int = 600,
        on_result: Optional[Callable[[int, List[Dict]], None]] = None,
        usage_out: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """
    Celeryタスクの結果を完了順に収集する。

    旧実装は投入順に task.get() でブロックしていたため、先頭の遅いタスクが
    完了済みの後続タスクの回収・逐次永続化を全て塞いでいた（HOLブロッキング。
    途中クラッシュ時に完了済み結果が JSONL に残らない）。本実装は ready() を
    ポーリングして完了したタスクから順に回収する。

    Args:
        tasks: AsyncResult のリスト
        timeout: 全タスクの収集が完了するまでのグローバルなタイムアウト（秒）。
            期限到達時点で未完了のタスクはタイムアウト扱いとなる
        on_result: タスク成功ごとに呼ばれるコールバック
            (タスクの投入時インデックス, Q/Aペアリスト) を受け取る。
            逐次永続化（JSONL追記）などに使用する。完了順に呼ばれる
        usage_out: 渡すとワーカー側のトークン使用量を集約して書き込む
            （input_tokens / output_tokens を加算）

    Returns:
        Q/Aペアのリスト（完了順。チャンクとの対応は各ペアの chunk_id で保持）
    """
    import time as _time

    logger.info(f"結果収集中: {len(tasks)}タスク, timeout={timeout}秒（完了順回収）")

    all_qa_pairs = []
    success_count = 0
    failed_count = 0
    timeout_count = 0
    error_details = []

    total = len(tasks)
    report_interval = max(1, total // 20)  # 5%刻みで進捗表示
    deadline = _time.monotonic() + timeout
    poll_interval = 0.5

    pending: Dict[int, object] = dict(enumerate(tasks))

    def _report(done: int) -> None:
        pct = done / total * 100 if total else 100.0
        logger.info(
            f"進捗: {done}/{total} ({pct:.0f}%) | Q/A累計: {len(all_qa_pairs)} "
            f"| 完了: {success_count} | 失敗: {failed_count}"
        )

    while pending and _time.monotonic() < deadline:
        completed_now = []
        for idx, task in pending.items():
            try:
                if not task.ready():
                    continue
            except Exception as e:
                # ブローカー接続断等。当該タスクは失敗扱い
                completed_now.append(idx)
                failed_count += 1
                error_details.append(f"タスク{idx + 1}: 状態取得エラー {str(e)[:80]}")
                continue

            completed_now.append(idx)
            try:
                # ready() 済みのため即時に取得できる
                result = task.get(timeout=10)

                # 新形式（dict: qa_pairs + usage）と旧形式（list）の両方を吸収
                if isinstance(result, dict):
                    qa_pairs = result.get("qa_pairs") or []
                    usage = result.get("usage") or {}
                    if usage_out is not None:
                        usage_out["input_tokens"] = usage_out.get("input_tokens", 0) \
                            + (usage.get("input_tokens", 0) or 0)
                        usage_out["output_tokens"] = usage_out.get("output_tokens", 0) \
                            + (usage.get("output_tokens", 0) or 0)
                else:
                    qa_pairs = result or []

                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                # qa_count=0（LLMがQ/A不要と判断）も正常終了
                success_count += 1

                # 逐次永続化フック（qa_count=0 も「処理済み」として通知する）
                if on_result is not None:
                    try:
                        on_result(idx, qa_pairs)
                    except Exception as cb_err:
                        logger.warning(f"on_result コールバックでエラー: {cb_err}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"タスク {idx + 1}/{total}: ❌ エラー: {error_msg}")
                error_details.append(f"タスク{idx + 1}: {error_msg[:100]}")
                failed_count += 1

            done = success_count + failed_count
            if done % report_interval == 0 or done == total:
                _report(done)

        for idx in completed_now:
            pending.pop(idx, None)

        if pending and not completed_now:
            _time.sleep(poll_interval)

    # 期限到達時点で未完了のタスクはタイムアウト扱い
    if pending:
        timeout_count = len(pending)
        failed_count += timeout_count
        logger.error(f"⏱️ タイムアウト: {timeout_count} タスクが期限内に完了しませんでした")
        for idx in sorted(pending):
            error_details.append(f"タスク{idx + 1}: タイムアウト")
        _report(success_count + failed_count)

    # サマリー
    logger.info("=" * 60)
    logger.info("結果収集完了")
    logger.info("=" * 60)
    logger.info(f"  成功: {success_count}/{len(tasks)}")
    logger.info(f"  失敗: {failed_count}/{len(tasks)}")
    logger.info(f"  タイムアウト: {timeout_count}/{len(tasks)}")
    logger.info(f"  Q/A総数: {len(all_qa_pairs)}")
    if usage_out is not None and (usage_out.get("input_tokens") or usage_out.get("output_tokens")):
        logger.info(
            f"  トークン（ワーカー集計）: 入力={usage_out.get('input_tokens', 0):,}, "
            f"出力={usage_out.get('output_tokens', 0):,}"
        )

    if error_details:
        logger.error("\n⚠️ エラー詳細:")
        for detail in error_details[:5]:
            logger.error(f"  - {detail}")

    logger.info("=" * 60)

    return all_qa_pairs


# ================================================================
# ワーカー状態確認（v2.7 改修）
# ================================================================

def get_worker_info() -> Dict:
    """
    Celeryワーカーの詳細情報を取得

    Returns:
        Dict: {
            'available': bool,           # ワーカーが利用可能か
            'worker_count': int,         # ワーカー数
            'total_concurrency': int,    # 総並列処理能力
            'workers': Dict[str, Dict],  # ワーカー名 → 詳細情報
            'error': Optional[str]       # エラーメッセージ（あれば）
        }

    Example:
        info = get_worker_info()
            if info['available']:
            print(f"並列処理能力: {info['total_concurrency']}")
    """
    result = {
        'available'        : False,
        'worker_count'     : 0,
        'total_concurrency': 0,
        'workers'          : {},
        'error'            : None
    }

    try:
        inspect = app.control.inspect()
        stats = inspect.stats()

        if stats is None:
            result['error'] = "Celeryワーカーが応答しません（stats=None）"
            return result

        if not stats:
            result['error'] = "アクティブなワーカーがありません"
            return result

        result['available'] = True
        result['worker_count'] = len(stats)

        total_concurrency = 0
        for worker_name, worker_stats in stats.items():
            pool = worker_stats.get('pool', {})
            concurrency = pool.get('max-concurrency', 1)
            total_concurrency += concurrency

            result['workers'][worker_name] = {
                'concurrency': concurrency,
                'pool_type'  : pool.get('implementation', 'unknown'),
                'pid'        : worker_stats.get('pid', 'N/A')
            }

        result['total_concurrency'] = total_concurrency
        return result

    except Exception as e:
        result['error'] = f"ワーカー情報取得エラー: {e}"
        logger.error(result['error'])
        return result


def get_total_concurrency() -> int:
    """
    Celeryワーカーの総並列処理能力を取得

    Returns:
        int: 総concurrency数（エラー時は0）

    Example:
        >>> concurrency = get_total_concurrency()
        >>> print(f"利用可能な並列数: {concurrency}")
    """
    info = get_worker_info()
    return info.get('total_concurrency', 0)


def check_celery_workers(
        min_workers: int = 1,
        required_concurrency: Optional[int] = None
) -> Union[bool, Dict]:
    """
    Celeryワーカーの状態確認（v2.7 改修版）

    Args:
        min_workers: 必要最小ワーカー数（デフォルト: 1）
        required_concurrency: 必要な並列数（Noneの場合はチェックしない）

    Returns:
        bool: 後方互換モード（required_concurrency=None の場合）
        Dict: 詳細モード（required_concurrency 指定時）
            {
                'ok': bool,                  # 要件を満たしているか
                'worker_count': int,         # ワーカー数
                'total_concurrency': int,    # 総並列処理能力
                'required_concurrency': int, # 要求された並列数
                'available_concurrency': int,# 実際に使用可能な並列数
                'workers': Dict,             # ワーカー詳細
                'message': str               # 状態メッセージ
            }

    Example（後方互換）:
        >>> if check_celery_workers():
        ...     print("ワーカー準備完了")

    Example（詳細モード）:
        >>> result = check_celery_workers(required_concurrency=4)
        >>> if result['ok']:
        ...     print(f"並列数 {result['available_concurrency']} で実行可能")
    """
    # ワーカー情報を取得
    info = get_worker_info()

    # ワーカーが利用不可の場合
    if not info['available']:
        error_msg = info.get('error', 'Celeryワーカーが応答しません')
        logger.error(f"❌ {error_msg}")

        if required_concurrency is not None:
            return {
                'ok'                   : False,
                'worker_count'         : 0,
                'total_concurrency'    : 0,
                'required_concurrency' : required_concurrency,
                'available_concurrency': 0,
                'workers'              : {},
                'message'              : error_msg
            }
        return False

    worker_count = info['worker_count']
    total_concurrency = info['total_concurrency']

    logger.info(f"アクティブなワーカー: {worker_count}個")
    logger.info(f"総並列処理能力: {total_concurrency}")

    # ワーカー詳細をログ出力
    for worker_name, worker_info in info['workers'].items():
        concurrency = worker_info.get('concurrency', 'N/A')
        logger.info(f"  - {worker_name}: concurrency={concurrency}")

    # ワーカー数チェック
    if worker_count < min_workers:
        error_msg = f"ワーカー数が不足: {worker_count} < {min_workers}"
        logger.error(f"❌ {error_msg}")

        if required_concurrency is not None:
            return {
                'ok'                   : False,
                'worker_count'         : worker_count,
                'total_concurrency'    : total_concurrency,
                'required_concurrency' : required_concurrency,
                'available_concurrency': 0,
                'workers'              : info['workers'],
                'message'              : error_msg
            }
        return False

    # required_concurrency が指定されていない場合（後方互換モード）
    if required_concurrency is None:
        logger.info("✅ Celeryワーカーの準備完了")
        return True

    # required_concurrency チェック（詳細モード）
    # 実際に使用する並列数を決定（要求値と利用可能値の小さい方）
    available_concurrency = min(required_concurrency, total_concurrency)

    if total_concurrency >= required_concurrency:
        message = f"要求された並列数 {required_concurrency} で実行可能"
        logger.info(f"✅ {message}")
        ok = True
    else:
        message = f"要求された並列数 {required_concurrency} に対し、利用可能は {total_concurrency}（制限して実行可能）"
        logger.warning(f"⚠️ {message}")
        ok = True  # 制限付きだが実行は可能

    return {
        'ok'                   : ok,
        'worker_count'         : worker_count,
        'total_concurrency'    : total_concurrency,
        'required_concurrency' : required_concurrency,
        'available_concurrency': available_concurrency,
        'workers'              : info['workers'],
        'message'              : message
    }


def validate_concurrency(requested: int) -> Dict:
    """
    要求された並列数が実行可能かを検証

    Args:
        requested: 要求する並列数

    Returns:
        Dict: {
            'valid': bool,          # 実行可能か
            'requested': int,        # 要求された並列数
            'available': int,        # 利用可能な並列数
            'effective': int,        # 実際に使用する並列数
            'message': str           # メッセージ
        }

    Example:
        result = validate_concurrency(8)
        if result['valid']:
        ...     effective = result['effective']
        ...     print(f"並列数 {effective} で実行します")
    """
    total = get_total_concurrency()

    if total == 0:
        return {
            'valid'    : False,
            'requested': requested,
            'available': 0,
            'effective': 0,
            'message'  : "Celeryワーカーが起動していません"
        }

    effective = min(requested, total)

    if total >= requested:
        message = f"要求通り {requested} 並列で実行可能"
        logger.info(f"✅ {message}")
    else:
        message = f"要求 {requested} → 利用可能 {total} に制限"
        logger.warning(f"⚠️ {message}")

    return {
        'valid'    : True,
        'requested': requested,
        'available': total,
        'effective': effective,
        'message'  : message
    }


# ================================================================
# ユーティリティ
# ================================================================

def get_active_tasks() -> Dict:
    """アクティブなタスクの情報を取得"""
    try:
        inspect = app.control.inspect()
        active = inspect.active()
        return active or {}
    except Exception as e:
        logger.error(f"アクティブタスク取得エラー: {e}")
        return {}


def purge_queue(queue_name: str = 'celery') -> int:
    """キューをクリア"""
    try:
        purged_count = app.control.purge()
        logger.warning(f"キュークリア: {purged_count}タスクを削除")
        return purged_count
    except Exception as e:
        logger.error(f"キュークリアエラー: {e}")
        return 0


# ================================================================
# デバッグ用（v3.0 修正版）
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Celeryタスクモジュール - 修正版 v3.0")
    print("=" * 60)
    print(f"プロジェクトルート: {project_root}")
    print(f"アプリケーション: {app.main}")
    print(f"ブローカー: {app.conf.broker_url}")
    print()

    # ✅ 修正: SmartQAGeneratorのインポートテスト
    print("SmartQAGeneratorのインポートテスト...")
    try:
        from qa_generation.smart_qa_generator import SmartQAGenerator

        print("✅ インポート成功")

        # クラスメソッド確認
        print(f"クラス: {SmartQAGenerator}")
        print("メソッド: __init__, analyze_chunk, generate_qa_pairs, process_chunk")

    except ImportError as e:
        print(f"❌ インポート失敗: {e}")
    print()

    # ワーカー確認（後方互換モード）
    print("=" * 60)
    print("ワーカー状態を確認中（後方互換モード）...")
    print("=" * 60)
    if check_celery_workers():
        print("✅ ワーカーが起動しています")
    else:
        print("❌ ワーカーが起動していません")
    print()

    # ワーカー確認（詳細モード）
    print("=" * 60)
    print("ワーカー状態を確認中（詳細モード: required_concurrency=8）...")
    print("=" * 60)
    result = check_celery_workers(required_concurrency=8)
    if isinstance(result, dict):
        print(f"  ok: {result['ok']}")
        print(f"  worker_count: {result['worker_count']}")
        print(f"  total_concurrency: {result['total_concurrency']}")
        print(f"  required_concurrency: {result['required_concurrency']}")
        print(f"  available_concurrency: {result['available_concurrency']}")
        print(f"  message: {result['message']}")
    print()

    # 並列数検証テスト
    print("=" * 60)
    print("並列数検証テスト（validate_concurrency）...")
    print("=" * 60)
    for req in [4, 8, 16]:
        result = validate_concurrency(req)
        print(f"  要求={req}: valid={result['valid']}, effective={result['effective']}, message={result['message']}")
    print()

    # 総並列数取得テスト
    print("=" * 60)
    print("総並列数取得テスト（get_total_concurrency）...")
    print("=" * 60)
    total = get_total_concurrency()
    print(f"  総並列処理能力: {total}")
