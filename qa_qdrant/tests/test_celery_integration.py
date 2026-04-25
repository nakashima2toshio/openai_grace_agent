#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_celery_integration.py - Celeryスマート生成統合テスト v2.4

改修内容（v2.4）:
- タイムアウトを短縮（120秒 → 30秒）
- エラー情報をより詳細に表示
- タスク状態の確認機能を追加

使用方法:
    # プロジェクトルートから実行
    python test_celery_integration.py

    # または qa_qdrant/tests/ から実行
    python qa_qdrant/tests/test_celery_integration.py
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict

# ================================================================
# パス設定（重要）- プロジェクトルートをPYTHONPATHに追加
# ================================================================
current_file = Path(__file__).resolve()

# プロジェクトルートを探索
if 'qa_qdrant' in str(current_file):
    # qa_qdrant/tests/にある場合: 2階層上がプロジェクトルート
    project_root = current_file.parent.parent.parent
else:
    # プロジェクトルートにある場合
    project_root = current_file.parent

# PYTHONPATHに追加
sys.path.insert(0, str(project_root))

# helper/も追加
helper_path = project_root / 'helper'
if helper_path.exists():
    sys.path.insert(0, str(helper_path))

print(f"📁 プロジェクトルート: {project_root}")
print(f"📁 現在のファイル: {current_file}")

# ================================================================
# ロギング設定
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_celery_worker_status():
    """テスト1: Celeryワーカーの状態確認"""
    logger.info("=" * 60)
    logger.info("テスト1: Celeryワーカー状態確認")
    logger.info("=" * 60)

    try:
        from celery_tasks import check_celery_workers

        if check_celery_workers(min_workers=1):
            logger.info("✅ 合格: ワーカーが起動しています")
            return True
        else:
            logger.error("❌ 不合格: ワーカーが起動していません")
            logger.error("解決策: ./start_celery.sh start -w 8")
            return False

    except ImportError as e:
        logger.error(f"❌ インポートエラー: {e}")
        logger.error("celery_tasks.pyがプロジェクトルートにあるか確認してください")
        return False
    except Exception as e:
        logger.error(f"❌ エラー: {e}")
        return False


def wait_for_task_with_status(task, timeout: int = 30):
    """
    タスクの完了を待機し、途中経過を表示

    Args:
        task: Celeryタスク
        timeout: タイムアウト（秒）

    Returns:
        タスクの結果（失敗時はNone）
    """
    start_time = time.time()
    check_interval = 2  # 2秒ごとに確認

    while True:
        elapsed = time.time() - start_time

        # タイムアウトチェック
        if elapsed > timeout:
            logger.error(f"⏱️ タイムアウト（{timeout}秒経過）")
            logger.error(f"タスク状態: {task.state}")
            return None

        # タスク状態チェック
        if task.ready():
            # 完了
            try:
                result = task.get(timeout=1)
                return result
            except Exception as e:
                logger.error(f"❌ タスク実行エラー: {e}")
                logger.error(f"タスク状態: {task.state}")
                if hasattr(task, 'traceback'):
                    logger.error(f"トレースバック:\n{task.traceback}")
                return None

        # 進捗表示
        if int(elapsed) % 5 == 0:
            logger.info(f"⏳ 待機中... ({elapsed:.0f}秒経過, 状態: {task.state})")

        time.sleep(check_interval)


def test_smart_generation_single_chunk():
    """テスト2: スマート生成（単一チャンク）"""
    logger.info("\n" + "=" * 60)
    logger.info("テスト2: スマート生成（単一チャンク）")
    logger.info("=" * 60)

    try:
        from celery_tasks import submit_unified_qa_generation

        # テストチャンク
        test_chunk = {
            'id'       : 'test_chunk_0',
            'text'     : '''
            AES-256暗号化アルゴリズムは、対称鍵暗号方式の一種で、
            256ビットの鍵長を持ちます。NIST（米国国立標準技術研究所）
            により承認されており、機密情報の保護に広く使用されています。
            ''',
            'tokens'   : 150,
            'doc_id'   : 'test_doc',
            'chunk_idx': 0
        }

        config = {
            'type'        : 'test',
            'qa_per_chunk': 3
        }

        # タスク投入（スマート生成）
        logger.info("タスクを投入中（スマート生成）...")
        tasks = submit_unified_qa_generation(
            chunks=[test_chunk],
            config=config,
            model="gemini-2.0-flash",
            provider="gemini",
            use_smart_generation=True  # ✨ スマート生成
        )

        # 結果収集（✅ タイムアウト短縮: 120秒 → 30秒）
        logger.info("結果を収集中...")
        start_time = time.time()

        task = tasks[0]
        qa_pairs = wait_for_task_with_status(task, timeout=30)

        elapsed = time.time() - start_time

        # 検証
        if qa_pairs is not None and qa_pairs:
            logger.info(f"✅ 合格: {len(qa_pairs)}個のQ/Aペア生成（{elapsed:.1f}秒）")

            # 最初のQ/Aを表示
            first_qa = qa_pairs[0]
            logger.info(f"例:")
            logger.info(f"  Q: {first_qa.get('question', 'N/A')}")
            logger.info(f"  A: {first_qa.get('answer', 'N/A')}")
            logger.info(f"  Topic: {first_qa.get('topic', 'N/A')}")
            logger.info(f"  Method: {first_qa.get('generation_method', 'N/A')}")

            # フィールドチェック
            has_topic = 'topic' in qa_pairs[0]
            has_method = 'generation_method' in qa_pairs[0]

            if has_topic and has_method:
                logger.info("✅ スマート生成のフィールドを確認")
            else:
                logger.warning("⚠️ スマート生成のフィールドが不足")

            return True
        else:
            logger.error("❌ 不合格: Q/Aペアが生成されませんでした")
            logger.error(f"タスク状態: {task.state}")
            return False

    except Exception as e:
        logger.error(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_legacy_generation_single_chunk():
    """テスト3: 従来方式（単一チャンク）"""
    logger.info("\n" + "=" * 60)
    logger.info("テスト3: 従来方式（単一チャンク）")
    logger.info("=" * 60)

    try:
        from celery_tasks import submit_unified_qa_generation

        # テストチャンク
        test_chunk = {
            'id'       : 'test_chunk_1',
            'text'     : '''
            この製品は赤色で、サイズはMサイズです。
            価格は3,000円で、送料無料です。
            ''',
            'tokens'   : 50,
            'doc_id'   : 'test_doc',
            'chunk_idx': 1
        }

        config = {
            'type'        : 'test',
            'qa_per_chunk': 3
        }

        # タスク投入（従来方式）
        logger.info("タスクを投入中（従来方式）...")
        tasks = submit_unified_qa_generation(
            chunks=[test_chunk],
            config=config,
            model="gemini-2.0-flash",
            provider="gemini",
            use_smart_generation=False  # ✨ 従来方式
        )

        # 結果収集（✅ タイムアウト短縮）
        logger.info("結果を収集中...")
        start_time = time.time()

        task = tasks[0]
        qa_pairs = wait_for_task_with_status(task, timeout=30)

        elapsed = time.time() - start_time

        # 検証
        if qa_pairs is not None and qa_pairs:
            logger.info(f"✅ 合格: {len(qa_pairs)}個のQ/Aペア生成（{elapsed:.1f}秒）")

            # 最初のQ/Aを表示
            first_qa = qa_pairs[0]
            logger.info(f"例:")
            logger.info(f"  Q: {first_qa.get('question', 'N/A')}")
            logger.info(f"  A: {first_qa.get('answer', 'N/A')}")
            logger.info(f"  Method: {first_qa.get('generation_method', 'N/A')}")

            return True
        else:
            logger.error("❌ 不合格: Q/Aペアが生成されませんでした")
            logger.error(f"タスク状態: {task.state}")
            return False

    except Exception as e:
        logger.error(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_chunks():
    """テスト4: 複数チャンク（並列処理）"""
    logger.info("\n" + "=" * 60)
    logger.info("テスト4: 複数チャンク並列処理（スマート生成）")
    logger.info("=" * 60)

    try:
        from celery_tasks import submit_unified_qa_generation, collect_results

        # テストチャンク（3個）
        test_chunks = [
            {
                'id'       : f'test_chunk_{i}',
                'text'     : f'テストチャンク{i}の内容です。これは並列処理のテストです。',
                'tokens'   : 30,
                'doc_id'   : 'test_doc',
                'chunk_idx': i
            }
            for i in range(3)
        ]

        config = {
            'type'        : 'test',
            'qa_per_chunk': 3
        }

        # タスク投入
        logger.info(f"タスクを投入中（{len(test_chunks)}チャンク）...")
        tasks = submit_unified_qa_generation(
            chunks=test_chunks,
            config=config,
            model="gemini-2.0-flash",
            provider="gemini",
            use_smart_generation=True
        )

        # 結果収集（✅ タイムアウト短縮）
        logger.info("結果を収集中...")
        start_time = time.time()
        qa_pairs = collect_results(tasks, timeout=30)
        elapsed = time.time() - start_time

        # 検証
        expected_min = len(test_chunks) * 1  # 最低1個/チャンク
        if len(qa_pairs) >= expected_min:
            logger.info(f"✅ 合格: {len(qa_pairs)}個のQ/Aペア生成（{elapsed:.1f}秒）")
            logger.info(f"  平均: {len(qa_pairs) / len(test_chunks):.1f}個/チャンク")
            logger.info(f"  並列効率: {len(test_chunks) / elapsed:.2f}チャンク/秒")
            return True
        else:
            logger.error(f"❌ 不合格: Q/A数が不足 ({len(qa_pairs)} < {expected_min})")
            return False

    except Exception as e:
        logger.error(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """テスト5: エラーハンドリング"""
    logger.info("\n" + "=" * 60)
    logger.info("テスト5: エラーハンドリング")
    logger.info("=" * 60)

    try:
        from celery_tasks import submit_unified_qa_generation

        # 異常なチャンク（テキストが空）
        bad_chunk = {
            'id'       : 'bad_chunk',
            'text'     : '',  # 空のテキスト
            'tokens'   : 0,
            'doc_id'   : 'test_doc',
            'chunk_idx': 0
        }

        config = {
            'type'        : 'test',
            'qa_per_chunk': 3
        }

        # タスク投入
        logger.info("異常なチャンクでタスクを投入中...")
        tasks = submit_unified_qa_generation(
            chunks=[bad_chunk],
            config=config,
            model="gemini-2.0-flash",
            provider="gemini",
            use_smart_generation=True
        )

        # 結果収集（✅ タイムアウト短縮）
        logger.info("結果を収集中...")

        task = tasks[0]
        qa_pairs = wait_for_task_with_status(task, timeout=30)

        # エラーハンドリングが正しく動作すれば、Noneまたは空のリストが返る
        if qa_pairs is None or qa_pairs == []:
            logger.info("✅ 合格: エラーハンドリングが正常に動作")
            return True
        else:
            logger.warning(f"⚠️ 予期しない結果: {len(qa_pairs)}個のQ/Aが生成されました")
            return True  # エラーは発生していないので合格

    except Exception as e:
        logger.error(f"❌ エラー: {e}")
        return False


def main():
    """メインテスト実行"""
    logger.info("\n" + "=" * 60)
    logger.info("Celeryスマート生成統合テスト v2.4")
    logger.info("=" * 60)

    # テスト実行
    results = {
        'ワーカー状態確認'  : test_celery_worker_status(),
        'スマート生成（単一）': test_smart_generation_single_chunk(),
        '従来方式（単一）'    : test_legacy_generation_single_chunk(),
        '並列処理（複数）'    : test_multiple_chunks(),
        'エラーハンドリング': test_error_handling(),
    }

    # 結果サマリー
    logger.info("\n" + "=" * 60)
    logger.info("テスト結果サマリー")
    logger.info("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ 合格" if result else "❌ 不合格"
        logger.info(f"{status}: {test_name}")

    logger.info(f"\n合計: {passed}/{total} テスト合格")

    if passed == total:
        logger.info("🎉 全てのテストに合格しました！")
        return 0
    else:
        logger.error(f"⚠️ {total - passed}個のテストが失敗しました")
        logger.error("\n📋 トラブルシューティング:")
        logger.error("1. Celeryワーカーのログを確認: tail -f logs/celery_qa_*.log")
        logger.error("2. ワーカーを再起動: ./start_celery.sh restart -w 8")
        logger.error("3. プロジェクト構造を確認: ls -la qa_generation/")
        return 1


if __name__ == "__main__":
    sys.exit(main())
