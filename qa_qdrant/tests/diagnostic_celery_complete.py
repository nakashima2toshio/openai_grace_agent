#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_celery_complete.py - Celery完全診断スクリプト

このスクリプトは以下を確認します：
1. Redisの状態
2. ワーカーが監視しているキュー
3. タスクが登録されているか
4. タスクが実際にキューに入っているか
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# プロジェクトルートを設定
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def run_command(cmd, description):
    """コマンドを実行して結果を表示"""
    print(f"\n{'=' * 60}")
    print(f"📋 {description}")
    print(f"{'=' * 60}")
    print(f"$ {cmd}")
    print("-" * 60)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"[stderr] {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏱️ タイムアウト")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def check_redis():
    """Redis接続確認"""
    print("\n" + "=" * 60)
    print("🔴 1. Redis接続確認")
    print("=" * 60)

    # Redis ping
    run_command("redis-cli ping", "Redis PING")

    # 全キューの状態
    run_command("redis-cli KEYS '*'", "Redis内の全キー")

    # celeryキューの長さ
    run_command("redis-cli LLEN celery", "celeryキューのタスク数")

    # high_priorityキューの長さ
    run_command("redis-cli LLEN high_priority", "high_priorityキューのタスク数")


def check_celery_workers():
    """Celeryワーカー確認"""
    print("\n" + "=" * 60)
    print("👷 2. Celeryワーカー確認")
    print("=" * 60)

    # ワーカープロセス確認
    run_command("ps aux | grep -E 'celery.*worker' | grep -v grep | head -5", "Celeryワーカープロセス")

    # Celery inspect
    try:
        from celery_config import app

        inspect = app.control.inspect()

        # アクティブなワーカー
        print("\n--- ワーカー統計 ---")
        stats = inspect.stats()
        if stats:
            for worker, info in stats.items():
                print(f"✅ ワーカー: {worker}")
                pool = info.get('pool', {})
                print(f"   - プール: {pool.get('implementation', 'N/A')}")
                print(f"   - 同時実行数: {pool.get('max-concurrency', 'N/A')}")
        else:
            print("❌ ワーカーが応答しません")

        # ★重要: ワーカーが監視しているキュー
        print("\n--- ワーカーが監視しているキュー ---")
        active_queues = inspect.active_queues()
        if active_queues:
            for worker, queues in active_queues.items():
                print(f"ワーカー: {worker}")
                for q in queues:
                    print(f"   ✅ キュー: {q.get('name', 'N/A')}")
        else:
            print("❌ キュー情報を取得できません")

        # 登録されているタスク
        print("\n--- 登録されているタスク ---")
        registered = inspect.registered()
        if registered:
            for worker, tasks in registered.items():
                print(f"ワーカー: {worker}")
                # generate_qa関連のみ表示
                for task in tasks:
                    if 'generate_qa' in task or 'celery_tasks' in task:
                        print(f"   ✅ {task}")
                # タスク総数
                print(f"   （合計 {len(tasks)} タスク）")
        else:
            print("❌ タスク情報を取得できません")

    except ImportError as e:
        print(f"❌ celery_configのインポートエラー: {e}")
    except Exception as e:
        print(f"❌ エラー: {e}")


def check_task_registration():
    """タスク登録確認"""
    print("\n" + "=" * 60)
    print("📝 3. タスク登録確認")
    print("=" * 60)

    try:
        from celery_config import app
        from celery_tasks import generate_qa_for_chunk_task

        print(f"タスク名: {generate_qa_for_chunk_task.name}")
        print(f"タスクオブジェクト: {generate_qa_for_chunk_task}")
        print(f"アプリ: {app.main}")

        # app.tasks で登録確認
        print("\n--- app.tasks内のgenerate_qa関連タスク ---")
        for name in app.tasks:
            if 'generate_qa' in name:
                print(f"   ✅ {name}")

    except Exception as e:
        print(f"❌ エラー: {e}")


def test_simple_task():
    """シンプルなタスクで動作確認"""
    print("\n" + "=" * 60)
    print("🧪 4. シンプルタスクテスト")
    print("=" * 60)

    try:
        from celery_config import app

        # デバッグ用の超シンプルなタスクを定義
        @app.task(name='test_ping_task')
        def test_ping():
            return "pong"

        print("タスクを投入中...")
        result = test_ping.apply_async()
        print(f"タスクID: {result.id}")
        print(f"初期状態: {result.state}")

        # 5秒待機
        for i in range(5):
            time.sleep(1)
            print(f"  {i + 1}秒後: {result.state}")
            if result.ready():
                print(f"✅ 結果: {result.get()}")
                return True

        print(f"❌ 5秒後もPENDING: {result.state}")
        return False

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_celery_config():
    """Celery設定確認"""
    print("\n" + "=" * 60)
    print("⚙️ 5. Celery設定確認")
    print("=" * 60)

    try:
        from celery_config import app, CeleryConfig

        print(f"broker_url: {app.conf.broker_url}")
        print(f"result_backend: {app.conf.result_backend}")
        print(f"task_default_queue: {app.conf.task_default_queue}")

        print("\n--- task_queues ---")
        for q in CeleryConfig.task_queues:
            print(f"   - {q.name} (exchange={q.exchange.name}, routing_key={q.routing_key})")

    except Exception as e:
        print(f"❌ エラー: {e}")


def check_worker_logs():
    """ワーカーログ確認"""
    print("\n" + "=" * 60)
    print("📄 6. ワーカーログ確認（最新20行）")
    print("=" * 60)

    log_patterns = [
        "logs/celery_qa_*.log",
        "celery_qa_*.log",
        "logs/celery*.log"
    ]

    import glob
    for pattern in log_patterns:
        files = glob.glob(pattern)
        if files:
            for f in files[:1]:  # 最初の1ファイルのみ
                run_command(f"tail -20 {f}", f"ログファイル: {f}")
            return

    print("⚠️ ログファイルが見つかりません")


def suggest_fix():
    """修正提案"""
    print("\n" + "=" * 60)
    print("🔧 修正提案")
    print("=" * 60)

    print("""
問題が解決しない場合、以下を順番に実行してください：

1. Celeryワーカーを完全停止:
   pkill -9 -f celery

2. Redisを完全クリア:
   redis-cli FLUSHALL

3. ワーカーを手動で起動（ログを直接見る）:
   cd /Users/nakashima_toshio/PycharmProjects/gemini_grace_agent
   PYTHONPATH=$PWD:$PWD/helper celery -A celery_config worker --loglevel=DEBUG -Q celery,high_priority,normal_priority,low_priority

4. 別のターミナルでテスト実行:
   python qa_qdrant/tests/test_celery_integration.py

5. ワーカーのログで「Received task」が表示されるか確認
""")


def main():
    print("=" * 60)
    print("🔍 Celery完全診断スクリプト")
    print("=" * 60)

    check_redis()
    check_celery_config()
    check_task_registration()
    check_celery_workers()
    test_simple_task()
    check_worker_logs()
    suggest_fix()

    print("\n" + "=" * 60)
    print("診断完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
