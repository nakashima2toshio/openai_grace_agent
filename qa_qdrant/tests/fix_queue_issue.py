#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_queue_issue.py - キュー問題の緊急修正スクリプト

問題: タスクが'high_priority'キューに投入されているが、
      ワーカーがデフォルトの'celery'キューを監視している

解決策: タスク投入時にキューを指定しない（デフォルトキューを使用）
"""

import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 60)
print("Celery キュー問題 - 緊急修正スクリプト")
print("=" * 60)

# ステップ1: 現在のワーカー設定を確認
print("\n[1] ワーカーのキュー設定を確認")
print("-" * 60)

try:
    from celery_config import app

    inspect = app.control.inspect()
    active_queues = inspect.active_queues()

    if active_queues:
        print("✅ ワーカーが監視しているキュー:")
        for worker_name, queues in active_queues.items():
            print(f"\n  ワーカー: {worker_name}")
            for queue in queues:
                print(f"    - {queue['name']}")
    else:
        print("⚠️ キュー情報を取得できませんでした")
        print("ワーカーはデフォルトの'celery'キューを監視している可能性があります")

except Exception as e:
    print(f"❌ エラー: {e}")

# ステップ2: 解決策の提示
print("\n[2] 解決策")
print("-" * 60)
print("""
問題: タスクが'high_priority'キューに投入されているが、
      ワーカーがそのキューを監視していない

解決策A: タスク投入時にキューを指定しない（推奨）
  - celery_tasks.pyとdiagnose_celery.pyを修正
  - apply_async()からqueue='high_priority'を削除

解決策B: ワーカーを正しいキューで起動
  - start_celery.shを修正してキューを明示的に指定
  - celery -A celery_config worker --queues=high_priority,normal_priority,low_priority
""")

# ステップ3: テスト（デフォルトキューを使用）
print("\n[3] デフォルトキューでテスト")
print("-" * 60)

try:
    from celery_tasks import generate_qa_for_chunk_task

    test_chunk = {
        'id'       : 'queue_test',
        'text'     : 'キューテスト用のチャンクです。',
        'tokens'   : 10,
        'doc_id'   : 'test',
        'chunk_idx': 0
    }

    config = {
        'type'        : 'test',
        'qa_per_chunk': 1
    }

    print("テストタスクを投入中（デフォルトキュー使用）...")

    # ✅ キューを指定しない（デフォルトの'celery'キューを使用）
    task = generate_qa_for_chunk_task.apply_async(
        args=(test_chunk, config, "gemini-2.0-flash", "gemini", True)
        # queue='high_priority' を削除
    )

    print(f"✅ タスク投入成功（デフォルトキュー）")
    print(f"  タスクID: {task.id}")
    print(f"  初期状態: {task.state}")

    # 10秒待機
    import time

    print("\n状態監視中（10秒間）:")
    for i in range(10):
        time.sleep(1)
        state = task.state

        if state == 'SUCCESS':
            result = task.get(timeout=1)
            print(f"  {i + 1}秒: {state} → ✅ 成功（{len(result)}件のQ/A）")
            print("\n🎉 デフォルトキューでは正常に動作します！")
            break
        elif state == 'FAILURE':
            print(f"  {i + 1}秒: {state} → ❌ 失敗")
            print(f"  エラー: {task.info}")
            break
        elif state == 'STARTED':
            print(f"  {i + 1}秒: {state} → ⚙️ 実行中")
        elif state == 'PENDING':
            print(f"  {i + 1}秒: {state} → ⏳ 待機中")
        else:
            print(f"  {i + 1}秒: {state}")

        if i == 9 and state == 'PENDING':
            print(f"\n⚠️ デフォルトキューでもPENDINGのまま")
            print("別の問題がある可能性があります")

except Exception as e:
    print(f"❌ テストエラー: {e}")
    import traceback

    traceback.print_exc()

# ステップ4: 修正手順
print("\n" + "=" * 60)
print("修正手順")
print("=" * 60)
print("""
✅ デフォルトキューで動作する場合:

1. celery_tasks.py を修正:

   # 修正前
   task = generate_qa_for_chunk_task.apply_async(
       args=(...),
       queue='high_priority',    # ← この2行を削除
       routing_key='high'
   )

   # 修正後
   task = generate_qa_for_chunk_task.apply_async(
       args=(...)
   )

2. diagnose_celery.py を修正:
   同様にqueue='high_priority'とrouting_key='high'を削除

3. テスト実行:
   python qa_qdrant/tests/test_celery_integration.py

---

❌ デフォルトキューでも動作しない場合:

別の問題があります。以下を確認:
1. ワーカーログ: tail -f logs/celery_qa_*.log
2. Redisの状態: redis-cli MONITOR
3. タスクの実行: ワーカーがタスクを受信しているか
""")

print("\n" + "=" * 60)
print("診断完了")
print("=" * 60)
