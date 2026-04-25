#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_celery.py - Celery診断スクリプト（修正版）

プロジェクトルートを正しく検出してCeleryの状態を診断します。
"""

import sys
import os
from pathlib import Path

# ================================================================
# プロジェクトルートを正しく検出
# ================================================================
current_file = Path(__file__).resolve()

# qa_qdrant/tests/から実行された場合
if 'qa_qdrant' in str(current_file) and 'tests' in str(current_file):
    # 2階層上がプロジェクトルート
    project_root = current_file.parent.parent.parent
elif 'qa_qdrant' in str(current_file):
    # 1階層上がプロジェクトルート
    project_root = current_file.parent.parent
else:
    # プロジェクトルート直下から実行された場合
    project_root = current_file.parent

# sys.pathに追加
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# helper/も追加
helper_path = project_root / 'helper'
if helper_path.exists() and str(helper_path) not in sys.path:
    sys.path.insert(0, str(helper_path))

print("=" * 60)
print("Celery診断スクリプト（修正版）")
print("=" * 60)

# ステップ1: 環境確認
print("\n[1] 環境確認")
print("-" * 60)
print(f"Python: {sys.version}")
print(f"実行ファイル: {current_file}")
print(f"プロジェクトルート: {project_root}")
print(f"sys.path[0]: {sys.path[0]}")
print(f"sys.path[1]: {sys.path[1] if len(sys.path) > 1 else 'N/A'}")

# プロジェクト構造の確認
print(f"\nプロジェクト構造:")
for item in sorted(project_root.iterdir())[:15]:
    if item.is_dir() and not item.name.startswith('.'):
        print(f"  📁 {item.name}/")
    elif item.is_file() and item.suffix == '.py':
        print(f"  📄 {item.name}")

# ステップ2: Redisの確認
print("\n[2] Redis接続確認")
print("-" * 60)
try:
    import redis

    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("✅ Redisに接続成功")

    # Redis情報
    info = r.info()
    print(f"  Redis version: {info['redis_version']}")
    print(f"  Connected clients: {info['connected_clients']}")

    # キューの状態
    try:
        celery_queue_length = r.llen('celery')
        print(f"  Celeryキューの長さ: {celery_queue_length}")

        # キューの内容をサンプル表示
        if celery_queue_length > 0:
            print(f"  ⚠️ 処理待ちタスク: {celery_queue_length}個")
    except:
        pass

except Exception as e:
    print(f"❌ Redis接続エラー: {e}")
    print("解決策: redis-server を起動してください")

# ステップ3: Celeryモジュールの確認
print("\n[3] Celeryモジュール確認")
print("-" * 60)

# celery_config.pyの存在確認
celery_config_path = project_root / 'celery_config.py'
print(f"celery_config.py: {celery_config_path.exists()}")

# celery_tasks.pyの存在確認
celery_tasks_path = project_root / 'celery_tasks.py'
print(f"celery_tasks.py: {celery_tasks_path.exists()}")

try:
    from celery_config import app, CeleryConfig

    print("✅ celery_config.pyインポート成功")
    print(f"  Broker URL: {app.conf.broker_url}")
    print(f"  Result Backend: {app.conf.result_backend}")
    print(f"  App name: {app.main}")
except ImportError as e:
    print(f"❌ celery_config.pyインポート失敗: {e}")
    print(f"  celery_config.pyの場所: {celery_config_path}")

try:
    from celery_tasks import generate_qa_for_chunk_task

    print("✅ celery_tasks.pyインポート成功")
    print(f"  タスク名: {generate_qa_for_chunk_task.name}")
except ImportError as e:
    print(f"❌ celery_tasks.pyインポート失敗: {e}")
    print(f"  celery_tasks.pyの場所: {celery_tasks_path}")

# ステップ4: ワーカーの状態確認
print("\n[4] ワーカー状態確認")
print("-" * 60)
try:
    from celery_config import app

    inspect = app.control.inspect()

    # アクティブなワーカー
    stats = inspect.stats()
    if stats:
        print(f"✅ アクティブなワーカー: {len(stats)}個")
        for worker_name, worker_stats in stats.items():
            print(f"\n  ワーカー: {worker_name}")
            print(f"    PID: {worker_stats.get('pid', 'N/A')}")
            pool_info = worker_stats.get('pool', {})
            print(f"    Concurrency: {pool_info.get('max-concurrency', 'N/A')}")
            print(f"    Pool: {pool_info.get('implementation', 'N/A')}")
    else:
        print("❌ アクティブなワーカーなし")
        print("解決策: ./start_celery.sh start -w 8")

    # 登録されているタスク
    registered = inspect.registered()
    if registered:
        print(f"\n✅ 登録されているタスク:")
        for worker_name, tasks in registered.items():
            print(f"\n  ワーカー: {worker_name}")

            # generate_qa関連のタスクを抽出
            qa_tasks = [t for t in tasks if 'generate_qa' in t or 'chunk' in t]

            if qa_tasks:
                print(f"  📋 Q/A生成タスク:")
                for task in qa_tasks:
                    print(f"    ✅ {task}")
            else:
                print(f"  ⚠️ Q/A生成タスクが登録されていません")

            # その他の主要なタスク
            print(f"  📋 全タスク数: {len(tasks)}個")
            print(f"  📋 サンプル（最初の5個）:")
            for task in list(tasks)[:5]:
                print(f"    - {task}")
    else:
        print("\n❌ 登録されているタスクなし")
        print("原因の可能性:")
        print("  1. ワーカーが celery_tasks.py をロードしていない")
        print("  2. タスクのデコレータ(@app.task)が正しく設定されていない")

    # アクティブなタスク
    active = inspect.active()
    if active:
        has_active = any(tasks for tasks in active.values())
        if has_active:
            print(f"\n⚠️ 実行中のタスク:")
            for worker_name, tasks in active.items():
                if tasks:
                    print(f"  ワーカー: {worker_name}")
                    for task in tasks:
                        print(f"    - {task['name']} (ID: {task['id'][:8]}...)")
        else:
            print(f"\n✅ 実行中のタスクなし")
    else:
        print(f"\n✅ 実行中のタスクなし")

    # 予約されているタスク
    reserved = inspect.reserved()
    if reserved:
        has_reserved = any(tasks for tasks in reserved.values())
        if has_reserved:
            print(f"\n⚠️ 予約されているタスク:")
            for worker_name, tasks in reserved.items():
                if tasks:
                    print(f"  ワーカー: {worker_name}")
                    for task in tasks:
                        print(f"    - {task['name']} (ID: {task['id'][:8]}...)")
        else:
            print(f"\n✅ 予約されているタスクなし")
    else:
        print(f"\n✅ 予約されているタスクなし")

except Exception as e:
    print(f"❌ ワーカー確認エラー: {e}")
    import traceback

    traceback.print_exc()

# ステップ5: qa_generationモジュールの確認
print("\n[5] qa_generationモジュール確認")
print("-" * 60)

qa_gen_path = project_root / 'qa_generation'
print(f"qa_generationディレクトリ: {qa_gen_path}")
print(f"存在: {qa_gen_path.exists()}")

if qa_gen_path.exists():
    gen_file = qa_gen_path / 'generation.py'
    print(f"generation.py: {gen_file.exists()}")

    # ファイルリスト
    py_files = list(qa_gen_path.glob('*.py'))
    print(f"Pythonファイル数: {len(py_files)}個")
    print(f"主要ファイル:")
    for f in ['__init__.py', 'generation.py', 'pipeline.py', 'structure.py']:
        path = qa_gen_path / f
        print(f"  {'✅' if path.exists() else '❌'} {f}")

    # インポートテスト
    try:
        from qa_generation.generation import generate_qa_dataset

        print("✅ qa_generation.generationインポート成功")
    except ImportError as e:
        print(f"❌ qa_generation.generationインポート失敗: {e}")
else:
    print("❌ qa_generationディレクトリが見つかりません")

# ステップ6: テストタスクの投入
print("\n[6] テストタスク投入")
print("-" * 60)

try:
    from celery_tasks import generate_qa_for_chunk_task

    test_chunk = {
        'id'       : 'diagnostic_test',
        'text'     : 'これは診断テストです。Celeryが正常に動作しているか確認します。',
        'tokens'   : 20,
        'doc_id'   : 'test',
        'chunk_idx': 0
    }

    config = {
        'type'        : 'test',
        'qa_per_chunk': 1
    }

    print("テストタスクを投入中...")
    task = generate_qa_for_chunk_task.apply_async(
        args=(test_chunk, config, "gemini-2.0-flash", "gemini", True),

    )

    print(f"✅ タスク投入成功")
    print(f"  タスクID: {task.id}")
    print(f"  初期状態: {task.state}")

    # 10秒待機して状態を確認
    import time

    print("\n状態監視中（10秒間）:")
    for i in range(10):
        time.sleep(1)
        state = task.state
        print(f"  {i + 1}秒: {state}", end='')

        if state == 'SUCCESS':
            try:
                result = task.get(timeout=1)
                print(f" → ✅ 成功（{len(result)}件のQ/A）")
                break
            except:
                print(f" → ❌ 結果取得失敗")
                break
        elif state == 'FAILURE':
            print(f" → ❌ 失敗")
            try:
                print(f"  エラー: {task.info}")
            except:
                pass
            break
        elif state == 'STARTED':
            print(f" → ⚙️ 実行中")
        elif state == 'PENDING':
            print(f" → ⏳ 待機中")
        else:
            print(f" → ❓ 不明な状態")

        if i == 9:
            print(f"\n\n⚠️ 10秒経過してもタスクが完了しません")
            print(f"最終状態: {task.state}")

            if task.state == 'PENDING':
                print("\n❌ 診断結果: タスクがワーカーに届いていません")
                print("原因の可能性:")
                print("  1. タスク名が登録されていない")
                print("  2. キューの設定が間違っている")
                print("  3. ワーカーがタスクを受信できていない")
                print("\n推奨対策:")
                print("  1. ワーカーログを確認: tail -f logs/celery_qa_*.log")
                print("  2. ワーカーを再起動: ./start_celery.sh restart -w 8")
                print("  3. 登録されているタスクを確認（上記の[4]を参照）")

except Exception as e:
    print(f"❌ テストタスク投入エラー: {e}")
    import traceback

    traceback.print_exc()

# ステップ7: ログファイルの確認
print("\n[7] ログファイル確認")
print("-" * 60)

log_dir = project_root / 'logs'
if log_dir.exists():
    print(f"ログディレクトリ: {log_dir}")
    log_files = list(log_dir.glob('celery_*.log'))
    if log_files:
        print(f"✅ ログファイル: {len(log_files)}個")
        for log_file in sorted(log_files)[-3:]:  # 最新3件
            print(f"\n  📄 {log_file.name}")
            print(f"     サイズ: {log_file.stat().st_size:,} bytes")

            # 最後の5行を表示
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"     最終行数: {len(lines)}")
                        print(f"     最後の3行:")
                        for line in lines[-3:]:
                            print(f"       {line.rstrip()}")
            except Exception as e:
                print(f"     ⚠️ 読み込みエラー: {e}")
    else:
        print("⚠️ ログファイルが見つかりません")
        print("ワーカーが起動していない可能性があります")
else:
    print("⚠️ logsディレクトリが存在しません")
    print(f"期待される場所: {log_dir}")

print("\n" + "=" * 60)
print("診断完了")
print("=" * 60)

# 最終判定
print("\n🎯 診断結果サマリー:")
print("-" * 60)

issues = []
success = []

# Redis
try:
    r.ping()
    success.append("✅ Redis接続")
except:
    issues.append("❌ Redis接続失敗")

# Celeryモジュール
try:
    from celery_config import app

    success.append("✅ celery_config.pyインポート")
except:
    issues.append("❌ celery_config.pyインポート失敗")

try:
    from celery_tasks import generate_qa_for_chunk_task

    success.append("✅ celery_tasks.pyインポート")
except:
    issues.append("❌ celery_tasks.pyインポート失敗")

# ワーカー
try:
    from celery_config import app

    stats = app.control.inspect().stats()
    if stats:
        success.append(f"✅ ワーカー起動中（{len(stats)}個）")
    else:
        issues.append("❌ ワーカーが起動していません")
except:
    issues.append("❌ ワーカー確認失敗")

# qa_generation
if qa_gen_path.exists():
    success.append("✅ qa_generationディレクトリ")
else:
    issues.append("❌ qa_generationディレクトリが見つかりません")

print("\n成功:")
for s in success:
    print(f"  {s}")

if issues:
    print("\n問題:")
    for i in issues:
        print(f"  {i}")

# 推奨アクション
print("\n📋 推奨アクション:")
if not issues:
    print("  🎉 すべての診断項目をクリアしました！")
    print("  次のステップ: python qa_qdrant/tests/test_celery_integration.py")
else:
    print("  1. ワーカーログを確認: tail -f logs/celery_qa_*.log")
    print("  2. Celeryワーカーを再起動: ./start_celery.sh restart -w 8")
    print("  3. 問題が解決しない場合: redis-cli FLUSHALL")
    print("  4. Pythonキャッシュをクリア: find . -name __pycache__ -exec rm -rf {} +")
