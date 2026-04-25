"""
Pytest configuration and hooks for custom output formatting.
"""
import pytest
import sys

# テストの総数と現在のカウントを追跡するためのグローバル変数
_test_count = 0
_total_tests = 0

def pytest_sessionstart(session):
    """セッション開始時にテスト総数を取得"""
    global _total_tests
    # 収集されたアイテム数はまだ確定していないため、collectionfinishで設定する手もあるが、
    # 簡易的に初期化のみ行う
    _total_tests = 0

def pytest_collection_modifyitems(session, config, items):
    """テスト収集完了後に総数を設定"""
    global _total_tests
    _total_tests = len(items)

def pytest_runtest_protocol(item, nextitem):
    """各テスト実行前にカスタムヘッダーを表示"""
    global _test_count
    _test_count += 1
    
    # 区切り線とカウントの表示
    header = f"\n----------------------------\nTest {_test_count}/{_total_tests}: {item.nodeid}\n----------------------------"
    sys.stdout.write(header + "\n")
    
    return None  # デフォルトの動作を続行
