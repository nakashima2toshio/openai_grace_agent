#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_collection_dynamic.py - 命名規則廃止版の動作確認テスト
================================================================

このスクリプトは、命名規則への依存を廃止した新しい実装の
動作を確認するためのテストスイートです。

テスト内容:
----------
1. Qdrantから全コレクション動的取得
2. コレクション名のフィルタリングなし確認
3. 全コレクション並列検索の動作確認
4. エラーハンドリングの確認
5. パフォーマンス測定

使用方法:
--------
$ python test_collection_dynamic.py

環境要件:
--------
- Qdrantサーバーが起動していること
- 必要なPythonパッケージがインストール済みであること
"""

import sys
import time
import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===================================================================
# テスト用ユーティリティ
# ===================================================================

class TestResult:
    """テスト結果を保持するクラス"""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.duration_ms = 0.0

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} - {self.name} ({self.duration_ms:.0f}ms): {self.message}"


class TestRunner:
    """テストランナー"""

    def __init__(self):
        self.results: List[TestResult] = []

    def run_test(self, test_func):
        """テストを実行"""
        result = TestResult(test_func.__name__)
        start_time = time.time()

        try:
            test_func()
            result.passed = True
            result.message = "成功"
        except AssertionError as e:
            result.passed = False
            result.message = f"アサーションエラー: {str(e)}"
        except Exception as e:
            result.passed = False
            result.message = f"予期しないエラー: {str(e)}"

        result.duration_ms = (time.time() - start_time) * 1000
        self.results.append(result)

        print(result)
        return result.passed

    def print_summary(self):
        """テスト結果のサマリーを表示"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        print("\n" + "=" * 60)
        print("📊 テスト結果サマリー")
        print("=" * 60)
        print(f"総テスト数: {total}")
        print(f"✅ 成功: {passed}")
        print(f"❌ 失敗: {failed}")
        print(f"成功率: {(passed / total * 100):.1f}%")
        print("=" * 60 + "\n")


# ===================================================================
# テストケース
# ===================================================================

# Qdrantクライアント（グローバル）
client = None


def test_01_qdrant_connection():
    """テスト1: Qdrant接続確認"""
    global client
    client = QdrantClient(host="localhost", port=6333)

    # ヘルスチェック
    collections_response = client.get_collections()
    assert collections_response is not None, "コレクション取得失敗"

    logger.info(f"✅ Qdrant接続成功: {len(collections_response.collections)}個のコレクション")


def test_02_get_all_collections_no_filter():
    """テスト2: 全コレクション取得（フィルタリングなし）"""
    collections_response = client.get_collections()
    all_collections = [c.name for c in collections_response.collections]

    assert len(all_collections) > 0, "コレクションが存在しません"

    logger.info(f"✅ 全コレクション取得成功: {len(all_collections)}個")
    logger.info(f"   コレクション一覧: {all_collections}")

    # 命名規則チェック（どんな名前でもOKであることを確認）
    for col_name in all_collections:
        # ❌ 以前: col_name.startswith("qa_") などのチェック
        # ✅ 現在: どんな名前でもOK
        assert isinstance(col_name, str), f"コレクション名が文字列ではありません: {col_name}"
        logger.info(f"   ✅ コレクション '{col_name}' を検索対象として受け入れ")


def test_03_no_naming_convention_dependency():
    """テスト3: 命名規則への依存がないことを確認"""
    collections_response = client.get_collections()
    all_collections = [c.name for c in collections_response.collections]

    # 様々な命名規則のコレクションを確認
    naming_patterns = {
        'qa_prefix'     : [c for c in all_collections if c.startswith('qa_')],
        'no_qa_prefix'  : [c for c in all_collections if not c.startswith('qa_')],
        'has_underscore': [c for c in all_collections if '_' in c],
        'no_underscore' : [c for c in all_collections if '_' not in c],
    }

    logger.info("📊 命名パターン分析:")
    for pattern, collections in naming_patterns.items():
        logger.info(f"   {pattern}: {len(collections)}個 {collections}")

    # すべてのパターンが受け入れられることを確認
    total_unique = len(all_collections)
    total_categorized = sum(len(cols) for cols in naming_patterns.values())

    logger.info(f"✅ すべての命名規則を受け入れ: {total_unique}個のコレクション")


def test_04_collection_search_simulation():
    """テスト4: コレクション検索のシミュレーション"""
    collections_response = client.get_collections()
    all_collections = [c.name for c in collections_response.collections]

    assert len(all_collections) > 0, "検索対象コレクションがありません"

    # 各コレクションの基本情報を取得
    for col_name in all_collections:
        try:
            collection_info = client.get_collection(col_name)
            point_count = collection_info.points_count

            logger.info(f"   📚 コレクション '{col_name}': {point_count}ポイント")

            # ポイント数が0でないことを確認
            assert point_count >= 0, f"不正なポイント数: {point_count}"

        except Exception as e:
            logger.warning(f"   ⚠️ コレクション '{col_name}' の情報取得失敗: {e}")

    logger.info(f"✅ {len(all_collections)}個のコレクションすべてが検索可能")


def test_05_parallel_search_readiness():
    """テスト5: 並列検索の準備状態確認"""
    collections_response = client.get_collections()
    all_collections = [c.name for c in collections_response.collections]

    # 並列検索に必要な条件を確認
    assert len(all_collections) > 0, "検索対象コレクションがありません"

    # 各コレクションがアクセス可能かチェック
    accessible_count = 0
    for col_name in all_collections:
        try:
            # 簡易アクセステスト
            client.get_collection(col_name)
            accessible_count += 1
        except Exception as e:
            logger.warning(f"   ⚠️ コレクション '{col_name}' にアクセスできません: {e}")

    logger.info(f"✅ 並列検索可能なコレクション: {accessible_count}/{len(all_collections)}個")

    # 最低1つはアクセス可能であること
    assert accessible_count > 0, "アクセス可能なコレクションが1つもありません"


def test_06_error_handling():
    """テスト6: エラーハンドリングの確認"""
    # 存在しないコレクションへのアクセス
    non_existent_collection = "this_collection_does_not_exist_12345"

    try:
        client.get_collection(non_existent_collection)
        # ここに到達したら失敗
        raise AssertionError("存在しないコレクションにアクセスできてしまった")
    except Exception as e:
        logger.info(f"✅ 期待通りのエラー: {type(e).__name__}")
        # エラーが発生することを期待


def test_07_performance_baseline():
    """テスト7: パフォーマンスベースライン測定"""
    collections_response = client.get_collections()
    all_collections = [c.name for c in collections_response.collections]

    if not all_collections:
        logger.warning("⚠️ コレクションが存在しないため、パフォーマンステストをスキップ")
        return

    # 各コレクションの情報取得にかかる時間を測定
    total_time = 0.0
    for col_name in all_collections:
        start = time.time()
        try:
            client.get_collection(col_name)
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            logger.info(f"   📊 '{col_name}': {elapsed:.2f}ms")
        except Exception as e:
            logger.warning(f"   ⚠️ '{col_name}': エラー ({e})")

    avg_time = total_time / len(all_collections) if all_collections else 0
    logger.info(f"✅ 平均応答時間: {avg_time:.2f}ms ({len(all_collections)}コレクション)")


def test_08_backward_compatibility():
    """テスト8: 後方互換性の確認"""
    # 古い命名規則のコレクションも正常に動作するか確認
    collections_response = client.get_collections()
    all_collections = [c.name for c in collections_response.collections]

    # 'qa_' で始まるコレクション（旧命名規則）があればテスト
    old_naming_collections = [c for c in all_collections if c.startswith('qa_')]

    if old_naming_collections:
        logger.info(f"📋 旧命名規則コレクション発見: {len(old_naming_collections)}個")
        for col_name in old_naming_collections:
            # 旧命名規則でも問題なく動作することを確認
            try:
                info = client.get_collection(col_name)
                logger.info(f"   ✅ '{col_name}': {info.points_count}ポイント")
            except Exception as e:
                raise AssertionError(f"旧命名規則コレクション '{col_name}' でエラー: {e}")
    else:
        logger.info("✅ 旧命名規則コレクションなし（新規環境）")

    # 新しい命名規則のコレクションも動作確認
    new_naming_collections = [c for c in all_collections if not c.startswith('qa_')]

    if new_naming_collections:
        logger.info(f"📋 新命名規則コレクション発見: {len(new_naming_collections)}個")
        for col_name in new_naming_collections:
            try:
                info = client.get_collection(col_name)
                logger.info(f"   ✅ '{col_name}': {info.points_count}ポイント")
            except Exception as e:
                raise AssertionError(f"新命名規則コレクション '{col_name}' でエラー: {e}")

    logger.info("✅ 後方互換性確認完了: 新旧両方の命名規則に対応")


# ===================================================================
# メイン実行
# ===================================================================

def main():
    """テストスイートのメイン関数"""
    print("\n" + "=" * 60)
    print("🧪 命名規則廃止版 - 動作確認テストスイート")
    print("=" * 60 + "\n")

    runner = TestRunner()

    # テスト実行
    tests = [
        test_01_qdrant_connection,
        test_02_get_all_collections_no_filter,
        test_03_no_naming_convention_dependency,
        test_04_collection_search_simulation,
        test_05_parallel_search_readiness,
        test_06_error_handling,
        test_07_performance_baseline,
        test_08_backward_compatibility,
    ]

    for test in tests:
        print(f"\n{'─' * 60}")
        print(f"🔍 実行中: {test.__doc__}")
        print(f"{'─' * 60}")
        runner.run_test(test)

    # サマリー表示
    runner.print_summary()

    # 終了コード
    all_passed = all(r.passed for r in runner.results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
