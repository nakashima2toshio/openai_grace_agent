#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qdrant_delete_collection.py - 指定コレクションを削除するコマンド

使用例:
    python qdrant_delete_collection.py cc_news_2per_openai
    python qdrant_delete_collection.py cc_news_2per_openai --yes   # 確認スキップ
    python qdrant_delete_collection.py --list                       # コレクション一覧表示
"""

import argparse
import sys

from qdrant_client_wrapper import create_qdrant_client, get_all_collections


def main():
    parser = argparse.ArgumentParser(description="Qdrant コレクション削除コマンド")
    parser.add_argument("collection_name", nargs="?", help="削除するコレクション名")
    parser.add_argument("--url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--yes", "-y", action="store_true", help="確認プロンプトをスキップ")
    parser.add_argument("--list", "-l", action="store_true", help="コレクション一覧を表示して終了")
    args = parser.parse_args()

    client = create_qdrant_client(url=args.url)

    # --list: コレクション一覧表示
    if args.list:
        collections = get_all_collections(client)
        if not collections:
            print("コレクションが存在しません。")
        else:
            print(f"コレクション一覧 ({len(collections)}件):")
            for col in collections:
                print(f"  - {col['name']:40s}  points={col['points_count']:>8,}  status={col['status']}")
        return

    # コレクション名が指定されていない場合
    if not args.collection_name:
        parser.print_help()
        sys.exit(1)

    collection_name = args.collection_name

    # 存在確認
    collections = get_all_collections(client)
    existing = [c["name"] for c in collections]
    if collection_name not in existing:
        print(f"エラー: コレクション '{collection_name}' は存在しません。")
        print(f"既存コレクション: {existing}")
        sys.exit(1)

    # 削除前の情報表示
    target = next(c for c in collections if c["name"] == collection_name)
    print(f"削除対象: {collection_name}")
    print(f"  points_count : {target['points_count']:,}")
    print(f"  status       : {target['status']}")

    # 確認プロンプト（--yes で省略可）
    if not args.yes:
        answer = input(f"\n'{collection_name}' を削除しますか？ [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("削除をキャンセルしました。")
            sys.exit(0)

    # 削除実行
    client.delete_collection(collection_name=collection_name)
    print(f"削除完了: コレクション '{collection_name}' を削除しました。")


if __name__ == "__main__":
    main()
