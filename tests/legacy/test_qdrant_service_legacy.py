#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_qdrant_service.py - Qdrantサービスのテスト
===============================================
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from services.qdrant_service import (
    batched,
    QdrantHealthChecker,
    QdrantDataFetcher,
    load_csv_for_qdrant,
    build_inputs_for_embedding,
    build_points_for_qdrant,
    QDRANT_CONFIG,
)


class TestBatched:
    """batched関数のテスト"""

    def test_batch_even_split(self):
        """均等分割のテスト"""
        items = [1, 2, 3, 4, 5, 6]
        batches = list(batched(items, 2))

        assert len(batches) == 3
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]
        assert batches[2] == [5, 6]

    def test_batch_uneven_split(self):
        """不均等分割のテスト"""
        items = [1, 2, 3, 4, 5]
        batches = list(batched(items, 2))

        assert len(batches) == 3
        assert batches[-1] == [5]

    def test_batch_single_item(self):
        """単一アイテムのテスト"""
        items = [1]
        batches = list(batched(items, 3))

        assert len(batches) == 1
        assert batches[0] == [1]

    def test_batch_empty(self):
        """空リストのテスト"""
        items = []
        batches = list(batched(items, 2))

        assert len(batches) == 0


class TestQdrantHealthChecker:
    """QdrantHealthCheckerのテスト"""

    def test_check_port_closed(self):
        """閉じたポートのチェック"""
        checker = QdrantHealthChecker(debug_mode=False)

        # 存在しないポートをチェック
        result = checker.check_port("127.0.0.1", 59999, timeout=0.1)

        assert result is False

    @patch("services.qdrant_service.QdrantClient")
    @patch.object(QdrantHealthChecker, "check_port", return_value=True)
    def test_check_qdrant_success(self, mock_port, mock_client):
        """Qdrant接続成功のテスト"""
        mock_instance = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test"
        mock_instance.get_collections.return_value.collections = [mock_collection]
        mock_client.return_value = mock_instance

        checker = QdrantHealthChecker(debug_mode=False)
        is_connected, message, metrics = checker.check_qdrant()

        assert is_connected is True
        assert message == "Connected"
        assert metrics is not None
        assert metrics["collection_count"] == 1

    @patch.object(QdrantHealthChecker, "check_port", return_value=False)
    def test_check_qdrant_port_closed(self, mock_port):
        """ポートが閉じている場合のテスト"""
        checker = QdrantHealthChecker(debug_mode=False)
        is_connected, message, metrics = checker.check_qdrant()

        assert is_connected is False
        assert "Connection refused" in message
        assert metrics is None


class TestQdrantDataFetcher:
    """QdrantDataFetcherのテスト"""

    def test_fetch_collections_success(self):
        """コレクション一覧取得の成功テスト"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client.get_collections.return_value.collections = [mock_collection]

        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.indexed_vectors_count = 100
        mock_info.status = "green"
        mock_client.get_collection.return_value = mock_info

        fetcher = QdrantDataFetcher(mock_client)
        result = fetcher.fetch_collections()

        assert isinstance(result, pd.DataFrame)
        assert "Collection" in result.columns
        assert result.iloc[0]["Collection"] == "test_collection"

    def test_fetch_collections_empty(self):
        """空のコレクション一覧のテスト"""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []

        fetcher = QdrantDataFetcher(mock_client)
        result = fetcher.fetch_collections()

        assert isinstance(result, pd.DataFrame)
        assert "Info" in result.columns

    def test_fetch_collections_error(self):
        """コレクション一覧取得エラーのテスト"""
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = Exception("Connection error")

        fetcher = QdrantDataFetcher(mock_client)
        result = fetcher.fetch_collections()

        assert isinstance(result, pd.DataFrame)
        assert "Error" in result.columns


class TestLoadCsvForQdrant:
    """load_csv_for_qdrant関数のテスト"""

    def test_load_csv_success(self, sample_qa_df):
        """CSVロード成功のテスト"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            sample_qa_df.to_csv(f, index=False)
            temp_path = f.name

        try:
            result = load_csv_for_qdrant(temp_path)

            assert isinstance(result, pd.DataFrame)
            assert "question" in result.columns
            assert "answer" in result.columns
            assert len(result) == 3
        finally:
            os.unlink(temp_path)

    def test_load_csv_with_limit(self, sample_qa_df):
        """件数制限付きCSVロードのテスト"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            sample_qa_df.to_csv(f, index=False)
            temp_path = f.name

        try:
            result = load_csv_for_qdrant(temp_path, limit=2)

            assert len(result) == 2
        finally:
            os.unlink(temp_path)

    def test_load_csv_file_not_found(self):
        """ファイル未存在エラーのテスト"""
        with pytest.raises(FileNotFoundError):
            load_csv_for_qdrant("/nonexistent/path.csv")

    def test_load_csv_missing_column(self):
        """必須カラム欠落エラーのテスト"""
        df = pd.DataFrame({"question": ["Q1"], "other": ["O1"]})

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="answer"):
                load_csv_for_qdrant(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_csv_column_mapping(self):
        """カラム名マッピングのテスト"""
        df = pd.DataFrame({
            "Question": ["Q1"],
            "Answer": ["A1"]
        })

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            result = load_csv_for_qdrant(temp_path)

            assert "question" in result.columns
            assert "answer" in result.columns
        finally:
            os.unlink(temp_path)


class TestBuildInputsForEmbedding:
    """build_inputs_for_embedding関数のテスト"""

    def test_question_only(self, sample_qa_df):
        """質問のみの埋め込み入力"""
        result = build_inputs_for_embedding(sample_qa_df, include_answer=False)

        assert len(result) == 3
        assert "Pythonとは何ですか？" in result[0]
        assert "Pythonは汎用" not in result[0]

    def test_question_and_answer(self, sample_qa_df):
        """質問と回答の埋め込み入力"""
        result = build_inputs_for_embedding(sample_qa_df, include_answer=True)

        assert len(result) == 3
        assert "Pythonとは何ですか？" in result[0]
        assert "Pythonは汎用" in result[0]


class TestBuildPointsForQdrant:
    """build_points_for_qdrant関数のテスト"""

    def test_build_points_success(self, sample_qa_df):
        """ポイント構築成功のテスト"""
        vectors = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

        result = build_points_for_qdrant(
            sample_qa_df, vectors, domain="test", source_file="test.csv"
        )

        assert len(result) == 3
        assert result[0].payload["domain"] == "test"
        assert result[0].payload["source"] == "test.csv"
        assert "question" in result[0].payload
        assert "answer" in result[0].payload

    def test_build_points_length_mismatch(self, sample_qa_df):
        """長さ不一致エラーのテスト"""
        vectors = [[0.1] * 1536, [0.2] * 1536]  # 2つだけ

        with pytest.raises(ValueError, match="mismatch"):
            build_points_for_qdrant(
                sample_qa_df, vectors, domain="test", source_file="test.csv"
            )


class TestQdrantConfig:
    """QDRANT_CONFIG定数のテスト"""

    def test_config_has_required_keys(self):
        """必須キーの存在確認"""
        required_keys = ["name", "host", "port", "url", "docker_image"]

        for key in required_keys:
            assert key in QDRANT_CONFIG

    def test_config_default_values(self):
        """デフォルト値の確認"""
        assert QDRANT_CONFIG["host"] == "localhost"
        assert QDRANT_CONFIG["port"] == 6333
        assert "localhost:6333" in QDRANT_CONFIG["url"]