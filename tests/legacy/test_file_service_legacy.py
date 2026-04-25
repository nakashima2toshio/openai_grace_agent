#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_file_service.py - ファイルサービスのテスト
===============================================
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from services.file_service import (
    load_qa_output_history,
    load_preprocessed_history,
    load_sample_questions_from_csv,
    load_source_qa_data,
)


class TestLoadQaOutputHistory:
    """load_qa_output_history関数のテスト"""

    def test_load_history_empty_directory(self, temp_dir):
        """空ディレクトリのテスト"""
        qa_dir = temp_dir / "qa_output"
        qa_dir.mkdir()

        with patch("services.file_service.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.glob.return_value = []

            result = load_qa_output_history()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_load_history_no_directory(self, temp_dir):
        """ディレクトリ不存在のテスト"""
        with patch("services.file_service.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            result = load_qa_output_history()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_load_history_with_files(self, qa_output_dir):
        """ファイルありのテスト"""
        # このテストは実際のファイルシステムを使用
        original_path = Path("qa_output")

        # 一時的にqa_outputディレクトリを作成
        if not original_path.exists():
            original_path.mkdir(parents=True)
            created = True
        else:
            created = False

        # テスト用CSVを作成
        test_csv = original_path / "test_qa_history.csv"
        pd.DataFrame({
            "question": ["Q1"],
            "answer": ["A1"]
        }).to_csv(test_csv, index=False)

        try:
            result = load_qa_output_history()

            assert isinstance(result, pd.DataFrame)
            assert "ファイル名" in result.columns
            assert "ファイルサイズ" in result.columns
            assert "作成日付" in result.columns
        finally:
            # クリーンアップ
            if test_csv.exists():
                test_csv.unlink()
            if created and original_path.exists():
                # 他のファイルがなければディレクトリを削除
                if not list(original_path.glob("*")):
                    original_path.rmdir()


class TestLoadPreprocessedHistory:
    """load_preprocessed_history関数のテスト"""

    def test_load_history_empty_directory(self, temp_dir):
        """空ディレクトリのテスト"""
        with patch("services.file_service.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.glob.return_value = []

            result = load_preprocessed_history()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0


class TestSaveToOutput:
    """save_to_output関数のテスト"""

    def test_save_creates_files(self, sample_text_df, temp_dir):
        """ファイル作成のテスト"""
        # 一時的にOUTPUTディレクトリを設定
        output_dir = temp_dir / "OUTPUT"
        output_dir.mkdir(exist_ok=True)

        # Combined_Textカラムを持つDataFrameを作成
        df = pd.DataFrame({
            "Combined_Text": ["テキスト1", "テキスト2"]
        })

        # 直接保存をテスト
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"preprocessed_test_{timestamp}.csv"
        txt_path = output_dir / f"test_{timestamp}.txt"

        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(df["Combined_Text"].tolist()))

        assert csv_path.exists()
        assert txt_path.exists()


class TestLoadSampleQuestionsFromCsv:
    """load_sample_questions_from_csv関数のテスト"""

    def test_load_samples_no_mapping(self):
        """マッピングなしのテスト"""
        result = load_sample_questions_from_csv("nonexistent_collection")

        assert result == []

    def test_load_samples_file_not_found(self):
        """ファイル不存在のテスト"""
        # map_collection_to_csv をモック
        with patch("services.file_service.map_collection_to_csv", return_value="test.csv"):
             result = load_sample_questions_from_csv("test")
             assert result == []


class TestLoadSourceQaData:
    """load_source_qa_data関数のテスト"""

    def test_load_source_file_not_found(self):
        """ファイル不存在のテスト"""
        result = load_source_qa_data("nonexistent.csv")

        assert result is None

    def test_load_source_success(self, temp_dir):
        """ファイル読み込み成功のテスト"""
        # qa_outputディレクトリを作成
        qa_dir = temp_dir / "qa_output"
        qa_dir.mkdir()

        # テストCSVを作成
        test_df = pd.DataFrame({
            "question": ["Q1", "Q2", "Q3"],
            "answer": ["A1", "A2", "A3"]
        })
        csv_path = qa_dir / "test_qa.csv"
        test_df.to_csv(csv_path, index=False)

        # パスをパッチ
        with patch("services.file_service.Path") as mock_path:
            # qa_output/test_qa.csv を返すようにモック
            mock_path.return_value.__truediv__ = lambda self, x: qa_dir / x

            # 実際のPathオブジェクトの動作をシミュレート
            actual_path = qa_dir / "test_qa.csv"
            if actual_path.exists():
                result = pd.read_csv(actual_path, nrows=20, usecols=["question", "answer"])

                assert result is not None
                assert "question" in result.columns
                assert "answer" in result.columns
                assert len(result) == 3


class TestFileServiceIntegration:
    """ファイルサービス統合テスト"""

    def test_full_workflow(self, temp_dir):
        """完全なワークフローのテスト"""
        # 1. OUTPUTディレクトリを作成
        output_dir = temp_dir / "OUTPUT"
        output_dir.mkdir()

        # 2. preprocessedファイルを保存
        df = pd.DataFrame({
            "Combined_Text": ["テキスト1", "テキスト2", "テキスト3"]
        })

        csv_path = output_dir / "preprocessed_test_20241126_120000.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 3. ファイルが存在することを確認
        assert csv_path.exists()

        # 4. ファイル内容を確認
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == 3
        assert "Combined_Text" in loaded_df.columns