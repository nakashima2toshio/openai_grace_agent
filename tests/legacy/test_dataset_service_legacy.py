#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_dataset_service.py - データセットサービスのテスト
======================================================
"""

import io
import json
import pytest
import pandas as pd

from services.dataset_service import (
    extract_text_content,
    load_uploaded_file,
)


class TestExtractTextContent:
    """extract_text_content関数のテスト"""

    def test_extract_with_text_field(self):
        """テキストフィールドからの抽出"""
        df = pd.DataFrame({
            "text": ["テキスト1", "テキスト2"],
            "other": ["その他1", "その他2"]
        })
        config = {"text_field": "text", "title_field": None}

        result = extract_text_content(df, config)

        assert "Combined_Text" in result.columns
        assert len(result) == 2
        assert "テキスト1" in result.iloc[0]["Combined_Text"]

    def test_extract_with_title_and_text(self):
        """タイトルとテキストの結合"""
        df = pd.DataFrame({
            "title": ["タイトル1", "タイトル2"],
            "content": ["本文1", "本文2"]
        })
        config = {"text_field": "content", "title_field": "title"}

        result = extract_text_content(df, config)

        assert "Combined_Text" in result.columns
        assert "タイトル1" in result.iloc[0]["Combined_Text"]
        assert "本文1" in result.iloc[0]["Combined_Text"]

    def test_extract_removes_empty_text(self):
        """空テキストの除外"""
        df = pd.DataFrame({
            "text": ["有効なテキスト", "", "   "]
        })
        config = {"text_field": "text", "title_field": None}

        result = extract_text_content(df, config)

        assert len(result) == 1
        assert "有効なテキスト" in result.iloc[0]["Combined_Text"]

    def test_extract_fallback_to_candidates(self):
        """テキスト候補フィールドへのフォールバック"""
        df = pd.DataFrame({
            "body": ["本文テキスト"],
            "metadata": ["メタデータ"]
        })
        config = {"text_field": "nonexistent", "title_field": None}

        result = extract_text_content(df, config)

        assert "Combined_Text" in result.columns
        assert "本文テキスト" in result.iloc[0]["Combined_Text"]


class TestLoadUploadedFile:
    """load_uploaded_file関数のテスト"""

    def test_load_csv_file(self):
        """CSVファイルの読み込み"""
        csv_content = "text,value\nテキスト1,100\nテキスト2,200"
        uploaded_file = io.BytesIO(csv_content.encode("utf-8"))
        uploaded_file.name = "test.csv"

        result = load_uploaded_file(uploaded_file)

        assert "Combined_Text" in result.columns
        assert len(result) == 2

    def test_load_txt_file(self):
        """テキストファイルの読み込み"""
        txt_content = "1行目のテキスト\n2行目のテキスト\n3行目のテキスト"
        uploaded_file = io.BytesIO(txt_content.encode("utf-8"))
        uploaded_file.name = "test.txt"

        result = load_uploaded_file(uploaded_file)

        assert "Combined_Text" in result.columns
        assert len(result) == 3

    def test_load_json_file_list(self):
        """JSONファイル（リスト形式）の読み込み"""
        json_data = [
            {"text": "テキスト1"},
            {"text": "テキスト2"}
        ]
        uploaded_file = io.BytesIO(json.dumps(json_data).encode("utf-8"))
        uploaded_file.name = "test.json"

        result = load_uploaded_file(uploaded_file)

        assert "Combined_Text" in result.columns
        assert len(result) == 2

    def test_load_json_file_object(self):
        """JSONファイル（オブジェクト形式）の読み込み"""
        json_data = {"text": "単一テキスト"}
        uploaded_file = io.BytesIO(json.dumps(json_data).encode("utf-8"))
        uploaded_file.name = "test.json"

        result = load_uploaded_file(uploaded_file)

        assert "Combined_Text" in result.columns
        assert len(result) == 1

    def test_load_jsonl_file(self):
        """JSON Linesファイルの読み込み"""
        jsonl_content = '{"text": "行1"}\n{"text": "行2"}'
        uploaded_file = io.BytesIO(jsonl_content.encode("utf-8"))
        uploaded_file.name = "test.jsonl"

        result = load_uploaded_file(uploaded_file)

        assert "Combined_Text" in result.columns
        assert len(result) == 2

    def test_load_unsupported_format(self):
        """未対応形式でのエラー"""
        uploaded_file = io.BytesIO(b"dummy content")
        uploaded_file.name = "test.xyz"

        with pytest.raises(ValueError, match="未対応のファイル形式"):
            load_uploaded_file(uploaded_file)

    def test_load_removes_empty_rows(self):
        """空行の除外"""
        csv_content = "text\nテキスト1\n\n   \nテキスト2"
        uploaded_file = io.BytesIO(csv_content.encode("utf-8"))
        uploaded_file.name = "test.csv"

        result = load_uploaded_file(uploaded_file)

        assert len(result) == 2