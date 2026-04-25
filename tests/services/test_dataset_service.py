import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
from services.dataset_service import (
    download_livedoor_corpus,
    load_livedoor_corpus,
    download_hf_dataset,
    extract_text_content,
    load_uploaded_file
)

class TestDatasetService:

    @patch("services.dataset_service.urllib.request.urlretrieve")
    @patch("services.dataset_service.tarfile.open")
    @patch("services.dataset_service.Path.exists")
    @patch("services.dataset_service.Path.mkdir")
    def test_download_livedoor_corpus(self, mock_mkdir, mock_exists, mock_tar, mock_urlretrieve):
        # Case: already exists
        mock_exists.return_value = True
        path = download_livedoor_corpus()
        assert "text" in path
        
        # Case: download needed
        mock_exists.side_effect = [False, False, False, False] # dir, tar, extract, text
        # Need to be careful with side_effect for path.exists which is called multiple times
        # 1. save_path / tar_filename -> False
        # 2. extract_path -> False
        # 3. text_dir -> False
        
        # Reset side effect
        mock_exists.side_effect = None
        mock_exists.return_value = False 
        
        download_livedoor_corpus()
        mock_urlretrieve.assert_called()
        mock_tar.assert_called()

    def test_extract_text_content(self):
        df = pd.DataFrame({
            "title": ["Title1", "Title2"],
            "body": ["Content1", "Content2"]
        })
        config = {"text_field": "body", "title_field": "title"}
        
        result_df = extract_text_content(df, config)
        assert "Combined_Text" in result_df.columns
        assert result_df.iloc[0]["Combined_Text"] == "Title1 Content1"

        # Fallback case
        df2 = pd.DataFrame({
            "unknown": ["Val1"]
        })
        result_df2 = extract_text_content(df2, {"text_field": "text"})
        assert "Combined_Text" in result_df2.columns
        # Since 'text' field is missing and no standard fields match, it joins all values
        assert result_df2.iloc[0]["Combined_Text"] == "Val1"

    def test_load_uploaded_file_csv(self):
        # Mock uploaded file object
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        # Since pandas read_csv expects a file-like object or path
        # If we pass MagicMock, read_csv might fail.
        # We should patch pd.read_csv
        
        with patch("services.dataset_service.pd.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({"text": ["abc"]})
            df = load_uploaded_file(mock_file)
            assert "Combined_Text" in df.columns
            assert df.iloc[0]["Combined_Text"] == "abc"

    def test_load_uploaded_file_json(self):
        mock_file = MagicMock()
        mock_file.name = "test.json"
        mock_file.read.return_value = b'[{"text": "abc"}]'
        
        df = load_uploaded_file(mock_file)
        assert "Combined_Text" in df.columns
        assert df.iloc[0]["Combined_Text"] == "abc"
