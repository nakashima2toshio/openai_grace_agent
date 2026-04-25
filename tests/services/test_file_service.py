import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
from services.file_service import (
    load_qa_output_history,
    load_preprocessed_history,
    save_to_output,
    load_sample_questions_from_csv
)

class TestFileService:

    @patch("services.file_service.Path.glob")
    @patch("services.file_service.Path.exists")
    def test_load_qa_output_history(self, mock_exists, mock_glob):
        mock_exists.return_value = True
        
        # Mock file
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        mock_file.stat.return_value.st_size = 100
        mock_file.stat.return_value.st_mtime = 100000
        
        mock_glob.return_value = [mock_file]
        
        df = load_qa_output_history()
        assert len(df) == 1
        assert df.iloc[0]["ファイル名"] == "test.csv"

    @patch("services.file_service.Path.glob")
    @patch("services.file_service.Path.exists")
    def test_load_preprocessed_history(self, mock_exists, mock_glob):
        mock_exists.return_value = True
        
        mock_file = MagicMock()
        mock_file.name = "preprocessed_test.csv"
        mock_file.stem = "preprocessed_test"
        mock_file.stat.return_value.st_size = 100
        mock_file.stat.return_value.st_mtime = 100000
        
        mock_glob.return_value = [mock_file]
        
        df = load_preprocessed_history()
        assert len(df) == 1
        assert df.iloc[0]["データセット名"] == "test"

    @patch("services.file_service.pd.DataFrame.to_csv")
    @patch("builtins.open", new_callable=mock_open)
    @patch("services.file_service.Path.mkdir")
    def test_save_to_output(self, mock_mkdir, mock_open_file, mock_to_csv):
        df = pd.DataFrame({"Combined_Text": ["text1"]})
        
        result = save_to_output(df, "dataset_type")
        
        assert "csv" in result
        assert "txt" in result
        assert "json" in result
        mock_to_csv.assert_called()
        assert mock_open_file.called

    @patch("services.file_service.map_collection_to_csv")
    @patch("services.file_service.pd.read_csv")
    @patch("services.file_service.Path.exists")
    def test_load_sample_questions_from_csv(self, mock_exists, mock_read_csv, mock_map):
        mock_map.return_value = "file.csv"
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({"question": ["q1", "q2", "q3", "q4"]})
        
        questions = load_sample_questions_from_csv("coll", num_samples=2)
        assert len(questions) == 2
