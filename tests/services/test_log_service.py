import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, mock_open
from services.log_service import (
    log_unanswered_question,
    load_unanswered_logs,
    clear_unanswered_logs
)

class TestLogService:

    @patch("services.log_service.UNANSWERED_LOG_FILE")
    @patch("services.log_service.LOG_DIR")
    @patch("builtins.open", new_callable=mock_open)
    @patch("services.log_service.csv.writer")
    def test_log_unanswered_question(self, mock_writer, mock_open_file, mock_log_dir, mock_log_file):
        mock_log_dir.exists.return_value = True
        mock_log_file.exists.return_value = True
        
        log_unanswered_question("query", ["coll1"], "reason")
        
        mock_open_file.assert_called()
        mock_writer.return_value.writerow.assert_called()

    @patch("services.log_service.pd.read_csv")
    @patch("services.log_service.UNANSWERED_LOG_FILE")
    @patch("services.log_service.LOG_DIR")
    def test_load_unanswered_logs(self, mock_log_dir, mock_log_file, mock_read_csv):
        mock_log_dir.exists.return_value = True
        mock_log_file.exists.return_value = True
        mock_log_file.stat.return_value.st_size = 100
        
        mock_read_csv.return_value = pd.DataFrame({
            "timestamp": ["2023-01-01"],
            "query": ["q"]
        })
        
        df = load_unanswered_logs()
        assert len(df) == 1
        assert df.iloc[0]["query"] == "q"

    @patch("services.log_service.UNANSWERED_LOG_FILE")
    @patch("builtins.open", new_callable=mock_open)
    def test_clear_unanswered_logs(self, mock_open_file, mock_log_file):
        clear_unanswered_logs()
        # Verify open was called
        assert mock_open_file.called
        # We can also check args more loosely if needed
        args, kwargs = mock_open_file.call_args
        assert args[0] == mock_log_file
        assert kwargs['mode'] == 'w'
