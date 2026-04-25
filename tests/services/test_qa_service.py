import pytest
from unittest.mock import MagicMock, patch
from services.qa_service import (
    generate_qa_pairs,
    save_qa_pairs_to_file,
    run_advanced_qa_generation,
    QAPair
)

class TestQAService:

    @patch("services.qa_service.create_llm_client")
    def test_generate_qa_pairs(self, mock_create_client):
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        # Mock structured output
        mock_response = MagicMock()
        mock_qa = MagicMock()
        mock_qa.question = "Q1"
        mock_qa.answer = "A1"
        mock_qa.question_type = "factual"
        
        mock_response.qa_pairs = [mock_qa]
        mock_client.generate_structured.return_value = mock_response
        
        result = generate_qa_pairs("text", "dataset", "chunk1")
        
        assert len(result) == 1
        assert isinstance(result[0], QAPair)
        assert result[0].question == "Q1"
        assert result[0].source_chunk_id == "chunk1"

    @patch("services.qa_service.pd.DataFrame.to_csv")
    @patch("services.qa_service.json.dump")
    @patch("builtins.open")
    @patch("services.qa_service.Path.mkdir")
    def test_save_qa_pairs_to_file(self, mock_mkdir, mock_open_file, mock_json_dump, mock_to_csv):
        qa_pairs = [
            QAPair(question="Q", answer="A", question_type="T", source_chunk_id="C", dataset_type="D")
        ]
        
        result = save_qa_pairs_to_file(qa_pairs, "dataset_type")
        
        assert "csv" in result
        assert "json" in result
        mock_to_csv.assert_called()
        mock_json_dump.assert_called()

    @patch("sys.path")
    def test_run_advanced_qa_generation(self, mock_sys_path):
        # We cannot easily mock the import inside function without mocking sys.modules or using patch.dict
        # But we can assume qa_generator_runner is mocked if we patch it where it is imported or used.
        # Since the function does `import qa_generator_runner` inside, standard patching 'services.qa_service.qa_generator_runner' might not work if it's not global.
        # However, we can patch 'builtins.__import__' or use 'sys.modules'.
        
        # Simplified approach: Mocking the runner via sys.modules injection
        mock_runner_module = MagicMock()
        mock_runner_module.run_qa_generator.return_value = {"success": True}
        
        with patch.dict("sys.modules", {"qa_generator_runner": mock_runner_module}):
            result = run_advanced_qa_generation(
                dataset="ds", input_file=None, use_celery=False, celery_workers=1,
                batch_chunks=1, max_docs=1, merge_chunks=False, min_tokens=10, max_tokens=100,
                coverage_threshold=0.5, model="m", analyze_coverage=False, log_callback=MagicMock()
            )
            
            assert result["success"] is True
            mock_runner_module.run_qa_generator.assert_called()
