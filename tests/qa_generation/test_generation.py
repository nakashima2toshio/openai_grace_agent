import pytest
from unittest.mock import MagicMock
from qa_generation.generation import QAGenerator
from models import QAPairsResponse, QAPair

def test_generate_for_chunk_success():
    mock_client = MagicMock()
    # Mock token count
    mock_client.count_tokens.return_value = 100
    
    # Mock structured response
    mock_qa = QAPair(
        question="Test Question?",
        answer="Test Answer.",
        question_type="fact",
        difficulty="easy",
        source_span="source"
    )
    mock_response = QAPairsResponse(qa_pairs=[mock_qa])
    mock_client.generate_structured.return_value = mock_response
    
    generator = QAGenerator(client=mock_client)
    chunk = {"text": "Source text content.", "id": "c1", "doc_id": "d1"}
    config = {"qa_per_chunk": 1, "lang": "en"}
    
    result = generator.generate_for_chunk(chunk, config)
    
    assert len(result) == 1
    assert result[0]["question"] == "Test Question?"
    assert result[0]["source_chunk_id"] == "c1"

def test_generate_for_chunk_fallback_failure():
    mock_client = MagicMock()
    mock_client.count_tokens.return_value = 100
    
    # Raise exception on first try
    mock_client.generate_structured.side_effect = Exception("Structured fail")
    # Raise exception on fallback text gen
    mock_client.generate_content.side_effect = Exception("Fallback fail")
    
    generator = QAGenerator(client=mock_client)
    chunk = {"text": "text", "id": "c1"}
    config = {"qa_per_chunk": 1, "lang": "en"}
    
    # It should log error and return empty list (since I changed it to re-raise, it should raise!)
    # Wait, did I verify the previous replace was effectively "raise"?
    # The replace instruction was "raise fallback_error".
    
    with pytest.raises(Exception) as excinfo:
        generator.generate_for_chunk(chunk, config)
    
    assert "Fallback fail" in str(excinfo.value)
