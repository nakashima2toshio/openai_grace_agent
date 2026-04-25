import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from qa_generation.evaluation import analyze_coverage

@patch("qa_generation.evaluation.SemanticCoverage")
def test_analyze_coverage(mock_semantic_coverage_cls):
    # Setup mock analyzer
    mock_analyzer = mock_semantic_coverage_cls.return_value
    
    # Mock embeddings (2 chunks, 2 QA pairs) -> dimension 2
    mock_analyzer.generate_embeddings.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    mock_analyzer.generate_embeddings_batch.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    # Mock cosine similarity
    # chunk0-qa0: 1.0 (perfect)
    # chunk0-qa1: 0.0
    # chunk1-qa0: 0.0
    # chunk1-qa1: 1.0
    def side_effect_similarity(v1, v2):
        return float(np.dot(v1, v2))
    
    mock_analyzer.cosine_similarity.side_effect = side_effect_similarity
    
    chunks = [{"id": "c1", "text": "t1"}, {"id": "c2", "text": "t2"}]
    qa_pairs = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
    
    result = analyze_coverage(chunks, qa_pairs, custom_threshold=0.5)
    
    assert result["coverage_rate"] == 1.0
    assert result["covered_chunks"] == 2
    assert result["total_chunks"] == 2
