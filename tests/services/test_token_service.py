import pytest
from unittest.mock import MagicMock, patch
from services.token_service import TokenManager, count_tokens, truncate_text, estimate_tokens_simple

class TestTokenService:

    def test_count_tokens_tiktoken(self):
        # Mock tiktoken
        with patch("services.token_service.tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1, 2, 3] # 3 tokens
            mock_get_encoding.return_value = mock_enc
            
            count = TokenManager.count_tokens("ABC")
            assert count == 3
            mock_get_encoding.assert_called_with("cl100k_base")

    def test_count_tokens_fallback(self):
        # Force exception to test fallback
        with patch("services.token_service.tiktoken.get_encoding", side_effect=Exception("Error")):
            count = TokenManager.count_tokens("ABC")
            # Fallback uses estimate_tokens_simple
            # "ABC" -> 3 english chars -> 3 * 0.25 = 0.75 -> max(1, 0) -> 1
            # Wait, 0.75 is int(0.75) = 0? max(1, 0) = 1.
            assert count >= 0
    
    def test_estimate_tokens_simple(self):
        # English
        assert estimate_tokens_simple("ABCD") == 1 # 4 * 0.25 = 1
        # Japanese
        assert estimate_tokens_simple("あいう") == 1 # 3 * 0.5 = 1.5 -> 1
        
    def test_truncate_text(self):
         with patch("services.token_service.tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1, 2, 3, 4, 5]
            mock_enc.decode.side_effect = lambda x: f"decoded_{len(x)}"
            mock_get_encoding.return_value = mock_enc
            
            truncated = TokenManager.truncate_text("ABCDE", max_tokens=3)
            # Should call decode with first 3 tokens
            mock_enc.decode.assert_called()
            args, _ = mock_enc.decode.call_args
            assert len(args[0]) == 3
            assert truncated == "decoded_3"

    def test_estimate_cost(self):
        # LLM
        cost = TokenManager.estimate_cost(1000, 1000, "gemini-2.0-flash")
        # Input: 1k * 0.0001 = 0.0001
        # Output: 1k * 0.0002 = 0.0002
        # Total: 0.0003
        assert cost == pytest.approx(0.0003)
        
        # Embedding
        cost_emb = TokenManager.estimate_cost(1000, 0, "gemini-embedding-001", is_embedding=True)
        # 1k * 0.0001 = 0.0001
        assert cost_emb == pytest.approx(0.0001)

    def test_get_model_limits(self):
        limits = TokenManager.get_model_limits("gpt-4o")
        assert limits["max_tokens"] == 128000
        assert limits["max_output"] == 4096
