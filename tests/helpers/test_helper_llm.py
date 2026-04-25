"""
helper_llm.py 単体テスト

テスト実行:
    pytest tests/helpers/test_helper_llm.py -v
"""

import pytest
import os
from typing import List
from unittest.mock import Mock, patch, MagicMock

from pydantic import BaseModel
import sys

# テスト対象のインポートパス解決
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import helper_llm # Import the module to inspect
from helper_llm import (
    LLMClient,
    OpenAIClient,
    GeminiClient,
    create_llm_client,
)

# テスト用Pydanticモデル
class MockResponseSchema(BaseModel):
    message: str
    score: int

class QAPair(BaseModel):
    question: str
    answer: str

class QAPairsResponse(BaseModel):
    qa_pairs: List[QAPair]

# ====================================
# ファクトリ関数テスト
# ====================================

class TestCreateLLMClient:
    def test_create_gemini_client(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_llm.genai"):
                client = create_llm_client("gemini")
                assert isinstance(client, GeminiClient)

    def test_create_openai_client(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper_llm.OpenAI"):
                client = create_llm_client("openai")
                assert isinstance(client, OpenAIClient)

    def test_invalid_provider(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
             with patch("helper_llm.genai"):
                client = create_llm_client("invalid_provider")
                assert isinstance(client, GeminiClient)

# ====================================
# OpenAIClient テスト
# ====================================

class TestOpenAIClient:
    @pytest.fixture
    def mock_openai_client(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper_llm.OpenAI") as mock_class:
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                client = OpenAIClient()
                return client, mock_instance

    def test_generate_content(self, mock_openai_client):
        client, mock_instance = mock_openai_client
        mock_choice = Mock()
        mock_choice.message.content = "Hello, world!"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_response

        result = client.generate_content("Say hello")
        assert result == "Hello, world!"

    def test_generate_content_with_system_instruction(self, mock_openai_client):
        client, mock_instance = mock_openai_client
        mock_choice = Mock()
        mock_choice.message.content = "Response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_response
        
        client.generate_content("Question", extra_param="test")
        call_args = mock_instance.chat.completions.create.call_args
        assert call_args.kwargs["extra_param"] == "test"

    def test_generate_structured(self, mock_openai_client):
        client, mock_instance = mock_openai_client
        mock_choice = Mock()
        mock_choice.message.parsed = MockResponseSchema(message="test", score=100)
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_instance.beta.chat.completions.parse.return_value = mock_response

        result = client.generate_structured("Generate test", MockResponseSchema)
        assert result.message == "test"
        assert result.score == 100

    def test_count_tokens(self, mock_openai_client):
        client, _ = mock_openai_client
        with patch("helper_llm.tiktoken") as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
            mock_tiktoken.encoding_for_model.return_value = mock_encoding
            mock_tiktoken.get_encoding.return_value = mock_encoding

            count = client.count_tokens("Hello world")
            assert count == 5

# ====================================
# GeminiClient テスト
# ====================================

class TestGeminiClient:

    @pytest.fixture
    def mock_gemini_client(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            # Patch google.generativeai directly
            # Note: We must patch where it is defined/imported from if we want to affect all usages,
            # but patching the library itself is safest if we can.
            
            with patch("google.generativeai.GenerativeModel") as mock_model_class, \
                 patch("google.generativeai.configure"):
                
                # Mock GenerativeModel instance
                mock_model_instance = Mock()
                mock_model_class.return_value = mock_model_instance
                
                client = GeminiClient()
                yield client, mock_model_instance

    def test_generate_content(self, mock_gemini_client):
        client, mock_model = mock_gemini_client
        
        mock_response = Mock()
        mock_response.text = "こんにちは"
        mock_model.generate_content.return_value = mock_response

        result = client.generate_content("Hello")
        assert result == "こんにちは"

    def test_generate_content_with_kwargs(self, mock_gemini_client):
        client, mock_model = mock_gemini_client
        
        mock_response = Mock()
        mock_response.text = "Response"
        mock_model.generate_content.return_value = mock_response

        client.generate_content("Question", thinking_level="high")
        
        args, kwargs = mock_model.generate_content.call_args
        assert kwargs["thinking_level"] == "high"

    def test_generate_structured(self, mock_gemini_client):
        client, mock_model = mock_gemini_client
        
        mock_response = Mock()
        mock_response.text = '{"message": "test", "score": 100}'
        mock_model.generate_content.return_value = mock_response

        result = client.generate_structured("Generate", MockResponseSchema)
        assert result.message == "test"
        assert result.score == 100

    def test_generate_structured_invalid_json(self, mock_gemini_client):
        client, mock_model = mock_gemini_client
        
        mock_response = Mock()
        mock_response.text = "not valid json"
        mock_model.generate_content.return_value = mock_response

        with pytest.raises(Exception): 
            client.generate_structured("Generate", MockResponseSchema)

    def test_count_tokens(self, mock_gemini_client):
        client, mock_model = mock_gemini_client
        
        mock_response = Mock()
        mock_response.total_tokens = 10
        mock_model.count_tokens.return_value = mock_response

        count = client.count_tokens("Hello world")
        assert count == 10