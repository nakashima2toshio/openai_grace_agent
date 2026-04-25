"""
helper_embedding.py 単体テスト

テスト実行:
    pytest tests/helpers/test_helper_embedding.py -v
"""

import pytest
import os
from typing import List
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helper_embedding import (
    EmbeddingClient,
    OpenAIEmbedding,
    GeminiEmbedding,
    create_embedding_client,
    get_embedding_dimensions,
)


class TestCreateEmbeddingClient:
    """create_embedding_client ファクトリ関数のテスト"""

    def test_create_gemini_client(self):
        """Geminiクライアント生成"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_embedding.genai") as mock_genai:
                client = create_embedding_client("gemini")
                assert isinstance(client, GeminiEmbedding)

    def test_create_openai_client(self):
        """OpenAIクライアント生成"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper_embedding.OpenAI") as mock_openai:
                client = create_embedding_client("openai")
                assert isinstance(client, OpenAIEmbedding)

    def test_invalid_provider(self):
        """不正なプロバイダー指定でエラー"""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_embedding_client("invalid_provider")


class TestOpenAIEmbedding:
    """OpenAIEmbedding クラスのテスト"""

    @pytest.fixture
    def mock_openai_client(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper_embedding.OpenAI") as mock_class:
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                client = OpenAIEmbedding()
                return client, mock_instance

    def test_embed_text(self, mock_openai_client):
        """単一テキストEmbedding"""
        client, mock_instance = mock_openai_client
        
        mock_data = Mock()
        mock_data.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_data]
        mock_instance.embeddings.create.return_value = mock_response

        result = client.embed_text("Hello")

        assert len(result) == 1536
        assert result[0] == 0.1
        mock_instance.embeddings.create.assert_called_once()

    def test_embed_texts(self, mock_openai_client):
        """バッチEmbedding"""
        client, mock_instance = mock_openai_client
        
        # モックレスポンス（3件分）
        mock_data1 = Mock()
        mock_data1.embedding = [0.1] * 1536
        mock_data1.index = 0
        
        mock_data2 = Mock()
        mock_data2.embedding = [0.2] * 1536
        mock_data2.index = 1
        
        mock_data3 = Mock()
        mock_data3.embedding = [0.3] * 1536
        mock_data3.index = 2

        mock_response = Mock()
        mock_response.data = [mock_data1, mock_data2, mock_data3]
        mock_instance.embeddings.create.return_value = mock_response

        result = client.embed_texts(["Hello", "World", "Test"])

        assert len(result) == 3
        assert len(result[0]) == 1536
        assert result[0][0] == 0.1
        assert result[1][0] == 0.2


class TestGeminiEmbedding:
    """GeminiEmbedding クラスのテスト"""

    @pytest.fixture
    def mock_gemini_client(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("helper_embedding.genai") as mock_genai:
                mock_instance = Mock()
                mock_genai.Client.return_value = mock_instance
                client = GeminiEmbedding()
                return client, mock_instance

    def test_embed_text(self, mock_gemini_client):
        """単一テキストEmbedding"""
        client, mock_instance = mock_gemini_client
        
        mock_embedding = Mock()
        mock_embedding.values = [0.1] * 3072
        mock_response = Mock()
        mock_response.embeddings = [mock_embedding]
        mock_instance.models.embed_content.return_value = mock_response

        result = client.embed_text("Hello")

        assert len(result) == 3072
        assert result[0] == 0.1
        mock_instance.models.embed_content.assert_called_once()

    def test_embed_texts(self, mock_gemini_client):
        """バッチEmbedding"""
        client, mock_instance = mock_gemini_client

        # モックレスポンス（3件分）
        mock_embedding = Mock()
        mock_embedding.values = [0.1] * 3072
        
        mock_response = Mock()
        # The code expects `response.embeddings` to be a list where each item has `.values`
        # Since we send 3 texts, we expect 3 embeddings in the response
        mock_response.embeddings = [mock_embedding, mock_embedding, mock_embedding]
        
        mock_instance.models.embed_content.return_value = mock_response

        result = client.embed_texts(["Hello", "World", "Test"])

        assert len(result) == 3
        assert len(result[0]) == 3072


class TestHelpers:
    """ヘルパー関数のテスト"""

    def test_get_embedding_dimensions(self):
        assert get_embedding_dimensions("gemini") == 3072
        assert get_embedding_dimensions("openai") == 1536
        
        with pytest.raises(ValueError):
            get_embedding_dimensions("invalid")
