"""
helper/helper_embedding.py 単体テスト（openai プロバイダー）

このテストは openai_grace_agent 専用に新規作成したもの。
anthropic_grace_agent の helper テストは Gemini Embedding を対象にしているため流用不可。

検証対象:
    - create_embedding_client("openai") が OpenAIEmbedding を返すこと
    - text-embedding-3-large のデフォルト次元数が 3072 であること
    - embed_text / embed_texts がモック SDK を介して正しく動作すること
    - get_embedding_dimensions("openai") == 3072

OpenAI SDK は unittest.mock でモックするため、実 API キーは不要。

[Usage]: pytest tests/helpers/test_helper_embedding.py -v
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helper.helper_embedding import (
    DEFAULT_OPENAI_EMBEDDING_DIMS,
    OpenAIEmbedding,
    create_embedding_client,
    get_embedding_dimensions,
)


class TestCreateEmbeddingClient:
    """create_embedding_client ファクトリ関数のテスト"""

    def test_create_openai_client(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper.helper_embedding.OpenAI"):
                client = create_embedding_client("openai")
                assert isinstance(client, OpenAIEmbedding)

    def test_default_provider_is_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper.helper_embedding.OpenAI"):
                client = create_embedding_client()
                assert isinstance(client, OpenAIEmbedding)

    def test_none_provider_defaults_to_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper.helper_embedding.OpenAI"):
                client = create_embedding_client(None)
                assert isinstance(client, OpenAIEmbedding)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            create_embedding_client("nope")


class TestEmbeddingDimensions:
    """次元数（3072 / text-embedding-3-large）の検証"""

    def test_default_openai_dims_constant(self):
        assert DEFAULT_OPENAI_EMBEDDING_DIMS == 3072

    def test_get_embedding_dimensions_openai(self):
        assert get_embedding_dimensions("openai") == 3072

    def test_get_embedding_dimensions_none_defaults_openai(self):
        assert get_embedding_dimensions(None) == 3072

    def test_client_dimensions_property(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper.helper_embedding.OpenAI"):
                client = OpenAIEmbedding()
                assert client.dimensions == 3072

    def test_default_model_is_3_large(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper.helper_embedding.OpenAI"):
                client = OpenAIEmbedding()
                assert client.model == "text-embedding-3-large"


class TestOpenAIEmbedding:
    """OpenAIEmbedding の埋め込み生成（モック SDK）"""

    @pytest.fixture
    def mock_embedding_client(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper.helper_embedding.OpenAI") as mock_class:
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                client = OpenAIEmbedding()
                return client, mock_instance

    def test_embed_text(self, mock_embedding_client):
        client, mock_instance = mock_embedding_client
        vec = [0.1] * 3072
        item = Mock()
        item.embedding = vec
        response = Mock()
        response.data = [item]
        mock_instance.embeddings.create.return_value = response

        result = client.embed_text("Hello world")
        assert len(result) == 3072
        # dimensions=3072 を明示指定して呼び出していること
        call_kwargs = mock_instance.embeddings.create.call_args.kwargs
        assert call_kwargs["dimensions"] == 3072
        assert call_kwargs["model"] == "text-embedding-3-large"

    def test_embed_texts_batch_sorted_by_index(self, mock_embedding_client):
        client, mock_instance = mock_embedding_client

        # index 順がレスポンスで前後しても、index でソートされること
        item0 = Mock()
        item0.index = 0
        item0.embedding = [0.0] * 3072
        item1 = Mock()
        item1.index = 1
        item1.embedding = [1.0] * 3072
        response = Mock()
        response.data = [item1, item0]  # わざと逆順
        mock_instance.embeddings.create.return_value = response

        results = client.embed_texts(["a", "b"], batch_size=100)
        assert len(results) == 2
        assert results[0][0] == 0.0  # index 0 が先頭
        assert results[1][0] == 1.0

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("helper.helper_embedding.OpenAI"):
                with pytest.raises(ValueError):
                    OpenAIEmbedding()
