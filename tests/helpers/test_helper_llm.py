"""
helper/helper_llm.py 単体テスト（openai プロバイダー）

このテストは openai_grace_agent 専用に新規作成したもの。
anthropic_grace_agent の helper テストは Anthropic/Gemini クライアントを
対象にしているため、そのままでは流用できない。

検証対象:
    - create_llm_client("openai") が OpenAIClient を返すこと
    - OpenAIClient.generate_content / generate_structured / count_tokens
    - モジュールレベルのトークン集計（reset_token_counter / get_token_counter /
      _token_accumulator）

OpenAI SDK は unittest.mock でモックするため、実 API キーは不要。
実 API テストは @pytest.mark.skipif で OPENAI_API_KEY 未設定時にスキップ。

[Usage]: pytest tests/helpers/test_helper_llm.py -v
"""

import os
import sys
from typing import List
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

# テスト対象のインポートパス解決
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helper.helper_llm import (
    OpenAIClient,
    _token_accumulator,  # noqa: E402
    create_llm_client,
    get_token_counter,
    reset_token_counter,
)


# テスト用 Pydantic モデル
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
    def test_create_openai_client(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper.helper_llm.OpenAI"):
                client = create_llm_client("openai")
                assert isinstance(client, OpenAIClient)

    def test_default_provider_is_openai(self):
        """provider 省略時は openai（DEFAULT_LLM_PROVIDER）になること"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper.helper_llm.OpenAI"):
                client = create_llm_client()
                assert isinstance(client, OpenAIClient)

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError):
            create_llm_client("invalid_provider")

    def test_openai_client_default_model(self):
        """OpenAIClient のデフォルトモデルが gpt-5-mini であること"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper.helper_llm.OpenAI"):
                client = create_llm_client("openai")
                assert client.default_model == "gpt-5-mini"


# ====================================
# OpenAIClient テスト
# ====================================

class TestOpenAIClient:
    @pytest.fixture
    def mock_openai_client(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper.helper_llm.OpenAI") as mock_class:
                mock_instance = Mock()
                mock_class.return_value = mock_instance
                client = OpenAIClient()
                return client, mock_instance

    def _make_response(self, *, content=None, parsed=None, usage=None):
        mock_choice = Mock()
        if content is not None:
            mock_choice.message.content = content
        if parsed is not None:
            mock_choice.message.parsed = parsed
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = usage
        return mock_response

    def test_methods_exist(self, mock_openai_client):
        """generate_content / generate_structured / count_tokens が存在すること"""
        client, _ = mock_openai_client
        assert callable(getattr(client, "generate_content", None))
        assert callable(getattr(client, "generate_structured", None))
        assert callable(getattr(client, "count_tokens", None))

    def test_generate_content(self, mock_openai_client):
        client, mock_instance = mock_openai_client
        mock_instance.chat.completions.create.return_value = self._make_response(
            content="Hello, world!"
        )

        result = client.generate_content("Say hello")
        assert result == "Hello, world!"
        # Responses/Chat: chat.completions.create が使われていること
        assert mock_instance.chat.completions.create.called

    def test_generate_content_passes_kwargs(self, mock_openai_client):
        client, mock_instance = mock_openai_client
        mock_instance.chat.completions.create.return_value = self._make_response(
            content="Response"
        )

        client.generate_content("Question", extra_param="test")
        call_args = mock_instance.chat.completions.create.call_args
        assert call_args.kwargs["extra_param"] == "test"

    def test_generate_content_max_tokens_converted(self, mock_openai_client):
        """max_tokens は max_completion_tokens に変換されること（gpt-5 系仕様）"""
        client, mock_instance = mock_openai_client
        mock_instance.chat.completions.create.return_value = self._make_response(
            content="ok"
        )

        client.generate_content("Q", max_tokens=123)
        call_kwargs = mock_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("max_completion_tokens") == 123
        assert "max_tokens" not in call_kwargs

    def test_generate_content_system_to_messages(self, mock_openai_client):
        """system= は messages 先頭の system ロールに変換されること"""
        client, mock_instance = mock_openai_client
        mock_instance.chat.completions.create.return_value = self._make_response(
            content="ok"
        )

        client.generate_content("Q", system="You are helpful.")
        messages = mock_instance.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."

    def test_generate_structured(self, mock_openai_client):
        client, mock_instance = mock_openai_client
        parsed = MockResponseSchema(message="test", score=100)
        mock_instance.beta.chat.completions.parse.return_value = self._make_response(
            parsed=parsed
        )

        result = client.generate_structured("Generate test", MockResponseSchema)
        assert result.message == "test"
        assert result.score == 100
        # response_format に Pydantic モデルが渡されていること
        call_kwargs = mock_instance.beta.chat.completions.parse.call_args.kwargs
        assert call_kwargs["response_format"] is MockResponseSchema

    def test_count_tokens(self, mock_openai_client):
        client, _ = mock_openai_client
        with patch("helper.helper_llm.tiktoken") as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
            mock_tiktoken.encoding_for_model.return_value = mock_encoding
            mock_tiktoken.get_encoding.return_value = mock_encoding

            count = client.count_tokens("Hello world")
            assert count == 5


# ====================================
# トークン集計テスト
# ====================================

class TestTokenAccumulator:
    def test_reset_and_get(self):
        """reset_token_counter / get_token_counter の往復"""
        _token_accumulator["input_tokens"] = 42
        _token_accumulator["output_tokens"] = 7

        snapshot = get_token_counter()
        assert snapshot == {"input_tokens": 42, "output_tokens": 7}
        # get_token_counter はコピーを返す（内部状態を共有しない）
        snapshot["input_tokens"] = 0
        assert _token_accumulator["input_tokens"] == 42

        reset_token_counter()
        assert get_token_counter() == {"input_tokens": 0, "output_tokens": 0}

    def test_generate_content_accumulates_usage(self):
        """generate_content の usage がアキュムレータに加算されること"""
        reset_token_counter()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("helper.helper_llm.OpenAI") as mock_class:
                mock_instance = Mock()
                mock_class.return_value = mock_instance

                usage = Mock()
                usage.prompt_tokens = 11
                usage.completion_tokens = 5
                mock_choice = Mock()
                mock_choice.message.content = "ok"
                mock_response = Mock()
                mock_response.choices = [mock_choice]
                mock_response.usage = usage
                mock_instance.chat.completions.create.return_value = mock_response

                client = OpenAIClient()
                client.generate_content("Q")

        counter = get_token_counter()
        assert counter["input_tokens"] == 11
        assert counter["output_tokens"] == 5
        reset_token_counter()


# ====================================
# 実 API テスト（OPENAI_API_KEY 設定時のみ）
# ====================================

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY 未設定のため実 API テストをスキップ",
)
class TestOpenAIClientRealAPI:
    def test_count_tokens_real(self):
        client = create_llm_client("openai")
        assert client.count_tokens("Hello world") > 0
