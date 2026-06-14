# tests/services/test_agent_service.py
#
# [MIGRATION anthropic(gemini)→openai]
# 元の anthropic 版テストは Gemini SDK (genai / chat_session / function_call /
# response.parts) を前提にしていたが、openai 版 ReActAgent は
#   - create_llm_client("openai") で LLM クライアントを生成
#   - self.llm.generate_with_tools(messages, tools, system, ...) が
#     (text, tool_calls, finish_reason) のタプルを返す
#   - 会話履歴は self._messages で自前管理
#   - ツール呼び出しは finish_reason == "tool_calls" + tool_calls リストで検出
#   - 検索は search_rag_knowledge_base_cached() 経由で実行
# という実装に変わっているため、openai 版の実 API に合わせて全面的に書き換えている。
from unittest.mock import MagicMock, patch

import pytest

from services.agent_service import ReActAgent


@pytest.fixture
def mock_llm():
    """create_llm_client をモック化し、LLM クライアントを差し替える"""
    with patch("services.agent_service.create_llm_client") as mock_create:
        client = MagicMock()
        mock_create.return_value = client
        yield client


class TestReActAgent:

    def test_init(self, mock_llm):
        """ReActAgent の初期化"""
        agent = ReActAgent(selected_collections=["coll1"], model_name="gpt-5-mini")

        assert agent.selected_collections == ["coll1"]
        assert agent.model_name == "gpt-5-mini"
        assert agent.thought_log == []
        # openai 版は create_llm_client("openai", ...) で LLM を生成する
        assert agent.llm is mock_llm
        # system instruction / tools が事前構築されること
        assert isinstance(agent.system_instruction, str)
        assert any(t["name"] == "search_rag_knowledge_base" for t in agent.tools)

    def test_execute_turn_simple_answer(self, mock_llm):
        """ツールを使わず直接回答するケース（ReAct → Reflection）"""
        agent = ReActAgent(selected_collections=[], model_name="gpt-5-mini")
        # KeywordExtractor の影響を排除（環境依存を避ける）
        agent.keyword_extractor = None

        # 1回目: ReAct ループ（finish_reason != "tool_calls" で即終了）
        react_resp = (
            "Thought: I know the answer.\nAnswer: The answer is 42.",
            [],
            "stop",
        )
        # 2回目: Reflection フェーズ
        reflection_resp = (
            "Thought: looks good.\nFinal Answer: The answer is 42.",
            [],
            "stop",
        )
        mock_llm.generate_with_tools.side_effect = [react_resp, reflection_resp]

        events = list(agent.execute_turn("What is the meaning of life?"))

        event_types = [e["type"] for e in events]
        assert "log" in event_types
        assert "final_text" in event_types
        assert "final_answer" in event_types

        final_event = events[-1]
        assert final_event["type"] == "final_answer"
        assert final_event["content"] == "The answer is 42."

    def test_execute_turn_with_tool_call(self, mock_llm):
        """ツール呼び出しを含むケース"""
        agent = ReActAgent(selected_collections=["coll1"], model_name="gpt-5-mini")
        agent.keyword_extractor = None

        # 1回目: Thought + Tool Call（finish_reason == "tool_calls"）
        resp1 = (
            "Thought: I need to search.",
            [{"id": "call_1", "name": "search_rag_knowledge_base",
              "input": {"query": "test query"}}],
            "tool_calls",
        )
        # 2回目: ツール結果を踏まえた回答（ReAct ループ終了）
        resp2 = (
            "Thought: I found it.\nAnswer: The result is X.",
            [],
            "stop",
        )
        # 3回目: Reflection
        resp3 = (
            "Thought: fine.\nFinal Answer: The result is X.",
            [],
            "stop",
        )
        mock_llm.generate_with_tools.side_effect = [resp1, resp2, resp3]

        # 検索は search_rag_knowledge_base_cached() 経由で実行される
        with patch("services.agent_service.search_rag_knowledge_base_cached") as mock_search:
            mock_search.return_value = "Search Result Content"
            events = list(agent.execute_turn("Search for test."))

        # ツールが呼ばれたこと
        mock_search.assert_called_once()
        _, kwargs = mock_search.call_args
        assert kwargs["query"] == "test query"

        types = [e["type"] for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "final_answer" in types

        # 思考ログに Thought が記録されること
        assert any("Thought: I need to search." in t for t in agent.thought_log)

    def test_format_final_answer(self, mock_llm):
        agent = ReActAgent(selected_collections=[], model_name="gpt-5-mini")

        assert agent._format_final_answer("Answer: Yes") == "Yes"
        assert agent._format_final_answer("Thought: Hmmm\nAnswer: Yes") == "Yes"
        assert agent._format_final_answer("Thought: Just a thought") == "Just a thought"
        assert agent._format_final_answer("考え: 日本語で") == "日本語で"
        assert agent._format_final_answer("Raw text") == "Raw text"
