# tests/services/test_agent_service.py
import pytest
from unittest.mock import MagicMock, patch
from services.agent_service import ReActAgent, SYSTEM_INSTRUCTION_TEMPLATE

# Mock config
@pytest.fixture
def mock_config():
    with patch("services.agent_service.AgentConfig") as mock:
        yield mock

@pytest.fixture
def mock_genai():
    with patch("services.agent_service.genai") as mock:
        yield mock

@pytest.fixture
def mock_agent_tools():
    with patch("services.agent_service.search_rag_knowledge_base") as mock_search, \
         patch("services.agent_service.list_rag_collections") as mock_list:
        yield mock_search, mock_list

class TestReActAgent:

    def test_init(self, mock_genai, mock_config):
        """Test initialization of ReActAgent"""
        # Mock environment variables
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
            agent = ReActAgent(selected_collections=["coll1"], model_name="gemini-pro")
            
            assert agent.selected_collections == ["coll1"]
            assert agent.model_name == "gemini-pro"
            assert agent.thought_log == []
            
            # Verify genai.configure called
            mock_genai.configure.assert_called_with(api_key='test_key')
            # Verify GenerativeModel created
            mock_genai.GenerativeModel.assert_called()

    def test_init_missing_key(self, mock_genai):
        """Test initialization failure when API key is missing"""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API Key missing"):
                ReActAgent(selected_collections=[], model_name="gemini-pro")

    def test_execute_turn_simple_answer(self, mock_genai):
        """Test execute_turn where model returns an answer directly"""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
            agent = ReActAgent(selected_collections=[], model_name="gemini-pro")
            
            # Mock chat session response
            mock_chat = agent.chat_session
            mock_response = MagicMock()
            mock_part = MagicMock()
            mock_part.text = "Thought: I know the answer.\nAnswer: The answer is 42."
            mock_part.function_call = None
            mock_response.parts = [mock_part]
            mock_chat.send_message.return_value = mock_response

            # Mock reflection response
            mock_reflection_response = MagicMock()
            mock_reflection_response.text = "Final Answer: The answer is 42."
            mock_chat.send_message.side_effect = [mock_response, mock_reflection_response]

            events = list(agent.execute_turn("What is the meaning of life?"))
            
            # Check events
            event_types = [e["type"] for e in events]
            assert "log" in event_types
            assert "final_text" in event_types
            assert "final_answer" in event_types
            
            # Check final answer
            final_event = events[-1]
            assert final_event["type"] == "final_answer"
            assert final_event["content"] == "The answer is 42."

    def test_execute_turn_with_tool_call(self, mock_genai, mock_agent_tools):
        """Test execute_turn with a tool call"""
        mock_search, mock_list = mock_agent_tools
        
        # Patch TOOLS_MAP to use our mocks
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}), \
             patch.dict('services.agent_service.TOOLS_MAP', {
                 'search_rag_knowledge_base': mock_search,
                 'list_rag_collections': mock_list
             }):
            
            agent = ReActAgent(selected_collections=["coll1"], model_name="gemini-pro")
            mock_chat = agent.chat_session

            # 1. First response: Thought + Tool Call
            mock_response1 = MagicMock()
            mock_part1_text = MagicMock()
            mock_part1_text.text = "Thought: I need to search."
            mock_part1_text.function_call = None
            
            mock_part1_tool = MagicMock()
            mock_part1_tool.text = None
            mock_part1_tool.function_call.name = "search_rag_knowledge_base"
            mock_part1_tool.function_call.args = {"query": "test query", "collection_name": "coll1"}
            
            mock_response1.parts = [mock_part1_text, mock_part1_tool]

            # 2. Second response: Answer based on tool result
            mock_response2 = MagicMock()
            mock_part2 = MagicMock()
            mock_part2.text = "Thought: I found it.\nAnswer: The result is X."
            mock_part2.function_call = None
            mock_response2.parts = [mock_part2]

            # 3. Reflection response
            mock_reflection = MagicMock()
            mock_reflection.text = "Final Answer: The result is X."

            mock_chat.send_message.side_effect = [mock_response1, mock_response2, mock_reflection]
            
            # Mock tool execution
            mock_search.return_value = "Search Result Content"

            events = list(agent.execute_turn("Search for test."))
            
            # Verify tool was called
            mock_search.assert_called_with(query="test query", collection_name="coll1")
            
            # Verify events
            types = [e["type"] for e in events]
            assert "tool_call" in types
            assert "tool_result" in types
            assert "final_answer" in types
            
            # Verify logs
            assert "Thought: I need to search." in agent.thought_log[0]

    def test_format_final_answer(self, mock_genai):
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
            agent = ReActAgent(selected_collections=[], model_name="gemini-pro")
            
            assert agent._format_final_answer("Answer: Yes") == "Yes"
            assert agent._format_final_answer("Thought: Hmmm\nAnswer: Yes") == "Yes"
            assert agent._format_final_answer("Thought: Just a thought") == "Just a thought"
            assert agent._format_final_answer("考え: 日本語で") == "日本語で"
            assert agent._format_final_answer("Raw text") == "Raw text"
