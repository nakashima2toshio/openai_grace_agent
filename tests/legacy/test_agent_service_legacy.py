#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_agent_service.py - Agent Service のテスト
=============================================

ReActAgentのユニットテストおよび統合テストを実装。
パリの世帯収入に関するWikipedia検索シナリオを含む。
"""

import os
from unittest.mock import MagicMock, patch, call, PropertyMock
from collections import namedtuple

import pytest

from services.agent_service import (
    ReActAgent,
    get_available_collections_from_qdrant_helper,
    TOOLS_MAP,
    SYSTEM_INSTRUCTION_TEMPLATE,
    REFLECTION_INSTRUCTION,
)
from agent_tools import (
    search_rag_knowledge_base,
    list_rag_collections,
    RAGToolError,
    QdrantConnectionError,
    CollectionNotFoundError,
)


# モックGeminiレスポンス用のヘルパークラス
Part = namedtuple('Part', ['text', 'function_call'])
FunctionCall = namedtuple('FunctionCall', ['name', 'args'])

class Response:
    """モックレスポンスクラス（.text属性を持つ）"""
    def __init__(self, parts):
        self.parts = parts
        # partsから最初のテキストを.text属性として設定
        self.text = ""
        for part in parts:
            if part.text:
                self.text = part.text
                break


class TestReActAgent:
    """ReActAgentクラスのユニットテスト"""

    @pytest.fixture
    def mock_genai(self):
        """Google Generative AIのモック"""
        with patch('services.agent_service.genai') as mock:
            yield mock

    @pytest.fixture
    def mock_env(self):
        """環境変数のモック"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
            yield

    @pytest.fixture
    def mock_qdrant_client(self):
        """Qdrantクライアントのモック"""
        with patch('services.agent_service.QdrantClient') as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client

            # コレクションの設定
            Collection = namedtuple('Collection', ['name'])
            mock_client.get_collections.return_value.collections = [
                Collection(name='wikipedia_ja'),
                Collection(name='livedoor'),
                Collection(name='cc_news'),
            ]
            yield mock_client

    @pytest.fixture
    def agent(self, mock_genai, mock_env, mock_qdrant_client):
        """テスト用エージェントインスタンス"""
        # ChatSessionのモック
        mock_chat = MagicMock()
        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        mock_genai.GenerativeModel.return_value = mock_model

        agent = ReActAgent(
            selected_collections=['wikipedia_ja'],
            model_name='gemini-pro'
        )
        agent.chat_session = mock_chat
        return agent

    def test_init_without_api_key(self):
        """APIキーなしの初期化エラーテスト"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API Key missing"):
                ReActAgent(['wikipedia_ja'], 'gemini-pro')

    def test_init_with_collections(self, mock_genai, mock_env):
        """コレクション付き初期化テスト"""
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        agent = ReActAgent(
            selected_collections=['wikipedia_ja', 'livedoor'],
            model_name='gemini-pro'
        )

        assert agent.selected_collections == ['wikipedia_ja', 'livedoor']
        assert agent.model_name == 'gemini-pro'
        assert agent.thought_log == []

    def test_setup_session(self, mock_genai, mock_env):
        """セッションセットアップのテスト"""
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        mock_genai.GenerativeModel.return_value = mock_model

        agent = ReActAgent(['wikipedia_ja'], 'gemini-pro')

        # GenerativeModelの呼び出し確認
        mock_genai.GenerativeModel.assert_called_once()
        call_args = mock_genai.GenerativeModel.call_args

        assert call_args.kwargs['model_name'] == 'gemini-pro'
        assert 'tools' in call_args.kwargs
        assert 'system_instruction' in call_args.kwargs

    def test_format_final_answer(self, agent):
        """最終回答フォーマットのテスト"""
        # Answer:形式
        result = agent._format_final_answer("Thought: 考えています\nAnswer: これが答えです")
        assert result == "これが答えです"

        # Thought:形式
        result = agent._format_final_answer("Thought: これが答えです")
        assert result == "これが答えです"

        # 通常テキスト
        result = agent._format_final_answer("これが答えです")
        assert result == "これが答えです"

    def test_execute_turn_simple_answer(self, agent):
        """単純な回答のターン実行テスト"""
        # モックレスポンスの設定
        response = Response(parts=[
            Part(text="Thought: 簡単な質問ですね\nAnswer: こんにちは！", function_call=None)
        ])
        agent.chat_session.send_message.return_value = response

        # ターン実行
        events = list(agent.execute_turn("こんにちは"))

        # イベント確認
        assert any(e['type'] == 'log' for e in events)
        assert any(e['type'] == 'final_answer' for e in events)

        final = [e for e in events if e['type'] == 'final_answer'][0]
        assert "こんにちは" in final['content']

    @patch('services.agent_service.TOOLS_MAP')
    def test_execute_turn_with_tool_call(self, mock_tools, agent):
        """ツール呼び出しを含むターン実行テスト"""
        # ツール結果のモック
        mock_tools['search_rag_knowledge_base'].return_value = (
            "Result 1 (Score: 0.85):\n"
            "Q: パリの世帯収入は？\n"
            "A: パリ市の平均世帯収入は約4万ユーロです。\n"
            "Source: wikipedia"
        )

        # レスポンスの設定
        tool_call = FunctionCall(
            name='search_rag_knowledge_base',
            args={'query': 'パリ 世帯収入', 'collection_name': 'wikipedia_ja'}
        )

        responses = [
            Response(parts=[Part(text="Thought: 検索が必要です", function_call=tool_call)]),
            Response(parts=[Part(text="Answer: パリの平均世帯収入は約4万ユーロです", function_call=None)])
        ]

        agent.chat_session.send_message.side_effect = responses

        # ターン実行
        events = list(agent.execute_turn("パリの世帯収入は？"))

        # イベント確認
        tool_events = [e for e in events if e['type'] == 'tool_call']
        assert len(tool_events) == 1
        assert tool_events[0]['name'] == 'search_rag_knowledge_base'

    def test_execute_reflection_phase(self, agent):
        """リフレクションフェーズのテスト"""
        draft = "パリの世帯収入は高いです"

        reflection_response = Response(parts=[
            Part(
                text="Thought: もう少し具体的にすべきです\nFinal Answer: パリの平均世帯収入は約4万ユーロで、フランス全体の平均より高いです",
                function_call=None
            )
        ])

        agent.chat_session.send_message.return_value = reflection_response

        # リフレクション実行（ジェネレータの戻り値を取得）
        events = []
        final = ""
        gen = agent._execute_reflection_phase(draft)

        try:
            while True:
                event = next(gen)
                events.append(event)
        except StopIteration as e:
            # ジェネレータのreturn値を取得
            final = e.value

        assert "フランス全体の平均より高い" in final

    def test_execute_turn_with_error(self, agent):
        """エラー処理のテスト"""
        # ツールエラーのシミュレーション
        tool_call = FunctionCall(
            name='search_rag_knowledge_base',
            args={'query': 'test', 'collection_name': 'nonexistent'}
        )

        # TOOLS_MAPの辞書アイテムを直接パッチ
        with patch.dict('services.agent_service.TOOLS_MAP',
                       {'search_rag_knowledge_base': MagicMock(side_effect=RAGToolError("Collection not found")),
                        'list_rag_collections': MagicMock()}):

            responses = [
                Response(parts=[Part(text="Thought: 検索します", function_call=tool_call)]),
                Response(parts=[Part(text="Answer: エラーが発生しました", function_call=None)])
            ]

            agent.chat_session.send_message.side_effect = responses

            events = list(agent.execute_turn("テスト"))

            # エラーメッセージの確認
            tool_results = [e for e in events if e['type'] == 'tool_result']
            assert len(tool_results) > 0
            assert any("エラー" in str(e['content']) for e in tool_results)


class TestWikipediaSearchScenario:
    """Wikipedia検索シナリオの統合テスト"""

    @pytest.fixture
    def mock_qdrant_search(self):
        """Qdrant検索結果のモック"""
        def search_mock(query, collection_name=None):
            if "パリ" in query and "収入" in query:
                return (
                    "Result 1 (Score: 0.92):\n"
                    "Q: パリ市の平均世帯所得はどのくらいですか？\n"
                    "A: パリ市の平均世帯所得は約42,000ユーロ（2020年）で、これはフランス全体の平均36,000ユーロを上回っています。\n"
                    "Source: wikipedia_ja\n\n"
                    "Result 2 (Score: 0.88):\n"
                    "Q: 日本の平均世帯所得は？\n"
                    "A: 日本の平均世帯所得は約545万円（2021年）です。これは約38,000ユーロに相当します。\n"
                    "Source: wikipedia_ja"
                )
            elif "フランス" in query and "収入" in query:
                return (
                    "Result 1 (Score: 0.90):\n"
                    "Q: フランスの平均世帯所得は？\n"
                    "A: フランス全体の平均世帯所得は約36,000ユーロ（2020年）です。\n"
                    "Source: wikipedia_ja"
                )
            else:
                return "[[NO_RAG_RESULT]] 検索結果が見つかりませんでした。"

        with patch('agent_tools.search_rag_knowledge_base', side_effect=search_mock):
            yield search_mock

    @pytest.fixture
    def integrated_agent(self, mock_qdrant_search):
        """統合テスト用エージェント"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            with patch('services.agent_service.genai') as mock_genai:
                # モックの設定
                mock_chat = MagicMock()
                mock_model = MagicMock()
                mock_model.start_chat.return_value = mock_chat
                mock_genai.GenerativeModel.return_value = mock_model

                agent = ReActAgent(['wikipedia_ja'], 'gemini-pro')
                agent.chat_session = mock_chat

                return agent

    def test_paris_income_comparison_scenario(self, integrated_agent):
        """パリの収入比較シナリオの完全テスト"""
        query = "パリ市の平均世帯所得は、フランス全体の平均と比べてどうですか？多いですか？また、日本と比較するとどうですか？"

        # モックレスポンスの構築
        search_call = FunctionCall(
            name='search_rag_knowledge_base',
            args={'query': 'パリ 世帯所得 収入', 'collection_name': 'wikipedia_ja'}
        )

        final_answer = (
            "パリ市の平均世帯所得について、検索結果をもとにお答えします。\n\n"
            "**フランス全体との比較:**\n"
            "- パリ市の平均世帯所得: 約42,000ユーロ（2020年）\n"
            "- フランス全体の平均: 約36,000ユーロ（2020年）\n"
            "- パリ市の方が約6,000ユーロ（約17%）多いです\n\n"
            "**日本との比較:**\n"
            "- 日本の平均世帯所得: 約545万円（約38,000ユーロ、2021年）\n"
            "- パリ市の方が約4,000ユーロ多いです\n\n"
            "結論: パリ市の平均世帯所得は、フランス全体の平均よりも高く、日本の平均と比較しても高い水準にあります。"
        )

        responses = [
            Response(parts=[Part(
                text="Thought: パリの世帯収入について検索する必要があります",
                function_call=search_call
            )]),
            Response(parts=[Part(
                text=f"Answer: {final_answer}",
                function_call=None
            )])
        ]

        integrated_agent.chat_session.send_message.side_effect = responses

        # 実行
        events = list(integrated_agent.execute_turn(query))

        # 検証
        tool_calls = [e for e in events if e['type'] == 'tool_call']
        assert len(tool_calls) == 1
        assert tool_calls[0]['args']['collection_name'] == 'wikipedia_ja'

        final_events = [e for e in events if e['type'] == 'final_answer']
        assert len(final_events) == 1
        assert "42,000ユーロ" in final_events[0]['content']
        assert "36,000ユーロ" in final_events[0]['content']


class TestHelperFunctions:
    """ヘルパー関数のテスト"""

    @patch('services.agent_service.QdrantClient')
    def test_get_available_collections_success(self, mock_client_class):
        """コレクション取得成功のテスト"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        Collection = namedtuple('Collection', ['name'])
        mock_client.get_collections.return_value.collections = [
            Collection(name='wikipedia_ja'),
            Collection(name='livedoor'),
        ]

        result = get_available_collections_from_qdrant_helper()

        assert result == ['wikipedia_ja', 'livedoor']

    @patch('services.agent_service.QdrantClient')
    def test_get_available_collections_error(self, mock_client_class):
        """コレクション取得エラーのテスト"""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_collections.side_effect = Exception("Connection failed")

        result = get_available_collections_from_qdrant_helper()

        assert result == []


class TestEdgeCases:
    """エッジケースのテスト"""

    @pytest.fixture
    def edge_agent(self):
        """エッジケーステスト用エージェント"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'}):
            with patch('services.agent_service.genai') as mock_genai:
                mock_chat = MagicMock()
                mock_model = MagicMock()
                mock_model.start_chat.return_value = mock_chat
                mock_genai.GenerativeModel.return_value = mock_model

                agent = ReActAgent([], 'gemini-pro')  # 空のコレクション
                agent.chat_session = mock_chat
                return agent

    def test_empty_collections(self, edge_agent):
        """空のコレクションリストでの動作テスト"""
        response = Response(parts=[
            Part(text="Answer: コレクションが設定されていません", function_call=None)
        ])
        edge_agent.chat_session.send_message.return_value = response

        events = list(edge_agent.execute_turn("テスト"))

        final = [e for e in events if e['type'] == 'final_answer'][0]
        assert "コレクション" in final['content']

    def test_max_turns_exceeded(self, edge_agent):
        """最大ターン数超過のテスト"""
        # 無限ループをシミュレート
        tool_call = FunctionCall(
            name='list_rag_collections',
            args={}
        )

        response = Response(parts=[
            Part(text="Thought: 検索中", function_call=tool_call)
        ])

        edge_agent.chat_session.send_message.return_value = response

        with patch.dict('services.agent_service.TOOLS_MAP',
                       {'list_rag_collections': MagicMock(return_value="Collections: test1, test2"),
                        'search_rag_knowledge_base': MagicMock()}):

            events = list(edge_agent.execute_turn("コレクション一覧"))

            # 最大10ターンで停止することを確認
            tool_calls = [e for e in events if e['type'] == 'tool_call']
            assert len(tool_calls) <= 10

    def test_malformed_response(self, edge_agent):
        """不正な形式のレスポンス処理テスト"""
        response = Response(parts=[
            Part(text="", function_call=None)  # 空のテキスト
        ])
        edge_agent.chat_session.send_message.return_value = response

        events = list(edge_agent.execute_turn("テスト"))

        # エラーなく処理できることを確認
        assert any(e['type'] == 'final_answer' for e in events)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])