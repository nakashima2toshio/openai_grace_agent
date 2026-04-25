import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grace.tools import RAGSearchTool, ToolResult

class TestRAGSearchToolDynamicThresholding(unittest.TestCase):
    def setUp(self):
        # Configのモック
        self.mock_config = MagicMock()
        self.mock_config.qdrant.url = "http://mock-qdrant:6333"
        self.mock_config.qdrant.search_priority = ["test_collection"]
        self.tool = RAGSearchTool(config=self.mock_config)

    @patch('agent_tools.search_rag_knowledge_base_structured')
    def test_dynamic_thresholding_high_score(self, mock_search):
        """Top 1スコアが非常に高い場合、絞り込みが行われるかテスト"""
        # モックの戻り値を設定 (ユーザー報告のケース)
        mock_search.return_value = [
            {
                "score": 0.99996805,
                "payload": {"question": "Q1", "answer": "A1"}
            },
            {
                "score": 0.81257004,
                "payload": {"question": "Q2", "answer": "A2"}
            }
        ]

        # 実行
        result = self.tool.execute(query="test query")

        # 検証
        self.assertTrue(result.success)
        self.assertEqual(len(result.output), 1, "結果は1件に絞り込まれるべき")
        self.assertEqual(result.output[0]["score"], 0.99996805, "最高スコアの結果が残るべき")
        print("\n[Test Case 1 passed] High score (0.999) triggerd thresholding.")

    @patch('agent_tools.search_rag_knowledge_base_structured')
    def test_dynamic_thresholding_low_score(self, mock_search):
        """Top 1スコアが閾値未満の場合、絞り込みが行われないかテスト"""
        # モックの戻り値を設定 (閾値0.98未満)
        mock_search.return_value = [
            {
                "score": 0.95000000,
                "payload": {"question": "Q1", "answer": "A1"}
            },
            {
                "score": 0.90000000,
                "payload": {"question": "Q2", "answer": "A2"}
            }
        ]

        # 実行
        result = self.tool.execute(query="test query")

        # 検証
        self.assertTrue(result.success)
        self.assertEqual(len(result.output), 2, "結果は2件のままであるべき")
        print("\n[Test Case 2 passed] Medium score (0.95) did NOT trigger thresholding.")

    @patch('agent_tools.search_rag_knowledge_base_structured')
    def test_dynamic_thresholding_single_result(self, mock_search):
        """結果が1件の場合、エラーにならないかテスト"""
        # モックの戻り値を設定
        mock_search.return_value = [
            {
                "score": 0.99000000,
                "payload": {"question": "Q1", "answer": "A1"}
            }
        ]

        # 実行
        result = self.tool.execute(query="test query")

        # 検証
        self.assertTrue(result.success)
        self.assertEqual(len(result.output), 1, "結果は1件のまま")
        print("\n[Test Case 3 passed] Single result handled correctly.")

if __name__ == '__main__':
    unittest.main()
