import unittest
import sys
import os

# プロジェクトルートをパスに追加 (先頭に追加して tests/grace などの競合を回避)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(f"DEBUG: sys.path: {sys.path}")

from unittest.mock import MagicMock, patch
from grace.confidence import ConfidenceCalculator, ConfidenceFactors, ConfidenceScore, create_confidence_calculator

class TestConfidenceFix(unittest.TestCase):
    def setUp(self):
        # Configのモック
        self.mock_config = MagicMock()
        # Weightsの設定 (合計が1.0になるように)
        weights = MagicMock()
        weights.search_quality = 0.2
        weights.source_agreement = 0.2
        weights.llm_self_eval = 0.2
        weights.tool_success = 0.2
        weights.query_coverage = 0.2
        self.mock_config.confidence.weights = weights
        
        self.calculator = create_confidence_calculator(config=self.mock_config)

    def test_search_score_override(self):
        """
        検索スコアが高い場合、LLMの低い評価を上書きすることを確認するテスト
        """
        # テストケース:
        # - 検索ステップである
        # - 検索スコアは非常に高い (0.99)
        # - しかしLLMは何らかの理由(バイアス等)で低めの評価 (0.8) を返すとする
        
        factors = ConfidenceFactors(
            is_search_step=True,
            search_max_score=0.99,
            search_result_count=1
        )
        
        step_description = "テスト用検索ステップ"
        tool_output = "テスト用検索結果"

        # LLMSelfEvaluator.evaluate_with_factors をモックして 0.8 を返すようにする
        with patch('grace.confidence.LLMSelfEvaluator.evaluate_with_factors') as mock_eval:
            mock_eval.return_value = {"score": 0.8, "reason": "LLMの低い評価"}
            
            # テスト対象のメソッドを実行
            result: ConfidenceScore = self.calculator.llm_calculate(
                factors=factors,
                step_description=step_description,
                tool_output=tool_output
            )

            print(f"\n[Test Result] LLM Score: 0.8, Search Score: 0.99 -> Final Score: {result.score}")
            print(f"[Test Reason] {result.reason}")

            # 検証
            self.assertEqual(result.score, 0.99, "検索スコア (0.99) が採用されるべき")
            self.assertIn("検索スコア 0.9900 を優先", result.reason, "理由に上書きの旨が含まれるべき")

    def test_no_override_when_low_search_score(self):
        """
        検索スコアが低い場合は上書きしないことを確認
        """
        factors = ConfidenceFactors(
            is_search_step=True,
            search_max_score=0.7, # 閾値 (0.9) 以下
            search_result_count=1
        )

        with patch('grace.confidence.LLMSelfEvaluator.evaluate_with_factors') as mock_eval:
            mock_eval.return_value = {"score": 0.8, "reason": "LLMの評価"}
            
            result = self.calculator.llm_calculate(
                factors=factors,
                step_description="...",
                tool_output="..."
            )
            
            self.assertEqual(result.score, 0.8, "LLMのスコア (0.8) がそのまま採用されるべき")

    def test_no_override_for_non_search_step(self):
        """
        検索ステップ以外では上書きしないことを確認
        """
        factors = ConfidenceFactors(
            is_search_step=False, # 検索ステップではない
            search_max_score=0.99, # ※何らかの理由で値が入っていても
        )

        with patch('grace.confidence.LLMSelfEvaluator.evaluate_with_factors') as mock_eval:
            mock_eval.return_value = {"score": 0.8, "reason": "LLMの評価"}
            
            result = self.calculator.llm_calculate(
                factors=factors,
                step_description="...",
                tool_output="..."
            )
            
            self.assertEqual(result.score, 0.8, "LLMのスコアが採用されるべき")

if __name__ == '__main__':
    unittest.main()
