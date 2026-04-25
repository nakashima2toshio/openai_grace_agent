"""
GRACE Planner Integration Tests
実際にLLM (Gemini) を呼び出して動作確認を行うテスト
Plannerのテスト
[Usage]: pytest --cov=grace.planner -vs tests/grace/test_planner_integration.py
"""

import pytest
import os
from grace.planner import Planner
from grace.schemas import ExecutionPlan

# APIキーがない環境（CIなど）で実行されないようにスキップ条件をつける
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set in environment"
)
class TestPlannerIntegration:
    """実際のLLMを使用した統合テスト"""

    def test_create_plan_real_llm(self):
        """実際のLLMを使って計画生成ができるか確認"""
        planner = Planner()
        
        # 実際にGeminiに問い合わせる
        query = "スペイン語の文法と単語はそれぞれ何語の影響を強く受けていますか？　日本語の影響は受けていますか？"
        print(f"\nSending query to LLM: {query}")
        
        plan = planner.create_plan(query)

        # 結果の検証（内容は変動するので、構造が正しいかチェック）
        print(f"\nGenerated Plan ID: {plan.plan_id}")
        print(f"Plan JSON:\n{plan.model_dump_json(indent=2)}")
        
        assert isinstance(plan, ExecutionPlan)
        assert plan.original_query == query
        assert len(plan.steps) > 0
        assert plan.complexity > 0.0
        
        # 最後のステップは必ず reasoning であるはず
        assert plan.steps[-1].action == "reasoning"

    def test_estimate_complexity_real_llm(self):
        """実際のLLMを使って複雑度推定ができるか確認"""
        planner = Planner()
        
        query = "スペイン語の文法と単語はそれぞれ何語の影響を強く受けていますか？ 日本語の影響は受けていますか？"
        complexity = planner.estimate_complexity_with_llm(query)
        
        print(f"\nQuery: {query}")
        print(f"Estimated Complexity: {complexity}")
        
        # 複雑な質問なので、ある程度高いスコアが出るはず
        assert 0.0 <= complexity <= 1.0
        # ※注: LLMの判断次第なので厳密な値のテストは難しいが、0.0ではないことを確認
        assert complexity > 0.1
