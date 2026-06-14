"""
GRACE Planner Integration Tests
実際にLLM (OpenAI GPT) を呼び出して動作確認を行うテスト
Plannerのテスト
[Usage]: pytest --cov=grace.planner -vs tests/grace/test_planner_integration.py
"""

import os

import pytest

from grace.planner import Planner
from grace.schemas import ExecutionPlan


# APIキーがない環境（CIなど）で実行されないようにスキップ条件をつける
@pytest.mark.skipif(
    # conftest がダミーキー（test-api-key）を設定するため、実キーの有無で判定する
    os.environ.get("OPENAI_API_KEY", "test-api-key") == "test-api-key",
    reason="Real OPENAI_API_KEY not set in environment"
)
class TestPlannerIntegration:
    """実際のLLMを使用した統合テスト"""

    def test_create_plan_real_llm(self):
        """実際のLLMを使って計画生成ができるか確認"""
        planner = Planner()
        # 二層計画生成のルールベース側に流れないよう、LLM計画生成を強制する
        planner.config.planner.force_llm_plan = True

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

    def test_create_plan_rule_based_no_llm(self):
        """単純なクエリはLLMを呼ばずルールベース計画が返ることを確認"""
        planner = Planner()

        plan = planner.create_plan("Pythonとは何ですか")

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "rag_search"
        assert plan.steps[-1].action == "reasoning"
