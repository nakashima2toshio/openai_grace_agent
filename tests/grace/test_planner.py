"""
GRACE Planner Tests
Plannerのテスト
[Usage]: pytest --cov=grace.planner -vs tests/grace/test_planner.py
"""

import pytest
from unittest.mock import MagicMock, patch
import json

from grace.planner import Planner, create_planner
from grace.schemas import ExecutionPlan, PlanStep
from grace.config import GraceConfig, reset_config


class TestPlanner:
    """Plannerのテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    @patch("grace.planner.genai.Client")
    @patch("grace.planner.QdrantClient")
    @patch("grace.planner.get_all_collections")
    def test_create_plan_success(self, mock_get_collections, mock_qdrant_client, mock_client_class):
        """計画生成の成功"""
        # Mock available collections
        mock_get_collections.return_value = [{"name": "wikipedia_ja"}, {"name": "livedoor"}]

        # Mock LLM responses
        # Call 1: Complexity estimation
        mock_response_complexity = MagicMock()
        mock_response_complexity.text = "0.5"

        # Call 2: Plan generation
        mock_response_plan = MagicMock()
        mock_response_plan.text = json.dumps({
            "original_query": "スペイン語の文法と単語はそれぞれ何語の影響を強く受けていますか？　日本語の影響は受けていますか？",
            "complexity": 0.5,
            "estimated_steps": 2,
            "requires_confirmation": False,
            "steps": [
                {
                    "step_id": 1,
                    "action": "rag_search",
                    "description": "関連情報を検索",
                    "query": "Python",
                    "expected_output": "検索結果",
                    "collection": "wikipedia_ja"
                },
                {
                    "step_id": 2,
                    "action": "reasoning",
                    "description": "回答を生成",
                    "depends_on": [1],
                    "expected_output": "回答"
                }
            ],
            "success_criteria": "質問に回答できている"
        })

        mock_client = MagicMock()
        # side_effect for multiple calls
        mock_client.models.generate_content.side_effect = [mock_response_complexity, mock_response_plan]
        mock_client_class.return_value = mock_client

        # Plannerをテスト
        planner = Planner()
        plan = planner.create_plan("スペイン語の文法と単語はそれぞれ何語の影響を強く受けていますか？　日本語の影響は受けていますか？")

        assert isinstance(plan, ExecutionPlan)
        assert plan.original_query == "スペイン語の文法と単語はそれぞれ何語の影響を強く受けていますか？　日本語の影響は受けていますか？"
        assert len(plan.steps) == 2
        assert plan.plan_id is not None
        assert plan.complexity == 0.5

    @patch("grace.planner.genai.Client")
    @patch("grace.planner.QdrantClient")
    @patch("grace.planner.get_all_collections")
    def test_create_plan_fallback(self, mock_get_collections, mock_qdrant_client, mock_client_class):
        """計画生成失敗時のフォールバック"""
        # Mock collections
        mock_get_collections.return_value = []

        mock_client = MagicMock()
        # Ensure calls fail
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        planner = Planner()
        plan = planner.create_plan("テスト")

        # フォールバック計画が返される
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "rag_search"
        assert plan.steps[0].collection == "wikipedia_ja"
        assert plan.steps[1].action == "reasoning"

    @patch("grace.planner.genai.Client")
    def test_estimate_complexity_with_llm_simple(self, mock_client_class):
        """単純な質問の複雑度推定 (LLM)"""
        mock_response = MagicMock()
        mock_response.text = "0.2"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        planner = Planner()

        # 単純な質問
        complexity = planner.estimate_complexity_with_llm("スペイン語の文法と単語はそれぞれ何語の影響を強く受けていますか？")
        assert complexity == 0.2

    @patch("grace.planner.genai.Client")
    def test_estimate_complexity_with_llm_complex(self, mock_client_class):
        """複雑な質問の複雑度推定 (LLM)"""
        mock_response = MagicMock()
        mock_response.text = "0.8"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        planner = Planner()

        # 複雑な質問
        complexity = planner.estimate_complexity_with_llm(
            "スペイン語の文法と単語はそれぞれ何語の影響を強く受けていますか？　日本語の影響は受けていますか？"
        )
        assert complexity == 0.8

    @patch("grace.planner.genai.Client")
    def test_estimate_complexity_with_llm_fallback(self, mock_client_class):
        """複雑度推定失敗時のフォールバック"""
        # LLM呼び出しでエラーを発生させる
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        planner = Planner()

        # フォールバック（ルールベース）が呼ばれる
        # "詳しく" (0.15) + ベース (0.5) = 0.65 程度になるはず
        complexity = planner.estimate_complexity_with_llm(
            "スペイン語の文法と単語に影響を与えた言語の影響ついて詳しく教えてください"
        )

        # ルールベースの計算結果であることを確認（エラーにならず値を返す）
        assert complexity > 0.0
        assert complexity <= 1.0

    @patch("grace.planner.genai.Client")
    def test_refine_plan(self, mock_client_class):
        """計画の修正"""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "original_query": "テスト",
            "complexity": 0.5,
            "estimated_steps": 1,
            "requires_confirmation": False,
            "steps": [
                {
                    "step_id": 1,
                    "action": "reasoning",
                    "description": "修正された説明",
                    "expected_output": "結果"
                }
            ],
            "success_criteria": "テスト"
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        planner = Planner()

        original_plan = ExecutionPlan(
            original_query="テスト",
            complexity=0.5,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[
                PlanStep(
                    step_id=1,
                    action="rag_search",
                    description="元の説明",
                    expected_output="結果"
                )
            ],
            success_criteria="テスト"
        )

        refined_plan = planner.refine_plan(original_plan, "もっと詳しく")

        assert isinstance(refined_plan, ExecutionPlan)
        assert refined_plan.plan_id != original_plan.plan_id


class TestCreatePlanner:
    """create_planner関数のテスト"""

    @patch("grace.planner.genai.Client")
    def test_create_planner_default(self, mock_client_class):
        """デフォルト設定でのPlanner作成"""
        mock_client_class.return_value = MagicMock()

        planner = create_planner()

        assert isinstance(planner, Planner)
        assert planner.model_name == "gemini-2.0-flash"

    @patch("grace.planner.genai.Client")
    def test_create_planner_custom_model(self, mock_client_class):
        """カスタムモデルでのPlanner作成"""
        mock_client_class.return_value = MagicMock()

        planner = create_planner(model_name="custom-model")

        assert planner.model_name == "custom-model"