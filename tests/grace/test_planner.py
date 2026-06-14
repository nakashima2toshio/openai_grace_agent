"""
GRACE Planner Tests
Plannerのテスト
[Usage]: pytest --cov=grace.planner -vs tests/grace/test_planner.py
"""

from unittest.mock import MagicMock, patch

import pytest

from grace.config import reset_config
from grace.planner import Planner, create_planner
from grace.schemas import ExecutionPlan, PlanStep


def _make_llm_plan(query: str = "テスト") -> ExecutionPlan:
    """LLMが返す想定の計画を作成"""
    return ExecutionPlan(
        original_query=query,
        complexity=0.8,
        estimated_steps=2,
        requires_confirmation=False,
        steps=[
            PlanStep(
                step_id=1,
                action="rag_search",
                description="関連情報を検索",
                query=query,
                expected_output="検索結果",
                fallback="web_search",
            ),
            PlanStep(
                step_id=2,
                action="reasoning",
                description="回答を生成",
                depends_on=[1],
                expected_output="回答",
            ),
        ],
        success_criteria="質問に回答できている",
    )


class TestPlanner:
    """Plannerのテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    @patch("grace.planner.create_llm_client")
    def test_create_plan_rule_based(self, mock_create_llm):
        """単純なクエリはルールベース計画（LLM呼び出しなし）"""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        planner = Planner()
        plan = planner.create_plan("Pythonとは何ですか")

        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "rag_search"
        assert plan.steps[0].fallback == "web_search"
        assert plan.steps[1].action == "reasoning"
        assert plan.plan_id is not None
        # 計画生成にLLMを使用していないこと
        mock_llm.generate_structured.assert_not_called()
        mock_llm.generate_content.assert_not_called()

    @patch("grace.planner.QdrantClient")
    @patch("grace.planner.get_all_collections")
    @patch("grace.planner.create_llm_client")
    def test_create_plan_llm_for_complex_query(
            self, mock_create_llm, mock_get_collections, mock_qdrant_client):
        """複雑なクエリはLLM計画生成を使用"""
        mock_get_collections.return_value = [{"name": "wikipedia_ja"}, {"name": "livedoor"}]

        query = "AとBの違いを複数の観点から比較して、なぜそうなるのか詳しく教えてください"
        mock_llm = MagicMock()
        mock_llm.generate_structured.return_value = _make_llm_plan(query)
        mock_create_llm.return_value = mock_llm

        planner = Planner()
        plan = planner.create_plan(query)

        assert isinstance(plan, ExecutionPlan)
        assert plan.original_query == query
        assert len(plan.steps) == 2
        assert plan.plan_id is not None
        # LLM計画生成の複雑度がそのまま使用される（別途LLM推定しない）
        assert plan.complexity == 0.8
        mock_llm.generate_structured.assert_called_once()
        mock_llm.generate_content.assert_not_called()

    @patch("grace.planner.QdrantClient")
    @patch("grace.planner.get_all_collections")
    @patch("grace.planner.create_llm_client")
    def test_create_plan_llm_for_explicit_web_search(
            self, mock_create_llm, mock_get_collections, mock_qdrant_client):
        """明示的なWeb検索指示はLLM計画生成を使用"""
        mock_get_collections.return_value = []

        query = "最新ニュースを検索して教えて"
        mock_llm = MagicMock()
        mock_llm.generate_structured.return_value = _make_llm_plan(query)
        mock_create_llm.return_value = mock_llm

        planner = Planner()
        planner.create_plan(query)

        mock_llm.generate_structured.assert_called_once()

    @patch("grace.planner.QdrantClient")
    @patch("grace.planner.get_all_collections")
    @patch("grace.planner.create_llm_client")
    def test_create_plan_fallback(
            self, mock_create_llm, mock_get_collections, mock_qdrant_client):
        """LLM計画生成失敗時のフォールバック"""
        mock_get_collections.return_value = []

        mock_llm = MagicMock()
        # 非一時的エラー（リトライせず即フォールバック）
        mock_llm.generate_structured.side_effect = Exception("API Error")
        mock_create_llm.return_value = mock_llm

        planner = Planner()
        query = "AとBの違いを複数の観点から比較して、なぜそうなるのか詳しく教えてください"
        plan = planner.create_plan(query)

        # フォールバック計画が返される
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "rag_search"
        assert plan.steps[0].collection is None
        assert plan.steps[1].action == "reasoning"

    @patch("grace.planner.create_llm_client")
    def test_retry_on_transient_error(self, mock_create_llm):
        """一時的エラーはリトライされる"""
        query = "AとBの違いを複数の観点から比較して、なぜそうなるのか詳しく教えてください"
        mock_llm = MagicMock()
        mock_llm.generate_structured.side_effect = [
            Exception("429 rate limit exceeded"),
            _make_llm_plan(query),
        ]
        mock_create_llm.return_value = mock_llm

        planner = Planner()
        # バックオフ待機を高速化
        planner.config.error.retry_delay_base = 0.01
        plan = planner._generate_plan_with_retry("dummy prompt")

        assert isinstance(plan, ExecutionPlan)
        assert mock_llm.generate_structured.call_count == 2

    @patch("grace.planner.create_llm_client")
    def test_no_retry_on_non_transient_error(self, mock_create_llm):
        """非一時的エラー（認証エラー等）はリトライしない"""
        mock_llm = MagicMock()
        mock_llm.generate_structured.side_effect = Exception("invalid x-api-key")
        mock_create_llm.return_value = mock_llm

        planner = Planner()
        with pytest.raises(Exception):
            planner._generate_plan_with_retry("dummy prompt")

        assert mock_llm.generate_structured.call_count == 1

    def test_is_transient_error(self):
        """一時的エラー判定"""
        assert Planner._is_transient_error(Exception("Request timeout"))
        assert Planner._is_transient_error(Exception("rate limit exceeded"))
        assert Planner._is_transient_error(Exception("503 Service Unavailable"))
        assert Planner._is_transient_error(TimeoutError("timed out"))
        assert not Planner._is_transient_error(Exception("invalid x-api-key"))
        assert not Planner._is_transient_error(ValueError("bad input"))

    @patch("grace.planner.create_llm_client")
    def test_estimate_complexity_heuristic(self, mock_create_llm):
        """ヒューリスティック複雑度推定"""
        mock_create_llm.return_value = MagicMock()
        planner = Planner()

        simple = planner.estimate_complexity("Pythonとは")
        complex_q = planner.estimate_complexity(
            "AとBの違いを複数の観点から比較して、なぜそうなるのか詳しく教えてください"
        )
        assert 0.0 <= simple <= 1.0
        assert complex_q > simple

    @patch("grace.planner.create_llm_client")
    def test_refine_plan(self, mock_create_llm):
        """計画の修正（元計画の完全JSONがプロンプトに含まれる）"""
        refined = _make_llm_plan("テスト")
        mock_llm = MagicMock()
        mock_llm.generate_structured.return_value = refined
        mock_create_llm.return_value = mock_llm

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
                    query="検索クエリXYZ",
                    expected_output="結果",
                )
            ],
            success_criteria="テスト",
            plan_id="original123",
        )

        refined_plan = planner.refine_plan(original_plan, "もっと詳しく")

        assert isinstance(refined_plan, ExecutionPlan)
        assert refined_plan.plan_id != original_plan.plan_id

        # プロンプトに元計画のクエリ等の詳細が含まれていること
        call_kwargs = mock_llm.generate_structured.call_args
        prompt = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]
        assert "検索クエリXYZ" in prompt

    @patch("grace.planner.create_llm_client")
    def test_refine_plan_failure_returns_original(self, mock_create_llm):
        """計画修正失敗時は元の計画を返す"""
        mock_llm = MagicMock()
        mock_llm.generate_structured.side_effect = Exception("API Error")
        mock_create_llm.return_value = mock_llm

        planner = Planner()
        original_plan = _make_llm_plan("テスト")
        original_plan.plan_id = "original123"

        refined_plan = planner.refine_plan(original_plan, "もっと詳しく")
        assert refined_plan.plan_id == "original123"


class TestCreatePlanner:
    """create_planner関数のテスト"""

    def setup_method(self):
        reset_config()

    @patch("grace.planner.create_llm_client")
    def test_create_planner_default(self, mock_create_llm):
        """デフォルト設定でのPlanner作成"""
        mock_create_llm.return_value = MagicMock()

        planner = create_planner()

        assert isinstance(planner, Planner)
        assert planner.model_name == "gpt-5-mini"

    @patch("grace.planner.create_llm_client")
    def test_create_planner_custom_model(self, mock_create_llm):
        """カスタムモデルでのPlanner作成"""
        mock_create_llm.return_value = MagicMock()

        planner = create_planner(model_name="custom-model")

        assert planner.model_name == "custom-model"
