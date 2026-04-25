"""
GRACE Schemas Tests
スキーマのテスト
"""

import pytest
from datetime import datetime

from grace.schemas import (
    PlanStep,
    ExecutionPlan,
    StepResult,
    ExecutionResult,
    ActionType,
    StepStatus,
    create_plan_id,
    validate_plan_dependencies,
)


class TestPlanStep:
    """PlanStepのテスト"""

    def test_create_basic_step(self):
        """基本的なステップ作成"""
        step = PlanStep(
            step_id=1,
            action="rag_search",
            description="関連情報を検索",
            query="Python async",
            expected_output="検索結果"
        )

        assert step.step_id == 1
        assert step.action == "rag_search"
        assert step.description == "関連情報を検索"
        assert step.query == "Python async"
        assert step.depends_on == []
        assert step.fallback is None

    def test_step_with_dependencies(self):
        """依存関係を持つステップ"""
        step = PlanStep(
            step_id=2,
            action="reasoning",
            description="情報を統合",
            depends_on=[1],
            expected_output="回答"
        )

        assert step.depends_on == [1]

    def test_step_with_fallback(self):
        """フォールバックを持つステップ"""
        step = PlanStep(
            step_id=1,
            action="rag_search",
            description="検索",
            expected_output="結果",
            fallback="reasoning"
        )

        assert step.fallback == "reasoning"

    def test_invalid_action(self):
        """無効なアクション"""
        with pytest.raises(ValueError):
            PlanStep(
                step_id=1,
                action="invalid_action",
                description="テスト",
                expected_output="結果"
            )

    def test_step_id_must_be_positive(self):
        """step_idは正の整数"""
        with pytest.raises(ValueError):
            PlanStep(
                step_id=0,
                action="rag_search",
                description="テスト",
                expected_output="結果"
            )


class TestExecutionPlan:
    """ExecutionPlanのテスト"""

    def test_create_basic_plan(self):
        """基本的な計画作成"""
        plan = ExecutionPlan(
            original_query="Pythonについて教えて",
            complexity=0.5,
            estimated_steps=2,
            requires_confirmation=False,
            steps=[
                PlanStep(
                    step_id=1,
                    action="rag_search",
                    description="検索",
                    expected_output="結果"
                ),
                PlanStep(
                    step_id=2,
                    action="reasoning",
                    description="回答生成",
                    depends_on=[1],
                    expected_output="回答"
                )
            ],
            success_criteria="質問に回答できている"
        )

        assert plan.original_query == "Pythonについて教えて"
        assert plan.complexity == 0.5
        assert len(plan.steps) == 2
        assert plan.created_at is not None

    def test_complexity_range(self):
        """complexityは0.0-1.0の範囲"""
        # 範囲外の値はエラー
        with pytest.raises(ValueError):
            ExecutionPlan(
                original_query="テスト",
                complexity=1.5,  # 範囲外
                estimated_steps=1,
                requires_confirmation=False,
                steps=[
                    PlanStep(
                        step_id=1,
                        action="reasoning",
                        description="テスト",
                        expected_output="結果"
                    )
                ],
                success_criteria="テスト"
            )

    def test_plan_must_have_steps(self):
        """計画は少なくとも1ステップ必要"""
        with pytest.raises(ValueError):
            ExecutionPlan(
                original_query="テスト",
                complexity=0.5,
                estimated_steps=0,
                requires_confirmation=False,
                steps=[],  # 空のステップ
                success_criteria="テスト"
            )


class TestStepResult:
    """StepResultのテスト"""

    def test_create_success_result(self):
        """成功結果の作成"""
        result = StepResult(
            step_id=1,
            status="success",
            output="検索結果",
            confidence=0.85,
            sources=["doc1", "doc2"]
        )

        assert result.step_id == 1
        assert result.status == "success"
        assert result.confidence == 0.85
        assert len(result.sources) == 2
        assert result.error is None

    def test_create_failed_result(self):
        """失敗結果の作成"""
        result = StepResult(
            step_id=1,
            status="failed",
            output=None,
            confidence=0.0,
            error="Connection timeout"
        )

        assert result.status == "failed"
        assert result.confidence == 0.0
        assert result.error == "Connection timeout"

    def test_confidence_range(self):
        """confidenceは0.0-1.0の範囲"""
        with pytest.raises(ValueError):
            StepResult(
                step_id=1,
                status="success",
                output="結果",
                confidence=1.5  # 範囲外
            )


class TestExecutionResult:
    """ExecutionResultのテスト"""

    def test_create_execution_result(self):
        """実行結果の作成"""
        result = ExecutionResult(
            plan_id="abc123",
            original_query="テスト",
            final_answer="回答です",
            step_results=[
                StepResult(
                    step_id=1,
                    status="success",
                    output="結果",
                    confidence=0.8
                )
            ],
            overall_confidence=0.8,
            overall_status="success"
        )

        assert result.plan_id == "abc123"
        assert result.final_answer == "回答です"
        assert result.overall_status == "success"
        assert result.replan_count == 0


class TestUtilities:
    """ユーティリティ関数のテスト"""

    def test_create_plan_id(self):
        """計画ID生成"""
        id1 = create_plan_id()
        id2 = create_plan_id()

        assert len(id1) == 12
        assert id1 != id2  # 一意性

    def test_validate_plan_dependencies_valid(self):
        """有効な依存関係の検証"""
        plan = ExecutionPlan(
            original_query="テスト",
            complexity=0.5,
            estimated_steps=2,
            requires_confirmation=False,
            steps=[
                PlanStep(
                    step_id=1,
                    action="rag_search",
                    description="検索",
                    expected_output="結果"
                ),
                PlanStep(
                    step_id=2,
                    action="reasoning",
                    description="推論",
                    depends_on=[1],
                    expected_output="回答"
                )
            ],
            success_criteria="テスト"
        )

        errors = validate_plan_dependencies(plan)
        assert len(errors) == 0

    def test_validate_plan_dependencies_invalid(self):
        """無効な依存関係の検証"""
        plan = ExecutionPlan(
            original_query="テスト",
            complexity=0.5,
            estimated_steps=2,
            requires_confirmation=False,
            steps=[
                PlanStep(
                    step_id=1,
                    action="rag_search",
                    description="検索",
                    depends_on=[99],  # 存在しないステップ
                    expected_output="結果"
                ),
                PlanStep(
                    step_id=2,
                    action="reasoning",
                    description="推論",
                    depends_on=[2],  # 自己参照
                    expected_output="回答"
                )
            ],
            success_criteria="テスト"
        )

        errors = validate_plan_dependencies(plan)
        assert len(errors) > 0


class TestEnums:
    """Enumのテスト"""

    def test_action_type(self):
        """ActionTypeの値"""
        assert ActionType.RAG_SEARCH.value == "rag_search"
        assert ActionType.REASONING.value == "reasoning"
        assert ActionType.ASK_USER.value == "ask_user"

    def test_step_status(self):
        """StepStatusの値"""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.SUCCESS.value == "success"
        assert StepStatus.FAILED.value == "failed"