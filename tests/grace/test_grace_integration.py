"""
GRACE エージェント統合テスト

Phase 1-4の統合テスト:
- Phase 1: Plan-and-Execute (Planner + Executor)
- Phase 2: Confidence (ConfidenceCalculator + LLMSelfEvaluator)
- Phase 3: HITL (InterventionHandler)
- Phase 4: Replan (ReplanOrchestrator)
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

# GRACE モジュール
from grace.planner import create_planner, Planner
from grace.executor import create_executor, Executor, ExecutionState
from grace.schemas import (
    ExecutionPlan,
    PlanStep,
    StepResult,
    ExecutionResult,
    StepStatus,
)
from grace.confidence import (
    ConfidenceCalculator,
    ConfidenceFactors,
    ConfidenceScore,
    ActionDecision,
    InterventionLevel,
    create_confidence_calculator,
)
from grace.intervention import (
    InterventionHandler,
    InterventionRequest,
    InterventionResponse,
    InterventionAction,
    create_intervention_handler,
)
from grace.replan import (
    ReplanManager,
    ReplanOrchestrator,
    ReplanTrigger,
    ReplanStrategy,
    create_replan_orchestrator,
)
from grace.config import get_config, GraceConfig
from grace.tools import ToolResult


# =============================================================================
# Phase 1 テスト: Plan-and-Execute
# =============================================================================

class TestPhase1PlanAndExecute:
    """Phase 1: Plan-and-Execute のテスト"""

    def test_planner_creates_valid_plan(self):
        """Plannerが有効な計画を生成できることをテスト"""
        config = get_config()

        # Plannerを作成（LLM呼び出しをモック）
        with patch.object(Planner, 'create_plan') as mock_create:
            mock_create.return_value = ExecutionPlan(
                original_query="Pythonの特徴を教えて",
                complexity=0.5,
                estimated_steps=2,
                requires_confirmation=False,
                steps=[
                    PlanStep(
                        step_id=1,
                        action="rag_search",
                        description="RAG検索",
                        expected_output="検索結果"
                    ),
                    PlanStep(
                        step_id=2,
                        action="reasoning",
                        description="回答生成",
                        depends_on=[1],
                        expected_output="最終回答"
                    ),
                ],
                success_criteria="質問に回答できている"
            )

            planner = create_planner(config=config)
            plan = planner.create_plan("Pythonの特徴を教えて")

            assert plan is not None
            assert plan.original_query == "Pythonの特徴を教えて"
            assert len(plan.steps) == 2
            assert plan.steps[0].action == "rag_search"
            assert plan.steps[1].action == "reasoning"

    def test_executor_runs_all_steps(self):
        """Executorが全ステップを実行できることをテスト"""
        config = get_config()

        # テスト用の計画
        plan = ExecutionPlan(
            original_query="テスト質問",
            complexity=0.3,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[
                PlanStep(
                    step_id=1,
                    action="reasoning",
                    description="回答生成",
                    query="テスト質問",
                    expected_output="回答"
                ),
            ],
            success_criteria="回答生成"
        )

        # コールバックをモック
        step_started = []
        step_completed = []

        def on_step_start(step):
            step_started.append(step.step_id)

        def on_step_complete(result):
            step_completed.append(result.step_id)

        # Executorを作成（ツール呼び出しをモック）
        executor = create_executor(
            config=config,
            on_step_start=on_step_start,
            on_step_complete=on_step_complete,
            enable_replan=False  # テストではリプラン無効
        )

        # ツール実行をモック
        with patch.object(executor.tool_registry, 'get') as mock_get:
            mock_tool = MagicMock()
            mock_tool.execute.return_value = ToolResult(
                success=True,
                output="テスト回答",
                confidence_factors={"avg_score": 0.8, "result_count": 3}
            )
            mock_get.return_value = mock_tool

            # LLMSelfEvaluatorをモック
            with patch.object(executor.llm_evaluator, 'evaluate') as mock_eval:
                mock_eval.return_value = ConfidenceScore(
                    score=0.8,
                    factors=ConfidenceFactors(llm_self_confidence=0.8)
                )

                result = executor.execute_plan(plan)

        assert 1 in step_started
        assert 1 in step_completed
        assert result.overall_status in ["success", "partial"]


# =============================================================================
# Phase 2 テスト: Confidence
# =============================================================================

class TestPhase2Confidence:
    """Phase 2: Confidence のテスト"""

    def test_confidence_calculator_computes_score(self):
        """ConfidenceCalculatorがスコアを計算できることをテスト"""
        config = get_config()
        calculator = create_confidence_calculator(config=config)

        factors = ConfidenceFactors(
            search_result_count=5,
            search_avg_score=0.85,
            search_score_variance=0.1,
            source_count=3,
            source_agreement=0.9,
            tool_success_rate=1.0,
            tool_execution_count=5,
            tool_success_count=5,
        )

        score = calculator.calculate(factors)

        assert score is not None
        assert 0.0 <= score.score <= 1.0
        assert score.level in ["very_low", "low", "medium", "high", "very_high"]

        def test_confidence_decides_action(self):
            """ConfidenceCalculator가アクションを決定できることをテスト"""
            config = get_config()
            calculator = create_confidence_calculator(config=config)
        
            # 高信頼度
            high_score = ConfidenceScore(score=0.9, factors=ConfidenceFactors())
            high_decision = calculator.decide_action(high_score)
            assert high_decision.level == InterventionLevel.SILENT
    
            # 低信頼度
            low_score = ConfidenceScore(score=0.2, factors=ConfidenceFactors())
            low_decision = calculator.decide_action(low_score)
            assert low_decision.level in [InterventionLevel.CONFIRM, InterventionLevel.ESCALATE]
    
# =============================================================================
# Phase 3 テスト: HITL
# =============================================================================

class TestPhase3HITL:
    """Phase 3: HITL のテスト"""

    def test_intervention_handler_handles_notify(self):
        """InterventionHandlerがNOTIFYを処理できることをテスト"""
        config = get_config()

        notifications = []

        def on_notify(message):
            notifications.append(message)

        handler = create_intervention_handler(
            config=config,
            on_notify=on_notify,
        )

        # NOTIFYレベルのアクション決定
        action_decision = ActionDecision(
            level=InterventionLevel.NOTIFY,
            confidence_score=0.8,
            reason="信頼度が低下しています"
        )

        # ダミーのステップと計画
        step = PlanStep(
            step_id=1,
            action="reasoning",
            description="テスト",
            expected_output="結果"
        )
        plan = ExecutionPlan(
            original_query="テスト",
            complexity=0.5,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[step],
            success_criteria="テスト"
        )

        response = handler.handle(action_decision, step, plan)

        # NOTIFYはログ出力のみで続行
        assert response.action == InterventionAction.PROCEED

    def test_intervention_handler_handles_confirm(self):
        """InterventionHandlerがCONFIRMを処理できることをテスト"""
        config = get_config()

        def on_confirm(request):
            # 自動で続行
            return InterventionResponse(action=InterventionAction.PROCEED)

        handler = create_intervention_handler(
            config=config,
            on_confirm=on_confirm,
        )

        action_decision = ActionDecision(
            level=InterventionLevel.CONFIRM,
            confidence_score=0.5,
            reason="続行しますか？"
        )

        step = PlanStep(
            step_id=1,
            action="reasoning",
            description="テスト",
            expected_output="結果"
        )
        plan = ExecutionPlan(
            original_query="テスト",
            complexity=0.5,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[step],
            success_criteria="テスト"
        )

        response = handler.handle(action_decision, step, plan)

        assert response.action == InterventionAction.PROCEED


# =============================================================================
# Phase 4 テスト: Replan
# =============================================================================

class TestPhase4Replan:
    """Phase 4: Replan のテスト"""

    def test_replan_manager_should_replan_on_failure(self):
        """ReplanManagerが失敗時にリプラン判定できることをテスト"""
        config = get_config()
        manager = ReplanManager(config=config)

        failed_result = StepResult(
            step_id=1,
            status="failed",
            output=None,
            confidence=0.0,
            error="テストエラー"
        )

        should, trigger = manager.should_replan(failed_result, replan_count=0)

        assert should is True
        assert trigger == ReplanTrigger.STEP_FAILED

    def test_replan_manager_respects_max_replans(self):
        """ReplanManagerが最大リプラン回数を尊重することをテスト"""
        config = get_config()
        manager = ReplanManager(config=config)

        failed_result = StepResult(
            step_id=1,
            status="failed",
            output=None,
            confidence=0.0,
            error="テストエラー"
        )

        # 最大回数を超えた場合
        should, trigger = manager.should_replan(
            failed_result,
            replan_count=manager.max_replans
        )

        assert should is False
        assert trigger is None

    def test_replan_orchestrator_handles_step_failure(self):
        """ReplanOrchestratorがステップ失敗を処理できることをテスト"""
        config = get_config()
        orchestrator = create_replan_orchestrator(config=config)

        # 失敗したステップ結果
        failed_result = StepResult(
            step_id=2,
            status="failed",
            output=None,
            confidence=0.0,
            error="検索エラー"
        )

        # 元の計画
        current_plan = ExecutionPlan(
            original_query="テスト質問",
            complexity=0.5,
            estimated_steps=2,
            requires_confirmation=False,
            steps=[
                PlanStep(
                    step_id=1,
                    action="rag_search",
                    description="検索",
                    expected_output="検索結果"
                ),
                PlanStep(
                    step_id=2,
                    action="reasoning",
                    description="回答",
                    depends_on=[1],
                    expected_output="回答"
                ),
            ],
            success_criteria="回答生成"
        )

        # 完了済み結果
        completed_results = {
            1: StepResult(
                step_id=1,
                status="success",
                output="検索結果",
                confidence=0.8
            )
        }

        # Plannerをモック
        with patch.object(orchestrator.replan_manager, '_get_planner') as mock_planner:
            mock_planner_instance = MagicMock()
            mock_planner_instance.create_plan.return_value = ExecutionPlan(
                original_query="テスト質問",
                complexity=0.5,
                estimated_steps=1,
                requires_confirmation=False,
                steps=[
                    PlanStep(
                        step_id=3,
                        action="reasoning",
                        description="代替回答",
                        expected_output="回答"
                    ),
                ],
                success_criteria="回答生成"
            )
            mock_planner.return_value = mock_planner_instance

            result = orchestrator.handle_step_failure(
                step_result=failed_result,
                current_plan=current_plan,
                completed_results=completed_results,
                replan_count=0
            )

        # リプラン結果を検証
        assert result is not None
        # 戦略が決定されている
        assert result.strategy in [
            ReplanStrategy.PARTIAL,
            ReplanStrategy.FULL,
            ReplanStrategy.FALLBACK,
            ReplanStrategy.SKIP,
            ReplanStrategy.ABORT
        ]


# =============================================================================
# 全体統合テスト
# =============================================================================

class TestFullIntegration:
    """全Phase統合テスト"""

    def test_full_grace_workflow(self):
        """GRACEの完全なワークフローをテスト"""
        config = get_config()

        # コールバック記録
        events = []

        def on_step_start(step):
            events.append(("step_start", step.step_id))

        def on_step_complete(result):
            events.append(("step_complete", result.step_id, result.status))

        def on_intervention(type_, data):
            events.append(("intervention", type_, data))
            return "はい、続行"

        def on_confidence_update(score, decision):
            events.append(("confidence", score.score, decision.level.value))

        # Executorを作成
        executor = create_executor(
            config=config,
            on_step_start=on_step_start,
            on_step_complete=on_step_complete,
            on_intervention_required=on_intervention,
            on_confidence_update=on_confidence_update,
            enable_replan=True
        )

        # テスト用の計画
        plan = ExecutionPlan(
            original_query="テスト",
            complexity=0.3,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[
                PlanStep(
                    step_id=1,
                    action="reasoning",
                    description="回答生成",
                    query="テスト",
                    expected_output="回答"
                ),
            ],
            success_criteria="回答生成"
        )

        # ツール実行をモック
        with patch.object(executor.tool_registry, 'get') as mock_get:
            mock_tool = MagicMock()
            mock_tool.execute.return_value = ToolResult(
                success=True,
                output="テスト回答です",
                confidence_factors={"avg_score": 0.85, "result_count": 5}
            )
            mock_get.return_value = mock_tool

            # LLMSelfEvaluatorをモック
            with patch.object(executor.llm_evaluator, 'evaluate') as mock_eval:
                mock_eval.return_value = ConfidenceScore(
                    score=0.85,
                    factors=ConfidenceFactors(llm_self_confidence=0.85)
                )

                result = executor.execute_plan(plan)

        # 結果を検証
        assert result is not None
        assert result.overall_status in ["success", "partial"]

        # イベントを検証
        assert ("step_start", 1) in events
        assert any(e[0] == "step_complete" and e[1] == 1 for e in events)
        assert any(e[0] == "confidence" for e in events)


# =============================================================================
# pytest エントリポイント
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])