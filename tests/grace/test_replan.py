"""
GRACE Replan Tests
動的リプランニングシステムのテスト
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from grace.replan import (
    ReplanTrigger,
    ReplanStrategy,
    ReplanContext,
    ReplanResult,
    ReplanManager,
    ReplanOrchestrator,
    create_replan_manager,
    create_replan_orchestrator,
)
from grace.schemas import ExecutionPlan, PlanStep, StepResult, ActionType
from grace.config import GraceConfig, reset_config


class TestReplanTrigger:
    """ReplanTriggerのテスト"""

    def test_step_failed(self):
        """STEP_FAILED値"""
        assert ReplanTrigger.STEP_FAILED == "step_failed"
        assert ReplanTrigger.STEP_FAILED.value == "step_failed"

    def test_low_confidence(self):
        """LOW_CONFIDENCE値"""
        assert ReplanTrigger.LOW_CONFIDENCE == "low_confidence"

    def test_user_feedback(self):
        """USER_FEEDBACK値"""
        assert ReplanTrigger.USER_FEEDBACK == "user_feedback"

    def test_new_information(self):
        """NEW_INFORMATION値"""
        assert ReplanTrigger.NEW_INFORMATION == "new_information"

    def test_timeout(self):
        """TIMEOUT値"""
        assert ReplanTrigger.TIMEOUT == "timeout"

    def test_all_triggers(self):
        """全トリガー種別の確認"""
        triggers = list(ReplanTrigger)
        assert len(triggers) == 5


class TestReplanStrategy:
    """ReplanStrategyのテスト"""

    def test_partial(self):
        """PARTIAL値"""
        assert ReplanStrategy.PARTIAL == "partial"

    def test_full(self):
        """FULL値"""
        assert ReplanStrategy.FULL == "full"

    def test_fallback(self):
        """FALLBACK値"""
        assert ReplanStrategy.FALLBACK == "fallback"

    def test_skip(self):
        """SKIP値"""
        assert ReplanStrategy.SKIP == "skip"

    def test_abort(self):
        """ABORT値"""
        assert ReplanStrategy.ABORT == "abort"

    def test_all_strategies(self):
        """全戦略の確認"""
        strategies = list(ReplanStrategy)
        assert len(strategies) == 5


class TestReplanContext:
    """ReplanContextのテスト"""

    def test_minimal_context(self):
        """最小限のコンテキスト"""
        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="テストクエリ"
        )

        assert context.trigger == ReplanTrigger.STEP_FAILED
        assert context.original_query == "テストクエリ"
        assert context.failed_step_id is None
        assert context.error_message is None
        assert context.completed_results == {}
        assert context.user_feedback is None
        assert context.replan_count == 0

    def test_full_context(self):
        """フルコンテキスト"""
        result1 = StepResult(step_id=1, status="success", confidence=0.9, output="結果1")
        result2 = StepResult(step_id=2, status="success", confidence=0.85, output="結果2")

        context = ReplanContext(
            trigger=ReplanTrigger.LOW_CONFIDENCE,
            original_query="元のクエリ",
            failed_step_id=3,
            error_message="エラー発生",
            completed_results={1: result1, 2: result2},
            user_feedback="フィードバック",
            new_information="新情報",
            replan_count=1
        )

        assert context.failed_step_id == 3
        assert context.error_message == "エラー発生"
        assert len(context.completed_results) == 2
        assert context.replan_count == 1

    def test_has_completed_steps_true(self):
        """完了済みステップあり"""
        result = StepResult(step_id=1, status="success", confidence=0.9)
        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            completed_results={1: result}
        )

        assert context.has_completed_steps is True

    def test_has_completed_steps_false(self):
        """完了済みステップなし"""
        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query"
        )

        assert context.has_completed_steps is False

    def test_completed_step_ids(self):
        """完了済みステップIDリスト"""
        result1 = StepResult(step_id=1, status="success", confidence=0.9)
        result3 = StepResult(step_id=3, status="success", confidence=0.85)

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            completed_results={3: result3, 1: result1}  # 順不同で追加
        )

        ids = context.completed_step_ids
        assert ids == [1, 3]  # ソートされている

    def test_get_completed_outputs(self):
        """完了済み出力の取得"""
        result1 = StepResult(step_id=1, status="success", confidence=0.9, output="出力1")
        result2 = StepResult(step_id=2, status="success", confidence=0.85, output=None)

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            completed_results={1: result1, 2: result2}
        )

        outputs = context.get_completed_outputs()
        assert outputs == {1: "出力1"}  # Noneは除外


class TestReplanResult:
    """ReplanResultのテスト"""

    def _create_sample_plan(self) -> ExecutionPlan:
        """サンプル計画を作成"""
        steps = [
            PlanStep(
                step_id=1,
                action="rag_search",
                description="検索",
                expected_output="検索結果"
            )
        ]
        return ExecutionPlan(
            original_query="query",
            complexity=0.5,
            estimated_steps=1,
            requires_confirmation=False,
            steps=steps,
            success_criteria="成功基準"
        )

    def test_success_result(self):
        """成功結果"""
        plan = self._create_sample_plan()

        result = ReplanResult(
            success=True,
            strategy=ReplanStrategy.PARTIAL,
            new_plan=plan,
            reason="部分再計画",
            replan_count=1
        )

        assert result.success is True
        assert result.strategy == ReplanStrategy.PARTIAL
        assert result.new_plan is not None
        assert result.reason == "部分再計画"
        assert result.replan_count == 1
        assert isinstance(result.created_at, datetime)

    def test_failure_result(self):
        """失敗結果"""
        result = ReplanResult(
            success=False,
            strategy=ReplanStrategy.ABORT,
            new_plan=None,
            reason="最大リプラン回数超過"
        )

        assert result.success is False
        assert result.new_plan is None


class TestReplanManager:
    """ReplanManagerのテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    def _create_sample_plan(self) -> ExecutionPlan:
        """サンプル計画を作成"""
        steps = [
            PlanStep(
                step_id=1,
                action="rag_search",
                description="検索1",
                expected_output="検索結果1"
            ),
            PlanStep(
                step_id=2,
                action="reasoning",
                description="推論1",
                depends_on=[1],
                expected_output="推論結果1"
            ),
            PlanStep(
                step_id=3,
                action="rag_search",
                description="検索2",
                depends_on=[2],
                expected_output="検索結果2"
            ),
        ]
        return ExecutionPlan(
            original_query="テストクエリ",
            complexity=0.5,
            estimated_steps=3,
            requires_confirmation=False,
            steps=steps,
            success_criteria="すべてのステップが成功"
        )

    def test_initialization(self):
        """初期化テスト"""
        manager = ReplanManager()

        assert manager.config is not None
        assert manager.max_replans > 0
        assert 0.0 <= manager.confidence_threshold <= 1.0
        assert manager.history == []

    def test_should_replan_step_failed(self):
        """ステップ失敗でリプラン"""
        manager = ReplanManager()
        result = StepResult(step_id=1, status="failed", confidence=0.0, error="エラー")

        should_replan, trigger = manager.should_replan(result, replan_count=0)

        assert should_replan is True
        assert trigger == ReplanTrigger.STEP_FAILED

    def test_should_replan_low_confidence(self):
        """低信頼度でリプラン"""
        manager = ReplanManager()
        manager.confidence_threshold = 0.5
        result = StepResult(step_id=1, status="success", confidence=0.3)

        should_replan, trigger = manager.should_replan(result, replan_count=0)

        assert should_replan is True
        assert trigger == ReplanTrigger.LOW_CONFIDENCE

    def test_should_replan_success_high_confidence(self):
        """成功・高信頼度ではリプランなし"""
        manager = ReplanManager()
        manager.confidence_threshold = 0.5
        result = StepResult(step_id=1, status="success", confidence=0.9)

        should_replan, trigger = manager.should_replan(result, replan_count=0)

        assert should_replan is False
        assert trigger is None

    def test_should_replan_max_reached(self):
        """最大リプラン回数到達"""
        manager = ReplanManager()
        manager.max_replans = 3
        result = StepResult(step_id=1, status="failed", confidence=0.0)

        should_replan, trigger = manager.should_replan(result, replan_count=3)

        assert should_replan is False
        assert trigger is None

    def test_should_replan_from_feedback_modification(self):
        """修正キーワードでリプラン"""
        manager = ReplanManager()

        # 修正を含むフィードバック
        should_replan, trigger = manager.should_replan_from_feedback(
            "この結果を修正してください", replan_count=0
        )

        assert should_replan is True
        assert trigger == ReplanTrigger.USER_FEEDBACK

    def test_should_replan_from_feedback_no_modification(self):
        """修正キーワードなしではリプランなし"""
        manager = ReplanManager()

        should_replan, trigger = manager.should_replan_from_feedback(
            "ありがとう、良い結果です", replan_count=0
        )

        assert should_replan is False
        assert trigger is None

    def test_determine_strategy_abort_max_reached(self):
        """最大回数超過でABORT"""
        manager = ReplanManager()
        manager.max_replans = 2

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            replan_count=2
        )
        plan = self._create_sample_plan()

        strategy = manager.determine_strategy(context, plan)

        assert strategy == ReplanStrategy.ABORT

    def test_determine_strategy_fallback(self):
        """代替アクションあり"""
        manager = ReplanManager()

        step_with_fallback = PlanStep(
            step_id=1,
            action="rag_search",
            description="検索",
            expected_output="結果",
            fallback="web_search"
        )
        plan = ExecutionPlan(
            original_query="query",
            complexity=0.3,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[step_with_fallback],
            success_criteria="成功"
        )

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            failed_step_id=1
        )

        strategy = manager.determine_strategy(context, plan)

        assert strategy == ReplanStrategy.FALLBACK

    def test_determine_strategy_timeout_full(self):
        """タイムアウトでFULL"""
        manager = ReplanManager()
        plan = self._create_sample_plan()

        context = ReplanContext(
            trigger=ReplanTrigger.TIMEOUT,
            original_query="query"
        )

        strategy = manager.determine_strategy(context, plan)

        assert strategy == ReplanStrategy.FULL

    def test_determine_strategy_user_feedback_full(self):
        """「最初から」でFULL"""
        manager = ReplanManager()
        plan = self._create_sample_plan()

        context = ReplanContext(
            trigger=ReplanTrigger.USER_FEEDBACK,
            original_query="query",
            user_feedback="最初からやり直して"
        )

        strategy = manager.determine_strategy(context, plan)

        assert strategy == ReplanStrategy.FULL

    def test_determine_strategy_user_feedback_partial(self):
        """通常フィードバックでPARTIAL"""
        manager = ReplanManager()
        plan = self._create_sample_plan()

        context = ReplanContext(
            trigger=ReplanTrigger.USER_FEEDBACK,
            original_query="query",
            user_feedback="ここを修正して"
        )

        strategy = manager.determine_strategy(context, plan)

        assert strategy == ReplanStrategy.PARTIAL

    def test_determine_strategy_early_failure_full(self):
        """序盤失敗でFULL"""
        manager = ReplanManager()
        plan = self._create_sample_plan()  # 3ステップ

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            failed_step_id=1  # 最初のステップ = 33%未満
        )

        strategy = manager.determine_strategy(context, plan)

        assert strategy == ReplanStrategy.FULL

    def test_determine_strategy_late_failure_partial(self):
        """後半失敗でPARTIAL"""
        manager = ReplanManager()
        plan = self._create_sample_plan()  # 3ステップ

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            failed_step_id=3  # 最後のステップ = 100%
        )

        strategy = manager.determine_strategy(context, plan)

        assert strategy == ReplanStrategy.PARTIAL

    @patch("grace.replan.create_planner")
    def test_create_new_plan_full(self, mock_create_planner):
        """FULL戦略での新計画作成"""
        mock_planner = MagicMock()
        mock_planner.create_plan.return_value = ExecutionPlan(
            original_query="新クエリ",
            complexity=0.5,
            estimated_steps=2,
            requires_confirmation=False,
            steps=[
                PlanStep(step_id=1, action="rag_search", description="検索", expected_output="結果")
            ],
            success_criteria="成功"
        )
        mock_create_planner.return_value = mock_planner

        manager = ReplanManager()
        plan = self._create_sample_plan()

        context = ReplanContext(
            trigger=ReplanTrigger.TIMEOUT,
            original_query="query",
            replan_count=0
        )

        result = manager.create_new_plan(context, ReplanStrategy.FULL, plan)

        assert result.success is True
        assert result.strategy == ReplanStrategy.FULL
        assert result.new_plan is not None
        assert result.replan_count == 1
        assert len(manager.history) == 1

    @patch("grace.replan.create_planner")
    def test_create_new_plan_partial(self, mock_create_planner):
        """PARTIAL戦略での新計画作成"""
        mock_planner = MagicMock()
        mock_planner.create_plan.return_value = ExecutionPlan(
            original_query="部分",
            complexity=0.3,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[
                PlanStep(step_id=1, action="reasoning", description="新推論", expected_output="結果")
            ],
            success_criteria="成功"
        )
        mock_create_planner.return_value = mock_planner

        manager = ReplanManager()
        plan = self._create_sample_plan()  # 3ステップ

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            failed_step_id=2,
            replan_count=0
        )

        result = manager.create_new_plan(context, ReplanStrategy.PARTIAL, plan)

        assert result.success is True
        assert result.strategy == ReplanStrategy.PARTIAL
        assert result.new_plan is not None

    def test_create_new_plan_fallback(self):
        """FALLBACK戦略"""
        manager = ReplanManager()

        step = PlanStep(
            step_id=1,
            action="rag_search",
            description="検索",
            expected_output="結果",
            fallback="web_search"
        )
        plan = ExecutionPlan(
            original_query="query",
            complexity=0.3,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[step],
            success_criteria="成功"
        )

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            failed_step_id=1
        )

        result = manager.create_new_plan(context, ReplanStrategy.FALLBACK, plan)

        assert result.success is True
        assert result.strategy == ReplanStrategy.FALLBACK
        assert result.new_plan is not None
        assert result.new_plan.steps[0].action == "web_search"
        assert "[代替]" in result.new_plan.steps[0].description

    def test_create_new_plan_skip(self):
        """SKIP戦略"""
        manager = ReplanManager()
        plan = self._create_sample_plan()

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            failed_step_id=2
        )

        result = manager.create_new_plan(context, ReplanStrategy.SKIP, plan)

        assert result.success is True
        assert result.strategy == ReplanStrategy.SKIP
        assert result.new_plan is not None
        # ステップ2がスキップされている
        step_ids = [s.step_id for s in result.new_plan.steps]
        assert 2 not in step_ids

    def test_create_new_plan_abort(self):
        """ABORT戦略"""
        manager = ReplanManager()
        plan = self._create_sample_plan()

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            replan_count=3
        )

        result = manager.create_new_plan(context, ReplanStrategy.ABORT, plan)

        assert result.success is False
        assert result.strategy == ReplanStrategy.ABORT
        assert result.new_plan is None
        assert "中断" in result.reason

    def test_can_replan_true(self):
        """リプラン可能"""
        manager = ReplanManager()
        manager.max_replans = 3

        assert manager.can_replan(0) is True
        assert manager.can_replan(2) is True

    def test_can_replan_false(self):
        """リプラン不可"""
        manager = ReplanManager()
        manager.max_replans = 3

        assert manager.can_replan(3) is False
        assert manager.can_replan(5) is False

    def test_history_management(self):
        """履歴管理"""
        manager = ReplanManager()

        # 初期状態
        assert manager.get_history() == []

        # ABORT結果を追加（plannerなしで作成可能）
        plan = self._create_sample_plan()
        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            replan_count=10
        )
        manager.create_new_plan(context, ReplanStrategy.ABORT, plan)

        # 履歴に追加されている
        history = manager.get_history()
        assert len(history) == 1

        # クリア
        manager.clear_history()
        assert manager.get_history() == []


class TestReplanOrchestrator:
    """ReplanOrchestratorのテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    def _create_sample_plan(self) -> ExecutionPlan:
        """サンプル計画を作成"""
        steps = [
            PlanStep(
                step_id=1,
                action="rag_search",
                description="検索1",
                expected_output="結果1"
            ),
            PlanStep(
                step_id=2,
                action="reasoning",
                description="推論1",
                expected_output="結果2"
            ),
        ]
        return ExecutionPlan(
            original_query="テストクエリ",
            complexity=0.3,
            estimated_steps=2,
            requires_confirmation=False,
            steps=steps,
            success_criteria="成功"
        )

    def test_initialization(self):
        """初期化テスト"""
        orchestrator = ReplanOrchestrator()

        assert orchestrator.config is not None
        assert orchestrator.replan_manager is not None

    def test_initialization_with_manager(self):
        """カスタムマネージャーでの初期化"""
        manager = ReplanManager()
        orchestrator = ReplanOrchestrator(replan_manager=manager)

        assert orchestrator.replan_manager is manager

    @patch("grace.replan.create_planner")
    def test_handle_step_failure_with_replan(self, mock_create_planner):
        """ステップ失敗でリプラン実行"""
        mock_planner = MagicMock()
        mock_planner.create_plan.return_value = ExecutionPlan(
            original_query="new",
            complexity=0.3,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[
                PlanStep(step_id=1, action="rag_search", description="新検索", expected_output="結果")
            ],
            success_criteria="成功"
        )
        mock_create_planner.return_value = mock_planner

        orchestrator = ReplanOrchestrator()
        plan = self._create_sample_plan()
        step_result = StepResult(step_id=1, status="failed", confidence=0.0, error="エラー")

        result = orchestrator.handle_step_failure(
            step_result=step_result,
            current_plan=plan,
            completed_results={},
            replan_count=0
        )

        assert result is not None
        assert result.success is True

    def test_handle_step_failure_no_replan(self):
        """成功ステップではリプランなし"""
        orchestrator = ReplanOrchestrator()
        orchestrator.replan_manager.confidence_threshold = 0.3

        plan = self._create_sample_plan()
        step_result = StepResult(step_id=1, status="success", confidence=0.9)

        result = orchestrator.handle_step_failure(
            step_result=step_result,
            current_plan=plan,
            completed_results={},
            replan_count=0
        )

        assert result is None

    def test_handle_step_failure_max_replans(self):
        """最大リプラン回数でリプランなし"""
        orchestrator = ReplanOrchestrator()
        orchestrator.replan_manager.max_replans = 2

        plan = self._create_sample_plan()
        step_result = StepResult(step_id=1, status="failed", confidence=0.0)

        result = orchestrator.handle_step_failure(
            step_result=step_result,
            current_plan=plan,
            completed_results={},
            replan_count=2
        )

        assert result is None

    @patch("grace.replan.create_planner")
    def test_handle_user_feedback_with_replan(self, mock_create_planner):
        """フィードバックでリプラン実行"""
        mock_planner = MagicMock()
        mock_planner.create_plan.return_value = ExecutionPlan(
            original_query="feedback",
            complexity=0.3,
            estimated_steps=1,
            requires_confirmation=False,
            steps=[
                PlanStep(step_id=1, action="rag_search", description="検索", expected_output="結果")
            ],
            success_criteria="成功"
        )
        mock_create_planner.return_value = mock_planner

        orchestrator = ReplanOrchestrator()
        plan = self._create_sample_plan()

        result = orchestrator.handle_user_feedback(
            feedback="この部分を修正してください",
            current_plan=plan,
            completed_results={},
            replan_count=0
        )

        assert result is not None
        assert result.success is True

    def test_handle_user_feedback_no_replan(self):
        """修正要求なしではリプランなし"""
        orchestrator = ReplanOrchestrator()
        plan = self._create_sample_plan()

        result = orchestrator.handle_user_feedback(
            feedback="とても良い結果です",
            current_plan=plan,
            completed_results={},
            replan_count=0
        )

        assert result is None


class TestFactoryFunctions:
    """ファクトリ関数のテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    def test_create_replan_manager_default(self):
        """デフォルト設定でマネージャー作成"""
        manager = create_replan_manager()

        assert isinstance(manager, ReplanManager)
        assert manager.config is not None

    def test_create_replan_manager_with_config(self):
        """カスタム設定でマネージャー作成"""
        config = GraceConfig()
        manager = create_replan_manager(config=config)

        assert manager.config is config

    def test_create_replan_orchestrator_default(self):
        """デフォルト設定でオーケストレーター作成"""
        orchestrator = create_replan_orchestrator()

        assert isinstance(orchestrator, ReplanOrchestrator)
        assert orchestrator.replan_manager is not None

    def test_create_replan_orchestrator_with_manager(self):
        """カスタムマネージャーでオーケストレーター作成"""
        manager = ReplanManager()
        orchestrator = create_replan_orchestrator(replan_manager=manager)

        assert orchestrator.replan_manager is manager


class TestEnhancedQueryGeneration:
    """クエリ強化のテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    def test_enhance_query_with_error(self):
        """エラー情報付きクエリ"""
        manager = ReplanManager()

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="元のクエリ",
            error_message="検索失敗"
        )

        enhanced = manager._enhance_query_with_context("元のクエリ", context)

        assert "元のクエリ" in enhanced
        assert "検索失敗" in enhanced
        assert "【追加情報】" in enhanced

    def test_enhance_query_with_completed_steps(self):
        """完了ステップ情報付きクエリ"""
        manager = ReplanManager()

        result1 = StepResult(step_id=1, status="success", confidence=0.9)
        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="元のクエリ",
            completed_results={1: result1}
        )

        enhanced = manager._enhance_query_with_context("元のクエリ", context)

        assert "ステップ1は完了済み" in enhanced

    def test_enhance_query_with_feedback(self):
        """フィードバック付きクエリ"""
        manager = ReplanManager()

        context = ReplanContext(
            trigger=ReplanTrigger.USER_FEEDBACK,
            original_query="元のクエリ",
            user_feedback="もっと詳しく"
        )

        enhanced = manager._enhance_query_with_context("元のクエリ", context)

        assert "もっと詳しく" in enhanced

    def test_enhance_query_no_hints(self):
        """追加情報なしの場合"""
        manager = ReplanManager()

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="元のクエリ"
        )

        enhanced = manager._enhance_query_with_context("元のクエリ", context)

        assert enhanced == "元のクエリ"


class TestDependencyHandling:
    """依存関係処理のテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    def test_skip_updates_dependencies(self):
        """SKIP時に依存関係が更新される"""
        manager = ReplanManager()

        steps = [
            PlanStep(
                step_id=1,
                action="rag_search",
                description="検索1",
                expected_output="結果1"
            ),
            PlanStep(
                step_id=2,
                action="reasoning",
                description="推論",
                depends_on=[1],
                expected_output="結果2"
            ),
            PlanStep(
                step_id=3,
                action="rag_search",
                description="検索2",
                depends_on=[2],
                expected_output="結果3"
            ),
        ]
        plan = ExecutionPlan(
            original_query="query",
            complexity=0.5,
            estimated_steps=3,
            requires_confirmation=False,
            steps=steps,
            success_criteria="成功"
        )

        context = ReplanContext(
            trigger=ReplanTrigger.STEP_FAILED,
            original_query="query",
            failed_step_id=2
        )

        result = manager.create_new_plan(context, ReplanStrategy.SKIP, plan)

        assert result.success is True
        # ステップ3の依存関係からステップ2が削除されている
        step3 = next((s for s in result.new_plan.steps if s.step_id == 3), None)
        assert step3 is not None
        assert 2 not in step3.depends_on