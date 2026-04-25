"""
GRACE Intervention Tests
介入システムのテスト
"""

import pytest
from unittest.mock import MagicMock

from grace.intervention import (
    InterventionRequest,
    InterventionResponse,
    InterventionAction,
    FeedbackRecord,
    InterventionHandler,
    DynamicThresholdAdjuster,
    ConfirmationFlow,
    create_intervention_handler,
    create_threshold_adjuster,
)
from grace.confidence import InterventionLevel, ActionDecision
from grace.schemas import ExecutionPlan, PlanStep
from grace.config import GraceConfig, reset_config


class TestInterventionRequest:
    """InterventionRequestのテスト"""

    def test_default_values(self):
        """デフォルト値"""
        request = InterventionRequest(
            level=InterventionLevel.CONFIRM,
            message="テスト"
        )

        assert request.level == InterventionLevel.CONFIRM
        assert request.message == "テスト"
        assert request.timeout_seconds == 300
        assert request.is_blocking is True
        assert request.created_at is not None

    def test_requires_response_for_confirm(self):
        """CONFIRMはレスポンス必要"""
        request = InterventionRequest(
            level=InterventionLevel.CONFIRM,
            message="テスト"
        )
        assert request.requires_response is True

    def test_requires_response_for_escalate(self):
        """ESCALATEはレスポンス必要"""
        request = InterventionRequest(
            level=InterventionLevel.ESCALATE,
            message="テスト"
        )
        assert request.requires_response is True

    def test_requires_response_for_silent(self):
        """SILENTはレスポンス不要"""
        request = InterventionRequest(
            level=InterventionLevel.SILENT,
            message="テスト"
        )
        assert request.requires_response is False

    def test_requires_response_for_notify(self):
        """NOTIFYはレスポンス不要"""
        request = InterventionRequest(
            level=InterventionLevel.NOTIFY,
            message="テスト"
        )
        assert request.requires_response is False


class TestInterventionResponse:
    """InterventionResponseのテスト"""

    def test_should_continue_for_proceed(self):
        """PROCEEDは継続"""
        response = InterventionResponse(action=InterventionAction.PROCEED)
        assert response.should_continue is True

    def test_should_continue_for_modify(self):
        """MODIFYは継続"""
        response = InterventionResponse(action=InterventionAction.MODIFY)
        assert response.should_continue is True

    def test_should_continue_for_cancel(self):
        """CANCELは継続しない"""
        response = InterventionResponse(action=InterventionAction.CANCEL)
        assert response.should_continue is False

    def test_should_continue_for_retry(self):
        """RETRYは継続"""
        response = InterventionResponse(action=InterventionAction.RETRY)
        assert response.should_continue is True

    def test_should_continue_for_skip(self):
        """SKIPは継続"""
        response = InterventionResponse(action=InterventionAction.SKIP)
        assert response.should_continue is True


class TestInterventionHandler:
    """InterventionHandlerのテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    def test_handle_silent(self):
        """SILENTレベル処理"""
        handler = InterventionHandler()
        decision = ActionDecision(
            level=InterventionLevel.SILENT,
            confidence_score=0.95,
            reason="高信頼度"
        )

        response = handler.handle(decision)

        assert response.action == InterventionAction.PROCEED

    def test_handle_notify_with_callback(self):
        """NOTIFYレベル処理（コールバックあり）"""
        notify_messages = []

        def on_notify(message):
            notify_messages.append(message)

        handler = InterventionHandler(on_notify=on_notify)
        decision = ActionDecision(
            level=InterventionLevel.NOTIFY,
            confidence_score=0.8,
            reason="中程度の信頼度"
        )

        response = handler.handle(decision)

        assert response.action == InterventionAction.PROCEED
        assert len(notify_messages) == 1

    def test_handle_confirm_with_callback(self):
        """CONFIRMレベル処理（コールバックあり）"""
        def on_confirm(request):
            return InterventionResponse(action=InterventionAction.PROCEED)

        handler = InterventionHandler(on_confirm=on_confirm)
        decision = ActionDecision(
            level=InterventionLevel.CONFIRM,
            confidence_score=0.5,
            reason="低信頼度"
        )

        response = handler.handle(decision)

        assert response.action == InterventionAction.PROCEED

    def test_handle_confirm_cancel(self):
        """CONFIRMレベルでキャンセル"""
        def on_confirm(request):
            return InterventionResponse(action=InterventionAction.CANCEL)

        handler = InterventionHandler(on_confirm=on_confirm)
        decision = ActionDecision(
            level=InterventionLevel.CONFIRM,
            confidence_score=0.5,
            reason="低信頼度"
        )

        response = handler.handle(decision)

        assert response.action == InterventionAction.CANCEL

    def test_handle_escalate_with_callback(self):
        """ESCALATEレベル処理（コールバックあり）"""
        def on_escalate(request):
            return InterventionResponse(
                action=InterventionAction.INPUT,
                user_input="追加情報"
            )

        handler = InterventionHandler(on_escalate=on_escalate)
        decision = ActionDecision(
            level=InterventionLevel.ESCALATE,
            confidence_score=0.2,
            reason="非常に低い信頼度"
        )

        response = handler.handle(decision)

        assert response.action == InterventionAction.INPUT
        assert response.user_input == "追加情報"

    def test_handle_escalate_without_callback_auto_proceed(self):
        """ESCALATEレベル（コールバックなし、自動進行設定）"""
        config = GraceConfig()
        config.intervention.auto_proceed_on_timeout = True

        handler = InterventionHandler(config=config)
        decision = ActionDecision(
            level=InterventionLevel.ESCALATE,
            confidence_score=0.2,
            reason="非常に低い信頼度"
        )

        response = handler.handle(decision)

        assert response.action == InterventionAction.PROCEED
        assert response.timeout_reached is True

    def test_handle_escalate_without_callback_cancel(self):
        """ESCALATEレベル（コールバックなし、キャンセル設定）"""
        config = GraceConfig()
        config.intervention.auto_proceed_on_timeout = False

        handler = InterventionHandler(config=config)
        decision = ActionDecision(
            level=InterventionLevel.ESCALATE,
            confidence_score=0.2,
            reason="非常に低い信頼度"
        )

        response = handler.handle(decision)

        assert response.action == InterventionAction.CANCEL
        assert response.timeout_reached is True

    def test_request_confirmation(self):
        """計画確認リクエスト"""
        def on_confirm(request):
            assert request.plan is not None
            return InterventionResponse(action=InterventionAction.PROCEED)

        handler = InterventionHandler(on_confirm=on_confirm)
        plan = ExecutionPlan(
            original_query="テスト",
            complexity=0.5,
            estimated_steps=1,
            requires_confirmation=True,
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

        response = handler.request_confirmation(plan, confidence=0.7)

        assert response.action == InterventionAction.PROCEED

    def test_request_clarification(self):
        """追加情報リクエスト"""
        def on_escalate(request):
            assert request.question is not None
            return InterventionResponse(
                action=InterventionAction.INPUT,
                user_input="回答"
            )

        handler = InterventionHandler(on_escalate=on_escalate)

        response = handler.request_clarification(
            question="どちらですか？",
            reason="複数の解釈が可能",
            options=["オプションA", "オプションB"]
        )

        assert response.action == InterventionAction.INPUT
        assert response.user_input == "回答"

    def test_notify_status(self):
        """ステータス通知"""
        messages = []

        def on_notify(message):
            messages.append(message)

        handler = InterventionHandler(on_notify=on_notify)
        handler.notify_status("処理中...")

        assert len(messages) == 1
        assert messages[0] == "処理中..."

    def test_history_recording(self):
        """履歴記録"""
        handler = InterventionHandler()
        decision = ActionDecision(
            level=InterventionLevel.SILENT,
            confidence_score=0.95,
            reason="高信頼度"
        )

        handler.handle(decision)

        history = handler.get_history()
        assert len(history) == 1
        assert history[0]["level"] == "silent"
        assert history[0]["action"] == "auto_proceed"

    def test_clear_history(self):
        """履歴クリア"""
        handler = InterventionHandler()
        decision = ActionDecision(
            level=InterventionLevel.SILENT,
            confidence_score=0.95,
            reason="高信頼度"
        )

        handler.handle(decision)
        handler.clear_history()

        assert len(handler.get_history()) == 0


class TestDynamicThresholdAdjuster:
    """DynamicThresholdAdjusterのテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    def test_initial_thresholds(self):
        """初期閾値"""
        adjuster = DynamicThresholdAdjuster()
        thresholds = adjuster.get_current_thresholds()

        assert thresholds["silent"] == 0.9
        assert thresholds["notify"] == 0.7
        assert thresholds["confirm"] == 0.4

    def test_get_level_silent(self):
        """SILENTレベル判定"""
        adjuster = DynamicThresholdAdjuster()
        level = adjuster.get_level(0.95)

        assert level == InterventionLevel.SILENT

    def test_get_level_notify(self):
        """NOTIFYレベル判定"""
        adjuster = DynamicThresholdAdjuster()
        level = adjuster.get_level(0.8)

        assert level == InterventionLevel.NOTIFY

    def test_get_level_confirm(self):
        """CONFIRMレベル判定"""
        adjuster = DynamicThresholdAdjuster()
        level = adjuster.get_level(0.5)

        assert level == InterventionLevel.CONFIRM

    def test_get_level_escalate(self):
        """ESCALATEレベル判定"""
        adjuster = DynamicThresholdAdjuster()
        level = adjuster.get_level(0.2)

        assert level == InterventionLevel.ESCALATE

    def test_record_feedback(self):
        """フィードバック記録"""
        adjuster = DynamicThresholdAdjuster(min_samples=5)

        adjuster.record_feedback(0.8, True)
        adjuster.record_feedback(0.7, True)

        assert len(adjuster.feedback_history) == 2

    def test_raise_thresholds_on_high_fp_rate(self):
        """高い偽陽性率で閾値を上げる"""
        adjuster = DynamicThresholdAdjuster(min_samples=5, learning_rate=0.05)
        original_silent = adjuster.silent_threshold

        # 高信頼度だが誤り（偽陽性）を多く記録
        for _ in range(5):
            adjuster.record_feedback(0.8, False)  # 高信頼度だが不正解

        # 閾値が上がっているはず
        assert adjuster.silent_threshold > original_silent

    def test_lower_thresholds_on_high_fn_rate(self):
        """高い偽陰性率で閾値を下げる"""
        adjuster = DynamicThresholdAdjuster(min_samples=5, learning_rate=0.05)
        original_confirm = adjuster.confirm_threshold

        # 低信頼度だが正解（偽陰性）を多く記録
        for _ in range(5):
            adjuster.record_feedback(0.3, True)  # 低信頼度だが正解

        # 閾値が下がっているはず
        assert adjuster.confirm_threshold < original_confirm

    def test_reset_thresholds(self):
        """閾値リセット"""
        adjuster = DynamicThresholdAdjuster(min_samples=5, learning_rate=0.1)

        # 閾値を変更
        for _ in range(5):
            adjuster.record_feedback(0.8, False)

        # リセット
        adjuster.reset_thresholds()
        thresholds = adjuster.get_current_thresholds()

        assert thresholds["silent"] == 0.9
        assert thresholds["notify"] == 0.7
        assert thresholds["confirm"] == 0.4
        assert len(adjuster.feedback_history) == 0

    def test_threshold_bounds(self):
        """閾値の上下限"""
        adjuster = DynamicThresholdAdjuster(min_samples=5, learning_rate=0.2)

        # 大量の偽陽性で閾値を上げ続ける
        for _ in range(20):
            adjuster.record_feedback(0.8, False)

        # 上限を超えないはず
        thresholds = adjuster.get_current_thresholds()
        assert thresholds["silent"] <= 0.95
        assert thresholds["notify"] <= 0.85
        assert thresholds["confirm"] <= 0.6


class TestConfirmationFlow:
    """ConfirmationFlowのテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    @pytest.fixture
    def sample_plan(self):
        """テスト用計画"""
        return ExecutionPlan(
            original_query="テスト",
            complexity=0.5,
            estimated_steps=1,
            requires_confirmation=True,
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

    def test_confirm_plan_proceed(self, sample_plan):
        """計画確認: 承認"""
        def on_confirm(request):
            return InterventionResponse(action=InterventionAction.PROCEED)

        handler = InterventionHandler(on_confirm=on_confirm)
        flow = ConfirmationFlow(handler)

        confirmed, result_plan = flow.confirm_plan(sample_plan, confidence=0.7)

        assert confirmed is True
        assert result_plan is sample_plan

    def test_confirm_plan_cancel(self, sample_plan):
        """計画確認: キャンセル"""
        def on_confirm(request):
            return InterventionResponse(action=InterventionAction.CANCEL)

        handler = InterventionHandler(on_confirm=on_confirm)
        flow = ConfirmationFlow(handler)

        confirmed, result_plan = flow.confirm_plan(sample_plan, confidence=0.7)

        assert confirmed is False
        assert result_plan is None

    def test_confirm_plan_modify(self, sample_plan):
        """計画確認: 修正して承認"""
        call_count = [0]

        def on_confirm(request):
            call_count[0] += 1
            if call_count[0] == 1:
                # 最初は修正
                modified = ExecutionPlan(
                    original_query="修正済み",
                    complexity=0.5,
                    estimated_steps=1,
                    requires_confirmation=True,
                    steps=sample_plan.steps,
                    success_criteria="テスト"
                )
                return InterventionResponse(
                    action=InterventionAction.MODIFY,
                    modified_plan=modified
                )
            else:
                # 2回目は承認
                return InterventionResponse(action=InterventionAction.PROCEED)

        handler = InterventionHandler(on_confirm=on_confirm)
        flow = ConfirmationFlow(handler)

        confirmed, result_plan = flow.confirm_plan(sample_plan, confidence=0.7)

        assert confirmed is True
        assert result_plan.original_query == "修正済み"
        assert call_count[0] == 2

    def test_confirm_plan_max_modifications(self, sample_plan):
        """計画確認: 最大修正回数超過"""
        def on_confirm(request):
            # 常に修正（修正計画なし = キャンセル扱い）
            return InterventionResponse(action=InterventionAction.MODIFY)

        handler = InterventionHandler(on_confirm=on_confirm)
        flow = ConfirmationFlow(handler, max_modifications=3)

        confirmed, result_plan = flow.confirm_plan(sample_plan, confidence=0.7)

        # 修正計画なしの場合はキャンセル
        assert confirmed is False


class TestFactoryFunctions:
    """ファクトリ関数のテスト"""

    def test_create_intervention_handler(self):
        """InterventionHandler作成"""
        handler = create_intervention_handler()
        assert isinstance(handler, InterventionHandler)

    def test_create_intervention_handler_with_callbacks(self):
        """コールバック付きInterventionHandler作成"""
        def on_notify(msg):
            pass

        handler = create_intervention_handler(on_notify=on_notify)
        assert handler.on_notify is not None

    def test_create_threshold_adjuster(self):
        """ThresholdAdjuster作成"""
        adjuster = create_threshold_adjuster()
        assert isinstance(adjuster, DynamicThresholdAdjuster)

    def test_create_threshold_adjuster_custom_rate(self):
        """カスタム学習率でThresholdAdjuster作成"""
        adjuster = create_threshold_adjuster(learning_rate=0.1)
        assert adjuster.learning_rate == 0.1