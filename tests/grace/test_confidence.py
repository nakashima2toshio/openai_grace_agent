"""
GRACE Confidence Tests
信頼度計算システムのテスト
"""

import pytest
from unittest.mock import MagicMock, patch

from grace.confidence import (
    ConfidenceFactors,
    ConfidenceScore,
    ActionDecision,
    InterventionLevel,
    ConfidenceCalculator,
    LLMSelfEvaluator,
    SourceAgreementCalculator,
    QueryCoverageCalculator,
    ConfidenceAggregator,
    create_confidence_calculator,
)
from grace.config import GraceConfig, reset_config


class TestConfidenceFactors:
    """ConfidenceFactorsのテスト"""

    def test_default_values(self):
        """デフォルト値"""
        factors = ConfidenceFactors()

        assert factors.search_result_count == 0
        assert factors.search_avg_score == 0.0
        assert factors.search_score_variance == 1.0
        assert factors.source_agreement == 0.0
        assert factors.source_count == 0
        assert factors.llm_self_confidence == 0.5
        assert factors.tool_success_rate == 1.0
        assert factors.query_coverage == 0.0

    def test_custom_values(self):
        """カスタム値"""
        factors = ConfidenceFactors(
            search_result_count=5,
            search_avg_score=0.85,
            search_score_variance=0.1,
            source_agreement=0.9,
            source_count=3,
            llm_self_confidence=0.8,
            tool_success_rate=1.0,
            query_coverage=0.95
        )

        assert factors.search_result_count == 5
        assert factors.search_avg_score == 0.85
        assert factors.source_agreement == 0.9


class TestConfidenceScore:
    """ConfidenceScoreのテスト"""

    def test_score_level_high(self):
        """高信頼度レベル"""
        score = ConfidenceScore(
            score=0.95,
            factors=ConfidenceFactors()
        )
        assert score.level == "high"

    def test_score_level_medium(self):
        """中信頼度レベル"""
        score = ConfidenceScore(
            score=0.75,
            factors=ConfidenceFactors()
        )
        assert score.level == "medium"

    def test_score_level_low(self):
        """低信頼度レベル"""
        score = ConfidenceScore(
            score=0.5,
            factors=ConfidenceFactors()
        )
        assert score.level == "low"

    def test_score_level_very_low(self):
        """非常に低い信頼度レベル"""
        score = ConfidenceScore(
            score=0.2,
            factors=ConfidenceFactors()
        )
        assert score.level == "very_low"


class TestActionDecision:
    """ActionDecisionのテスト"""

    def test_silent_should_proceed(self):
        """SILENTレベルは自動進行"""
        decision = ActionDecision(
            level=InterventionLevel.SILENT,
            confidence_score=0.95,
            reason="Test"
        )
        assert decision.should_proceed is True
        assert decision.needs_confirmation is False
        assert decision.needs_user_input is False

    def test_notify_should_proceed(self):
        """NOTIFYレベルは自動進行"""
        decision = ActionDecision(
            level=InterventionLevel.NOTIFY,
            confidence_score=0.8,
            reason="Test"
        )
        assert decision.should_proceed is True
        assert decision.needs_confirmation is False

    def test_confirm_needs_confirmation(self):
        """CONFIRMレベルは確認が必要"""
        decision = ActionDecision(
            level=InterventionLevel.CONFIRM,
            confidence_score=0.5,
            reason="Test"
        )
        assert decision.should_proceed is False
        assert decision.needs_confirmation is True
        assert decision.needs_user_input is False

    def test_escalate_needs_user_input(self):
        """ESCALATEレベルはユーザー入力が必要"""
        decision = ActionDecision(
            level=InterventionLevel.ESCALATE,
            confidence_score=0.2,
            reason="Test"
        )
        assert decision.should_proceed is False
        assert decision.needs_confirmation is False
        assert decision.needs_user_input is True


class TestConfidenceCalculator:
    """ConfidenceCalculatorのテスト"""

    def setup_method(self):
        """各テスト前の準備"""
        reset_config()

    def test_calculate_high_confidence(self):
        """高信頼度の計算"""
        calculator = ConfidenceCalculator()
        factors = ConfidenceFactors(
            search_result_count=5,
            search_avg_score=0.9,
            search_max_score=0.95,  # 最高スコアを設定
            search_score_variance=0.05,
            source_agreement=0.95,
            source_count=3,
            llm_self_confidence=0.9,
            tool_success_rate=1.0,
            query_coverage=0.95
        )

        result = calculator.calculate(factors)

        assert isinstance(result, ConfidenceScore)
        assert result.score > 0.8
        assert len(result.breakdown) == 5

    def test_calculate_low_confidence_no_results(self):
        """検索結果0件の場合は低信頼度"""
        calculator = ConfidenceCalculator()
        # 検索ステップとして明示
        factors = ConfidenceFactors(
            search_result_count=0,
            search_avg_score=0.0,
            search_max_score=0.0,
            llm_self_confidence=0.5,
            tool_success_rate=1.0,
            is_search_step=True
        )

        result = calculator.calculate(factors)

        assert result.score < 0.5
        assert "no_search_results" in result.penalties_applied

    def test_calculate_penalty_no_sources(self):
        """ソースなしはペナルティ"""
        calculator = ConfidenceCalculator()
        factors = ConfidenceFactors(
            search_result_count=3,
            search_avg_score=0.8,
            search_max_score=0.85,
            source_count=0,
            llm_self_confidence=0.5,
            tool_success_rate=1.0,
            is_search_step=False
        )

        result = calculator.calculate(factors)

        assert "no_sources" in result.penalties_applied

    def test_calculate_penalty_tool_failures(self):
        """ツール失敗はペナルティ"""
        calculator = ConfidenceCalculator()
        factors = ConfidenceFactors(
            search_result_count=3,
            search_avg_score=0.8,
            source_count=2,
            llm_self_confidence=0.8,
            tool_success_rate=0.5
        )

        result = calculator.calculate(factors)

        assert any("tool_failures" in p for p in result.penalties_applied)

    def test_decide_action_silent(self):
        """SILENT判定"""
        calculator = ConfidenceCalculator()
        score = ConfidenceScore(score=0.95, factors=ConfidenceFactors())

        decision = calculator.decide_action(score)

        assert decision.level == InterventionLevel.SILENT
        assert decision.should_proceed is True

    def test_decide_action_notify(self):
        """NOTIFY判定"""
        calculator = ConfidenceCalculator()
        score = ConfidenceScore(score=0.8, factors=ConfidenceFactors())

        decision = calculator.decide_action(score)

        assert decision.level == InterventionLevel.NOTIFY

    def test_decide_action_confirm(self):
        """CONFIRM判定"""
        calculator = ConfidenceCalculator()
        score = ConfidenceScore(score=0.5, factors=ConfidenceFactors())

        decision = calculator.decide_action(score)

        assert decision.level == InterventionLevel.CONFIRM
        assert decision.needs_confirmation is True

    def test_decide_action_escalate(self):
        """ESCALATE判定"""
        calculator = ConfidenceCalculator()
        score = ConfidenceScore(score=0.2, factors=ConfidenceFactors())

        decision = calculator.decide_action(score)

        assert decision.level == InterventionLevel.ESCALATE
        assert decision.needs_user_input is True

    def test_weights_validation(self):
        """重み合計の検証"""
        config = GraceConfig()
        # 重みの合計が1.0であることを確認
        weights = config.confidence.weights
        total = (
            weights.search_quality +
            weights.source_agreement +
            weights.llm_self_eval +
            weights.tool_success +
            weights.query_coverage
        )
        assert abs(total - 1.0) < 0.01


class TestLLMSelfEvaluator:
    """LLMSelfEvaluatorのテスト"""

    @patch("grace.confidence.genai.Client")
    def test_evaluate_success(self, mock_client_class):
        """評価成功"""
        mock_response = MagicMock()
        mock_response.text = "0.85"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        evaluator = LLMSelfEvaluator()
        result = evaluator.evaluate(
            query="Pythonとは",
            answer="Pythonはプログラミング言語です",
            sources=["doc1", "doc2"]
        )

        assert result == 0.85

    @patch("grace.confidence.genai.Client")
    def test_evaluate_parse_error(self, mock_client_class):
        """パースエラー時はデフォルト値"""
        mock_response = MagicMock()
        mock_response.text = "invalid"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        evaluator = LLMSelfEvaluator()
        result = evaluator.evaluate(
            query="テスト",
            answer="回答"
        )

        assert result == 0.5  # デフォルト値

    @patch("grace.confidence.genai.Client")
    def test_evaluate_clamp_values(self, mock_client_class):
        """値は0.0-1.0の範囲にクランプ"""
        mock_response = MagicMock()
        mock_response.text = "1.5"  # 範囲外

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        evaluator = LLMSelfEvaluator()
        result = evaluator.evaluate(query="テスト", answer="回答")

        assert result == 1.0  # クランプされる


class TestSourceAgreementCalculator:
    """SourceAgreementCalculatorのテスト"""

    def test_single_source_full_agreement(self):
        """単一ソースは完全一致"""
        with patch("grace.confidence.genai.Client"):
            calculator = SourceAgreementCalculator()
            result = calculator.calculate(["answer1"])

            assert result == 1.0

    @patch("grace.confidence.genai.Client")
    def test_calculate_with_mock(self, mock_client_class):
        """複数ソースの一致度計算"""
        # Embeddingレスポンスをモック
        mock_embedding_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_embedding_response.embeddings = [mock_embedding]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_embedding_response
        mock_client_class.return_value = mock_client

        calculator = SourceAgreementCalculator()
        result = calculator.calculate(["answer1", "answer2"])

        # 同じEmbeddingなのでコサイン類似度は1.0
        assert 0.0 <= result <= 1.0


class TestQueryCoverageCalculator:
    """QueryCoverageCalculatorのテスト"""

    @patch("grace.confidence.genai.Client")
    def test_calculate_success(self, mock_client_class):
        """網羅度計算成功"""
        mock_response = MagicMock()
        mock_response.text = "0.9"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        calculator = QueryCoverageCalculator()
        result = calculator.calculate(
            query="Pythonの特徴を教えて",
            answer="Pythonは動的型付け言語で..."
        )

        assert result == 0.9


class TestConfidenceAggregator:
    """ConfidenceAggregatorのテスト"""

    def test_aggregate_mean(self):
        """平均集計"""
        aggregator = ConfidenceAggregator()
        scores = [
            ConfidenceScore(score=0.8, factors=ConfidenceFactors()),
            ConfidenceScore(score=0.6, factors=ConfidenceFactors()),
            ConfidenceScore(score=0.7, factors=ConfidenceFactors()),
        ]

        result = aggregator.aggregate(scores, method="mean")

        assert abs(result - 0.7) < 0.01

    def test_aggregate_min(self):
        """最小値集計"""
        aggregator = ConfidenceAggregator()
        scores = [
            ConfidenceScore(score=0.8, factors=ConfidenceFactors()),
            ConfidenceScore(score=0.6, factors=ConfidenceFactors()),
            ConfidenceScore(score=0.9, factors=ConfidenceFactors()),
        ]

        result = aggregator.aggregate(scores, method="min")

        assert result == 0.6

    def test_aggregate_weighted(self):
        """重み付き集計（後半重視）"""
        aggregator = ConfidenceAggregator()
        scores = [
            ConfidenceScore(score=0.5, factors=ConfidenceFactors()),  # 重み1
            ConfidenceScore(score=0.9, factors=ConfidenceFactors()),  # 重み2
        ]

        result = aggregator.aggregate(scores, method="weighted")

        # (0.5*1 + 0.9*2) / (1+2) = 2.3 / 3 ≒ 0.767
        assert 0.76 < result < 0.78

    def test_aggregate_empty(self):
        """空リスト"""
        aggregator = ConfidenceAggregator()
        result = aggregator.aggregate([], method="mean")

        assert result == 0.0

    def test_aggregate_with_critical_check_no_failure(self):
        """重要ステップ失敗なし"""
        aggregator = ConfidenceAggregator()
        scores = [
            ConfidenceScore(score=0.8, factors=ConfidenceFactors()),
            ConfidenceScore(score=0.7, factors=ConfidenceFactors()),
        ]

        score, has_failure = aggregator.aggregate_with_critical_check(scores)

        assert has_failure is False
        assert abs(score - 0.75) < 0.01

    def test_aggregate_with_critical_check_has_failure(self):
        """重要ステップ失敗あり"""
        aggregator = ConfidenceAggregator()
        scores = [
            ConfidenceScore(score=0.8, factors=ConfidenceFactors()),
            ConfidenceScore(score=0.2, factors=ConfidenceFactors()),  # 閾値以下
        ]

        score, has_failure = aggregator.aggregate_with_critical_check(
            scores, critical_threshold=0.3
        )

        assert has_failure is True
        # ペナルティ適用: 0.5 * 0.7 = 0.35
        assert score < 0.5


class TestCreateConfidenceCalculator:
    """create_confidence_calculator関数のテスト"""

    def test_create_default(self):
        """デフォルト設定で作成"""
        calculator = create_confidence_calculator()

        assert isinstance(calculator, ConfidenceCalculator)