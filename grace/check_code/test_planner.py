"""
test_planner.py - GRACE Planner ユニットテスト

テスト対象: grace/planner.py
テストフレームワーク: pytest + unittest.mock

実行方法:
    pytest test_planner.py -v
    pytest test_planner.py -v -k "test_estimate_complexity"  # 特定テストのみ
"""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

# =============================================================================
# テスト用フィクスチャ
# =============================================================================

# --- LLM応答のモック用データ ---

VALID_PLAN_JSON = json.dumps({
    "original_query": "東京タワーの高さは？",
    "complexity": 0.2,
    "estimated_steps": 2,
    "requires_confirmation": False,
    "steps": [
        {
            "step_id": 1,
            "action": "rag_search",
            "description": "関連情報を検索",
            "query": "東京タワーの高さは？",
            "collection": None,
            "expected_output": "関連するドキュメント",
            "fallback": "reasoning",
            "depends_on": [],
        },
        {
            "step_id": 2,
            "action": "reasoning",
            "description": "取得した情報を元に回答を生成",
            "query": None,
            "collection": None,
            "depends_on": [1],
            "expected_output": "ユーザーへの回答",
            "fallback": None,
        },
    ],
    "success_criteria": "ユーザーの質問に適切に回答できている",
}, ensure_ascii=False)

INVALID_JSON = "{ this is not valid json !!!"

# 依存関係エラーを含む計画JSON（step2がstep3に依存 = 後方依存）
PLAN_WITH_BAD_DEPS_JSON = json.dumps({
    "original_query": "テスト",
    "complexity": 0.5,
    "estimated_steps": 2,
    "requires_confirmation": False,
    "steps": [
        {
            "step_id": 1,
            "action": "rag_search",
            "description": "検索",
            "query": "テスト",
            "expected_output": "結果",
            "depends_on": [],
        },
        {
            "step_id": 2,
            "action": "reasoning",
            "description": "推論",
            "expected_output": "回答",
            "depends_on": [3],  # 存在しないステップへの依存
        },
    ],
    "success_criteria": "回答できている",
}, ensure_ascii=False)

REFINED_PLAN_JSON = json.dumps({
    "original_query": "AIについて教えてください",
    "complexity": 0.6,
    "estimated_steps": 3,
    "requires_confirmation": False,
    "steps": [
        {
            "step_id": 1,
            "action": "rag_search",
            "description": "AI技術の詳細を検索",
            "query": "AIについて教えてください",
            "expected_output": "技術的な情報",
            "depends_on": [],
        },
        {
            "step_id": 2,
            "action": "rag_search",
            "description": "AI応用例を検索",
            "query": "AIの具体的な応用例",
            "expected_output": "応用例の情報",
            "depends_on": [],
        },
        {
            "step_id": 3,
            "action": "reasoning",
            "description": "統合して回答",
            "expected_output": "詳細な回答",
            "depends_on": [1, 2],
        },
    ],
    "success_criteria": "技術的詳細と応用例を含む回答",
}, ensure_ascii=False)


def _make_llm_response(text: str) -> MagicMock:
    """Gemini API応答のモックを生成"""
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


def _make_llm_response_none() -> MagicMock:
    """response.text が None の応答モック"""
    mock_response = MagicMock()
    mock_response.text = None
    return mock_response


# =============================================================================
# パッチ対象の定数
# =============================================================================

PATCH_GENAI_CLIENT = "grace.planner.genai.Client"
PATCH_QDRANT_CLIENT = "grace.planner.QdrantClient"
PATCH_GET_ALL_COLLECTIONS = "grace.planner.get_all_collections"
PATCH_GET_CONFIG = "grace.planner.get_config"
PATCH_KEYWORD_EXTRACTOR = "grace.planner.KeywordExtractor"


# =============================================================================
# フィクスチャ
# =============================================================================

@pytest.fixture
def mock_config():
    """テスト用のGraceConfig"""
    from grace.config import GraceConfig
    return GraceConfig()  # デフォルト値を使用


@pytest.fixture
def mock_genai_client():
    """Gemini APIクライアントのモック"""
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def planner_instance(mock_config, mock_genai_client):
    """
    外部依存を全てモックしたPlannerインスタンスを生成

    モック対象:
    - genai.Client → mock_genai_client
    - KeywordExtractor → MagicMock
    - get_config → mock_config
    """
    with patch(PATCH_GENAI_CLIENT, return_value=mock_genai_client), \
         patch(PATCH_KEYWORD_EXTRACTOR, return_value=MagicMock()):
        from grace.planner import Planner
        planner = Planner(config=mock_config)
        # client属性をモックに差し替え（__init__内で genai.Client() した結果）
        planner.client = mock_genai_client
        return planner


# =============================================================================
# 1. estimate_complexity テスト（キーワードベース複雑度推定）
# =============================================================================

class TestEstimateComplexity:
    """Planner.estimate_complexity() のテスト"""

    def test_no_keywords_short_query(self, planner_instance):
        """1-1: キーワードなし・短い質問 → ベーススコア 0.5"""
        result = planner_instance.estimate_complexity("東京タワーの高さは？")
        assert result == 0.5

    def test_single_keyword(self, planner_instance):
        """1-2: 「違い」キーワード1つ → 0.65"""
        result = planner_instance.estimate_complexity("PythonとJavaの違いは？")
        assert result == pytest.approx(0.65, abs=0.01)

    @pytest.mark.parametrize("query, expected", [
        # 「比較」(0.15) + 「違い」(0.15)
        ("AとBの違いを比較して", 0.8),
        # 「違い」(0.15) + 「比較」(0.15) + 「詳しく」(0.15)
        ("PythonとJavaの違いを比較して詳しく教えて", 0.95),
        # 「複数」(0.2) + 「方法」(0.1)
        ("複数の方法を教えて", 0.8),
    ])
    def test_multiple_keywords(self, planner_instance, query, expected):
        """1-3: 複数キーワード検出"""
        result = planner_instance.estimate_complexity(query)
        assert result == pytest.approx(expected, abs=0.01)

    def test_long_query_100(self, planner_instance):
        """1-4: 100文字超の質問 → +0.1"""
        query = "あ" * 101  # キーワードなし、101文字
        result = planner_instance.estimate_complexity(query)
        assert result == pytest.approx(0.6, abs=0.01)

    def test_long_query_200(self, planner_instance):
        """1-5: 200文字超の質問 → +0.2"""
        query = "あ" * 201  # キーワードなし、201文字
        result = planner_instance.estimate_complexity(query)
        assert result == pytest.approx(0.7, abs=0.01)

    def test_upper_clamp(self, planner_instance):
        """1-6: 上限クランプ → max 1.0"""
        # 全キーワードを含む超長文（理論スコアが1.0を超える）
        query = "比較 違い 複数 最新 理由 方法 詳しく ステップ 手順 なぜ どのように " + "あ" * 201
        result = planner_instance.estimate_complexity(query)
        assert result == 1.0

    def test_empty_string(self, planner_instance):
        """1-7: 空文字列 → ベーススコア 0.5"""
        result = planner_instance.estimate_complexity("")
        assert result == 0.5


# =============================================================================
# 2. _create_fallback_plan テスト
# =============================================================================

class TestCreateFallbackPlan:
    """Planner._create_fallback_plan() のテスト"""

    def test_basic_structure(self, planner_instance):
        """2-1: 基本構造の確認"""
        plan = planner_instance._create_fallback_plan("テスト質問")
        assert plan.complexity == 0.5
        assert plan.estimated_steps == 2
        assert plan.requires_confirmation is False
        assert len(plan.steps) == 2

    def test_original_query_preserved(self, planner_instance):
        """2-2: original_queryが保持される"""
        query = "元の質問文をそのまま保持するか？"
        plan = planner_instance._create_fallback_plan(query)
        assert plan.original_query == query

    def test_step_composition(self, planner_instance):
        """2-3: ステップ構成（rag_search → reasoning、依存関係）"""
        plan = planner_instance._create_fallback_plan("質問")
        step1, step2 = plan.steps

        assert step1.action == "rag_search"
        assert step1.step_id == 1
        assert step1.depends_on == []

        assert step2.action == "reasoning"
        assert step2.step_id == 2
        assert step2.depends_on == [1]

    def test_collection_is_wikipedia_ja(self, planner_instance):
        """2-4: フォールバックはwikipedia_jaを明示指定"""
        plan = planner_instance._create_fallback_plan("質問")
        assert plan.steps[0].collection == "wikipedia_ja"

    def test_plan_id_generated(self, planner_instance):
        """2-5: plan_idが生成されている"""
        plan = planner_instance._create_fallback_plan("質問")
        assert plan.plan_id is not None
        assert len(plan.plan_id) > 0

    def test_query_passed_to_search_step(self, planner_instance):
        """2-6: クエリが検索ステップに渡される"""
        query = "特定のクエリ文"
        plan = planner_instance._create_fallback_plan(query)
        assert plan.steps[0].query == query


# =============================================================================
# 3. _create_plan_legacy テスト
# =============================================================================

class TestCreatePlanLegacy:
    """Planner._create_plan_legacy() のテスト"""

    def test_basic_structure(self, planner_instance):
        """3-1: Legacy計画の基本構造"""
        plan = planner_instance._create_plan_legacy("テスト質問")
        assert plan.complexity == 0.1
        assert plan.estimated_steps == 1
        assert plan.requires_confirmation is False
        assert len(plan.steps) == 1

    def test_action_is_run_legacy_agent(self, planner_instance):
        """3-2: アクションが run_legacy_agent"""
        plan = planner_instance._create_plan_legacy("テスト")
        assert plan.steps[0].action == "run_legacy_agent"

    def test_plan_id_generated(self, planner_instance):
        """3-3: plan_idが生成されている"""
        plan = planner_instance._create_plan_legacy("テスト")
        assert plan.plan_id is not None


# =============================================================================
# 4. estimate_complexity_with_llm テスト
# =============================================================================

class TestEstimateComplexityWithLLM:
    """Planner.estimate_complexity_with_llm() のテスト"""

    def test_normal_response(self, planner_instance):
        """4-1: 正常系 — LLMが数値を返す"""
        planner_instance.client.models.generate_content.return_value = _make_llm_response("0.3")
        result = planner_instance.estimate_complexity_with_llm("簡単な質問")
        assert result == pytest.approx(0.3, abs=0.01)

    def test_clamp_upper(self, planner_instance):
        """4-2: 上限超過 → 1.0にクランプ"""
        planner_instance.client.models.generate_content.return_value = _make_llm_response("1.5")
        result = planner_instance.estimate_complexity_with_llm("質問")
        assert result == 1.0

    def test_clamp_lower(self, planner_instance):
        """4-3: 下限未満 → 0.0にクランプ"""
        planner_instance.client.models.generate_content.return_value = _make_llm_response("-0.3")
        result = planner_instance.estimate_complexity_with_llm("質問")
        assert result == 0.0

    def test_none_response_fallback(self, planner_instance):
        """4-4: 空レスポンス（Noneガード）→ キーワードベースにフォールバック"""
        planner_instance.client.models.generate_content.return_value = _make_llm_response_none()
        result = planner_instance.estimate_complexity_with_llm("東京タワーの高さは？")
        # キーワードベースのestimate_complexityが呼ばれる → 0.5（キーワードなし）
        assert result == pytest.approx(0.5, abs=0.01)

    def test_api_exception_fallback(self, planner_instance):
        """4-5: API例外発生 → キーワードベースにフォールバック"""
        planner_instance.client.models.generate_content.side_effect = Exception("API Error")
        result = planner_instance.estimate_complexity_with_llm("東京タワーの高さは？")
        assert result == pytest.approx(0.5, abs=0.01)

    def test_non_numeric_response_fallback(self, planner_instance):
        """4-6: 非数値レスポンス → キーワードベースにフォールバック"""
        planner_instance.client.models.generate_content.return_value = _make_llm_response("高い")
        result = planner_instance.estimate_complexity_with_llm("東京タワーの高さは？")
        assert result == pytest.approx(0.5, abs=0.01)

    def test_whitespace_trimmed(self, planner_instance):
        """4-7: 前後の空白がトリムされる"""
        planner_instance.client.models.generate_content.return_value = _make_llm_response("  0.7  \n")
        result = planner_instance.estimate_complexity_with_llm("質問")
        assert result == pytest.approx(0.7, abs=0.01)


# =============================================================================
# 5. create_plan テスト（メイン計画生成）
# =============================================================================

class TestCreatePlan:
    """Planner.create_plan() のテスト"""

    def test_normal_plan_generation(self, planner_instance):
        """5-1: 正常系 — 有効なExecutionPlanが返る"""
        # estimate_complexity_with_llm の応答
        # create_plan内部で2回 generate_content が呼ばれる:
        #   1回目: estimate_complexity_with_llm
        #   2回目: 計画生成本体
        planner_instance.client.models.generate_content.side_effect = [
            _make_llm_response("0.2"),    # complexity推定
            _make_llm_response(VALID_PLAN_JSON),  # 計画生成
        ]

        with patch(PATCH_GET_ALL_COLLECTIONS, return_value=[{"name": "wikipedia_ja"}]):
            with patch(PATCH_QDRANT_CLIENT):
                plan = planner_instance.create_plan("東京タワーの高さは？")

        assert plan.original_query == "東京タワーの高さは？"
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "rag_search"
        assert plan.steps[1].action == "reasoning"

    def test_plan_id_is_set(self, planner_instance):
        """5-2: plan_idが設定される"""
        planner_instance.client.models.generate_content.side_effect = [
            _make_llm_response("0.2"),
            _make_llm_response(VALID_PLAN_JSON),
        ]

        with patch(PATCH_GET_ALL_COLLECTIONS, return_value=[{"name": "col1"}]):
            with patch(PATCH_QDRANT_CLIENT):
                plan = planner_instance.create_plan("テスト")

        assert plan.plan_id is not None
        assert len(plan.plan_id) > 0

    def test_complexity_overwritten_by_llm(self, planner_instance):
        """5-3: complexityがLLM推定値で上書きされる"""
        # LLM推定値は0.8、JSON内のcomplexityは0.2
        planner_instance.client.models.generate_content.side_effect = [
            _make_llm_response("0.8"),    # complexity = 0.8
            _make_llm_response(VALID_PLAN_JSON),  # JSON内は0.2
        ]

        with patch(PATCH_GET_ALL_COLLECTIONS, return_value=[]):
            with patch(PATCH_QDRANT_CLIENT):
                plan = planner_instance.create_plan("テスト")

        # LLM推定値 0.8 で上書きされる
        assert plan.complexity == pytest.approx(0.8, abs=0.01)

    def test_llm_failure_returns_fallback(self, planner_instance):
        """5-4: LLM失敗 → フォールバック計画"""
        planner_instance.client.models.generate_content.side_effect = [
            _make_llm_response("0.5"),    # complexity推定は成功
            Exception("LLM API Error"),   # 計画生成で失敗
        ]

        with patch(PATCH_GET_ALL_COLLECTIONS, return_value=[]):
            with patch(PATCH_QDRANT_CLIENT):
                plan = planner_instance.create_plan("テスト質問")

        # フォールバック計画の特徴を確認
        assert plan.complexity == 0.5
        assert len(plan.steps) == 2
        assert plan.steps[0].collection == "wikipedia_ja"

    def test_json_parse_failure_returns_fallback(self, planner_instance):
        """5-5: JSONパース失敗 → フォールバック計画"""
        planner_instance.client.models.generate_content.side_effect = [
            _make_llm_response("0.3"),
            _make_llm_response(INVALID_JSON),
        ]

        with patch(PATCH_GET_ALL_COLLECTIONS, return_value=[]):
            with patch(PATCH_QDRANT_CLIENT):
                plan = planner_instance.create_plan("テスト")

        # フォールバック計画
        assert plan.complexity == 0.5
        assert plan.steps[0].collection == "wikipedia_ja"

    def test_dependency_errors_warn_only(self, planner_instance):
        """5-6: 依存関係エラー → 警告のみで計画は返却される"""
        planner_instance.client.models.generate_content.side_effect = [
            _make_llm_response("0.5"),
            _make_llm_response(PLAN_WITH_BAD_DEPS_JSON),
        ]

        with patch(PATCH_GET_ALL_COLLECTIONS, return_value=[]):
            with patch(PATCH_QDRANT_CLIENT):
                plan = planner_instance.create_plan("テスト")

        # 計画自体は返却される（エラーではない）
        assert plan is not None
        assert len(plan.steps) == 2


# =============================================================================
# 6. _get_available_collections テスト
# =============================================================================

class TestGetAvailableCollections:
    """Planner._get_available_collections() のテスト"""

    def test_normal_retrieval(self, planner_instance):
        """6-1: 正常系 — コレクション名リストを返す"""
        mock_collections = [
            {"name": "wikipedia_ja", "points_count": 100, "status": "green"},
            {"name": "livedoor", "points_count": 50, "status": "green"},
        ]

        with patch(PATCH_QDRANT_CLIENT) as mock_qc:
            with patch(PATCH_GET_ALL_COLLECTIONS, return_value=mock_collections):
                result = planner_instance._get_available_collections()

        assert result == ["wikipedia_ja", "livedoor"]

    def test_qdrant_failure_returns_default(self, planner_instance):
        """6-2: Qdrant接続失敗 → config.qdrant.search_priority を返す"""
        with patch(PATCH_QDRANT_CLIENT, side_effect=Exception("Connection refused")):
            result = planner_instance._get_available_collections()

        expected = planner_instance.config.qdrant.search_priority
        assert result == expected


# =============================================================================
# 7. refine_plan テスト
# =============================================================================

class TestRefinePlan:
    """Planner.refine_plan() のテスト"""

    def _create_initial_plan(self):
        """テスト用の初期計画を生成"""
        from grace.schemas import ExecutionPlan, PlanStep, create_plan_id
        return ExecutionPlan(
            original_query="AIについて教えてください",
            complexity=0.3,
            estimated_steps=2,
            requires_confirmation=False,
            steps=[
                PlanStep(
                    step_id=1,
                    action="rag_search",
                    description="検索",
                    query="AIについて教えてください",
                    expected_output="情報",
                ),
                PlanStep(
                    step_id=2,
                    action="reasoning",
                    description="回答生成",
                    depends_on=[1],
                    expected_output="回答",
                ),
            ],
            success_criteria="回答できている",
            plan_id=create_plan_id(),
        )

    def test_normal_refinement(self, planner_instance):
        """7-1: 正常系 — 修正計画が返る"""
        initial_plan = self._create_initial_plan()
        planner_instance.client.models.generate_content.return_value = (
            _make_llm_response(REFINED_PLAN_JSON)
        )

        refined = planner_instance.refine_plan(initial_plan, "もっと詳しく")

        assert refined.plan_id is not None
        assert refined.plan_id != initial_plan.plan_id  # 新しいIDが付与

    def test_llm_failure_returns_original(self, planner_instance):
        """7-2: LLM失敗時 → 元の計画をそのまま返却"""
        initial_plan = self._create_initial_plan()
        planner_instance.client.models.generate_content.side_effect = Exception("API Error")

        refined = planner_instance.refine_plan(initial_plan, "フィードバック")

        # 元の計画がそのまま返る
        assert refined.plan_id == initial_plan.plan_id
        assert refined.original_query == initial_plan.original_query


# =============================================================================
# 8. __init__ / create_planner テスト
# =============================================================================

class TestPlannerInit:
    """Planner.__init__() および create_planner() のテスト"""

    def test_default_model_from_config(self, mock_config):
        """8-1: デフォルト設定 → config.llm.model が使われる"""
        with patch(PATCH_GENAI_CLIENT), \
             patch(PATCH_KEYWORD_EXTRACTOR):
            from grace.planner import Planner
            planner = Planner(config=mock_config)
            assert planner.model_name == mock_config.llm.model

    def test_custom_model_name(self, mock_config):
        """8-2: カスタムモデル名指定"""
        with patch(PATCH_GENAI_CLIENT), \
             patch(PATCH_KEYWORD_EXTRACTOR):
            from grace.planner import Planner
            planner = Planner(config=mock_config, model_name="custom-model-v2")
            assert planner.model_name == "custom-model-v2"

    def test_keyword_extractor_failure(self, mock_config):
        """8-3: KeywordExtractor初期化失敗 → keyword_extractor = None"""
        with patch(PATCH_GENAI_CLIENT), \
             patch(PATCH_KEYWORD_EXTRACTOR, side_effect=Exception("MeCab not found")):
            from grace.planner import Planner
            planner = Planner(config=mock_config)
            assert planner.keyword_extractor is None

    def test_create_planner_factory(self, mock_config):
        """8-4: create_planner ファクトリ関数"""
        with patch(PATCH_GENAI_CLIENT), \
             patch(PATCH_KEYWORD_EXTRACTOR):
            from grace.planner import create_planner
            planner = create_planner(config=mock_config, model_name="test-model")
            assert planner.model_name == "test-model"

    def test_create_planner_default(self):
        """8-5: create_planner デフォルト引数"""
        with patch(PATCH_GENAI_CLIENT), \
             patch(PATCH_KEYWORD_EXTRACTOR), \
             patch(PATCH_GET_CONFIG) as mock_get_config:
            from grace.config import GraceConfig
            mock_get_config.return_value = GraceConfig()

            from grace.planner import create_planner
            planner = create_planner()
            assert planner.model_name == "gemini-3-flash-preview"


# =============================================================================
# 補足: ログ出力の検証（オプション）
# =============================================================================

class TestPlannerLogging:
    """ログ出力の検証テスト"""

    def test_fallback_logs_warning(self, planner_instance, caplog):
        """LLM失敗時にERROR + INFO（Falling back）ログが出力される"""
        import logging
        planner_instance.client.models.generate_content.side_effect = [
            _make_llm_response("0.5"),
            Exception("LLM Error"),
        ]

        with patch(PATCH_GET_ALL_COLLECTIONS, return_value=[]):
            with patch(PATCH_QDRANT_CLIENT):
                with caplog.at_level(logging.INFO, logger="grace.planner"):
                    planner_instance.create_plan("テスト")

        # フォールバックのログが出ていることを確認
        assert any("Falling back" in msg or "fallback" in msg.lower()
                    for msg in caplog.messages)

    def test_complexity_estimation_logs_time(self, planner_instance, caplog):
        """LLM複雑度推定でAPI時間がログ出力される"""
        import logging
        planner_instance.client.models.generate_content.return_value = _make_llm_response("0.5")

        with caplog.at_level(logging.INFO, logger="grace.planner"):
            planner_instance.estimate_complexity_with_llm("テスト")

        assert any("API時間" in msg or "estimate_complexity" in msg
                    for msg in caplog.messages)
