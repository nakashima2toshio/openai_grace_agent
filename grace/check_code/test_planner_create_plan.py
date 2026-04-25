"""
=============================================================================
test_planner_create_plan.py
GRACE Planner.create_plan() / estimate_complexity_with_llm() / refine_plan() 単体テスト
=============================================================================

■ テスト方針
  - Gemini API: 実APIコール（GOOGLE_API_KEY 環境変数が必要）
  - Qdrant:     モック（サーバー不要）
  - MeCab:      モック（KeywordExtractor）
  - prompts.py: そのままimport

■ モック対象（最小限）
  ┌──────────────────────────────────┬──────────────────────────────────┐
  │ 対象                              │ 理由                             │
  ├──────────────────────────────────┼──────────────────────────────────┤
  │ grace.planner.QdrantClient       │ Qdrantサーバー不要にする           │
  │ grace.planner.get_all_collections│ 同上                              │
  │ grace.planner.KeywordExtractor   │ MeCab環境依存を排除                │
  └──────────────────────────────────┴──────────────────────────────────┘

■ テストケース一覧
  ┌────────┬──────────────────────────────────────────────────┬──────────┐
  │ TC-ID  │ テスト内容                                         │ 種別     │
  ├────────┼──────────────────────────────────────────────────┼──────────┤
  │ TC-01  │ 単純クエリで ExecutionPlan が正しく返る               │ 正常系   │
  │ TC-02  │ 複雑クエリで複数ステップが生成される                    │ 正常系   │
  │ TC-03  │ 最終ステップが必ず reasoning である                   │ 正常系   │
  │ TC-04  │ depends_on の依存関係が正しい                        │ 正常系   │
  │ TC-05  │ plan_id が割り当てられ、2回の呼び出しで異なる           │ 正常系   │
  │ TC-06  │ complexity が estimate_complexity_with_llm の      │ 正常系   │
  │        │ 値で上書きされる（LLM JSON内の値ではない）              │          │
  │ TC-07  │ original_query がユーザー入力と一致する                │ 正常系   │
  │ TC-08  │ LLM APIエラー時にフォールバック計画が返る               │ 異常系   │
  │ TC-09  │ LLMが不正JSON返却時にフォールバック計画が返る            │ 異常系   │
  │ TC-10  │ コレクション取得失敗時もデフォルトで動作する              │ 境界値   │
  │ TC-11  │ estimate_complexity_with_llm: 単純クエリ→低い値      │ 正常系   │
  │ TC-12  │ estimate_complexity_with_llm: 複雑クエリ→高い値     │ 正常系   │
  │ TC-13  │ estimate_complexity_with_llm: 戻り値が0.0-1.0内    │ 境界値   │
  │ TC-14  │ estimate_complexity_with_llm: API失敗→キーワード    │ 異常系   │
  │        │ ベースのフォールバック                               │          │
  ├────────┼──────────────────────────────────────────────────┼──────────┤
  │ TC-15  │ refine_plan: フィードバックで修正された計画が返る      │ 正常系   │
  │ TC-16  │ refine_plan: 修正計画に新しい plan_id が付与される    │ 正常系   │
  │ TC-17  │ refine_plan: 最終ステップが reasoning を維持する     │ 正常系   │
  │ TC-18  │ refine_plan: API失敗時に元の計画がそのまま返る        │ 異常系   │
  └────────┴──────────────────────────────────────────────────┴──────────┘

■ 実行方法
  cd <project_root>
  GOOGLE_API_KEY=xxxxx pytest test_planner_create_plan.py -v -s

■ 注意
  - 実APIを叩くため、テスト1回あたり数秒～十数秒かかる
  - API課金が発生する（少額）
  - ネットワーク接続が必要
  - GOOGLE_API_KEY が未設定の場合、全テストがスキップされる
"""

import os
import time
import pytest
from unittest.mock import patch, MagicMock

# =============================================================================
# GOOGLE_API_KEY チェック — 未設定なら全テストスキップ
# =============================================================================
SKIP_REASON = "GOOGLE_API_KEY not set — real API tests require it"
api_key_available = os.environ.get("GOOGLE_API_KEY") is not None

pytestmark = pytest.mark.skipif(not api_key_available, reason=SKIP_REASON)


# =============================================================================
# Imports (テスト対象)
# =============================================================================
from grace.schemas import (
    ExecutionPlan,
    PlanStep,
    validate_plan_dependencies,
    create_plan_id,
)
from grace.config import GraceConfig, LLMConfig, QdrantConfig
from grace.planner import Planner


# =============================================================================
# 共通フィクスチャ
# =============================================================================

@pytest.fixture(scope="module")
def grace_config():
    """テスト用の GraceConfig（Qdrant設定はダミー）"""
    return GraceConfig(
        llm=LLMConfig(
            model="gemini-3-flash-preview",
            temperature=0.7,
            max_tokens=4096,
        ),
        qdrant=QdrantConfig(
            url="http://localhost:6333",  # モックするので実接続しない
            search_priority=["wikipedia_ja", "livedoor", "cc_news"],
        ),
    )


@pytest.fixture(scope="module")
def planner(grace_config):
    """
    モック済み Planner インスタンス（モジュールスコープで1回だけ生成）

    モック対象:
    - QdrantClient → Qdrantサーバー不要
    - get_all_collections → 固定コレクションリスト返却
    - KeywordExtractor → MeCab不要
    """
    mock_collections = [
        {"name": "wikipedia_ja", "points_count": 5000, "status": "green"},
        {"name": "livedoor", "points_count": 3000, "status": "green"},
        {"name": "cc_news", "points_count": 2000, "status": "green"},
    ]

    with patch("grace.planner.QdrantClient") as mock_qdrant, \
         patch("grace.planner.get_all_collections", return_value=mock_collections), \
         patch("grace.planner.KeywordExtractor") as mock_kw:

        # KeywordExtractor のモック（extract が空リストを返す）
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract.return_value = []
        mock_kw.return_value = mock_extractor_instance

        p = Planner(config=grace_config)
        yield p


# =============================================================================
# ヘルパー
# =============================================================================

def _print_plan_summary(plan: ExecutionPlan, label: str = ""):
    """テスト結果のデバッグ表示"""
    prefix = f"[{label}] " if label else ""
    print(f"\n{'='*60}")
    print(f"{prefix}Plan Summary")
    print(f"  plan_id:              {plan.plan_id}")
    print(f"  original_query:       {plan.original_query[:60]}...")
    print(f"  complexity:           {plan.complexity}")
    print(f"  estimated_steps:      {plan.estimated_steps}")
    print(f"  requires_confirmation:{plan.requires_confirmation}")
    print(f"  actual steps:         {len(plan.steps)}")
    for s in plan.steps:
        deps = f" (depends_on={s.depends_on})" if s.depends_on else ""
        print(f"    Step {s.step_id}: [{s.action}] {s.description[:50]}{deps}")
    print(f"  success_criteria:     {plan.success_criteria[:60]}")
    print(f"{'='*60}\n")


# =============================================================================
# TC-01: 正常系 — 単純クエリで ExecutionPlan が正しく返る
# =============================================================================
# INPUT:  query = "Pythonとは何ですか？"
# 期待:   ExecutionPlan インスタンスが返り、基本フィールドが妥当
# 合格条件:
#   ① isinstance(result, ExecutionPlan)
#   ② len(result.steps) >= 1
#   ③ result.plan_id is not None かつ空文字でない
#   ④ result.complexity は 0.0〜1.0 の範囲
# =============================================================================

class TestCreatePlanNormal:
    """正常系テスト群"""

    def test_tc01_simple_query_returns_execution_plan(self, planner):
        """TC-01: 単純クエリで ExecutionPlan が正しく返る"""
        query = "Pythonとは何ですか？"
        plan = planner.create_plan(query)

        _print_plan_summary(plan, "TC-01")

        assert isinstance(plan, ExecutionPlan), "戻り値が ExecutionPlan でない"
        assert len(plan.steps) >= 1, "ステップが0件"
        assert plan.plan_id is not None and plan.plan_id != "", "plan_id が未設定"
        assert 0.0 <= plan.complexity <= 1.0, f"complexity が範囲外: {plan.complexity}"

    # =============================================================================
    # TC-02: 正常系 — 複雑クエリで複数ステップが生成される
    # =============================================================================
    # INPUT:  長く複雑なクエリ
    # 期待:   2ステップ以上の計画が生成される
    # 合格条件:
    #   ① len(result.steps) >= 2
    #   ② result.estimated_steps >= 2
    # =============================================================================

    def test_tc02_complex_query_generates_multiple_steps(self, planner):
        """TC-02: 複雑クエリで複数ステップが生成される"""
        query = (
            "機械学習とディープラーニングの違いを比較し、"
            "それぞれの適用事例を詳しく説明してください。"
            "また、最新のトレンドについても言及してください。"
        )
        plan = planner.create_plan(query)

        _print_plan_summary(plan, "TC-02")

        assert len(plan.steps) >= 2, f"ステップ数が2未満: {len(plan.steps)}"
        assert plan.estimated_steps >= 2, f"estimated_steps が2未満: {plan.estimated_steps}"

    # =============================================================================
    # TC-03: 正常系 — 最終ステップが必ず reasoning である
    # =============================================================================
    # 根拠: プロンプトに「最後のステップは必ず reasoning で回答を生成」と指示
    # 合格条件:
    #   ① result.steps[-1].action == "reasoning"
    # =============================================================================

    def test_tc03_last_step_is_reasoning(self, planner):
        """TC-03: 最終ステップが reasoning である"""
        query = "日本の首都はどこですか？"
        plan = planner.create_plan(query)

        _print_plan_summary(plan, "TC-03")

        last_step = plan.steps[-1]
        assert last_step.action == "reasoning", (
            f"最終ステップが reasoning でない: {last_step.action}"
        )

    # =============================================================================
    # TC-04: 正常系 — depends_on の依存関係が正しい
    # =============================================================================
    # 合格条件:
    #   ① validate_plan_dependencies(plan) がエラーを返さない（空リスト）
    #   ② 各ステップの depends_on は自身より小さい step_id のみを参照
    # =============================================================================

    def test_tc04_dependencies_are_valid(self, planner):
        """TC-04: depends_on の依存関係が正しい"""
        query = "量子コンピュータの仕組みを説明してください"
        plan = planner.create_plan(query)

        _print_plan_summary(plan, "TC-04")

        errors = validate_plan_dependencies(plan)
        assert errors == [], f"依存関係エラー: {errors}"

        # 追加検証: depends_on の各IDが自身のstep_idより小さいこと
        for step in plan.steps:
            for dep_id in step.depends_on:
                assert dep_id < step.step_id, (
                    f"Step {step.step_id} が後方のステップ {dep_id} に依存している"
                )

    # =============================================================================
    # TC-05: 正常系 — plan_id が一意である
    # =============================================================================
    # 合格条件:
    #   ① 同一クエリで2回呼び出し、plan_id が異なる
    # =============================================================================

    def test_tc05_plan_id_is_unique(self, planner):
        """TC-05: 2回の呼び出しで plan_id が異なる"""
        query = "テスト用の質問です"
        plan1 = planner.create_plan(query)
        time.sleep(0.1)  # create_plan_id のタイムスタンプ衝突回避
        plan2 = planner.create_plan(query)

        print(f"\n[TC-05] plan1.plan_id={plan1.plan_id}, plan2.plan_id={plan2.plan_id}")

        assert plan1.plan_id != plan2.plan_id, (
            f"plan_id が同一: {plan1.plan_id}"
        )

    # =============================================================================
    # TC-06: 正常系 — complexity が estimate_complexity_with_llm の値で上書き
    # =============================================================================
    # 根拠: create_plan() 内の L195: plan.complexity = estimated_complexity
    #        → LLMが生成したJSON内の complexity ではなく、
    #          事前に estimate_complexity_with_llm() で計算した値が採用される
    # 合格条件:
    #   ① result.complexity が 0.0〜1.0 の float である
    #   ② estimate_complexity_with_llm の戻り値と一致する
    #      (estimate_complexity_with_llm をスパイして検証)
    # =============================================================================

    def test_tc06_complexity_overwritten_by_llm_estimation(self, planner):
        """TC-06: complexity が estimate_complexity_with_llm の値で上書きされる"""
        query = "Pythonのリスト内包表記とは何ですか？"

        # estimate_complexity_with_llm の実際の戻り値を記録
        original_method = planner.estimate_complexity_with_llm
        recorded_complexity = {}

        def spy_complexity(q):
            result = original_method(q)
            recorded_complexity["value"] = result
            return result

        with patch.object(planner, "estimate_complexity_with_llm", side_effect=spy_complexity):
            plan = planner.create_plan(query)

        _print_plan_summary(plan, "TC-06")

        expected = recorded_complexity.get("value")
        assert expected is not None, "estimate_complexity_with_llm が呼ばれなかった"
        assert plan.complexity == expected, (
            f"complexity不一致: plan={plan.complexity}, estimated={expected}"
        )

    # =============================================================================
    # TC-07: 正常系 — original_query がユーザー入力と一致する
    # =============================================================================
    # 合格条件:
    #   ① result.original_query == 入力クエリ文字列
    # =============================================================================

    def test_tc07_original_query_matches_input(self, planner):
        """TC-07: original_query がユーザー入力と一致する"""
        query = "『金色夜叉:尾崎紅葉不如帰:徳富蘆花』の構成者は誰ですか？"
        plan = planner.create_plan(query)

        _print_plan_summary(plan, "TC-07")

        assert plan.original_query == query, (
            f"original_query 不一致:\n  期待: {query}\n  実際: {plan.original_query}"
        )


# =============================================================================
# TC-08, TC-09: 異常系テスト群
# =============================================================================

class TestCreatePlanFallback:
    """異常系 — フォールバック計画テスト群"""

    # フォールバック計画の共通検証
    @staticmethod
    def _assert_is_fallback_plan(plan: ExecutionPlan, query: str):
        """フォールバック計画の共通アサーション"""
        assert isinstance(plan, ExecutionPlan), "戻り値が ExecutionPlan でない"
        assert plan.complexity == 0.5, f"fallback complexity != 0.5: {plan.complexity}"
        assert len(plan.steps) == 2, f"fallback steps != 2: {len(plan.steps)}"
        assert plan.steps[0].action == "rag_search", (
            f"fallback step[0] != rag_search: {plan.steps[0].action}"
        )
        assert plan.steps[0].collection == "wikipedia_ja", (
            f"fallback step[0].collection != wikipedia_ja: {plan.steps[0].collection}"
        )
        assert plan.steps[1].action == "reasoning", (
            f"fallback step[1] != reasoning: {plan.steps[1].action}"
        )
        assert plan.plan_id is not None, "fallback plan_id が None"

    # =============================================================================
    # TC-08: 異常系 — LLM API エラー時にフォールバック計画が返る
    # =============================================================================
    # 条件:  generate_content が Exception を送出
    # 合格条件:
    #   ① ExecutionPlan が返る（例外で落ちない）
    #   ② フォールバック計画の形式である
    # =============================================================================

    def test_tc08_api_error_returns_fallback(self, grace_config):
        """TC-08: LLM APIエラー → フォールバック計画"""
        mock_collections = [
            {"name": "wikipedia_ja", "points_count": 5000, "status": "green"},
        ]

        with patch("grace.planner.QdrantClient"), \
             patch("grace.planner.get_all_collections", return_value=mock_collections), \
             patch("grace.planner.KeywordExtractor"):

            p = Planner(config=grace_config)

            # generate_content を強制的に例外にする
            p.client.models.generate_content = MagicMock(
                side_effect=Exception("Simulated API Error")
            )

            query = "テスト質問"
            plan = p.create_plan(query)

            _print_plan_summary(plan, "TC-08")
            self._assert_is_fallback_plan(plan, query)

    # =============================================================================
    # TC-09: 異常系 — LLMが不正JSON返却 → フォールバック計画
    # =============================================================================
    # 条件:  generate_content が不正なJSON文字列を返す
    # 合格条件:
    #   ① Pydantic ValidationError が内部で捕捉される
    #   ② フォールバック計画が返る
    # =============================================================================

    def test_tc09_invalid_json_returns_fallback(self, grace_config):
        """TC-09: 不正JSON → フォールバック計画"""
        mock_collections = [
            {"name": "wikipedia_ja", "points_count": 5000, "status": "green"},
        ]

        with patch("grace.planner.QdrantClient"), \
             patch("grace.planner.get_all_collections", return_value=mock_collections), \
             patch("grace.planner.KeywordExtractor"):

            p = Planner(config=grace_config)

            # 不正なJSON応答を返すモック
            mock_response = MagicMock()
            mock_response.text = '{ "invalid": "not an ExecutionPlan"'  # 壊れたJSON

            # estimate_complexity_with_llm は正常に動作させるが、
            # 計画生成の generate_content だけ不正応答にする
            original_generate = p.client.models.generate_content
            call_count = {"n": 0}

            def mock_generate(*args, **kwargs):
                call_count["n"] += 1
                # 1回目: estimate_complexity_with_llm（正常に動かす）
                if call_count["n"] == 1:
                    return original_generate(*args, **kwargs)
                # 2回目: 計画生成（不正JSON）
                return mock_response

            p.client.models.generate_content = mock_generate

            query = "テスト質問"
            plan = p.create_plan(query)

            _print_plan_summary(plan, "TC-09")
            self._assert_is_fallback_plan(plan, query)


# =============================================================================
# TC-10: 境界値 — コレクション取得失敗時
# =============================================================================

class TestCreatePlanBoundary:
    """境界値テスト群"""

    # =============================================================================
    # TC-10: コレクション取得失敗時もデフォルトコレクションで動作する
    # =============================================================================
    # 条件:  _get_available_collections 内の QdrantClient が例外を送出
    # 根拠:  planner.py L254-255:
    #         except → return self.config.qdrant.search_priority
    # 合格条件:
    #   ① ExecutionPlan が正常に返る（フォールバックではなくLLM生成版）
    #   ② plan.steps が1件以上
    # =============================================================================

    def test_tc10_collection_fetch_failure_uses_defaults(self, grace_config):
        """TC-10: コレクション取得失敗 → デフォルトリストで動作"""

        with patch("grace.planner.QdrantClient", side_effect=Exception("Connection refused")), \
             patch("grace.planner.get_all_collections", side_effect=Exception("Connection refused")), \
             patch("grace.planner.KeywordExtractor"):

            p = Planner(config=grace_config)
            query = "日本の四季について教えてください"
            plan = p.create_plan(query)

            _print_plan_summary(plan, "TC-10")

            assert isinstance(plan, ExecutionPlan)
            assert len(plan.steps) >= 1


# =============================================================================
# TC-11〜TC-14: estimate_complexity_with_llm テスト群
# =============================================================================

class TestEstimateComplexity:
    """estimate_complexity_with_llm テスト群"""

    # =============================================================================
    # TC-11: 単純クエリ → 低い複雑度（0.0〜0.4）
    # =============================================================================
    # INPUT:  "Pythonとは？"
    # 合格条件:
    #   ① 戻り値が float
    #   ② 0.0 <= result <= 0.4（単純質問なので低い値が期待される）
    # =============================================================================

    def test_tc11_simple_query_low_complexity(self, planner):
        """TC-11: 単純クエリ → 低い複雑度"""
        query = "Pythonとは？"
        result = planner.estimate_complexity_with_llm(query)

        print(f"\n[TC-11] query='{query}', complexity={result}")

        assert isinstance(result, float), f"戻り値が float でない: {type(result)}"
        assert 0.0 <= result <= 0.4, (
            f"単純クエリの複雑度が高すぎる: {result} (期待: 0.0〜0.4)"
        )

    # =============================================================================
    # TC-12: 複雑クエリ → 高い複雑度（0.5〜1.0）
    # =============================================================================
    # INPUT:  複数のトピックを含む長い質問
    # 合格条件:
    #   ① 0.5 <= result <= 1.0
    # =============================================================================

    def test_tc12_complex_query_high_complexity(self, planner):
        """TC-12: 複雑クエリ → 高い複雑度"""
        query = (
            "量子コンピューティングと古典的コンピューティングの根本的な違いを、"
            "量子ビットの重ね合わせ・量子もつれの観点から詳しく比較し、"
            "さらに量子優位性が実証された具体的な問題領域と、"
            "現在の技術的課題および解決の見通しについて、"
            "複数の研究論文を引用しながら包括的に論じてください。"
        )
        result = planner.estimate_complexity_with_llm(query)

        print(f"\n[TC-12] query='{query[:50]}...', complexity={result}")

        assert isinstance(result, float)
        assert 0.5 <= result <= 1.0, (
            f"複雑クエリの複雑度が低すぎる: {result} (期待: 0.5〜1.0)"
        )

    # =============================================================================
    # TC-13: 戻り値が必ず 0.0〜1.0 の範囲に収まる
    # =============================================================================
    # 根拠:  planner.py L370:
    #         return min(1.0, max(0.0, complexity))
    # 合格条件:
    #   ① 複数の異なるクエリで全て 0.0 <= result <= 1.0
    # =============================================================================

    def test_tc13_complexity_always_in_range(self, planner):
        """TC-13: 複雑度が常に 0.0〜1.0 の範囲"""
        queries = [
            "",                              # 空文字列
            "はい",                           # 極短
            "A" * 500,                       # 極長
            "What is Python?",               # 英語
            "比較して詳しく複数の方法で",      # 複雑キーワード集中
        ]

        for query in queries:
            result = planner.estimate_complexity_with_llm(query)
            print(f"  query='{query[:30]}...' -> complexity={result}")

            assert isinstance(result, float), f"float でない: {type(result)}"
            assert 0.0 <= result <= 1.0, (
                f"範囲外: {result} (query='{query[:30]}')"
            )

    # =============================================================================
    # TC-14: API失敗 → キーワードベースのフォールバック
    # =============================================================================
    # 条件:  generate_content が例外を送出
    # 根拠:  planner.py L372-374:
    #         except → return self.estimate_complexity(query)
    # 合格条件:
    #   ① フォールバック値 (estimate_complexity) が返る
    #   ② 例外で落ちない
    # =============================================================================

    def test_tc14_api_failure_falls_back_to_keyword_estimation(self, grace_config):
        """TC-14: API失敗 → キーワードベースの estimate_complexity にフォールバック"""

        with patch("grace.planner.QdrantClient"), \
             patch("grace.planner.get_all_collections", return_value=[]), \
             patch("grace.planner.KeywordExtractor"):

            p = Planner(config=grace_config)

            # generate_content を強制例外にする
            p.client.models.generate_content = MagicMock(
                side_effect=Exception("Simulated API Error")
            )

            query = "比較して詳しく説明してください"
            result = p.estimate_complexity_with_llm(query)

            print(f"\n[TC-14] complexity (fallback) = {result}")

            # キーワードベース estimate_complexity の期待値を計算
            expected = p.estimate_complexity(query)

            assert isinstance(result, float)
            assert result == expected, (
                f"フォールバック値不一致: result={result}, expected={expected}"
            )
            assert 0.0 <= result <= 1.0


# =============================================================================
# TC-15〜TC-18: refine_plan テスト群
# =============================================================================

class TestRefinePlan:
    """refine_plan テスト群（実API）"""

    @staticmethod
    def _create_initial_plan(planner) -> ExecutionPlan:
        """
        refine_plan テスト用の初期計画を実APIで生成する。

        create_plan() を実行して「本物の計画」を取得し、それを
        refine_plan() の入力として使う。これにより初期計画の構造が
        常に妥当であることが保証される。
        """
        return planner.create_plan("AIについて教えてください")

    # =============================================================================
    # TC-15: 正常系 — フィードバックで修正された計画が返る
    # =============================================================================
    # INPUT:
    #   plan:     create_plan("AIについて教えてください") の結果
    #   feedback: "もっと技術的な詳細と、具体的な応用例が欲しいです"
    # 合格条件:
    #   ① 戻り値が ExecutionPlan インスタンス
    #   ② ステップが1件以上
    #   ③ complexity が 0.0〜1.0 の範囲
    # =============================================================================

    def test_tc15_feedback_returns_refined_plan(self, planner):
        """TC-15: フィードバックで修正された計画が返る"""
        initial_plan = self._create_initial_plan(planner)
        _print_plan_summary(initial_plan, "TC-15 initial")

        feedback = "もっと技術的な詳細と、具体的な応用例が欲しいです"
        refined = planner.refine_plan(initial_plan, feedback)

        _print_plan_summary(refined, "TC-15 refined")

        assert isinstance(refined, ExecutionPlan), "戻り値が ExecutionPlan でない"
        assert len(refined.steps) >= 1, "ステップが0件"
        assert 0.0 <= refined.complexity <= 1.0, (
            f"complexity が範囲外: {refined.complexity}"
        )

    # =============================================================================
    # TC-16: 正常系 — 修正計画に新しい plan_id が付与される
    # =============================================================================
    # 根拠: planner.py L423: refined_plan.plan_id = create_plan_id()
    # 合格条件:
    #   ① refined.plan_id is not None
    #   ② refined.plan_id != initial.plan_id（新しいIDが生成される）
    # =============================================================================

    def test_tc16_refined_plan_has_new_plan_id(self, planner):
        """TC-16: 修正計画に新しい plan_id が付与される"""
        initial_plan = self._create_initial_plan(planner)

        feedback = "検索範囲を広げて、複数の観点から分析してください"
        refined = planner.refine_plan(initial_plan, feedback)

        print(f"\n[TC-16] initial.plan_id={initial_plan.plan_id}")
        print(f"[TC-16] refined.plan_id={refined.plan_id}")

        assert refined.plan_id is not None, "refined plan_id が None"
        assert refined.plan_id != initial_plan.plan_id, (
            f"plan_id が同一（新IDが付与されていない）: {refined.plan_id}"
        )

    # =============================================================================
    # TC-17: 正常系 — 修正計画の最終ステップが reasoning を維持する
    # =============================================================================
    # 根拠: refine_plan の応答も ExecutionPlan スキーマに準拠し、
    #        LLMが PLAN_GENERATION_PROMPT のルール（最終=reasoning）を踏襲する
    # 合格条件:
    #   ① refined.steps[-1].action == "reasoning"
    # =============================================================================

    def test_tc17_refined_plan_last_step_is_reasoning(self, planner):
        """TC-17: 修正計画の最終ステップが reasoning"""
        initial_plan = self._create_initial_plan(planner)

        feedback = "歴史的な背景も含めて、より包括的に説明してください"
        refined = planner.refine_plan(initial_plan, feedback)

        _print_plan_summary(refined, "TC-17")

        last_step = refined.steps[-1]
        assert last_step.action == "reasoning", (
            f"修正計画の最終ステップが reasoning でない: {last_step.action}"
        )

    # =============================================================================
    # TC-18: 異常系 — API失敗時に元の計画がそのまま返る
    # =============================================================================
    # 条件:  generate_content が Exception を送出
    # 根拠:  planner.py L428-430:
    #         except Exception as e → return plan（元の計画）
    # 合格条件:
    #   ① 戻り値が元の計画と同一（plan_id が一致）
    #   ② 例外で落ちない
    # =============================================================================

    def test_tc18_api_failure_returns_original_plan(self, grace_config):
        """TC-18: API失敗 → 元の計画がそのまま返る"""

        with patch("grace.planner.QdrantClient"), \
             patch("grace.planner.get_all_collections", return_value=[]), \
             patch("grace.planner.KeywordExtractor"):

            p = Planner(config=grace_config)

            # テスト用の初期計画を手動作成
            initial_plan = ExecutionPlan(
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

            # generate_content を強制例外にする
            p.client.models.generate_content = MagicMock(
                side_effect=Exception("Simulated API Error")
            )

            feedback = "もっと詳しく教えて"
            result = p.refine_plan(initial_plan, feedback)

            print(f"\n[TC-18] initial.plan_id={initial_plan.plan_id}")
            print(f"[TC-18] result.plan_id ={result.plan_id}")

            # 元の計画がそのまま返ること
            assert result.plan_id == initial_plan.plan_id, (
                f"plan_id が異なる（フォールバックされていない）: "
                f"initial={initial_plan.plan_id}, result={result.plan_id}"
            )
            assert result.original_query == initial_plan.original_query


# =============================================================================
# エントリポイント（直接実行用）
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
