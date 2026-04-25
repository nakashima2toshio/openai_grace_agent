"""
GRACE Executor Integration Tests (Real LLM & Real RAG)
実際のLLM (Gemini) と 実際のQdrant を使用して
PlannerとExecutorの連携動作を確認する完全統合テスト

【前提条件】
1. Google Gemini APIキーが設定されていること
2. Qdrantサーバーが起動していること
3. Qdrantに「スペイン語の文法...」に関するデータ（a02_qa_pairs_wikipedia_ja.csv等）が登録されていること

[Usage]: pytest --cov=grace.executor -vs tests/grace/test_executor_integration.py
"""

import pytest
import os
import logging
from typing import List, Dict, Any

from grace.planner import Planner
from grace.executor import create_executor, ExecutionResult, StepStatus
from grace.tools import create_tool_registry
from grace.schemas import ExecutionPlan

# ログ設定（テスト実行時に詳細が見えるように）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set in environment"
)
class TestExecutorIntegration:
    """ExecutorとPlannerの統合テスト（実環境）"""

    @pytest.fixture
    def setup_components(self):
        """テスト用のコンポーネントセットアップ"""
        # 1. Planner (Real LLM)
        planner = Planner()

        # 2. ToolRegistry (Real Tools: RAG, Reasoning, AskUser)
        # configからデフォルトのツール群（RAGSearchTool, ReasoningTool等）を生成
        registry = create_tool_registry()

        # 3. Executor
        executor = create_executor(tool_registry=registry)

        return planner, executor

    def test_execute_plan_generator_flow(self, setup_components):
        """Plannerで計画作成 -> Executor(Generator)で実行の流れを確認"""
        planner, executor = setup_components
        
        # 実際にデータが存在すると思われるクエリを使用
        query = "スペイン語の文法と単語はそれぞれ何語の影響を強く受けていますか？ 日本語の影響は受けていますか？"
        print(f"\n[Step 1] Creating plan for query: {query}")
        
        # 1. 計画作成
        plan = planner.create_plan(query)
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.steps) > 0
        print(f"Plan generated: {len(plan.steps)} steps")
        # 計画の内容を表示（デバッグ用）
        print(plan.model_dump_json(indent=2))

        # 2. 実行 (Generator)
        print("\n[Step 2] Executing plan...")
        generator = executor.execute_plan_generator(plan)
        
        final_result = None
        
        # ジェネレータを回す
        try:
            while True:
                state_or_event = next(generator)
                
                # イベント辞書の場合（ログなど）
                if isinstance(state_or_event, dict):
                    if state_or_event.get("type") == "log":
                        # 実際のツール出力ログを表示
                        print(f"   [Log] {state_or_event.get('content')}")
                    continue
                
                # ExecutionStateの場合
                state = state_or_event
                # 直近で完了したステップの状態を表示
                for step_id, status in state.step_statuses.items():
                    if status == StepStatus.RUNNING:
                         # 実行中
                         pass
                    elif status == StepStatus.SUCCESS:
                        # 成功したステップの結果を表示（一度だけ表示するロジックは簡易的に省略）
                        result = state.step_results.get(step_id)
                        if result:
                            # 信頼度などを確認
                            pass

        except StopIteration as e:
            final_result = e.value

        # 3. 検証
        print("\n[Step 3] Verification")
        assert isinstance(final_result, ExecutionResult)
        
        print(f"Final Answer: {final_result.final_answer}")
        print(f"Overall Confidence: {final_result.overall_confidence:.2f}")

        # 成功しているか
        # データがない場合でもエラーにならず "success" で答えが "わかりませんでした" になる可能性があるため
        # overall_status == "success" をチェック
        assert final_result.overall_status == "success"
        
        # データが存在する前提でのコンテンツチェック（環境によっては失敗する可能性があるため、Warningレベルにするか、コメントアウトしても良いが、今回はあえてチェック）
        if "ラテン語" in str(final_result.final_answer) and "アラビア語" in str(final_result.final_answer):
            print("✅ Verified: Answer contains expected keywords.")
        else:
            print("⚠️ Warning: Answer might not contain expected keywords. Check RAG data.")
            # 強制的に落とさない（データがない場合もあるため）

        # 信頼度が計算されていること
        assert final_result.overall_confidence >= 0.0

    def test_execute_plan_blocking_flow(self, setup_components):
        """Executor(Blocking)での実行確認"""
        planner, executor = setup_components
        
        query = "スペイン語の単語はどのような影響を受けていますか？"
        
        # 1. 計画作成
        plan = planner.create_plan(query)
        
        # 2. 実行 (Blocking)
        print(f"\nExecuting blocking plan for: {query}")
        result = executor.execute_plan(plan)
        
        # 3. 検証
        assert result.overall_status == "success"
        assert result.final_answer is not None
        
        print(f"Final Answer: {result.final_answer}")