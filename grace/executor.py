# grace/executor.py
"""
GRACE Executor - 計画実行エージェント
生成された計画を順次実行し、結果を管理
"""

import ast
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, cast

from .calibration import Calibrator  # S1: confidence 較正
from .confidence import (
    ActionDecision,
    ConfidenceFactors,
    ConfidenceScore,
    InterventionLevel,
    create_confidence_aggregator,
    create_confidence_calculator,
    create_llm_evaluator,
    create_query_coverage_calculator,
    create_source_agreement_calculator,  # TODO #5: 追加
)
from .config import GraceConfig, get_config
from .intervention import (
    InterventionAction,
    InterventionRequest,
    InterventionResponse,
    create_intervention_handler,
)
from .memory import create_execution_memory  # P4: 実行メモリ層
from .schemas import (
    ExecutionPlan,
    ExecutionResult,
    PlanStep,
    StepResult,
    StepStatus,
    create_plan_id,
)
from .tools import ToolRegistry, ToolResult, create_tool_registry

# === Legacy Agent Integration ===
try:
    from services.agent_service import ReActAgent, get_available_collections_from_qdrant_helper

    LEGACY_AGENT_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)  # Ensure logger exists before warning
    logger.warning("Failed to import services.agent_service. Legacy agent execution will fail.")
    LEGACY_AGENT_AVAILABLE = False
# ================================

# [MIGRATION] create_llm_client を追加（_check_rag_relevance_with_llm で使用）
try:
    from helper.helper_llm import create_llm_client  # [FIXED] helper_llm → helper.helper_llm
    _LLM_CLIENT_AVAILABLE = True
except ImportError:
    _LLM_CLIENT_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# 実行状態管理
# =============================================================================

@dataclass
class ExecutionState:
    """実行状態管理"""

    plan: ExecutionPlan
    current_step_id: int = 0
    step_results: Dict[int, StepResult] = field(default_factory=dict)
    step_statuses: Dict[int, StepStatus] = field(default_factory=dict)
    overall_confidence: float = 0.0
    is_cancelled: bool = False
    is_paused: bool = False
    intervention_request: Optional[Any] = None  # InterventionRequest
    replan_count: int = 0
    max_replans: int = 3
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    # P4: 実行中に使用した RAG コレクション（実行メモリ記録用）
    used_collections: List[str] = field(default_factory=list)

    def __post_init__(self):
        """初期化後の処理"""
        # 全ステップをPENDINGで初期化
        for step in self.plan.steps:
            self.step_statuses[step.step_id] = StepStatus.PENDING

    def get_completed_outputs(self) -> Dict[int, str]:
        """完了済みステップの出力を取得"""
        return {
            step_id: result.output
            for step_id, result in self.step_results.items()
            if result.status == "success"
        }

    def get_completed_sources(self) -> List[str]:
        """完了済みステップのソースを取得"""
        sources = []
        for result in self.step_results.values():
            if result.status == "success" and result.sources:
                sources.extend(result.sources)
        return sources

    def can_replan(self) -> bool:
        """リプラン可能か判定"""
        return self.replan_count < self.max_replans and not self.is_cancelled

    def get_execution_time_ms(self) -> Optional[int]:
        """実行時間を取得（ミリ秒）"""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return int((end - self.start_time) * 1000)


# =============================================================================
# Executor クラス
# =============================================================================

from .replan import ReplanOrchestrator, create_replan_orchestrator  # noqa: E402


class Executor:
    """計画実行エージェント（GRACEネイティブ実装）"""

    def __init__(
            self,
            config: Optional[GraceConfig] = None,
            tool_registry: Optional[ToolRegistry] = None,
            on_step_start: Optional[Callable[[PlanStep], None]] = None,
            on_step_complete: Optional[Callable[[StepResult], None]] = None,
            on_intervention_required: Optional[Callable[[str, Dict], Any]] = None,
            on_confidence_update: Optional[Callable[[ConfidenceScore, ActionDecision], None]] = None,
            on_replan: Optional[Callable[[str, int], None]] = None,
            replan_orchestrator: Optional[ReplanOrchestrator] = None,
            enable_replan: bool = True,
    ):
        self.config = config or get_config()

        # ToolRegistry（指定がなければデフォルト作成）
        self.tool_registry = tool_registry or create_tool_registry(config=self.config)

        # Confidence関連コンポーネント（Phase 2）
        self.confidence_calculator = create_confidence_calculator(config=self.config)
        self.llm_evaluator = create_llm_evaluator(config=self.config)
        self.query_coverage_calculator = create_query_coverage_calculator(config=self.config)
        self.confidence_aggregator = create_confidence_aggregator(config=self.config)

        # S1: confidence 較正器（較正ファイルが無ければ恒等 T=1.0 = 無変換）
        self._calibrator = Calibrator.load(
            getattr(self.config.confidence, "calibration_path", "config/calibration.json")
        )
        # P4: 実行メモリ層（コレクション事前分布の学習）。memory.enabled=False で無効化。
        self._memory = None
        if getattr(self.config, "memory", None) and self.config.memory.enabled:
            self._memory = create_execution_memory(self.config.memory.path)

        # コールバック
        self.on_step_start = on_step_start
        self.on_step_complete = on_step_complete
        self.on_intervention_required = on_intervention_required
        self.on_confidence_update = on_confidence_update
        self.on_replan = on_replan  # リプラン発生時のコールバック

        # InterventionHandler（Phase 3）
        self.intervention_handler = create_intervention_handler(
            config=self.config,
            on_notify=self._handle_intervention_notify,
            on_confirm=self._handle_intervention_confirm,
            on_escalate=self._handle_intervention_escalate,
        )

        # ReplanOrchestrator（Phase 4）
        if replan_orchestrator is not None:
            self.replan_orchestrator = replan_orchestrator
        elif enable_replan:
            self.replan_orchestrator = create_replan_orchestrator(config=self.config)
        else:
            self.replan_orchestrator = None

        # ステップごとのConfidenceScoreを保持
        self.step_confidence_scores: Dict[int, ConfidenceScore] = {}

        # 並列プリフェッチ結果のキャッシュ（#60）: step_id -> ToolResult | Exception
        self._prefetched_tool_results: Dict[int, Any] = {}

        replan_status = "enabled" if self.replan_orchestrator else "disabled"
        logger.info(
            f"Executor (GRACE Native) initialized: "
            f"tools={self.tool_registry.list_tools()}, replan={replan_status}"
        )

    def execute_plan_generator(
            self,
            plan: ExecutionPlan,
            state: Optional[ExecutionState] = None
    ) -> Generator[ExecutionState, None, ExecutionResult]:
        """
        計画をステップごとに実行（ジェネレータ版）
        UIなどで進捗をリアルタイム表示するために使用
        Args:
            plan: 実行する計画
            state: 既存の状態（再開時などに指定）
        Yields:
            ExecutionState: 各ステップ完了後の状態
        Returns:
            ExecutionResult: 最終実行結果
        """
        logger.info(f"Executing plan (generator): {plan.plan_id}, steps={len(plan.steps)}")

        # 受け取ったプラン内容をログ出力
        logger.info(f"Received Execution Plan in Executor (generator):\n{plan.model_dump_json(indent=2)}")

        # 実行状態を初期化（未指定の場合）
        if state is None:
            state = ExecutionState(plan=plan)
            state.start_time = time.time()
            self._prefetched_tool_results.clear()

        try:
            # 各ステップを順次実行
            # 注: リプランなどでステップ数が増減する可能性があるため、インデックス管理が必要
            # ここでは簡易的に、現在のステップID以降を実行するロジック

            # 実行すべきステップのリストを取得（現在の計画に基づく）
            # 既に完了しているステップはスキップ
            steps_to_execute = [
                s for s in plan.steps
                if state.step_statuses.get(s.step_id) != StepStatus.SUCCESS
            ]

            for step in steps_to_execute:
                # 状態更新: 現在のステップID
                state.current_step_id = step.step_id

                # キャンセルチェック
                if state.is_cancelled:
                    logger.info("Execution cancelled")
                    break

                # スキップ済みチェック（RAGスコア十分時にweb_searchを動的スキップした場合）
                if state.step_statuses.get(step.step_id) == StepStatus.SKIPPED:
                    logger.info(f"Step {step.step_id}: Already marked as SKIPPED, skipping")
                    yield state
                    continue

                # 依存関係チェック
                if not self._check_dependencies(step, state):
                    logger.warning(f"Step {step.step_id}: Dependencies not met, skipping")
                    state.step_statuses[step.step_id] = StepStatus.SKIPPED
                    yield state
                    continue

                # ステップ開始コールバック
                state.step_statuses[step.step_id] = StepStatus.RUNNING
                if self.on_step_start:
                    self.on_step_start(step)

                # 並列実行（#60）: 現在ステップと依存関係のない後続検索ステップの
                # ツール呼び出しを先行して並列実行（結果はプリフェッチして各ステップ処理時に消費）
                if self.config.executor.parallel_search:
                    self._prefetch_parallel_searches(step, steps_to_execute, state)

                # ステップ実行
                # _execute_step は StepResult または Generator[Any, None, StepResult] を返す可能性がある
                step_execution = self._execute_step(step, state)

                result = None
                if isinstance(step_execution, Generator):
                    # ジェネレータの場合はイベントを中継し、最終結果(return value)を取得
                    # yield from は return value を返す
                    result = yield from step_execution
                else:
                    # 直接結果が返ってきた場合
                    result = step_execution

                # 結果を保存
                state.step_results[step.step_id] = result
                state.step_statuses[step.step_id] = (
                    StepStatus.SUCCESS if result.status == "success" else StepStatus.FAILED
                )

                # --- RAG検索結果に基づく動的条件分岐 ---
                if step.action == "rag_search" and result.status == "success":
                    rag_max_score = 0.0
                    if step.step_id in self.step_confidence_scores:
                        rag_max_score = self.step_confidence_scores[step.step_id].factors.search_max_score

                    rag_threshold = self.config.qdrant.rag_sufficient_score  # デフォルト 0.7

                    # web_searchを実行すべきかの判定
                    need_web_search = False

                    if rag_max_score < rag_threshold:
                        # スコア不足 → 無条件でweb_search
                        logger.info(
                            f"RAG score insufficient ({rag_max_score:.4f} < {rag_threshold}), "
                            f"need web_search"
                        )
                        need_web_search = True
                    else:
                        # スコア十分 → LLMで意味的適合性を検証
                        logger.info(
                            f"RAG score sufficient ({rag_max_score:.4f} >= {rag_threshold}), "
                            f"checking semantic relevance with LLM"
                        )
                        is_relevant = self._evaluate_rag_relevance(
                            query=step.query or step.description,
                            rag_output=result.output or "",
                        )
                        if not is_relevant:
                            logger.info("RAG result not semantically relevant, need web_search")
                            need_web_search = True
                        else:
                            logger.info("RAG result semantically relevant, skipping web_search")

                    if need_web_search:
                        # パターン(2)(3): web_search を動的実行
                        web_result = yield from self._execute_dynamic_web_search(step, state)

                        if web_result is None or web_result.status == "failed":
                            # パターン(3): Webも失敗 → ask_user を動的実行
                            logger.info("Web search also failed, executing ask_user")
                            yield from self._execute_dynamic_ask_user(step, state)
                    else:
                        # パターン(1): web_search スキップ
                        for future_step in steps_to_execute:
                            if future_step.action == "web_search" and future_step.step_id > step.step_id:
                                state.step_statuses[future_step.step_id] = StepStatus.SKIPPED
                                logger.info(f"Skipping planned web_search step {future_step.step_id}")

                # ステップ完了コールバック
                if self.on_step_complete:
                    self.on_step_complete(result)

                # 信頼度に基づく介入チェック (Phase 3)
                if step.step_id in self.step_confidence_scores:
                    confidence_score = self.step_confidence_scores[step.step_id]
                    action_decision = self.confidence_calculator.decide_action(confidence_score)

                    # ESCALATE のみ一時停止（CONFIRM は通知して自動続行）
                    if action_decision.level == InterventionLevel.ESCALATE:
                        logger.info(f"Pausing for intervention: {action_decision.level} (Step {step.step_id})")

                        state.is_paused = True

                        message = f"信頼度が非常に低いためエスカレーションが必要です ({confidence_score.score:.2f})"
                        if action_decision.reason:
                            message += f"\n理由: {action_decision.reason}"

                        state.intervention_request = InterventionRequest(
                            level=action_decision.level,
                            step_id=step.step_id,
                            message=message,
                            reason=action_decision.reason,
                            confidence_score=confidence_score.score,
                            plan=plan
                        )

                        # Yield: 一時停止状態を通知
                        yield state

                        # ジェネレータを終了（再開時は新しいジェネレータを作成）
                        return self._create_execution_result(state)

                    elif action_decision.level == InterventionLevel.CONFIRM:
                        # CONFIRM は警告ログのみ・自動続行
                        message = f"信頼度が低いため確認推奨 ({confidence_score.score:.2f}) — 自動続行"
                        if action_decision.reason:
                            message += f" / 理由: {action_decision.reason}"
                        logger.info(f"[CONFIRM] {message}")
                        self._handle_intervention_notify(message)

                    else:
                        # SILENT / NOTIFY
                        self._handle_intervention_if_needed(action_decision, step, state)

                # Yield: ステップ完了状態を通知
                yield state

                # ask_user の場合の処理（既存ロジック）
                if step.action == "ask_user" and result.status == "success":
                    # ...既存のask_user処理...
                    pass

                # 失敗時またはステップ信頼度低下時のリプラン
                # 低信頼度トリガーは検索系ステップに限定する（#64）。
                if self._should_trigger_replan(step, result, state):
                    replan_result = self.replan_orchestrator.handle_step_failure(
                        step_result=result,
                        current_plan=plan,
                        completed_results=state.step_results,
                        replan_count=state.replan_count
                    )
                    if replan_result and replan_result.success and replan_result.new_plan:
                        logger.info(f"Replanning: {replan_result.reason}")
                        state.replan_count += 1

                        # 新しい計画に差し替え
                        # Generatorを再帰呼び出しするか、ループを再構成する必要がある
                        # ここでは、新しい計画で再帰的にGeneratorを作成し、その値をYieldする
                        state.plan = replan_result.new_plan
                        # 再帰呼び出し
                        yield from self.execute_plan_generator(replan_result.new_plan, state)
                        # 再帰から戻ったら終了（新しい計画が完了しているため）
                        return self._create_execution_result(state)

            # 全体の信頼度を計算
            state.overall_confidence = self._calculate_overall_confidence(state)
            state.end_time = time.time()

            # P4: 実行メモリへ記録（使用コレクションごとの成否・信頼度）
            self._record_memory(state)

            # 最終結果
            return self._create_execution_result(state)

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            state.end_time = time.time()
            # エラー時も結果を返す
            return ExecutionResult(
                plan_id=plan.plan_id or create_plan_id(),
                original_query=plan.original_query,
                final_answer=f"実行エラー: {str(e)}",
                step_results=list(state.step_results.values()),
                overall_confidence=0.0,
                overall_status="failed",
                replan_count=state.replan_count,
                total_execution_time_ms=state.get_execution_time_ms(),
                total_token_usage=None,
                total_cost_usd=None,
            )

    def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        """
        計画を実行（ブロッキング版）

        execute_plan_generator() をドレインする薄いラッパー。
        動的 web_search・介入・SKIP・リプラン処理を含め、ジェネレータ版と
        完全に同一のロジックで実行される（#59 二重実行ループの解消）。

        Args:
            plan: 実行する計画
        Returns:
            ExecutionResult: 実行結果
        """
        logger.info(f"Executing plan (blocking): {plan.plan_id}, steps={len(plan.steps)}")

        gen = self.execute_plan_generator(plan)
        try:
            while True:
                event = next(gen)
                # 中間イベント（ログなど）はブロッキング版ではログ出力のみ
                if isinstance(event, dict) and event.get("type") == "log":
                    logger.info(event.get("content"))
        except StopIteration as e:
            return e.value

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """execute_plan() の統一エントリーポイント（benchmark.py 互換）"""
        return self.execute_plan(plan)

    def _should_trigger_replan(
            self,
            step: PlanStep,
            result: StepResult,
            state: ExecutionState
    ) -> bool:
        """リプランを発火すべきか判定する。

        - ステップ失敗: 常にリプラン対象
        - 低信頼度: 検索系ステップ（rag_search / web_search）のみ対象
          （reasoning 成功後に計画全体を再実行すると同一検索を繰り返し
          コストが max_replans 倍に膨らむため限定する）
        - リプラン回数上限（config.replan.max_replans）超過時は発火しない
        """
        if not self.replan_orchestrator:
            return False
        if not state.can_replan():
            return False
        if result.status == "failed":
            return True
        is_search_step = step.action in ("rag_search", "web_search")
        return is_search_step and result.confidence < self.config.replan.confidence_threshold

    def _handle_ask_user_response(
            self,
            step: PlanStep,
            result: StepResult,
            state: ExecutionState
    ) -> None:
        """ask_user ステップの出力を UI コールバックへ渡し、ユーザー応答を結果へ反映する"""
        if not self.on_intervention_required:
            return

        output = result.output
        if isinstance(output, dict):
            output_data = output
        elif isinstance(output, str):
            # [SECURITY] eval() ではなく ast.literal_eval() を使用（任意コード実行を防止）
            try:
                parsed = ast.literal_eval(output)
                output_data = parsed if isinstance(parsed, dict) else {"question": output}
            except (ValueError, SyntaxError):
                output_data = {"question": output}
        else:
            output_data = {"question": str(output)}

        user_response = self.on_intervention_required("ask_user", output_data)
        if user_response:
            # ユーザー応答を後続ステップ（reasoning等）で利用可能にする
            result.output = f"ユーザー応答: {user_response}"
            state.step_results[step.step_id] = result

    _SEARCH_ACTIONS = ("rag_search", "web_search")

    def _prefetch_parallel_searches(
            self,
            current_step: PlanStep,
            steps_to_execute: List[PlanStep],
            state: ExecutionState
    ) -> None:
        """現在のステップと依存関係のない後続検索ステップを並列に先行実行する（#60）。

        現在のステップが検索系であり、後続にも依存関係のない検索ステップが
        ある場合のみ並列化する（依存DAGの同一ウェーブ）。結果は
        _prefetched_tool_results にキャッシュされ、各ステップの
        _execute_step で消費される。例外もキャッシュされ、消費時に再送出
        されるため fallback 処理は逐次実行と同じ経路を通る。

        注: 並列実行されたステップのトークン集計は逐次実行時と異なり
        ステップ単位で分離されない（検索ツールは通常LLMを使用しないため影響は軽微）。
        """
        if current_step.action not in self._SEARCH_ACTIONS:
            return
        if current_step.step_id in self._prefetched_tool_results:
            return  # 既にプリフェッチ済み

        # 同一ウェーブの検索ステップを収集
        batch = [current_step]
        batch_ids = {current_step.step_id}
        for s in steps_to_execute:
            if s.step_id <= current_step.step_id or s.action not in self._SEARCH_ACTIONS:
                continue
            if s.step_id in self._prefetched_tool_results:
                continue
            if state.step_statuses.get(s.step_id) not in (StepStatus.PENDING, None):
                continue
            # バッチ内ステップ・未完了ステップへの依存があれば並列化不可
            deps_ok = all(
                dep not in batch_ids
                and dep in state.step_results
                and state.step_results[dep].status == "success"
                for dep in s.depends_on
            )
            if not deps_ok:
                continue
            batch.append(s)
            batch_ids.add(s.step_id)
            if len(batch) >= self.config.executor.max_parallel_steps:
                break

        if len(batch) < 2:
            return

        logger.info(f"Parallel search execution: steps={[s.step_id for s in batch]}")

        from concurrent.futures import ThreadPoolExecutor as _Pool

        pool = _Pool(max_workers=len(batch), thread_name_prefix="grace-parallel")
        try:
            futures = {}
            for s in batch:
                tool = self.tool_registry.get(s.action)
                if tool is None:
                    continue
                kwargs = self._prepare_tool_kwargs(s, state)
                futures[s.step_id] = (s, pool.submit(tool.execute, **kwargs))

            for sid, (s, future) in futures.items():
                try:
                    self._prefetched_tool_results[sid] = future.result(
                        timeout=s.timeout_seconds or None
                    )
                except Exception as e:
                    logger.warning(f"Parallel execution of step {sid} failed: {e}")
                    self._prefetched_tool_results[sid] = e
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _run_tool_with_timeout(
            self,
            tool: Any,
            kwargs: Dict[str, Any],
            step: PlanStep
    ) -> ToolResult:
        """ツールを timeout_seconds 制限付きで実行する。

        タイムアウト時は TimeoutError を送出し、呼び出し元の
        フォールバック/失敗処理に委ねる。
        （実行中のスレッド自体は中断できないためバックグラウンドで放置される）
        """
        timeout = step.timeout_seconds
        if not timeout:
            return tool.execute(**kwargs)

        from concurrent.futures import ThreadPoolExecutor as _Pool
        from concurrent.futures import TimeoutError as _FutureTimeout

        pool = _Pool(max_workers=1, thread_name_prefix=f"grace-step-{step.step_id}")
        try:
            future = pool.submit(tool.execute, **kwargs)
            return future.result(timeout=timeout)
        except _FutureTimeout:
            logger.warning(
                f"Step {step.step_id}: tool '{step.action}' timed out after {timeout}s"
            )
            raise TimeoutError(
                f"ステップ {step.step_id} ({step.action}) が {timeout} 秒でタイムアウトしました"
            )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _check_dependencies(self, step: PlanStep, state: ExecutionState) -> bool:
        """依存ステップの完了確認"""
        for dep_id in step.depends_on:
            if dep_id not in state.step_results:
                return False
            if state.step_results[dep_id].status == "failed":
                return False
        return True

    def _execute_step(self, step: PlanStep, state: ExecutionState) -> Any:
        """
        個別ステップの実行
        Args:
            step: 実行するステップ
            state: 現在の実行状態
        Returns:
            StepResult or Generator: ステップ実行結果（またはジェネレータ）
        """
        logger.info(f"Executing step {step.step_id}: {step.action} - {step.description}")

        start_time = time.time()

        try:
            # ツールを取得
            tool = self.tool_registry.get(step.action)

            # --- 互換性維持のための特別なハンドリング ---
            if tool is None and step.action == "run_legacy_agent":
                # ツールとして登録されていないが、以前のLegacyプランが残っている場合
                return self._execute_legacy_agent_step(step, state, start_time)

            if tool is None:
                raise ValueError(f"Unknown action: {step.action}")

            # ツール実行引数を準備
            kwargs = self._prepare_tool_kwargs(step, state)

            # 実行（並列プリフェッチ済みの結果があれば消費・#60、
            # なければ PlanStep.timeout_seconds 制限付きで実行・#57）
            prefetched = self._prefetched_tool_results.pop(step.step_id, None)
            if isinstance(prefetched, Exception):
                raise prefetched
            elif prefetched is not None:
                tool_result: ToolResult = prefetched
            else:
                tool_result = self._run_tool_with_timeout(tool, kwargs, step)

            # --- UIへの中間結果通知 (思考プロセス表示用) ---
            if tool_result.success and tool_result.output:
                import json
                try:
                    # RAG検索結果などはリスト/辞書なので整形する
                    out_display = json.dumps(tool_result.output, indent=2, ensure_ascii=False) if isinstance(
                        tool_result.output, (list, dict)) else str(tool_result.output)
                except Exception:
                    out_display = str(tool_result.output)

                # IPO風のラベルをつけて通知
                yield {
                    "type"   : "log",
                    "content": f"📝 【ツール実行結果: {step.action}】\n{out_display}"
                }

            # P4: 使用した RAG コレクションを実行メモリ用に記録
            if step.action == "rag_search" and isinstance(tool_result.confidence_factors, dict):
                uc = tool_result.confidence_factors.get("used_collection")
                if uc and uc not in state.used_collections:
                    state.used_collections.append(uc)

            # 実行時間
            execution_time = int((time.time() - start_time) * 1000)

            # ----------------------
            # 信頼度を計算（state引数を渡す）
            # ----------------------
            # confidence = self._calculate_step_confidence(tool_result, step, state)
            confidence = self._llm_calculate_step_confidence(tool_result, step, state)

            # ソースを抽出
            sources = self._extract_sources(tool_result)

            return StepResult(
                step_id=step.step_id,
                status="success" if tool_result.success else "failed",
                output=self._format_output(tool_result.output),
                confidence=confidence,
                sources=sources,
                error=tool_result.error if not tool_result.success else None,
                execution_time_ms=execution_time,
                token_usage=None,
            )

        except Exception as e:
            logger.error(f"Step {step.step_id} failed: {e}")
            execution_time = int((time.time() - start_time) * 1000)

            # フォールバック処理
            if step.fallback:
                logger.info(f"Attempting fallback: {step.fallback}")
                fallback_result = self._execute_fallback(step, state)
                if fallback_result.status == "success":
                    return fallback_result

            return StepResult(
                step_id=step.step_id,
                status="failed",
                output=None,
                confidence=0.0,
                error=str(e),
                execution_time_ms=execution_time,
                token_usage=None,
            )

    def _execute_legacy_agent_step(self, step: PlanStep, state: ExecutionState, start_time: float) -> Generator[
        Any, None, StepResult]:
        """Legacy ReActAgent を使用したステップ実行（ジェネレータ版）"""
        if not LEGACY_AGENT_AVAILABLE:
            raise ImportError("agent_service module not found")

        # 1. コレクション準備
        available_collections = get_available_collections_from_qdrant_helper()
        if not available_collections:
            available_collections = self.config.qdrant.search_priority

        # 2. Agent初期化
        agent = ReActAgent(
            selected_collections=available_collections,
            model_name=self.config.llm.model
        )

        query = step.query or step.description
        logger.info(f"Running Legacy Agent with query: {query}")

        final_answer = ""
        sources = []

        # 3. エージェント実行（ジェネレータ）
        # ストリーミングイベントを拾いながら、ツール結果からソースを収集
        for event in agent.execute_turn(query):
            # イベントをそのまま上位へ流す（UI表示用）
            yield event

            # ログ出力（デバッグ用）
            if event["type"] == "log":
                logger.info(f"[LegacyAgent] {event['content']}")
            elif event["type"] == "tool_call":
                logger.info(f"[LegacyAgent] Tool Call: {event['name']} args={event['args']}")
            elif event["type"] == "tool_result":
                logger.info(f"[LegacyAgent] Tool Result (len={len(event['content'])})")
                # ソース抽出 (簡易的な文字列解析)
                if "Source:" in event["content"]:
                    import re
                    # Source: filename.csv のパターンを抽出
                    found_sources = re.findall(r"Source:\s*([a-zA-Z0-9_.\-]+)", event["content"])
                    if found_sources:
                        sources.extend(found_sources)
            elif event["type"] == "final_answer":
                final_answer = event["content"]

        # 4. 結果構築
        execution_time = int((time.time() - start_time) * 1000)

        # ソースの重複排除
        sources = list(set(sources))

        # Confidence計算 (簡易版)
        confidence = 0.8 if final_answer and "申し訳ありません" not in final_answer else 0.3

        # ConfidenceScoreオブジェクトを作成して保存
        conf_score_obj = ConfidenceScore(
            score=confidence,
            factors=ConfidenceFactors(
                source_count=len(sources),
                search_result_count=len(sources),
                llm_self_confidence=confidence
            )
        )
        self.step_confidence_scores[step.step_id] = conf_score_obj

        # アクション判定
        if self.on_confidence_update:
            action = self.confidence_calculator.decide_action(conf_score_obj)
            self.on_confidence_update(conf_score_obj, action)

        return StepResult(
            step_id=step.step_id,
            status="success",
            output=final_answer,
            confidence=confidence,
            sources=sources,
            error=None,
            execution_time_ms=execution_time,
            token_usage=None,
        )

    def _prepare_tool_kwargs(
            self,
            step: PlanStep,
            state: ExecutionState
    ) -> Dict[str, Any]:
        """ツール実行引数を準備"""
        kwargs: Dict[str, Any] = {
            "query": step.query or step.description,
        }

        if step.action == "rag_search":
            kwargs["collection"] = step.collection

        elif step.action == "web_search":
            kwargs["num_results"] = self.config.web_search.num_results
            kwargs["language"] = self.config.web_search.language

        elif step.action == "reasoning":
            # 全成功ステップの結果をコンテキストとして追加
            # （depends_on ではなく state.step_results 全体を参照）
            # → 動的挿入された web_search やリプラン後の結果も取得可能
            context_parts = []
            sources = []
            logger.info("--- Reasoning Step ---")
            logger.info(f"Step: {step}")
            logger.info(f"Available step_results: {list(state.step_results.keys())}")

            for dep_id in sorted(state.step_results.keys()):
                dep_result = state.step_results[dep_id]
                if dep_result.status != "success":
                    continue
                dep_output = dep_result.output

                if dep_output:
                    # 文字列化されたリストを復元する試み
                    if isinstance(dep_output, str):
                        try:
                            # RAG検索結果は "[{...}, {...}]" 形式で文字列化されている
                            if dep_output.startswith("[{") or dep_output.startswith("[{'"):
                                import ast
                                parsed = ast.literal_eval(dep_output)
                                if isinstance(parsed, list):
                                    sources.extend(parsed)
                                    continue
                        except (ValueError, SyntaxError):
                            pass
                        # パースできない場合はコンテキストとして追加
                        context_parts.append(f"--- 参照情報 (Step {dep_id}) ---\n{dep_output}")
                    elif isinstance(dep_output, list):
                        sources.extend(dep_output)

            if sources:
                kwargs["sources"] = sources
            if context_parts:
                kwargs["context"] = "\n\n".join(context_parts)

        elif step.action == "ask_user":
            kwargs.update({
                "question": step.query or step.description,
                "reason"  : f"ステップ {step.step_id}: {step.description}",
                "urgency" : "blocking"
            })

        return kwargs

    def _evaluate_rag_relevance(
            self,
            query: str,
            rag_output: str,
    ) -> bool:
        """
        LLMを使用してRAG検索結果がユーザーの質問に意味的に適合しているかを判定する。

        コサイン類似度は文構造の類似性を反映するが、意味的な適合性は保証しない。
        例: 「日本の多義性」と「言語の多義性」は文構造が似ているが主題が異なる。

        Args:
            query: ユーザーの元の質問文
            rag_output: RAG検索結果の出力文字列

        Returns:
            bool: 適合していればTrue、不適合ならFalse
        """
        prompt = (
            "以下の【検索結果】が、【ユーザーの質問】に対する回答として使えるかを判定してください。\n"
            "\n"
            "【判定基準】\n"
            "- 検索結果の主題が質問の主題と一致しているか\n"
            "- 質問に対する回答に必要な情報が含まれているか\n"
            "\n"
            f"【ユーザーの質問】\n{query}\n"
            f"\n"
            f"【検索結果】\n{rag_output[:500]}\n"
            "\n"
            "回答として使える場合は YES、使えない場合は NO とだけ回答してください。"
        )

        try:
            # [MIGRATION] from google import genai / types → create_llm_client("openai")
            # Gemini: genai.Client().models.generate_content() + GenerateContentConfig + AFC無効化
            # OpenAI:  create_llm_client("openai").generate_content() で代替
            import time as _time

            if not _LLM_CLIENT_AVAILABLE:
                raise ImportError("helper_llm.create_llm_client が利用できません")

            llm = create_llm_client("openai", default_model=self.config.llm.model)
            t0 = _time.time()

            # YES/NO のみ返させる。gpt-5系は推論モデルで出力枠をまず推論トークンに消費するため、
            # max_completion_tokens=5 では本文が空になる。十分な枠を確保する。
            answer = (llm.generate_content(
                prompt=prompt,
                temperature=0.0,
                max_completion_tokens=256,
            ) or "").strip().upper()

            elapsed = _time.time() - t0
            if not answer:
                # 推論モデルが出力枠を使い切る等で空応答になった場合は判定不能とみなし、
                # 例外時と同じく既存動作（スコアのみで判定＝適合扱い）を維持して True を返す。
                logger.warning("RAG relevance check returned empty answer, defaulting to True")
                return True
            is_relevant = "YES" in answer
            logger.info(
                f"RAG relevance check: '{answer}' -> {is_relevant} ({elapsed:.1f}s)"
            )
            return is_relevant

        except Exception as e:
            logger.warning(f"RAG relevance check failed: {e}, defaulting to True")
            return True  # フォールバック: 判定失敗時は既存動作（スコアのみで判定）を維持

    def _execute_dynamic_web_search(
            self,
            rag_step: PlanStep,
            state: ExecutionState
    ) -> Generator:
        """
        RAGスコア不足時に web_search を動的に実行する。

        Args:
            rag_step: 直前に実行された rag_search ステップ
            state: 現在の実行状態

        Yields:
            state: 中間状態

        Returns:
            StepResult or None: web_search の結果
        """
        web_step_id = rag_step.step_id + 100  # 動的挿入用に大きなIDを付与
        web_step = PlanStep(
            step_id=web_step_id,
            action="web_search",
            description="[動的挿入] RAGスコア不足のためWeb検索を実行",
            query=rag_step.query,
            collection=None,
            depends_on=[rag_step.step_id],
            expected_output="Web検索結果",
            fallback=None,
            timeout_seconds=15,  # タイムアウト短め
        )

        logger.info(f"Dynamic web_search: step_id={web_step_id}, query={rag_step.query[:50]}")

        state.current_step_id = web_step_id
        state.step_statuses[web_step_id] = StepStatus.RUNNING

        if self.on_step_start:
            self.on_step_start(web_step)

        try:
            step_execution = self._execute_step(web_step, state)
            web_result = None
            if isinstance(step_execution, Generator):
                web_result = yield from step_execution
            else:
                web_result = step_execution

            state.step_results[web_step_id] = web_result
            state.step_statuses[web_step_id] = (
                StepStatus.SUCCESS if web_result.status == "success" else StepStatus.FAILED
            )

            if self.on_step_complete:
                self.on_step_complete(web_result)

            yield state
            return web_result

        except Exception as e:
            logger.error(f"Dynamic web_search failed: {e}")
            failed_result = StepResult(
                step_id=web_step_id,
                status="failed",
                output=None,
                confidence=0.0,
                error=str(e),
                execution_time_ms=0,
                token_usage=None,
            )
            state.step_results[web_step_id] = failed_result
            state.step_statuses[web_step_id] = StepStatus.FAILED
            yield state
            return failed_result

    def _execute_dynamic_ask_user(
            self,
            rag_step: PlanStep,
            state: ExecutionState
    ) -> Generator:
        """
        RAG・Web検索の両方が不十分な場合に ask_user を動的に実行する。

        Args:
            rag_step: 元の rag_search ステップ
            state: 現在の実行状態

        Yields:
            state: 中間状態
        """
        ask_step_id = rag_step.step_id + 200  # 動的挿入用ID
        ask_step = PlanStep(
            step_id=ask_step_id,
            action="ask_user",
            description="[動的挿入] 検索結果が不十分なためユーザーに確認",
            query=(
                f"「{rag_step.query[:100]}」について検索しましたが、"
                f"十分な情報が見つかりませんでした。\n"
                f"追加の情報があれば教えてください。"
                f"または、現在の情報で回答を試みますか？"
            ),
            collection=None,
            depends_on=[rag_step.step_id],
            expected_output="ユーザーの指示",
            fallback=None,
        )

        logger.info(f"Dynamic ask_user: step_id={ask_step_id}")

        state.current_step_id = ask_step_id
        state.step_statuses[ask_step_id] = StepStatus.RUNNING

        if self.on_step_start:
            self.on_step_start(ask_step)

        try:
            step_execution = self._execute_step(ask_step, state)
            ask_result = None
            if isinstance(step_execution, Generator):
                ask_result = yield from step_execution
            else:
                ask_result = step_execution

            state.step_results[ask_step_id] = ask_result
            state.step_statuses[ask_step_id] = (
                StepStatus.SUCCESS if ask_result.status == "success" else StepStatus.FAILED
            )

            if self.on_step_complete:
                self.on_step_complete(ask_result)

            yield state

        except Exception as e:
            logger.error(f"Dynamic ask_user failed: {e}")
            yield state

    def _execute_fallback(
            self,
            step: PlanStep,
            state: ExecutionState
    ) -> StepResult:
        """フォールバックアクションを実行"""
        fallback_step = PlanStep(
            step_id=step.step_id,
            action=cast(Literal["rag_search", "web_search", "reasoning", "ask_user", "code_execute", "run_legacy_agent"], step.fallback),
            description=f"[Fallback] {step.description}",
            query=step.query,
            collection=step.collection,
            depends_on=step.depends_on,
            expected_output=step.expected_output,
            fallback=None,  # 二重フォールバックは無し
            timeout_seconds=step.timeout_seconds,
        )
        step_execution = self._execute_step(fallback_step, state)
        if isinstance(step_execution, Generator):
            try:
                while True:
                    next(step_execution)
            except StopIteration as e:
                return e.value
        return step_execution

    def _build_confidence_factors(
            self,
            tool_result: ToolResult,
            step: PlanStep,
            state: ExecutionState
    ) -> ConfidenceFactors:
        """ツール結果とステップ情報から ConfidenceFactors を構築する（共通部・#66）

        LLM評価版/ヒューリスティック版の両経路で重複していた構築ロジックを集約する。
        """
        factors = tool_result.confidence_factors

        # source_count の決定: ツールが明示的に返した値を優先
        extracted_sources = self._extract_sources(tool_result)
        source_count = factors.get("source_count", len(extracted_sources))

        # ソース一致度 (Source Agreement) の計算
        source_agreement = 1.0
        if source_count > 1:
            texts = []
            if isinstance(tool_result.output, list):
                for item in tool_result.output:
                    if isinstance(item, dict):
                        payload = item.get("payload", {})
                        content = payload.get("content") or payload.get("text") or payload.get("answer")
                        if content:
                            texts.append(str(content))

            if len(texts) > 1:
                try:
                    sa_calc = create_source_agreement_calculator(config=self.config)
                    source_agreement = sa_calc.calculate(texts)
                    logger.info(f"[confidence] Calculated source_agreement: {source_agreement:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate source_agreement: {e}")
                    source_agreement = 0.5

        # 依存ステップからのスコア継承ロジック
        current_result_count = factors.get("result_count", 0)
        current_max_score = factors.get("max_score", factors.get("avg_score", 0.0))
        current_avg_score = factors.get("avg_score", 0.0)

        # 自身で検索しておらず、かつ推論ステップなどの場合、依存元のスコアを引き継ぐ
        if current_result_count == 0 and step.action not in ("rag_search", "web_search"):
            inherited_max = 0.0
            inherited_found = False
            for dep_id in step.depends_on:
                if dep_id in state.step_results:
                    dep_res = state.step_results[dep_id]
                    if dep_res.confidence > inherited_max:
                        inherited_max = dep_res.confidence
                        inherited_found = True

            if inherited_found:
                logger.info(f"[confidence] Inherited scores from dependency: max={inherited_max}")
                current_max_score = inherited_max
                current_avg_score = inherited_max
                current_result_count = 1  # 仮想的に1件あったとみなす

        return ConfidenceFactors(
            # RAG検索関連
            search_result_count=current_result_count,
            search_avg_score=current_avg_score,
            search_max_score=current_max_score,
            search_score_variance=factors.get("score_variance", 1.0),
            # ソース関連
            source_count=source_count,
            source_agreement=source_agreement,
            # ツール実行関連
            tool_success_rate=1.0 if tool_result.success else 0.0,
            tool_execution_count=1,
            tool_success_count=1 if tool_result.success else 0,
            # ステップタイプ
            is_search_step=(step.action in ["rag_search", "web_search"])
        )

    def _llm_calculate_step_confidence(
            self,
            tool_result: ToolResult,
            step: PlanStep,
            state: ExecutionState
    ) -> float:
        """
        LLMを使用したステップ信頼度の計算
        """
        if not tool_result.success:
            return 0.0

        # ConfidenceFactors を構築（共通部・#66）
        confidence_factors = self._build_confidence_factors(tool_result, step, state)
        logger.info(f"[_llm_calculate_step_confidence] Constructed ConfidenceFactors: {confidence_factors}")

        # ConfidenceCalculatorで計算（LLM評価 + Heuristicフォールバック）
        try:
            confidence_score = self.confidence_calculator.llm_calculate(
                factors=confidence_factors,
                step_description=step.description,
                tool_output=str(tool_result.output)
            )

            # LLM評価が低すぎる場合、Heuristicで再計算して比較
            if confidence_score.score < 0.6 and confidence_factors.is_search_step:
                heuristic_score = self.confidence_calculator.calculate(confidence_factors)
                if heuristic_score.score > confidence_score.score:
                    logger.info(
                        f"Using heuristic score {heuristic_score.score:.2f} "
                        f"instead of LLM score {confidence_score.score:.2f}"
                    )
                    confidence_score = heuristic_score

        except Exception as e:
            logger.error(f"LLM confidence calculation failed: {e}, falling back to heuristic")
            confidence_score = self.confidence_calculator.calculate(confidence_factors)

        # ステップごとのConfidenceScoreを保存
        self.step_confidence_scores[step.step_id] = confidence_score

        # アクション決定を取得
        action_decision = self.confidence_calculator.decide_action(confidence_score)

        # コールバックで通知（Phase 3のHITLと連携）
        if self.on_confidence_update:
            self.on_confidence_update(confidence_score, action_decision)

        logger.info(
            f"Step {step.step_id} confidence: {confidence_score.score:.2f} "
            f"(level={confidence_score.level}, action={action_decision.level.value})"
        )

        return confidence_score.score

    # -------------------
    # Step 3の評価を担当する関数
    # -------------------
    def _calculate_step_confidence(
            self,
            tool_result: ToolResult,
            step: PlanStep,
            state: ExecutionState
    ) -> float:
        """
        ステップの信頼度を計算（ConfidenceCalculator使用 - Heuristic版）
        Args:
            tool_result: ツール実行結果
            step: 実行したステップ
            state: 現在の実行状態
        Returns:
            float: 信頼度スコア (0.0-1.0)
        """
        if not tool_result.success:
            return 0.0

        # ConfidenceFactors を構築（共通部・#66）
        confidence_factors = self._build_confidence_factors(tool_result, step, state)
        logger.info(f"[_calculate_step_confidence] Constructed ConfidenceFactors: {confidence_factors}")

        # ConfidenceCalculatorで計算
        confidence_score = self.confidence_calculator.calculate(confidence_factors)

        # ステップごとのConfidenceScoreを保存
        self.step_confidence_scores[step.step_id] = confidence_score

        # アクション決定を取得
        action_decision = self.confidence_calculator.decide_action(confidence_score)

        # コールバックで通知（Phase 3のHITLと連携）
        if self.on_confidence_update:
            self.on_confidence_update(confidence_score, action_decision)

        logger.info(
            f"Step {step.step_id} confidence: {confidence_score.score:.2f} "
            f"(level={confidence_score.level}, action={action_decision.level.value})"
        )

        return confidence_score.score

    def _extract_sources(self, tool_result: ToolResult) -> List[str]:
        """ツール結果からソースを抽出"""
        sources = []

        if isinstance(tool_result.output, list):
            for item in tool_result.output:
                if isinstance(item, dict):
                    payload = item.get("payload", {})
                    source = payload.get("source", "")
                    if source and source not in sources:
                        sources.append(source)

        return sources

    def _format_output(self, output: Any) -> Optional[str]:
        """出力を文字列にフォーマット"""
        if output is None:
            return None
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            return str(output)
        if isinstance(output, list):
            # RAG検索結果の場合
            if output and isinstance(output[0], dict):
                return str(output)
            return "\n".join(str(item) for item in output)
        return str(output)

    def _calculate_overall_confidence(self, state: ExecutionState) -> float:
        """全体の信頼度を計算し、S1 較正（temperature scaling）を適用して返す。

        生の集約値は `_compute_raw_overall_confidence` が計算する。較正ファイルが
        無い（既定）場合は恒等変換（T=1.0）なので挙動は変わらない。
        """
        raw = self._compute_raw_overall_confidence(state)
        if not getattr(self.config.confidence, "calibration_enabled", True):
            return round(min(1.0, max(0.0, raw)), 3)
        calibrated = self._calibrator.transform(raw)
        if not self._calibrator.is_identity():
            logger.info(f"Calibrated confidence: {raw:.3f} -> {calibrated:.3f} "
                        f"(T={self._calibrator.temperature})")
        return round(min(1.0, max(0.0, calibrated)), 3)

    def _compute_raw_overall_confidence(self, state: ExecutionState) -> float:
        """
        全体の信頼度を計算（ConfidenceAggregator + LLMSelfEvaluator使用）

        Args:
            state: 実行状態

        Returns:
            float: 全体の信頼度スコア (0.0-1.0)
        """
        if not state.step_results:
            return 0.0

        # 各ステップのConfidenceScoreを収集
        step_scores = list(self.step_confidence_scores.values())

        # 最新のbreakdownを取得（ベースとして使用）
        current_breakdown = {}
        if step_scores:
            # 最後のステップのbreakdownをコピー
            current_breakdown = step_scores[-1].breakdown.copy()

        # 最終回答を取得（最後のreasoningまたはlegacy_agentステップの出力）
        final_answer: Optional[str] = None
        for step in reversed(state.plan.steps):
            if (step.action in ["reasoning", "run_legacy_agent"]) and step.step_id in state.step_results:
                result = state.step_results[step.step_id]
                if result.status == "success":
                    final_answer = result.output
                    break

        # LLMSelfEvaluatorで最終回答を評価（#65: 自己評価＋クエリ網羅度を1回の呼び出しに統合）
        if final_answer is not None:
            try:
                final_eval = self.llm_evaluator.evaluate_final(
                    query=state.plan.original_query,
                    answer=final_answer,
                    sources=state.get_completed_sources()
                )

                # 1. LLM自己評価 (Accuracy/Style etc.)
                current_breakdown["llm_self_eval"] = final_eval.self_eval_score
                llm_score = ConfidenceScore(
                    score=final_eval.self_eval_score,
                    factors=ConfidenceFactors(llm_self_confidence=final_eval.self_eval_score),
                    breakdown=current_breakdown.copy(),
                    reason=final_eval.reason,
                )
                step_scores.append(llm_score)
                logger.info(f"LLM self-evaluation: {final_eval.self_eval_score:.2f}")

                # 2. クエリ網羅度評価 (Query Coverage)
                current_breakdown["query_coverage"] = final_eval.coverage_score
                coverage_obj = ConfidenceScore(
                    score=final_eval.coverage_score,
                    factors=ConfidenceFactors(query_coverage=final_eval.coverage_score),
                    breakdown=current_breakdown.copy()
                )
                step_scores.append(coverage_obj)
                logger.info(f"Query coverage evaluation: {final_eval.coverage_score:.2f}")

                # UIへの反映のために、最後の信頼度更新として通知する
                if self.on_confidence_update:
                    decision = ActionDecision(
                        level=InterventionLevel.SILENT,
                        confidence_score=final_eval.coverage_score,
                        reason="Final coverage evaluation completed"
                    )
                    self.on_confidence_update(coverage_obj, decision)

            except Exception as e:
                logger.warning(f"Final evaluation (evaluate_final) failed: {e}")

        # ConfidenceAggregatorで統合
        if step_scores:
            aggregated_score = self.confidence_aggregator.aggregate(
                scores=step_scores,
                method="weighted"
            )
            logger.info(f"Aggregated confidence: {aggregated_score:.2f}")
            return aggregated_score

        # フォールバック: 単純平均
        confidences = [r.confidence for r in state.step_results.values()]
        return sum(confidences) / len(confidences)

    def _record_memory(self, state: ExecutionState) -> None:
        """P4: 実行結果を実行メモリへ記録する（best-effort）。

        使用したコレクションごとに (質問, 成否, overall_confidence) を蓄積し、
        以降の Planner のコレクション優先順位に反映する。
        """
        if self._memory is None:
            return
        try:
            statuses = [r.status for r in state.step_results.values()]
            success = bool(statuses) and all(s == "success" for s in statuses)
            collections = list(state.used_collections)
            if not collections:
                return  # コレクション未使用（web のみ等）は記録対象外
            self._memory.record_many(
                query=state.plan.original_query,
                collections=collections,
                success=success,
                confidence=state.overall_confidence,
            )
            logger.info(f"[memory] recorded {len(collections)} collection outcome(s) "
                        f"(success={success}, conf={state.overall_confidence:.2f})")
        except Exception as e:  # 記録失敗は実行を止めない
            logger.warning(f"_record_memory failed: {e}")

    def _create_execution_result(self, state: ExecutionState) -> ExecutionResult:
        """実行結果を生成"""
        # 全体ステータスを判定
        statuses = [r.status for r in state.step_results.values()]

        overall_status: Literal["success", "partial", "failed", "cancelled"]
        if state.is_cancelled:
            overall_status = "cancelled"
        elif all(s == "success" for s in statuses):
            overall_status = "success"
        elif any(s == "success" for s in statuses):
            overall_status = "partial"
        else:
            overall_status = "failed"

        # 最終回答を取得（最後のreasoningまたはlegacy_agentステップの出力）
        final_answer = None
        for step in reversed(state.plan.steps):
            if (step.action in ["reasoning", "run_legacy_agent"]) and step.step_id in state.step_results:
                result = state.step_results[step.step_id]
                if result.status == "success":
                    final_answer = result.output
                    break

        return ExecutionResult(
            plan_id=state.plan.plan_id or create_plan_id(),
            original_query=state.plan.original_query,
            final_answer=final_answer,
            step_results=list(state.step_results.values()),
            overall_confidence=state.overall_confidence,
            overall_status=overall_status,
            replan_count=state.replan_count,
            total_execution_time_ms=state.get_execution_time_ms(),
            total_token_usage=None,
            total_cost_usd=None,
        )

    def cancel(self, state: ExecutionState):
        """実行をキャンセル"""
        state.is_cancelled = True
        logger.info("Execution cancelled")

    def resume(self, state: ExecutionState):
        """実行を再開"""
        state.is_paused = False
        logger.info("Execution resumed")

    # =========================================================================
    # Intervention Handler コールバック（Phase 3）
    # =========================================================================

    def _handle_intervention_notify(self, message: str) -> None:
        """通知レベルの介入処理（ログ出力のみ）"""
        logger.info(f"[NOTIFY] {message}")
        # UIへの通知（オプション）
        if self.on_intervention_required:
            self.on_intervention_required("notify", {"message": message})

    def _handle_intervention_confirm(
            self,
            request: InterventionRequest
    ) -> InterventionResponse:
        """確認レベルの介入処理"""
        logger.info(f"[CONFIRM] {request.message}")

        # on_intervention_requiredコールバックでUIに確認を要求
        if self.on_intervention_required:
            user_response = self.on_intervention_required("confirm", {
                "message"   : request.message,
                "reason"    : request.reason,
                "options"   : request.options,
                "confidence": request.confidence_score,
            })

            if user_response:
                # ユーザー応答を解析
                if user_response in ["はい、続行", "proceed", "yes"]:
                    return InterventionResponse(action=InterventionAction.PROCEED)
                elif user_response in ["計画を修正", "modify"]:
                    return InterventionResponse(action=InterventionAction.MODIFY)
                elif user_response in ["キャンセル", "cancel", "no"]:
                    return InterventionResponse(action=InterventionAction.CANCEL)
                else:
                    # ユーザー入力として扱う
                    return InterventionResponse(
                        action=InterventionAction.INPUT,
                        user_input=str(user_response)
                    )

        # コールバックがない場合はデフォルトで続行
        return InterventionResponse(action=InterventionAction.PROCEED)

    def _handle_intervention_escalate(
            self,
            request: InterventionRequest
    ) -> InterventionResponse:
        """エスカレーションレベルの介入処理"""
        logger.info(f"[ESCALATE] {request.message}")

        # on_intervention_requiredコールバックでUIにユーザー入力を要求
        if self.on_intervention_required:
            user_response = self.on_intervention_required("escalate", {
                "message"   : request.message,
                "question"  : request.question,
                "reason"    : request.reason,
                "confidence": request.confidence_score,
            })

            if user_response:
                return InterventionResponse(
                    action=InterventionAction.INPUT,
                    user_input=str(user_response)
                )

        # コールバックがない場合はタイムアウト扱い
        return InterventionResponse(
            action=InterventionAction.PROCEED,
            timeout_reached=True
        )

    def _handle_intervention_if_needed(
            self,
            action_decision: ActionDecision,
            step: PlanStep,
            state: ExecutionState
    ) -> Optional[InterventionResponse]:
        """
        必要に応じて介入を処理

        Args:
            action_decision: 信頼度に基づくアクション決定
            step: 現在のステップ
            state: 実行状態

        Returns:
            InterventionResponse: 介入レスポンス（介入が発生した場合）、またはNone
        """
        # SILENT/NOTIFYは自動続行
        if action_decision.level in [InterventionLevel.SILENT, InterventionLevel.NOTIFY]:
            if action_decision.level == InterventionLevel.NOTIFY:
                self.intervention_handler.handle(action_decision, step, state.plan)
            return None

        # CONFIRM/ESCALATEはユーザー介入が必要
        response = self.intervention_handler.handle(action_decision, step, state.plan)

        # キャンセルの場合は実行を中止
        if response.action == InterventionAction.CANCEL:
            state.is_cancelled = True

        return response


# =============================================================================
# ファクトリ関数
# =============================================================================

def create_executor(
        config: Optional[GraceConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        **kwargs
) -> Executor:
    """Executorインスタンスを作成"""
    return Executor(
        config=config,
        tool_registry=tool_registry,
        **kwargs
    )


# =============================================================================
# エクスポート
# =============================================================================

__all__ = [
    "ExecutionState",
    "Executor",
    "create_executor",
]
