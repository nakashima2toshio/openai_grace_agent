"""
GRACE - Guided Reasoning with Adaptive Confidence Execution
適応型計画実行エージェント

Main Components:
- schemas: Pydantic models for ExecutionPlan, PlanStep, StepResult
- config: Configuration loader with YAML and environment variable support
- planner: Plan generation using Anthropic API
- executor: Plan execution with state management
- tools: Tool definitions (RAG search, reasoning, ask_user, etc.)
- confidence: Confidence score calculation (Phase 2)
- intervention: Human-in-the-loop intervention logic (Phase 3)
- replan: Adaptive replanning strategies (Phase 4)
"""

__version__ = "0.1.0"
__author__ = "GRACE Team"

# Schemas
# Confidence (Phase 2)
from grace.confidence import (
    ActionDecision,
    ConfidenceAggregator,
    ConfidenceCalculator,
    ConfidenceFactors,
    ConfidenceScore,
    InterventionLevel,
    LLMSelfEvaluator,
    QueryCoverageCalculator,
    SourceAgreementCalculator,
    create_confidence_aggregator,
    create_confidence_calculator,
    create_llm_evaluator,
    create_query_coverage_calculator,
    create_source_agreement_calculator,
)

# Config
from grace.config import (
    GraceConfig,
    get_config,
    reload_config,
)

# Executor
from grace.executor import (
    ExecutionState,
    Executor,
    create_executor,
)

# Intervention (Phase 3)
from grace.intervention import (
    ConfirmationFlow,
    DynamicThresholdAdjuster,
    FeedbackRecord,
    InterventionAction,
    InterventionHandler,
    InterventionRequest,
    InterventionResponse,
    create_confirmation_flow,
    create_intervention_handler,
    create_threshold_adjuster,
)

# Planner
from grace.planner import (
    Planner,
    create_planner,
)

# Replan (Phase 4)
from grace.replan import (
    ReplanContext,
    ReplanManager,
    ReplanOrchestrator,
    ReplanResult,
    ReplanStrategy,
    ReplanTrigger,
    create_replan_manager,
    create_replan_orchestrator,
)
from grace.schemas import (
    ActionType,
    ExecutionPlan,
    ExecutionResult,
    PlanStep,
    SearchResultItem,
    SearchResultPayload,
    StepResult,
    StepStatus,
    create_plan_id,
    validate_plan_dependencies,
)

# Tools
from grace.tools import (
    AskUserTool,
    BaseTool,
    RAGSearchTool,
    ReasoningTool,
    ToolRegistry,
    ToolResult,
    WebSearchTool,
    create_tool_registry,
)

__all__ = [
    # Version
    "__version__",

    # Schemas
    "ExecutionPlan",
    "PlanStep",
    "StepResult",
    "ExecutionResult",
    "ActionType",
    "StepStatus",
    "SearchResultPayload",
    "SearchResultItem",
    "create_plan_id",
    "validate_plan_dependencies",

    # Config
    "GraceConfig",
    "get_config",
    "reload_config",

    # Planner
    "Planner",
    "create_planner",

    # Tools
    "ToolResult",
    "BaseTool",
    "RAGSearchTool",
    "WebSearchTool",
    "ReasoningTool",
    "AskUserTool",
    "ToolRegistry",
    "create_tool_registry",

    # Executor
    "ExecutionState",
    "Executor",
    "create_executor",

    # Confidence (Phase 2)
    "ConfidenceFactors",
    "ConfidenceScore",
    "ActionDecision",
    "InterventionLevel",
    "ConfidenceCalculator",
    "LLMSelfEvaluator",
    "SourceAgreementCalculator",
    "QueryCoverageCalculator",
    "ConfidenceAggregator",
    "create_confidence_calculator",
    "create_llm_evaluator",
    "create_source_agreement_calculator",
    "create_query_coverage_calculator",
    "create_confidence_aggregator",

    # Intervention (Phase 3)
    "InterventionRequest",
    "InterventionResponse",
    "InterventionAction",
    "FeedbackRecord",
    "InterventionHandler",
    "DynamicThresholdAdjuster",
    "ConfirmationFlow",
    "create_intervention_handler",
    "create_threshold_adjuster",
    "create_confirmation_flow",

    # Replan (Phase 4)
    "ReplanTrigger",
    "ReplanStrategy",
    "ReplanContext",
    "ReplanResult",
    "ReplanManager",
    "ReplanOrchestrator",
    "create_replan_manager",
    "create_replan_orchestrator",
]
