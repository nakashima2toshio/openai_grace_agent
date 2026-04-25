"""
GRACE Test Fixtures
pytest共通フィクスチャ
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import os
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 環境変数設定（テスト用）
os.environ.setdefault("GOOGLE_API_KEY", "test-api-key")
os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("EMBEDDING_PROVIDER", "gemini")


@pytest.fixture
def mock_gemini_client():
    """Gemini APIモック"""
    with patch("google.genai.Client") as mock:
        client_instance = MagicMock()

        # models.generate_content のモック
        mock_response = MagicMock()
        mock_response.text = '{"original_query": "test", "complexity": 0.5, "estimated_steps": 2, "requires_confirmation": false, "steps": [], "success_criteria": "test"}'
        client_instance.models.generate_content.return_value = mock_response

        mock.return_value = client_instance
        yield mock


@pytest.fixture
def mock_qdrant_client():
    """Qdrantクライアントモック"""
    with patch("qdrant_client.QdrantClient") as mock:
        client_instance = MagicMock()

        # search結果のモック
        mock_result1 = MagicMock()
        mock_result1.score = 0.85
        mock_result1.payload = {"question": "Q1", "answer": "A1", "source": "doc1"}

        mock_result2 = MagicMock()
        mock_result2.score = 0.75
        mock_result2.payload = {"question": "Q2", "answer": "A2", "source": "doc2"}

        client_instance.search.return_value = [mock_result1, mock_result2]

        # collection_exists のモック
        client_instance.collection_exists.return_value = True

        mock.return_value = client_instance
        yield mock


@pytest.fixture
def mock_embedding_client():
    """Embeddingクライアントモック"""
    with patch("helper_embedding.create_embedding_client") as mock:
        embedding_instance = MagicMock()

        # 3072次元のダミーベクトル
        dummy_vector = [0.1] * 3072
        embedding_instance.embed_text.return_value = dummy_vector
        embedding_instance.embed_texts.return_value = [dummy_vector, dummy_vector]
        embedding_instance.dimensions = 3072

        mock.return_value = embedding_instance
        yield mock


@pytest.fixture
def sample_plan():
    """テスト用ExecutionPlan"""
    from grace.schemas import ExecutionPlan, PlanStep

    return ExecutionPlan(
        original_query="Pythonの非同期処理について教えて",
        complexity=0.5,
        estimated_steps=2,
        requires_confirmation=False,
        steps=[
            PlanStep(
                step_id=1,
                action="rag_search",
                description="RAG検索で関連情報を取得",
                query="Python 非同期処理 async await",
                expected_output="非同期処理の説明文"
            ),
            PlanStep(
                step_id=2,
                action="reasoning",
                description="取得情報を整理して回答作成",
                depends_on=[1],
                expected_output="ユーザー向けの回答"
            )
        ],
        success_criteria="非同期処理の基本概念を説明できている"
    )


@pytest.fixture
def sample_step_result():
    """テスト用StepResult"""
    from grace.schemas import StepResult

    return StepResult(
        step_id=1,
        status="success",
        output="Pythonの非同期処理は、asyncとawaitキーワードを使用して...",
        confidence=0.85,
        sources=["doc1", "doc2"]
    )


@pytest.fixture
def high_confidence_factors():
    """高信頼度シナリオ用のConfidenceFactors"""
    from grace.confidence import ConfidenceFactors

    return ConfidenceFactors(
        search_result_count=5,
        search_avg_score=0.85,
        search_score_variance=0.05,
        source_agreement=0.9,
        source_count=3,
        llm_self_confidence=0.85,
        tool_success_rate=1.0,
        query_coverage=0.9
    )


@pytest.fixture
def low_confidence_factors():
    """低信頼度シナリオ用のConfidenceFactors"""
    from grace.confidence import ConfidenceFactors

    return ConfidenceFactors(
        search_result_count=0,
        search_avg_score=0.0,
        search_score_variance=1.0,
        source_agreement=0.0,
        source_count=0,
        llm_self_confidence=0.3,
        tool_success_rate=0.5,
        query_coverage=0.2
    )


@pytest.fixture
def mock_config():
    """設定モック"""
    from grace.config import GraceConfig, LLMConfig, EmbeddingConfig

    return GraceConfig(
        llm=LLMConfig(
            provider="gemini",
            model="gemini-2.0-flash",
            temperature=0.7,
            max_tokens=4096
        ),
        embedding=EmbeddingConfig(
            provider="gemini",
            model="gemini-embedding-001",
            dimensions=3072
        )
    )