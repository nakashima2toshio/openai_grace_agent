import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from qdrant_client.http import models
from services.qdrant_service import (
    QdrantHealthChecker,
    QdrantDataFetcher,
    map_collection_to_csv,
    get_dynamic_collection_mapping,
    get_collection_embedding_params,
    load_csv_for_qdrant,
    build_inputs_for_embedding,
    embed_texts_for_qdrant,
    build_points_for_qdrant,
    embed_query_for_search,
    merge_collections
)

@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    return client

class TestQdrantService:

    def test_map_collection_to_csv(self):
        with patch("os.path.exists") as mock_exists:
            # Case 1: Exact match
            mock_exists.return_value = True
            assert map_collection_to_csv("test") == "test.csv"
            
            # Case 2: Stripped match (qa_test -> test.csv)
            # 1. qa_test.csv (False)
            # 2. test.csv (True)
            mock_exists.side_effect = [False, True]
            assert map_collection_to_csv("qa_test") == "test.csv"

    def test_get_dynamic_collection_mapping(self, mock_qdrant_client):
        # Mock collections
        mock_c1 = MagicMock()
        mock_c1.name = "col1"
        mock_qdrant_client.get_collections.return_value.collections = [mock_c1]
        
        # Mock payload scroll
        mock_point = MagicMock()
        mock_point.payload = {"source": "source.csv"}
        mock_qdrant_client.scroll.return_value = ([mock_point], None)
        
        mapping = get_dynamic_collection_mapping(mock_qdrant_client)
        assert mapping["col1"] == "source.csv"

    def test_get_collection_embedding_params(self, mock_qdrant_client):
        # Case 1: 3072 dim
        mock_info = MagicMock()
        mock_info.config.params.vectors.size = 3072
        mock_qdrant_client.get_collection.return_value = mock_info
        
        params = get_collection_embedding_params(mock_qdrant_client, "c")
        assert params["dims"] == 3072
        assert params["model"] == "gemini-embedding-001"

    def test_health_checker(self):
        checker = QdrantHealthChecker()
        with patch.object(checker, "check_port", return_value=True), \
             patch("services.qdrant_service.QdrantClient") as MockClient:
            
            MockClient.return_value.get_collections.return_value.collections = []
            success, msg, metrics = checker.check_qdrant()
            assert success is True
            assert metrics is not None

    def test_data_fetcher(self, mock_qdrant_client):
        fetcher = QdrantDataFetcher(mock_qdrant_client)
        
        # fetch_collections
        mock_c = MagicMock()
        mock_c.name = "c1"
        mock_qdrant_client.get_collections.return_value.collections = [mock_c]
        mock_qdrant_client.get_collection.return_value.points_count = 100
        
        df = fetcher.fetch_collections()
        assert len(df) == 1
        assert df.iloc[0]["Collection"] == "c1"
        
        # fetch_collection_points
        mock_point = MagicMock()
        mock_point.id = 1
        mock_point.payload = {"k": "v"}
        mock_qdrant_client.scroll.return_value = ([mock_point], None)
        
        df_points = fetcher.fetch_collection_points("c1")
        assert len(df_points) == 1
        assert df_points.iloc[0]["k"] == "v"

    def test_load_csv_for_qdrant(self):
        with patch("os.path.exists", return_value=True), \
             patch("pandas.read_csv") as mock_read:
            
            mock_read.return_value = pd.DataFrame({
                "Question": ["q"], "Answer": ["a"]
            })
            
            df = load_csv_for_qdrant("dummy.csv")
            assert "question" in df.columns
            assert "answer" in df.columns

    def test_build_inputs_for_embedding(self):
        df = pd.DataFrame({"question": ["q"], "answer": ["a"]})
        inputs = build_inputs_for_embedding(df, include_answer=True)
        assert inputs[0] == "q\na"

    @patch("services.qdrant_service.create_embedding_client")
    def test_embed_texts_for_qdrant(self, mock_create):
        mock_client = MagicMock()
        mock_client.embed_texts.return_value = [[0.1]*3072]
        mock_create.return_value = mock_client
        
        vecs = embed_texts_for_qdrant(["text"], model="gemini")
        assert len(vecs) == 1
        assert len(vecs[0]) == 3072

    def test_build_points_for_qdrant(self):
        df = pd.DataFrame({"question": ["q"], "answer": ["a"]})
        vectors = [[0.1]*3072]
        
        points = build_points_for_qdrant(df, vectors, "domain", "source.csv")
        assert len(points) == 1
        assert isinstance(points[0], models.PointStruct)
        assert points[0].payload["question"] == "q"

    @patch("services.qdrant_service.create_embedding_client")
    def test_embed_query_for_search(self, mock_create):
        mock_client = MagicMock()
        mock_client.embed_text.return_value = [0.1]*3072
        mock_create.return_value = mock_client
        
        vec = embed_query_for_search("q", dims=3072)
        assert len(vec) == 3072

    def test_merge_collections(self, mock_qdrant_client):
        # Mock scroll
        p1 = models.Record(id=1, vector=[0.1]*3072, payload={"a": 1})
        mock_qdrant_client.scroll.side_effect = [([p1], None), ([p1], None)] # Called for each source col
        mock_qdrant_client.get_collection.return_value.points_count = 1
        
        result = merge_collections(mock_qdrant_client, ["s1"], "target")
        
        assert result["success"] is True
        mock_qdrant_client.upsert.assert_called()
