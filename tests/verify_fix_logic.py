
import unittest
from unittest.mock import MagicMock

def get_collection_embedding_params_logic_verification(client, collection_name):
    """
    検証対象の修正ロジック
    """
    default_params = {"model": "gemini-embedding-001", "dims": 3072}
    try:
        # Payloadからのメタデータ取得を試みる
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_payload=["embedding_provider", "embedding_model"],
            with_vectors=False
        )
        if points and points[0].payload:
            payload = points[0].payload
            provider = payload.get("embedding_provider")
            model = payload.get("embedding_model")
            
            if provider == "gemini" and model:
                default_params["model"] = model
    except Exception:
        pass
    return default_params

class TestFixLogic(unittest.TestCase):
    def test_payload_priority(self):
        """Payloadにモデル情報がある場合に正しく採用されるか検証"""
        mock_client = MagicMock()
        mock_point = MagicMock()
        mock_point.payload = {
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-verified-model"
        }
        mock_client.scroll.return_value = ([mock_point], None)
        
        result = get_collection_embedding_params_logic_verification(mock_client, "test_col")
        self.assertEqual(result["model"], "gemini-embedding-verified-model")

if __name__ == "__main__":
    unittest.main()
