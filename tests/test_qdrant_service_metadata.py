
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 実際の関数をインポート
from services.qdrant_service import get_collection_embedding_params

class TestQdrantServiceMetadata(unittest.TestCase):

    @patch('services.qdrant_service.QdrantClient')
    def test_get_collection_embedding_params_with_payload(self, MockClient):
        """qdrant_service.py がPayloadからプロバイダー情報を読み取れるか検証"""
        
        # モックの設定
        mock_client = MockClient()
        
        # scroll の戻り値をモック (Geminiのケース)
        mock_point_gemini = MagicMock()
        mock_point_gemini.payload = {
            "embedding_provider": "gemini",
            "embedding_model": "gemini-embedding-001"
        }
        # scroll は (points, next_offset) を返す
        mock_client.scroll.side_effect = [
            ([mock_point_gemini], None),  # 1回目の呼び出し (Geminiテスト用)
            ([MagicMock(payload={"embedding_provider": "openai", "embedding_model": "text-embedding-3-small"})], None) # 2回目の呼び出し (OpenAIテスト用)
        ]
        
        # 1. Geminiのケース検証
        params_gemini = get_collection_embedding_params(mock_client, "test_collection_gemini")
        self.assertEqual(params_gemini['model'], 'gemini-embedding-001')
        
        # 2. OpenAIのケース検証
        params_openai = get_collection_embedding_params(mock_client, "test_collection_openai")
        self.assertEqual(params_openai['model'], 'text-embedding-3-small')

if __name__ == '__main__':
    unittest.main()
