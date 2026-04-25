
import sys
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 検証対象の修正ロジック
def get_collection_embedding_params_real_api_verification(client, collection_name):
    default_params = {"model": "gemini-embedding-001", "dims": 3072}
    try:
        # 実際の API を使用した Payload 取得
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
            
            print(f"  [Real API] Extracted Payload: provider={provider}, model={model}")
            
            if provider == "gemini" and model:
                return {"model": model, "dims": 3072}
    except Exception as e:
        print(f"  [Real API] Error during scroll: {e}")
    return default_params

def main():
    qdrant_url = "http://localhost:6333"
    client = QdrantClient(url=qdrant_url)
    col_name = "test_verification_collection"
    
    print(f"1. Creating temp collection: {col_name}")
    client.recreate_collection(
        collection_name=col_name,
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE)
    )
    
    print("2. Upserting point with metadata...")
    test_model_name = "gemini-embedding-REAL-API-VERIFIED"
    client.upsert(
        collection_name=col_name,
        points=[
            models.PointStruct(
                id=1,
                vector=[0.1] * 3072,
                payload={
                    "embedding_provider": "gemini",
                    "embedding_model": test_model_name,
                    "question": "test question"
                }
            )
        ]
    )
    
    print("3. Running logic verification...")
    result = get_collection_embedding_params_real_api_verification(client, col_name)
    
    print(f"4. Result Model: {result['model']}")
    
    # 検証
    if result['model'] == test_model_name:
        print("\n✅ SUCCESS: Logic verified with REAL Qdrant API.")
    else:
        print(f"\n❌ FAILURE: Expected {test_model_name}, but got {result['model']}")
        sys.exit(1)
    
    print(f"5. Cleaning up collection: {col_name}")
    client.delete_collection(col_name)

if __name__ == "__main__":
    main()
