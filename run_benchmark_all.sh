#!/usr/bin/env bash
# ==============================================================
# run_benchmark_all.sh - openai_grace_agent ベンチマーク実行
# ==============================================================
# 使用法:
#   chmod +x run_benchmark_all.sh
#   ./run_benchmark_all.sh
#
# 前提条件:
#   - Qdrant が起動済み (localhost:6333)
#   - cc_news_2per_openai コレクションが作成・embedding 済み
#   - .env または環境変数に OPENAI_API_KEY が設定済み
#
# データ準備フロー:
#   (1) down_load_non_qa_rag_data_from_huggingface.py  → OUTPUT/cc_news_2per.csv
#   (2) chunking/csv_text_to_chunks_text_csv.py        → output_chunked/cc_news_2per_chunks.csv
#   (3) qa_qdrant/make_qa_register_qdrant.py           → Qdrant: cc_news_2per_openai
#   (4) ./run_benchmark_all.sh                         → logs/benchmark_results.csv
# ==============================================================

set -euo pipefail

COLLECTION="cc_news_2per_openai"
PROJECT="openai_grace_agent"
MODEL="gpt-5-mini"

echo "================================================================"
echo "  GRACE Benchmark Runner"
echo "  Project   : ${PROJECT}"
echo "  Model     : ${MODEL}"
echo "  Collection: ${COLLECTION}"
echo "  Start     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

uv run python - << PYEOF
from grace.benchmark import BenchmarkRunner

runner = BenchmarkRunner(qdrant_collection="${COLLECTION}")
sessions = runner.run_query_set(runs_per_query=3)
count = len(sessions)
print(f"\n完了: {count} セッション -> logs/benchmark_results.csv")
PYEOF

echo "================================================================"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
