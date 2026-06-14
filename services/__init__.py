#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
services - ビジネスロジック分離モジュール
==========================================
agent_rag.pyから分離したビジネスロジック

モジュール構成:
- config_service.py: 設定管理（YAML、環境変数）
- cache_service.py: メモリキャッシュ（TTL対応）
- json_service.py: JSON処理（シリアライズ、ファイルI/O）
- token_service.py: トークン管理（カウント、コスト推定）
- dataset_service.py: データセット操作（ダウンロード、前処理）
- qdrant_service.py: Qdrant操作（CRUD、ヘルスチェック）
- file_service.py: ファイル操作（履歴読み込み、保存）
- qa_service.py: Q/A生成（OpenAI API、サブプロセス実行）
"""

from services.cache_service import (
    MemoryCache,
    cache,
    cache_result,
    get_global_cache,
    init_cache_from_config,
)
from services.config_service import (
    ConfigManager,
    config,
    get_config,
    logger,
    reload_config,
    set_config,
)
from services.dataset_service import (
    download_hf_dataset,
    download_livedoor_corpus,
    extract_text_content,
    load_livedoor_corpus,
    load_uploaded_file,
)
from services.file_service import (
    load_collection_qa_preview,
    load_preprocessed_history,
    load_qa_output_history,
    load_sample_questions_from_csv,
    load_source_qa_data,
    save_to_output,
)
from services.json_service import (
    compact_json,
    is_valid_json,
    load_json_file,
    load_json_file_or_default,
    merge_json_files,
    pretty_print_json,
    safe_json_dumps,
    safe_json_loads,
    safe_json_serializer,
    save_json_file,
)
from services.qa_service import (
    generate_qa_pairs,
    run_advanced_qa_generation,
    save_qa_pairs_to_file,
)
from services.qdrant_service import (
    COLLECTION_CSV_MAPPING,
    COLLECTION_EMBEDDINGS_SEARCH,
    QDRANT_CONFIG,
    QdrantDataFetcher,
    QdrantHealthChecker,
    build_inputs_for_embedding,
    build_points_for_qdrant,
    create_or_recreate_collection_for_qdrant,
    delete_all_collections,
    embed_query_for_search,
    embed_texts_for_qdrant,
    get_all_collections,
    get_collection_stats,
    load_csv_for_qdrant,
    upsert_points_to_qdrant,
)
from services.token_service import (
    DEFAULT_ENCODING,
    EMBEDDING_PRICING,
    LLM_PRICING,
    MODEL_ENCODINGS,
    MODEL_LIMITS,
    TokenManager,
    count_tokens,
    estimate_tokens_simple,
    get_embedding_pricing,
    get_llm_pricing,
    get_model_limits,
    truncate_text,
)

__all__ = [
    # dataset_service
    "download_livedoor_corpus",
    "load_livedoor_corpus",
    "download_hf_dataset",
    "extract_text_content",
    "load_uploaded_file",
    # qdrant_service
    "QdrantHealthChecker",
    "QdrantDataFetcher",
    "get_collection_stats",
    "get_all_collections",
    "delete_all_collections",
    "load_csv_for_qdrant",
    "build_inputs_for_embedding",
    "embed_texts_for_qdrant",
    "create_or_recreate_collection_for_qdrant",
    "build_points_for_qdrant",
    "upsert_points_to_qdrant",
    "embed_query_for_search",
    "QDRANT_CONFIG",
    "COLLECTION_EMBEDDINGS_SEARCH",
    "COLLECTION_CSV_MAPPING",
    # file_service
    "load_qa_output_history",
    "load_preprocessed_history",
    "save_to_output",
    "load_sample_questions_from_csv",
    "load_source_qa_data",
    "load_collection_qa_preview",
    # qa_service
    "run_advanced_qa_generation",
    "generate_qa_pairs",
    "save_qa_pairs_to_file",
    # token_service
    "TokenManager",
    "count_tokens",
    "estimate_tokens_simple",
    "truncate_text",
    "get_llm_pricing",
    "get_embedding_pricing",
    "get_model_limits",
    "DEFAULT_ENCODING",
    "MODEL_ENCODINGS",
    "LLM_PRICING",
    "EMBEDDING_PRICING",
    "MODEL_LIMITS",
    # config_service
    "ConfigManager",
    "config",
    "logger",
    "get_config",
    "set_config",
    "reload_config",
    # cache_service
    "MemoryCache",
    "cache_result",
    "cache",
    "get_global_cache",
    "init_cache_from_config",
    # json_service
    "safe_json_serializer",
    "safe_json_dumps",
    "safe_json_loads",
    "load_json_file",
    "save_json_file",
    "load_json_file_or_default",
    "merge_json_files",
    "is_valid_json",
    "pretty_print_json",
    "compact_json",
]
