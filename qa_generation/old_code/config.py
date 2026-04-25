#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/config.py - Q/A生成の設定管理
"""

from typing import Dict, Any

# データセット別最適閾値設定
OPTIMAL_THRESHOLDS = {
    "cc_news": {
        "strict": 0.80,
        "standard": 0.70,
        "lenient": 0.60
    },
    "japanese_text": {
        "strict": 0.75,
        "standard": 0.65,
        "lenient": 0.55
    },
    "wikipedia_ja": {
        "strict": 0.85,   # 専門的な内容 → 高い類似度要求
        "standard": 0.75,
        "lenient": 0.65
    },
    "livedoor": {
        "strict": 0.78,   # ニュース記事は具体的で厳密性が中程度
        "standard": 0.68, # cc_newsとwikipedia_jaの中間
        "lenient": 0.58
    }
}

# データセット設定の拡張（ローカル固有設定）
LOCAL_DATASET_EXTENSIONS = {
    "cc_news": {
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "en",
    },
    "japanese_text": {
        "text_column": "Combined_Text",
        "title_column": None,
        "lang": "ja",
    },
    "wikipedia_ja": {
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
    },
    "livedoor": {
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
    }
}
