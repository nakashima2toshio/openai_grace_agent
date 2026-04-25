#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from unittest.mock import MagicMock, patch
from qa_generation.keyword_extraction import (
    BestKeywordSelector,
    SmartKeywordSelector,
    get_best_keywords,
    get_smart_keywords
)

# Mock data
MOCK_TEXT = "これはテスト用のテキストです。人工知能と機械学習について学習します。"
MOCK_KEYWORDS_SCORES = {
    "tfidf": [("人工知能", 0.9), ("機械学習", 0.8), ("テスト", 0.7), ("テキスト", 0.6), ("学習", 0.5)],
    "textrank": [("機械学習", 0.85), ("人工知能", 0.8), ("学習", 0.6), ("テスト", 0.5), ("テキスト", 0.4)],
    "simple": [("テスト", 0.5), ("テキスト", 0.5), ("人工知能", 0.5), ("機械学習", 0.5), ("学習", 0.5)]
}

@pytest.fixture
def mock_extractor():
    with patch('qa_generation.keyword_extraction.KeywordExtractor') as MockClass:
        instance = MockClass.return_value
        # extract_with_details returns a dict of method -> list of (keyword, score)
        instance.extract_with_details.return_value = MOCK_KEYWORDS_SCORES
        yield instance

class TestBestKeywordSelector:

    def test_evaluate_keywords(self, mock_extractor):
        selector = BestKeywordSelector(prefer_mecab=False)
        keywords = ["人工知能", "機械学習", "テスト"]
        
        metrics = selector.evaluate_keywords(keywords, MOCK_TEXT)
        
        # Check if all metrics are present
        expected_metrics = ['coverage', 'diversity', 'technicality', 'coherence', 'length_balance']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0.0 <= metrics[metric] <= 1.0
            
        # Basic check for coverage: all keywords are in MOCK_TEXT
        assert metrics['coverage'] == 1.0

    def test_calculate_total_score(self, mock_extractor):
        selector = BestKeywordSelector()
        metrics = {
            'coverage': 1.0,
            'diversity': 0.5,
            'technicality': 0.8,
            'coherence': 0.5,
            'length_balance': 0.8
        }
        score = selector.calculate_total_score(metrics)
        assert 0.0 <= score <= 1.0
        # Manual calculation check based on default weights
        # 0.25*1 + 0.15*0.5 + 0.25*0.8 + 0.20*0.5 + 0.15*0.8 = 0.25 + 0.075 + 0.2 + 0.1 + 0.12 = 0.745
        assert abs(score - 0.745) < 1e-5

    def test_extract_best(self, mock_extractor):
        selector = BestKeywordSelector()
        result = selector.extract_best(MOCK_TEXT, top_n=3)
        
        assert 'keywords' in result
        assert 'best_method' in result
        assert 'total_score' in result
        assert 'reason' in result
        assert len(result['keywords']) == 3
        # Should pick one of the mocked methods
        assert result['best_method'] in ["tfidf", "textrank", "simple"]

class TestSmartKeywordSelector:

    def test_calculate_auto_top_n(self, mock_extractor):
        selector = SmartKeywordSelector()
        
        # Short text
        short_text = "短いテキストです。" * 2
        n, reason = selector.calculate_auto_top_n(short_text)
        assert n == 3
        assert "超短文" in reason
        
        # Standard text
        standard_text = "普通の長さのテキストです。" * 20
        n, reason = selector.calculate_auto_top_n(standard_text)
        assert 5 <= n <= 12
        
        # Long text
        long_text = "長いテキストです。" * 200
        n, reason = selector.calculate_auto_top_n(long_text)
        assert n >= 15

    def test_extract_best_auto_mode(self, mock_extractor):
        selector = SmartKeywordSelector()
        
        # Test "auto" mode
        result = selector.extract_best_auto(MOCK_TEXT, mode="auto")
        assert result['mode'] == "auto"
        assert result['top_n'] == 3 # MOCK_TEXT is short ("超短文")
        
        # Test "summary" mode
        result = selector.extract_best_auto(MOCK_TEXT, mode="summary")
        assert result['mode'] == "summary"
        assert result['top_n'] == 5

    def test_extract_best_auto_coverage(self, mock_extractor):
        selector = SmartKeywordSelector()
        # Mock extract_best to return keywords that simulate coverage
        with patch.object(selector, 'extract_best') as mock_extract_best:
            # Use a side_effect function to handle variable number of calls
            def side_effect(text, top_n, **kwargs):
                # Generate fake keywords
                keywords = ["人工知能", "機械学習", "テスト", "テキスト", "学習"]
                # Repeat list if top_n > len(keywords)
                while len(keywords) < top_n:
                    keywords.extend([f"kw_{i}" for i in range(top_n - len(keywords))])
                
                return {
                    'keywords': keywords[:top_n],
                    'total_score': 0.5 + (top_n * 0.05),
                    'best_method': 'tfidf'
                }
            
            mock_extract_best.side_effect = side_effect
            
            result = selector.extract_best_auto(MOCK_TEXT, mode="coverage", min_keywords=1, max_keywords=5)
            
            assert result['mode'] == "coverage"
            assert result['top_n'] >= 1
            # Check if reason mentions coverage
            assert "カバレッジ" in result['reason'] or "coverage" in str(result)

class TestUtilityFunctions:
    
    def test_get_best_keywords(self, mock_extractor):
        keywords = get_best_keywords(MOCK_TEXT, top_n=3)
        assert isinstance(keywords, list)
        assert len(keywords) == 3

    def test_get_smart_keywords(self, mock_extractor):
        result = get_smart_keywords(MOCK_TEXT, mode="summary")
        assert isinstance(result, dict)
        assert result['mode'] == "summary"
        assert len(result['keywords']) == 5

if __name__ == "__main__":
    pytest.main()
