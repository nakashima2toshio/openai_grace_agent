#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from qa_generation.semantic import SemanticCoverage

# Mock data
SAMPLE_TEXT = """第一段落です。ここにはいくつかの文があります。
これは二番目の文です。

第二段落は少し長めにします。
セマンティック分割のテストを行っています。
MeCabや正規表現での分割を確認します。"""

@pytest.fixture
def mock_clients():
    with patch('qa_generation.semantic.create_llm_client') as mock_llm_factory, \
         patch('qa_generation.semantic.create_embedding_client') as mock_emb_factory, \
         patch('qa_generation.semantic.get_embedding_dimensions') as mock_dims, \
         patch('qa_generation.semantic.tiktoken.get_encoding') as mock_tiktoken:
        
        # Mock LLM Client
        mock_llm = MagicMock()
        mock_llm.count_tokens.return_value = 10
        mock_llm_factory.return_value = mock_llm
        
        # Mock Embedding Client
        mock_emb = MagicMock()
        mock_emb.embed_texts.return_value = [np.random.rand(3072).tolist() for _ in range(2)]
        mock_emb.embed_text.return_value = np.random.rand(3072).tolist()
        mock_emb_factory.return_value = mock_emb
        
        mock_dims.return_value = 3072
        
        # Mock Tokenizer
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1, 2, 3, 4, 5]
        mock_enc.decode.return_value = "decoded text"
        mock_tiktoken.return_value = mock_enc
        
        yield {
            'llm': mock_llm,
            'emb': mock_emb,
            'enc': mock_enc
        }

class TestSemanticCoverage:

    def test_init(self, mock_clients):
        sc = SemanticCoverage()
        assert sc.has_api_key is True
        assert sc.embedding_dims == 3072
        assert sc.embedding_model == "gemini-embedding-001"

    def test_split_into_paragraphs(self, mock_clients):
        sc = SemanticCoverage()
        paragraphs = sc._split_into_paragraphs(SAMPLE_TEXT)
        assert len(paragraphs) == 2
        assert paragraphs[0].startswith("第一段落")
        assert paragraphs[1].startswith("第二段落")

    def test_split_into_sentences_regex(self, mock_clients):
        sc = SemanticCoverage()
        # MeCabを無効化して正規表現での動作を確認
        sc.mecab_available = False
        
        # スペースあり
        text1 = "こんにちは。 元気ですか？ はい、元気です！"
        sentences1 = sc._split_into_sentences(text1)
        assert len(sentences1) == 3
        assert "こんにちは。" in sentences1
        assert "元気ですか？" in sentences1
        
        # スペースなし
        text2 = "こんにちは。元気ですか？はい、元気です！"
        sentences2 = sc._split_into_sentences(text2)
        assert len(sentences2) == 3
        assert "こんにちは。" in sentences2
        assert "元気ですか？" in sentences2

    @patch('qa_generation.semantic.SemanticCoverage._split_sentences_mecab')
    def test_split_into_sentences_mecab_trigger(self, mock_mecab_split, mock_clients):
        sc = SemanticCoverage()
        sc.mecab_available = True
        mock_mecab_split.return_value = ["文1", "文2"]
        
        text = "日本語の文章です。テストします。"
        sentences = sc._split_into_sentences(text)
        
        assert sentences == ["文1", "文2"]
        mock_mecab_split.assert_called_once()

    def test_force_split_sentence(self, mock_clients):
        sc = SemanticCoverage()
        sentence = "非常に長い文章のダミーです。"
        # Mock tokens to be 10 tokens, max_tokens=3
        mock_clients['enc'].encode.return_value = list(range(10))
        
        chunks = sc._force_split_sentence(sentence, max_tokens=3)
        # 10 / 3 = 4 chunks (3, 3, 3, 1)
        assert len(chunks) == 4
        assert chunks[0]['type'] == 'forced_split'

    def test_chunk_by_paragraphs(self, mock_clients):
        sc = SemanticCoverage()
        # Mock token counts: para1=100, para2=300 (exceeds max 200)
        mock_clients['llm'].count_tokens.side_effect = [100, 300, 50, 50, 50, 50, 50, 50]
        
        chunks = sc._chunk_by_paragraphs(SAMPLE_TEXT, max_tokens=200)
        
        assert len(chunks) >= 2
        # First paragraph should be kept as 'paragraph' type
        assert any(c['type'] == 'paragraph' for c in chunks)
        # Second paragraph should be split into 'sentence_group'
        assert any(c['type'] == 'sentence_group' for c in chunks)

    def test_adjust_chunks_for_topic_continuity(self, mock_clients):
        sc = SemanticCoverage()
        chunks = [
            {"text": "Chunk 1 is long enough.", "sentences": ["S1"], "end_sentence_idx": 0, "type": "paragraph"},
            {"text": "Short.", "sentences": ["S2"], "end_sentence_idx": 1, "type": "sentence_group"}
        ]
        # Mock token count: first=100, second=10, combined=110
        mock_clients['llm'].count_tokens.side_effect = [100, 10, 110]
        
        adjusted = sc._adjust_chunks_for_topic_continuity(chunks, min_tokens=50)
        
        assert len(adjusted) == 1
        assert adjusted[0]["type"] == "merged"
        assert "Short." in adjusted[0]["text"]

    def test_cosine_similarity(self, mock_clients):
        sc = SemanticCoverage()
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        assert sc.cosine_similarity(v1, v2) == pytest.approx(1.0)
        
        v3 = np.array([0.0, 1.0, 0.0])
        assert sc.cosine_similarity(v1, v3) == pytest.approx(0.0)

    def test_generate_embeddings(self, mock_clients):
        sc = SemanticCoverage()
        chunks = [{"text": "text1"}, {"text": "text2"}]
        
        embeddings = sc.generate_embeddings(chunks)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 3072
        # Check normalization
        norm = np.linalg.norm(embeddings[0])
        assert norm == pytest.approx(1.0)

    def test_create_semantic_chunks_flow(self, mock_clients):
        sc = SemanticCoverage()
        # Paragraph mode
        chunks = sc.create_semantic_chunks(SAMPLE_TEXT, max_tokens=200, verbose=False)
        
        assert len(chunks) > 0
        assert "id" in chunks[0]
        assert "sentences" in chunks[0]
        assert "type" in chunks[0]

if __name__ == "__main__":
    pytest.main()
