import pytest
from qa_generation.structure import merge_small_chunks

def test_merge_small_chunks_logic():
    # Mocking tiktoken behavior implicitly by relying on text length roughly, 
    # but tokens are recalculated inside. 
    # Assuming standard encoding, "Short" is very few tokens.
    
    chunks = [
        {"id": "1", "text": "Short text.", "doc_id": "d1"},
        {"id": "2", "text": "Another short text.", "doc_id": "d1"},
        {"id": "3", "text": "A very long text " * 50, "doc_id": "d1"},
    ]
    
    # We rely on the internal tokenizer of merge_small_chunks.
    # We expect 1 and 2 to merge if min_tokens is high enough.
    
    merged = merge_small_chunks(chunks, min_tokens=50, max_tokens=500)
    
    # Depending on token counts:
    # "Short text." is ~3 tokens.
    # "A very long text " * 50 is ~200 tokens.
    # So 1 and 2 should merge. 3 is large enough.
    
    assert len(merged) == 2
    assert merged[0]["merged"] is True
    assert "Short text." in merged[0]["text"]
    assert "Another short text." in merged[0]["text"]
    assert merged[1]["id"] == "3"

def test_merge_different_docs():
    chunks = [
        {"id": "1", "text": "Short.", "doc_id": "d1"},
        {"id": "2", "text": "Short.", "doc_id": "d2"},
    ]
    # Should not merge different docs
    merged = merge_small_chunks(chunks, min_tokens=50, max_tokens=500)
    assert len(merged) == 2
