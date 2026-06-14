"""
チャンキング（文書単位処理）のテスト

P1-P2 改修の回帰テスト:
- 文境界を保ったブロック分割
- 文書境界をまたがない結合
- LLM失敗時のテキスト保全（カバレッジ保証）
- ルールベース連続性判定
- manifest 出力
[Usage]: pytest -vs tests/chunking/test_document_chunking.py
"""

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("OPENAI_API_KEY", "test-api-key")

import chunking.csv_text_to_chunks_text_csv as chunking_mod
from chunking.checkpoint_manager import CheckpointManager
from chunking.csv_text_to_chunks_text_csv import (
    _as_chunk_dicts,
    _count_tokens,
    _rule_based_continuity,
    _split_document_into_blocks,
    load_documents_from_csv,
)


class FakeAPIClient:
    """LLM呼び出しを模倣するクライアント（入力をそのまま段落/チャンクとして返す）"""

    def __init__(self, **kwargs):
        self.step3_calls = 0

    async def generate_content(self, model, contents, response_schema, task_id=None, system=None):
        body = contents.split("\n", 1)[1] if "\n" in contents else contents
        if task_id.startswith("step3"):
            self.step3_calls += 1
            return json.dumps({"is_connected": True})
        if "FAIL_MARKER" in contents:
            return None  # フォールバック経路の検証用
        return json.dumps(
            {"paragraphs": [{"id": 1, "sentences": [{"text": body}]}]},
            ensure_ascii=False,
        )

    def get_stats(self):
        return {"total_requests": 0, "failed_requests": 0,
                "truncated_responses": 0, "usage": {}}


class TestBlockSplitting:
    """文境界ブロック分割のテスト"""

    def test_splits_at_sentence_boundaries(self):
        text = "これは一文目です。これは二文目です。これは三文目です。"
        blocks = _split_document_into_blocks(text, block_size=25)

        assert len(blocks) >= 2
        # 全ブロックが文末（句点）で終わること = 文の途中で切られていない
        for block in blocks:
            assert block.rstrip().endswith("。")

    def test_oversized_sentence_kept_intact(self):
        """block_size を超える1文は切断せず単独ブロックにする"""
        long_sentence = "あ" * 100 + "。"
        blocks = _split_document_into_blocks(long_sentence, block_size=50)
        assert len(blocks) == 1
        assert blocks[0].rstrip().endswith("。")

    def test_all_text_preserved(self):
        text = "短文一。短文二。短文三。短文四。"
        blocks = _split_document_into_blocks(text, block_size=10)
        joined = "".join(b.replace("\n", "") for b in blocks)
        assert joined == text


class TestRuleBasedContinuity:
    """ルールベース連続性判定のテスト"""

    LONG_NEXT = (
        "全く別の話題であるデータベースの選定基準について述べます。"
        "性能・運用性・コストの三点を比較することが重要であり、"
        "それぞれの観点で定量的な評価基準を設けることが推奨されます。"
    )

    def test_demonstrative_prefix_connects(self):
        assert _rule_based_continuity("前の文。", "この方式は強力です。" + self.LONG_NEXT)

    def test_short_next_chunk_connects(self):
        assert _rule_based_continuity("前の文。", "補足です。")

    def test_independent_long_chunk_separates(self):
        assert not _rule_based_continuity("前の文。", self.LONG_NEXT)


class TestTokenCounting:
    def test_japanese_not_underestimated(self):
        """フォールバック概算が日本語を1/4に過小評価しないこと"""
        japanese = "あ" * 100
        # tiktoken が使えない環境でも 100文字 ≈ 100トークン相当になる
        with patch.object(chunking_mod, "_TOKENIZER", None), \
             patch.object(chunking_mod, "_TOKENIZER_FAILED", True):
            assert _count_tokens(japanese) >= 90


class TestChunkDicts:
    def test_mixed_input_normalized(self):
        out = _as_chunk_dicts(["plain", {"text": "t", "doc_id": 3}])
        assert out[0] == {"text": "plain", "doc_id": None}
        assert out[1] == {"text": "t", "doc_id": 3}


class TestLoadDocuments:
    def test_one_row_one_document(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "in.csv")
            pd.DataFrame({"text": ["記事A。", "", "記事B。"]}).to_csv(csv_path, index=False)

            docs = load_documents_from_csv(csv_path)

            assert len(docs) == 2  # 空行は除外
            assert docs[0]["text"] == "記事A。"
            assert docs[0]["doc_id"] == 0
            assert docs[1]["doc_id"] == 2  # 元の行番号を保持


class TestEndToEnd:
    """3段階パイプラインのモックE2Eテスト"""

    DOCS = [
        {"doc_id": 0, "text": "文書Aの一文目です。文書Aの二文目です。"},
        {"doc_id": 1, "text": "FAIL_MARKERを含む文書Bです。"},
        {"doc_id": 2, "text": "文書Cの内容です。"},
    ]

    def _run(self, tmpdir, continuity_mode, output_file=None):
        client = FakeAPIClient()
        with patch.object(chunking_mod, "AsyncAPIClient", lambda **kw: client):
            cm = CheckpointManager(checkpoint_dir=os.path.join(tmpdir, f"ckpt_{continuity_mode}"))
            chunks = asyncio.run(chunking_mod.chunks_all_async(
                documents=self.DOCS,
                checkpoint_manager=cm,
                continuity_mode=continuity_mode,
                output_file=output_file,
            ))
        return chunks, client

    def test_document_boundary_preserved(self):
        """LLMが常に「連続」と判定しても文書をまたいで結合されない"""
        with tempfile.TemporaryDirectory() as td:
            chunks, _ = self._run(td, "llm")
            assert len(chunks) == 3

    def test_failed_block_text_preserved(self):
        """LLM失敗ブロックのテキストが捨てられない（カバレッジ保証）"""
        with tempfile.TemporaryDirectory() as td:
            chunks, _ = self._run(td, "off")
            assert any("FAIL_MARKER" in c for c in chunks)

    def test_rule_mode_no_step3_llm_calls(self):
        with tempfile.TemporaryDirectory() as td:
            _, client = self._run(td, "rule")
            assert client.step3_calls == 0

    def test_manifest_written_with_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "test_chunks.csv")
            self._run(td, "off", output_file=out)

            df = pd.read_csv(out)
            assert "doc_id" in df.columns
            assert sorted(df["doc_id"].tolist()) == [0, 1, 2]

            manifest_path = out.replace(".csv", ".manifest.json")
            assert os.path.exists(manifest_path)
            manifest = json.load(open(manifest_path, encoding="utf-8"))
            assert manifest["schema_version"] == "chunks:v2"
            assert manifest["coverage"]["ratio"] > 0.9
            assert manifest["counts"]["documents"] == 3
            assert manifest["fallbacks"]["step1_fallbacks"] == 1


class TestMaxChunkTokenEnforcement:
    """P2: チャンク最大トークン数の強制分割（Embedding入力上限との連携）"""

    def test_oversized_chunk_is_split_at_sentence_boundary(self):
        from chunking.csv_text_to_chunks_text_csv import _enforce_max_chunk_tokens

        # 日本語は概算で1文字≈1トークン。各文約40トークン×5文 ≈ 200トークン
        sentence = "これはテスト用の文章でありおおよそ四十文字程度の長さを持つ一文です。"
        big_text = "".join(sentence for _ in range(5))
        chunks = [{"doc_id": 7, "text": big_text}]

        result = _enforce_max_chunk_tokens(chunks, max_tokens=100)

        assert len(result) > 1  # 分割された
        # doc_id が引き継がれる
        assert all(c["doc_id"] == 7 for c in result)
        # テキストは保全される（文の欠落なし）
        joined = "".join(c["text"].replace(" ", "") for c in result)
        assert joined == big_text

    def test_within_limit_chunk_is_untouched(self):
        from chunking.csv_text_to_chunks_text_csv import _enforce_max_chunk_tokens

        chunks = [{"doc_id": 0, "text": "短いチャンク。"}]
        result = _enforce_max_chunk_tokens(chunks, max_tokens=512)
        assert result == chunks

    def test_single_oversized_sentence_kept_whole(self):
        """1文で上限超の場合は文の途中で切らず保持（警告のみ）"""
        from chunking.csv_text_to_chunks_text_csv import _enforce_max_chunk_tokens

        one_long_sentence = "あ" * 300 + "。"
        chunks = [{"doc_id": 1, "text": one_long_sentence}]
        result = _enforce_max_chunk_tokens(chunks, max_tokens=100)
        assert len(result) == 1
        assert result[0]["text"] == one_long_sentence

    def test_split_pieces_respect_token_limit(self):
        from chunking.csv_text_to_chunks_text_csv import (
            _count_tokens,
            _enforce_max_chunk_tokens,
        )

        sentence = "これは四十文字程度のテスト文章でありそれなりの長さを持っています。"
        big_text = "".join(sentence for _ in range(10))
        result = _enforce_max_chunk_tokens([{"doc_id": 0, "text": big_text}], max_tokens=120)

        for c in result:
            assert _count_tokens(c["text"]) <= 120
