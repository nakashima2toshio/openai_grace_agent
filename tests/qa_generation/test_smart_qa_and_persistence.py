"""
Q/A生成（SmartQAGenerator）と逐次永続化のテスト

[PORT] anthropic_grace_agent → openai_grace_agent

⚠️ プロバイダー差異による調整:
- openai の SmartQAGenerator は #52 で「分析+生成 構造化出力1回」方式
  （generate_structured / SmartQAResult / SmartQAPair）に統一済み。
  構造化出力が失敗した場合は success=False を返し、呼び出し側でスキップする
  （2段階方式・raw_decode フォールバックは廃止）。
- env キー: ANTHROPIC_API_KEY/GOOGLE_API_KEY → OPENAI_API_KEY
- モデル: claude-sonnet-4-6 → gpt-5-mini
- anthropic の TestManifestValidation は openai の pipeline.load_data() に manifest 検証
  ロジックが存在しないため移植せず（スキップ）。

[Usage]: pytest -vs tests/qa_generation/test_smart_qa_and_persistence.py
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("OPENAI_API_KEY", "test-api-key")


class TestSmartQAGenerator:
    """統合1回呼び出し（#52）のテスト（openai: generate_structured ベース）"""

    def _make_generator(self, mock_create):
        from qa_generation.smart_qa_generator import SmartQAGenerator
        mock_llm = MagicMock()
        mock_create.return_value = mock_llm
        return SmartQAGenerator(model="gpt-5-mini"), mock_llm

    @patch("qa_generation.smart_qa_generator.create_llm_client")
    def test_single_structured_call(self, mock_create):
        """分析+生成がLLM 1回の構造化出力で完了すること（旧実装は2回）"""
        from qa_generation.smart_qa_generator import SmartQAPair, SmartQAResult

        gen, mock_llm = self._make_generator(mock_create)
        mock_llm.generate_structured.return_value = SmartQAResult(
            qa_count=2,
            key_topics=["topic1"],
            importance_score=0.8,
            complexity="medium",
            reasoning="test",
            qa_pairs=[
                SmartQAPair(question="Q1?", answer="A1", topic="T1"),
                SmartQAPair(question="Q2?", answer="A2", topic="T2"),
            ],
        )

        result = gen.process_chunk("テストチャンク")

        assert result["success"]
        assert len(result["qa_pairs"]) == 2
        assert result["analysis"]["qa_count"] == 2
        # 構造化出力1回のみ。旧2段階の generate_content は呼ばれない
        assert mock_llm.generate_structured.call_count == 1
        assert mock_llm.generate_content.call_count == 0

    @patch("qa_generation.smart_qa_generator.create_llm_client")
    def test_qa_count_zero(self, mock_create):
        """qa_count=0（Q/A不要）でも success=True で空リストが返ること"""
        from qa_generation.smart_qa_generator import SmartQAResult

        gen, mock_llm = self._make_generator(mock_create)
        mock_llm.generate_structured.return_value = SmartQAResult(
            qa_count=0, reasoning="メタ情報のみ", qa_pairs=[]
        )

        result = gen.process_chunk("ページ番号: 42")

        assert result["success"]
        assert result["qa_pairs"] == []
        assert mock_llm.generate_structured.call_count == 1

    @patch("qa_generation.smart_qa_generator.create_llm_client")
    def test_structured_failure_returns_unsuccessful(self, mock_create):
        """構造化出力失敗時は success=False を返し、呼び出し側でスキップ可能なこと

        #52 で2段階方式（generate_content + raw_decode）は廃止したため、
        フォールバックせず success=False を返す。
        """
        gen, mock_llm = self._make_generator(mock_create)
        mock_llm.generate_structured.side_effect = Exception("structured output failed")

        result = gen.process_chunk("テストチャンク")

        assert result["success"] is False
        assert result["qa_pairs"] == []
        assert result["analysis"] == {}

    @patch("qa_generation.smart_qa_generator.create_llm_client")
    def test_topic_defaulted_when_empty(self, mock_create):
        """topic が空文字のとき 'その他' で補完されること"""
        from qa_generation.smart_qa_generator import SmartQAPair, SmartQAResult

        gen, mock_llm = self._make_generator(mock_create)
        mock_llm.generate_structured.return_value = SmartQAResult(
            qa_count=1,
            reasoning="r",
            qa_pairs=[SmartQAPair(question="Q?", answer="A", topic="")],
        )

        result = gen.process_chunk("テストチャンク")

        assert result["success"]
        assert result["qa_pairs"][0]["topic"] == "その他"


class TestPipelinePersistence:
    """逐次永続化・再開のテスト"""

    def _make_pipeline(self, tmpdir):
        from qa_generation.pipeline import QAPipeline

        csv_path = os.path.join(tmpdir, "in_chunks.csv")
        pd.DataFrame({
            "chunk_id": ["c1", "c2", "c3"],
            "text": ["text1", "text2", "text3"],
        }).to_csv(csv_path, index=False)

        with patch("qa_generation.pipeline.SmartQAGenerator") as MockGen:
            inst = MockGen.return_value
            inst.process_chunk.return_value = {
                "success": True,
                "qa_pairs": [{"question": "Q", "answer": "A", "topic": "T"}],
                "analysis": {},
            }
            pipeline = QAPipeline(input_file=csv_path, output_dir=tmpdir)
            pipeline.smart_generator = inst
        return pipeline, inst

    def test_progress_written_per_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            pipeline, _ = self._make_pipeline(td)
            chunks = pipeline._load_chunks_from_csv(pipeline.load_data())

            pairs = pipeline.generate_qa(chunks, use_celery=False)

            assert len(pairs) == 3
            progress_path = pipeline._progress_path()
            assert progress_path.exists()
            lines = progress_path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 3
            assert all("chunk_id" in json.loads(line) for line in lines)

    def test_resume_skips_processed_chunks(self):
        with tempfile.TemporaryDirectory() as td:
            pipeline, inst = self._make_pipeline(td)
            chunks = pipeline._load_chunks_from_csv(pipeline.load_data())

            # 2チャンク処理済み（うち1つは qa_count=0）の状態を作る
            progress_path = pipeline._progress_path()
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"chunk_id": "c1",
                                    "qa_pairs": [{"question": "Q1", "answer": "A1"}]}) + "\n")
                f.write(json.dumps({"chunk_id": "c2", "qa_pairs": []}) + "\n")

            pairs = pipeline.generate_qa(chunks, use_celery=False)

            # 未処理の c3 のみ処理される
            assert inst.process_chunk.call_count == 1
            # 復元1件（c1）+ 新規1件（c3）。c2 は qa_count=0 として再処理されない
            assert len(pairs) == 2

    def test_corrupted_progress_line_skipped(self):
        """壊れた行（途中クラッシュ）はスキップされ、そのチャンクは再処理される"""
        with tempfile.TemporaryDirectory() as td:
            pipeline, inst = self._make_pipeline(td)
            chunks = pipeline._load_chunks_from_csv(pipeline.load_data())

            progress_path = pipeline._progress_path()
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"chunk_id": "c1", "qa_pairs": []}) + "\n")
                f.write('{"chunk_id": "c2", "qa_pa')  # 壊れた行

            pipeline.generate_qa(chunks, use_celery=False)

            # c1 はスキップ、c2（壊れた行）と c3 は処理される
            assert inst.process_chunk.call_count == 2

    def test_clear_progress(self):
        with tempfile.TemporaryDirectory() as td:
            pipeline, _ = self._make_pipeline(td)
            pipeline._append_progress("c1", [])
            assert pipeline._progress_path().exists()
            pipeline._clear_progress()
            assert not pipeline._progress_path().exists()
