#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/generation.py - Q/Aペア生成モジュール（スマート生成統合版）

改修内容:
- SmartQAGeneratorを統合
- use_smart_generationフラグで従来方式と切り替え可能
- 後方互換性を維持
"""

import logging
import time
import json
from typing import List, Dict, Optional
from helper.helper_llm import LLMClient, create_llm_client
from models import QAPairsResponse
from config import DATASET_CONFIGS
from qa_generation.structure import merge_small_chunks

# ✨ スマート生成のインポート
from qa_generation.smart_qa_generator import SmartQAGenerator

logger = logging.getLogger(__name__)


class QAGenerator:
    """Q/Aペア生成クラス（スマート生成統合版）"""

    def __init__(
            self,
            client: Optional[LLMClient] = None,
            model: str = "gemini-2.0-flash",
            use_smart_generation: bool = False  # ✨ 新規追加
    ):
        """
        Args:
            client: LLMクライアント
            model: 使用するモデル名
            use_smart_generation: スマート生成を使用するか（デフォルト: False = 従来方式）
        """
        self.client = client if client else create_llm_client(provider="gemini")
        self.model = model
        self.use_smart_generation = use_smart_generation

        # ✨ スマートジェネレーターの初期化
        if self.use_smart_generation:
            logger.info("🆕 スマート生成モードを有効化")
            self.smart_generator = SmartQAGenerator(model=model)
        else:
            logger.info("🔧 従来の固定Q/A数生成モードを使用")
            self.smart_generator = None

    def determine_qa_count(self, chunk: Dict, config: Dict) -> int:
        """
        チャンクに最適なQ/A数を決定

        従来方式（use_smart_generation=False）:
            トークン数ベースの固定計算

        スマート方式（use_smart_generation=True）:
            SmartQAGeneratorによる動的決定
        """
        if self.use_smart_generation:
            # ✨ スマート生成: LLMによる動的決定
            try:
                analysis = self.smart_generator.analyze_chunk(chunk['text'])
                qa_count = analysis['qa_count']

                # メタデータを保存（後で使用）
                chunk['_smart_analysis'] = analysis

                logger.debug(
                    f"スマート分析: chunk_id={chunk.get('id', 'N/A')}, "
                    f"qa_count={qa_count}, "
                    f"importance={analysis['importance_score']:.2f}"
                )

                return qa_count

            except Exception as e:
                logger.warning(f"スマート分析エラー、従来方式にフォールバック: {e}")
                # フォールバック: 従来方式
                return self._legacy_determine_qa_count(chunk, config)

        else:
            # 🔧 従来方式: トークン数ベース
            return self._legacy_determine_qa_count(chunk, config)

    def _legacy_determine_qa_count(self, chunk: Dict, config: Dict) -> int:
        """従来方式のQ/A数決定（後方互換性）"""
        base_count = config["qa_per_chunk"]
        token_count = self.client.count_tokens(chunk['text'], model=self.model)
        chunk_position = chunk.get('chunk_idx', 0)

        # トークン数に基づく基本Q&A数決定
        if token_count < 50:
            qa_count = 2
        elif token_count < 100:
            qa_count = 3
        elif token_count < 200:
            qa_count = base_count + 1
        elif token_count < 300:
            qa_count = base_count + 2
        else:
            qa_count = base_count + 3

        # 文書後半の位置バイアス補正
        if isinstance(chunk_position, int) and chunk_position >= 5:
            qa_count += 1

        return min(qa_count, 8)

    def generate_for_chunk(self, chunk: Dict, config: Dict) -> List[Dict]:
        """
        単一チャンクからQ/Aペアを生成

        スマート生成モードの場合、SmartQAGeneratorを使用
        従来モードの場合、既存のロジックを使用
        """
        if self.use_smart_generation:
            # ✨ スマート生成
            return self._generate_smart(chunk, config)
        else:
            # 🔧 従来方式
            return self._generate_legacy(chunk, config)

    def _generate_smart(self, chunk: Dict, config: Dict) -> List[Dict]:
        """
        🆕 スマート生成メソッド

        SmartQAGeneratorを使用してQ/Aペアを生成
        """
        try:
            # 既に分析済みの場合は再利用
            if '_smart_analysis' in chunk:
                analysis = chunk['_smart_analysis']
            else:
                analysis = self.smart_generator.analyze_chunk(chunk['text'])

            # Q/A生成
            qa_pairs = self.smart_generator.generate_qa_pairs(
                chunk['text'],
                analysis=analysis
            )

            # Q/Aペアが0個の場合
            if not qa_pairs:
                logger.info(
                    f"スマート生成: chunk_id={chunk.get('id', 'N/A')} - "
                    f"Q/A生成スキップ（qa_count=0）"
                )
                return []

            # メタデータの追加
            enriched_qa_pairs = []
            for qa in qa_pairs:
                enriched_qa = {
                    "question"         : qa['question'],
                    "answer"           : qa['answer'],
                    "question_type"    : "fact",  # デフォルト
                    "topic"            : qa.get('topic', 'その他'),  # ✨ 新規フィールド
                    "source_chunk_id"  : chunk.get('id', ''),
                    "doc_id"           : chunk.get('doc_id', ''),
                    "dataset_type"     : chunk.get('dataset_type', ''),
                    "chunk_idx"        : chunk.get('chunk_idx', 0),
                    # ✨ スマート生成メタデータ
                    "generation_method": "smart",
                    "importance_score" : analysis['importance_score'],
                    "complexity"       : analysis['complexity']
                }
                enriched_qa_pairs.append(enriched_qa)

            logger.info(
                f"スマート生成完了: chunk_id={chunk.get('id', 'N/A')}, "
                f"qa_count={len(enriched_qa_pairs)}, "
                f"importance={analysis['importance_score']:.2f}"
            )

            return enriched_qa_pairs

        except Exception as e:
            logger.error(
                f"スマート生成エラー（chunk_id={chunk.get('id', 'N/A')}）: {e}"
            )
            logger.info("従来方式にフォールバック...")
            return self._generate_legacy(chunk, config)

    def _generate_legacy(self, chunk: Dict, config: Dict) -> List[Dict]:
        """
        🔧 従来方式の生成メソッド（後方互換性）

        既存のロジックをそのまま保持
        """
        num_pairs = self.determine_qa_count(chunk, config)
        lang = config["lang"]

        # 言語別のプロンプト設定
        if lang == "ja":
            system_prompt = """あなたは教育コンテンツ作成の専門家です。
与えられた日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成"""

            question_types_desc = """
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）"""
        else:
            system_prompt = """You are an expert in educational content creation.
Generate high-quality Q&A pairs from the given English text.

Generation rules:
1. Questions should be clear and specific
2. Answers should be concise and accurate (1-2 sentences)
3. Stay faithful to the text content
4. Create questions from diverse perspectives"""

            question_types_desc = """
- fact: Factual questions (What is...?) 
- reason: Explanatory questions (Why...?) 
- comparison: Comparative questions (What's the difference...?) 
- application: Application questions (How is... used?)"""

        # チャンクが長すぎる場合は短縮
        max_chunk_length = 2000
        chunk_text = chunk['text']
        if len(chunk_text) > max_chunk_length:
            chunk_text = chunk_text[:max_chunk_length] + "..."
            logger.debug(f"チャンクを{max_chunk_length}文字に短縮")

        # 言語に応じたユーザープロンプト
        if lang == "ja":
            user_prompt = f"""以下のテキストから{num_pairs}個のQ&Aペアを生成してください。

質問タイプ:
{question_types_desc}

テキスト:
{chunk_text}

JSON形式で出力:
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""
        else:
            user_prompt = f"""Generate {num_pairs} Q&A pairs from the following text.

Question types:
{question_types_desc}

Text:
{chunk_text}

Output in JSON format:
{{
  "qa_pairs": [
    {{
      "question": "question text",
      "answer": "answer text",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""

        try:
            combined_input = f"{system_prompt}\n\n{user_prompt}"
            logger.debug(f"Gemini構造化出力試行中... (chunk: {chunk.get('id')})")
            parsed_data = self.client.generate_structured(
                prompt=combined_input,
                response_schema=QAPairsResponse,
                model=self.model,
                max_output_tokens=1000
            )

            qa_pairs = []
            for qa_data in parsed_data.qa_pairs:
                qa = {
                    "question"         : qa_data.question,
                    "answer"           : qa_data.answer,
                    "question_type"    : qa_data.question_type,
                    "source_chunk_id"  : chunk.get('id', ''),
                    "doc_id"           : chunk.get('doc_id', ''),
                    "dataset_type"     : chunk.get('dataset_type', ''),
                    "chunk_idx"        : chunk.get('chunk_idx', 0),
                    # 🔧 従来方式メタデータ
                    "generation_method": "legacy"
                }
                qa_pairs.append(qa)

            if len(qa_pairs) == 0:
                logger.error(
                    f"Gemini APIから解析可能なレスポンスが返されませんでした (chunk {chunk.get('id', 'unknown')})")
                raise ValueError("No parseable response from Gemini API")

            return qa_pairs

        except Exception as e:
            logger.warning(f"構造化出力失敗、テキスト生成にフォールバック (chunk: {chunk.get('id')}): {str(e)[:100]}")

            try:
                # フォールバック: テキスト生成してJSON解析
                combined_input = f"{system_prompt}\n\n{user_prompt}"
                logger.debug(f"Geminiテキスト生成試行中... (chunk: {chunk.get('id')})")
                response_text = self.client.generate_content(
                    prompt=combined_input,
                    model=self.model
                )

                # JSONを抽出して解析
                import re
                json_match = re.search(r'\{.*}', response_text, re.DOTALL)
                if json_match:
                    parsed_data = json.loads(json_match.group())
                    qa_pairs = []
                    for qa_data in parsed_data.get('qa_pairs', []):
                        qa = {
                            "question"         : qa_data.get('question', ''),
                            "answer"           : qa_data.get('answer', ''),
                            "question_type"    : qa_data.get('question_type', 'fact'),
                            "source_chunk_id"  : chunk.get('id', ''),
                            "doc_id"           : chunk.get('doc_id', ''),
                            "dataset_type"     : chunk.get('dataset_type', ''),
                            "chunk_idx"        : chunk.get('chunk_idx', 0),
                            "generation_method": "legacy"
                        }
                        qa_pairs.append(qa)
                    return qa_pairs
                else:
                    raise ValueError("JSON not found in response")
            except Exception as fallback_error:
                logger.error(f"フォールバックも失敗 (chunk {chunk.get('id', 'unknown')}): {fallback_error}")
                raise fallback_error

    def generate_for_batch(self, chunks: List[Dict], config: Dict) -> List[Dict]:
        """
        複数チャンクから一度にQ/Aペアを生成（バッチ処理）

        注意: スマート生成モードではバッチ処理は非推奨
              各チャンクで異なるQ/A数が生成されるため、
              個別処理にフォールバックします
        """
        if self.use_smart_generation:
            # ✨ スマート生成: バッチ処理は非推奨、個別処理にフォールバック
            logger.info("スマート生成モード: バッチ処理を個別処理にフォールバック")
            all_qa_pairs = []
            for chunk in chunks:
                try:
                    qa_pairs = self.generate_for_chunk(chunk, config)
                    if qa_pairs:
                        all_qa_pairs.extend(qa_pairs)
                except Exception as e:
                    logger.error(f"チャンク処理エラー（chunk_id={chunk.get('id', 'N/A')}）: {e}")
            return all_qa_pairs

        # 🔧 従来方式: バッチ処理
        if len(chunks) == 0:
            return []
        if len(chunks) == 1:
            return self.generate_for_chunk(chunks[0], config)

        lang = config["lang"]
        all_qa_pairs = []

        if lang == "ja":
            system_prompt = """あなたは教育コンテンツ作成の専門家です。
複数の日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成"""

            combined_text = ""
            chunks_data = {}
            total_pairs = 0

            for i, chunk in enumerate(chunks, 1):
                num_pairs = self.determine_qa_count(chunk, config)
                total_pairs += num_pairs
                chunk_text = chunk['text']
                if len(chunk_text) > 1000:
                    chunk_text = chunk_text[:1000] + "..."
                combined_text += f"\n\n【テキスト{i}】\n{chunk_text}"
                chunks_data[f"chunk_{i}"] = {"num_pairs": num_pairs, "chunk": chunk}

            user_prompt = f"""以下の{len(chunks)}個のテキストから、合計{total_pairs}個のQ&Aペアを生成してください。
{combined_text}

質問タイプ:
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）

JSON形式で出力:
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""
        else:
            system_prompt = """You are an expert in educational content creation.
Generate high-quality Q&A pairs from multiple English texts.

Generation rules:
1. Questions should be clear and specific
2. Answers should be concise and accurate (1-2 sentences)
3. Stay faithful to the text content
4. Create questions from diverse perspectives"""

            combined_text = ""
            chunks_data = {}
            total_pairs = 0

            for i, chunk in enumerate(chunks, 1):
                num_pairs = self.determine_qa_count(chunk, config)
                total_pairs += num_pairs
                chunk_text = chunk['text']
                if len(chunk_text) > 1000:
                    chunk_text = chunk_text[:1000] + "..."
                combined_text += f"\n\n【Text {i}】\n{chunk_text}"
                chunks_data[f"chunk_{i}"] = {"num_pairs": num_pairs, "chunk": chunk}

            user_prompt = f"""Generate {total_pairs} Q&A pairs from the following {len(chunks)} texts.
{combined_text}

Question types:
- fact: Factual questions (What is...?) 
- reason: Explanatory questions (Why...?) 
- comparison: Comparative questions (What's the difference...?) 
- application: Application questions (How is... used?)

Output in JSON format:
{{
  "qa_pairs": [
    {{
      "question": "question text",
      "answer": "answer text",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""

        try:
            combined_input = f"{system_prompt}\n\n{user_prompt}"
            parsed_data = self.client.generate_structured(
                prompt=combined_input,
                response_schema=QAPairsResponse,
                model=self.model,
                max_output_tokens=4000
            )

            qa_index = 0
            for i, chunk in enumerate(chunks, 1):
                chunk_key = f"chunk_{i}"
                expected_pairs = chunks_data[chunk_key]["num_pairs"]

                for _ in range(expected_pairs):
                    if qa_index < len(parsed_data.qa_pairs):
                        qa_data = parsed_data.qa_pairs[qa_index]
                        qa = {
                            "question"         : qa_data.question,
                            "answer"           : qa_data.answer,
                            "question_type"    : qa_data.question_type,
                            "source_chunk_id"  : chunk.get('id', ''),
                            "doc_id"           : chunk.get('doc_id', ''),
                            "dataset_type"     : chunk.get('dataset_type', ''),
                            "chunk_idx"        : chunk.get('chunk_idx', 0),
                            "generation_method": "legacy"
                        }
                        all_qa_pairs.append(qa)
                        qa_index += 1

            if len(all_qa_pairs) == 0:
                logger.error("Gemini APIから解析可能なレスポンスが返されませんでした")
                raise ValueError("No parseable response from Gemini API")

            return all_qa_pairs

        except Exception as e:
            logger.error(f"バッチQ/A生成エラー: {e}")
            import traceback
            logger.debug(f"スタックトレース: {traceback.format_exc()}")
            logger.info("フォールバック: チャンクを個別処理します")
            for chunk in chunks:
                try:
                    qa_pairs = self.generate_for_chunk(chunk, config)
                    all_qa_pairs.extend(qa_pairs)
                except Exception as chunk_error:
                    logger.error(f"チャンク個別処理エラー: {chunk_error}")
            return all_qa_pairs


def generate_qa_dataset(
        chunks: List[Dict],
        dataset_type: str,
        model: str = "gemini-2.0-flash",
        chunk_batch_size: int = 3,
        merge_chunks: bool = True,
        min_tokens: int = 150,
        max_tokens: int = 400,
        config: Optional[Dict] = None,
        client: Optional[LLMClient] = None,
        use_smart_generation: bool = False  # ✨ 新規追加
) -> List[Dict]:
    """
    データセット全体のQ/Aペア生成

    Args:
        use_smart_generation: スマート生成を使用するか（デフォルト: False）
    """
    if config is None:
        config = DATASET_CONFIGS.get(dataset_type)
        if not config:
            raise ValueError(f"未対応のデータセット: {dataset_type}")

    # クライアント生成（指定がなければ作成）
    if client is None:
        client = create_llm_client(provider="gemini")

    # ✨ QAGenerator初期化（スマート生成フラグ付き）
    generator = QAGenerator(
        client=client,
        model=model,
        use_smart_generation=use_smart_generation
    )
    all_qa_pairs = []

    # チャンクの前処理（小さいチャンクの統合）
    if merge_chunks:
        processed_chunks = merge_small_chunks(chunks, min_tokens, max_tokens)
    else:
        processed_chunks = chunks

    total_chunks = len(processed_chunks)
    api_calls = (total_chunks + chunk_batch_size - 1) // chunk_batch_size

    generation_mode = "スマート生成" if use_smart_generation else "従来方式"
    logger.info(f"""
    Q/Aペア生成開始 ({generation_mode}):
    - 元チャンク数: {len(chunks)}
    - 処理チャンク数: {total_chunks}
    - バッチサイズ: {chunk_batch_size}
    - API呼び出し予定: {api_calls}回
    - モデル: {model}
    """)

    # バッチ処理
    for i in range(0, total_chunks, chunk_batch_size):
        batch = processed_chunks[i:i + chunk_batch_size]
        batch_num = i // chunk_batch_size + 1
        total_batches = api_calls

        logger.info(f"バッチ {batch_num}/{total_batches} 処理中 ({len(batch)}チャンク)...")

        # リトライ機能付きQ/A生成
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if chunk_batch_size == 1:
                    qa_pairs = generator.generate_for_chunk(batch[0], config)
                else:
                    qa_pairs = generator.generate_for_batch(batch, config)

                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                    logger.debug(f"バッチ {batch_num}: {len(qa_pairs)}個のQ/Aペア生成")
                break

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"バッチ {batch_num} 生成失敗: {e}")
                    logger.info("個別処理にフォールバック...")
                    for chunk in batch:
                        try:
                            qa_pairs = generator.generate_for_chunk(chunk, config)
                            if qa_pairs:
                                all_qa_pairs.extend(qa_pairs)
                        except Exception as chunk_error:
                            logger.error(f"チャンク処理エラー: {chunk_error}")
                else:
                    wait_time = 2 ** attempt
                    logger.warning(f"リトライ {attempt + 1}/{max_retries} (待機: {wait_time}秒)")
                    time.sleep(wait_time)

        # API制限対策
        if i + chunk_batch_size < total_chunks:
            time.sleep(0.2)

    logger.info(f"""
    Q/Aペア生成完了 ({generation_mode}):
    - 生成されたQ/Aペア: {len(all_qa_pairs)}個
    - 実行されたAPI呼び出し: 約{api_calls}回
    """)

    return all_qa_pairs
