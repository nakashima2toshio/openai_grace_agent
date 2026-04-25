#!/usr/bin/env python3
"""
helper_rag_qa.py - RAG Q&A用ユーティリティモジュール（後方互換レイヤー）
=========================================================================
このモジュールは後方互換性のために維持されています。
新しいコードでは qa_generation/ パッケージを直接使用してください。

統合先:
- qa_generation/models.py: QAPair, QAPairsList, ChainOfThought*, EnhancedQAPair*
- qa_generation/keyword_extraction.py: BestKeywordSelector, SmartKeywordSelector
- qa_generation/semantic.py: SemanticCoverage

クラス一覧（このファイルに残存）:
- QACountOptimizer
- QAOptimizedExtractor
- LLMBasedQAGenerator
- ChainOfThoughtQAGenerator
- RuleBasedQAGenerator
- TemplateBasedQAGenerator
- HybridQAGenerator
- AdvancedQAGenerationTechniques
- QAGenerationOptimizer
- OptimizedHybridQAGenerator
- BatchHybridQAGenerator
"""

from regex_mecab import KeywordExtractor
from typing import List, Dict, Tuple, Optional, Any
import re
import math
import json
import os
from collections import defaultdict
import numpy as np
import tiktoken
from helper_llm import create_llm_client
from helper_embedding import create_embedding_client, get_embedding_dimensions
from pydantic import BaseModel
import spacy

# .envファイルから環境変数を読み込む
from dotenv import load_dotenv
load_dotenv()

# ===================================================================
# qa_generation/ からの統合インポート（後方互換性）
# ===================================================================
from qa_generation.semantic import SemanticCoverage
from qa_generation.models import (
    QAPair as _QAPair,
    QAPairsList as _QAPairsList,
    ChainOfThoughtAnalysis as _ChainOfThoughtAnalysis,
    ChainOfThoughtQAPair as _ChainOfThoughtQAPair,
    ChainOfThoughtResponse as _ChainOfThoughtResponse,
    EnhancedQAPair as _EnhancedQAPair,
    EnhancedQAPairsList as _EnhancedQAPairsList,
    QAGenerationConsiderations as _QAGenerationConsiderations,
)
from qa_generation.keyword_extraction import (
    BestKeywordSelector as _BestKeywordSelector,
    SmartKeywordSelector as _SmartKeywordSelector,
    get_best_keywords,
    get_smart_keywords,
)

"""
[キーワード抽出関連クラス]

1. BestKeywordSelector - 3手法（MeCab/正規表現/統合版）から最良のキーワードを選択するクラス
2. SmartKeywordSelector - テキスト特性に応じて最適なtop_nを自動決定してキーワードを抽出するクラス
3. QAOptimizedExtractor - Q&Aペア生成に最適化されたキーワード抽出クラス（関係性抽出、難易度分類、文脈情報付与機能を含む）

[セマンティックカバレッジ関連クラス]

4. SemanticCoverage - 意味的な網羅性を測定するクラス（文書のセマンティックチャンク分割、埋め込みベクトル生成、コサイン類似度計算）
5. QAGenerationConsiderations - Q/A生成前のチェックリストクラス（文書特性分析、Q/A要件定義、品質基準設定）

[データモデル pydanticクラス]

6. QAPair - Q/Aペアのデータモデル（Pydantic BaseModel）
7. QAPairsList - Q/Aペアのリスト構造を定義（Pydantic BaseModel）

[Q/A生成クラス]

8. LLMBasedQAGenerator - LLM（GPT-5-mini）を使用したQ/A生成クラス（基本Q/A生成、多様な種類のQ/A生成）
9. ChainOfThoughtQAGenerator -
思考の連鎖（Chain-of-Thought）を使った高品質Q/A生成クラス（推論過程付きQ/A生成、信頼度スコア算出）
10. RuleBasedQAGenerator - ルールベースのQ/A生成クラス（定義文抽出、事実情報抽出、列挙パターン抽出）
11. TemplateBasedQAGenerator - テンプレートを使用したQ/A生成クラス（質問テンプレート管理、エンティティベースQ/A生成）
12. HybridQAGenerator - 複数の手法を組み合わせた高度なQ/A生成クラス（包括的パイプライン、品質検証、重複除去）

[高度なQ/A生成技術クラス]

13. AdvancedQAGenerationTechniques - 高度なQ/A生成技術クラス（敵対的Q/A、マルチホップ推論Q/A、反事実的Q/A生成）
14. QAGenerationOptimizer - Q/A生成の最適化クラス（カバレッジ最大化戦略、適応的生成、コスト最適化）


"""
class BestKeywordSelector:
    """3手法から最良のキーワードを選択するクラス"""

    def __init__(self, prefer_mecab: bool = True):
        """
        Args:
            prefer_mecab: MeCabを優先的に使用するか
        """
        self.extractor = KeywordExtractor(prefer_mecab=prefer_mecab)

        # 評価重み付け（調整可能）
        self.weights = {
            'coverage': 0.25,      # カバレージ率
            'diversity': 0.15,     # 多様性
            'technicality': 0.25,  # 専門性
            'coherence': 0.20,     # 一貫性
            'length_balance': 0.15 # 長さのバランス
        }

    def evaluate_keywords(self, keywords: List[str], text: str) -> Dict[str, float]:
        """
        キーワードセットの品質を多面的に評価

        Args:
            keywords: 評価対象のキーワードリスト
            text: 元のテキスト

        Returns:
            評価指標の辞書
        """
        if not keywords:
            return {metric: 0.0 for metric in self.weights.keys()}

        metrics = {}

        # 1. カバレージ率（キーワードがテキストに存在する割合）
        coverage_count = sum(1 for kw in keywords if kw in text)
        metrics['coverage'] = coverage_count / len(keywords)

        # 2. 多様性（文字数の分散）
        lengths = [len(kw) for kw in keywords]
        avg_len = sum(lengths) / len(lengths)
        if len(lengths) > 1:
            variance = sum((l - avg_len) ** 2 for l in lengths) / (len(lengths) - 1)
            # 適度な分散を評価（標準偏差2-4文字が理想）
            std_dev = variance ** 0.5
            metrics['diversity'] = min(1.0, (std_dev / 3.0) if std_dev < 3 else (6 - std_dev) / 3.0)
        else:
            metrics['diversity'] = 0.5

        # 3. 専門性（カタカナ・英語・漢字複合語の割合）
        technical_patterns = [
            (r'^[ァ-ヴー]{3,}$', 1.0),      # カタカナ3文字以上
            (r'^[A-Z]{2,}[A-Z0-9]*$', 1.2), # 英大文字（略語）
            (r'^[一-龥]{4,}$', 0.9),        # 漢字4文字以上
            (r'^[A-Za-z]+[A-Za-z0-9]*$', 0.8) # 英単語
        ]

        tech_score = 0
        for kw in keywords:
            kw_tech = 0
            for pattern, weight in technical_patterns:
                if re.match(pattern, kw):
                    kw_tech = max(kw_tech, weight)
            tech_score += kw_tech
        metrics['technicality'] = min(1.0, tech_score / len(keywords))

        # 4. 一貫性（キーワード間の関連性）
        # 同じ文字を含むキーワードのペア数で評価
        coherence_score = 0
        for i, kw1 in enumerate(keywords):
            for kw2 in keywords[i+1:]:
                # 部分文字列の共有
                if len(kw1) >= 2 and len(kw2) >= 2:
                    if any(sub in kw2 for sub in [kw1[i:i+2] for i in range(len(kw1)-1)]):
                        coherence_score += 1
        max_pairs = len(keywords) * (len(keywords) - 1) / 2
        metrics['coherence'] = coherence_score / max_pairs if max_pairs > 0 else 0

        # 5. 長さのバランス（2-8文字が理想）
        ideal_length_ratio = sum(1 for kw in keywords if 2 <= len(kw) <= 8) / len(keywords)
        metrics['length_balance'] = ideal_length_ratio

        return metrics

    def calculate_total_score(self, metrics: Dict[str, float]) -> float:
        """
        評価指標から総合スコアを計算

        Args:
            metrics: 各評価指標の辞書

        Returns:
            総合スコア（0.0-1.0）
        """
        total = sum(metrics.get(metric, 0) * weight
                   for metric, weight in self.weights.items())
        return min(1.0, total)

    def extract_best(self, text: str, top_n: int = 10,
                     return_details: bool = False) -> Dict[str, Any]:
        """
        3つの手法で抽出し、最良の結果を選択
        Args:
            text: 分析対象テキスト
            top_n: 抽出するキーワード数
            return_details: 詳細情報を返すか
        Returns:
            最良のキーワードと選択理由
        """
        # 各手法で抽出
        all_results = self.extractor.extract_with_details(text, top_n)

        # 各手法の評価
        evaluations = {}
        for method, keywords_scores in all_results.items():
            keywords = [kw for kw, _ in keywords_scores[:top_n]]

            # 評価指標を計算
            metrics = self.evaluate_keywords(keywords, text)
            total_score = self.calculate_total_score(metrics)

            evaluations[method] = {
                'keywords': keywords,
                'metrics': metrics,
                'total_score': total_score,
                'keyword_scores': keywords_scores[:top_n]
            }

        # 最良の手法を選択
        best_method = max(evaluations.items(),
                         key=lambda x: x[1]['total_score'])

        result = {
            'best_method': best_method[0],
            'keywords': best_method[1]['keywords'],
            'total_score': best_method[1]['total_score'],
            'reason': self._generate_reason(best_method[0], evaluations)
        }

        if return_details:
            result['all_evaluations'] = evaluations

        return result

    def _generate_reason(self, best_method: str,
                        evaluations: Dict[str, Dict]) -> str:
        """選択理由を生成"""
        best_eval = evaluations[best_method]
        metrics = best_eval['metrics']

        # 最も優れた指標を特定
        best_metric = max(metrics.items(), key=lambda x: x[1])

        reasons = {
            'coverage': 'テキストカバレージが最も高い',
            'diversity': 'キーワードの多様性が優れている',
            'technicality': '専門用語の抽出精度が高い',
            'coherence': 'キーワード間の一貫性が優れている',
            'length_balance': 'キーワード長のバランスが良い'
        }

        return f"{reasons.get(best_metric[0], '総合的に優れている')} (スコア: {best_eval['total_score']:.3f})"


class SmartKeywordSelector(BestKeywordSelector):
    """テキスト特性に応じた最適なキーワード抽出を行うクラス"""

    def __init__(self, prefer_mecab: bool = True):
        super().__init__(prefer_mecab=prefer_mecab)

        # モード別のデフォルトtop_n
        self.mode_defaults = {
            "summary": 5,        # 要約・概要把握用
            "standard": 10,      # 標準的な分析
            "detailed": 15,      # 詳細分析
            "exhaustive": 20,    # 網羅的抽出
            "tag": 3,           # タグ付け用
        }

    def calculate_auto_top_n(self, text: str) -> Tuple[int, str]:
        """
        テキスト特性から最適なtop_nを自動計算

        Returns:
            (top_n, 判定理由)
        """
        # 基本メトリクス
        text_length = len(text)
        sentences = len(re.split(r'[。！？\.\!\?]+', text))

        # 専門用語の密度を推定（カタカナ・英字・漢字複合語）
        technical_pattern = r'[ァ-ヴー]{3,}|[A-Z]{2,}|[一-龥]{4,}'
        technical_terms = len(re.findall(technical_pattern, text))
        technical_density = technical_terms / (text_length / 100) if text_length > 0 else 0

        # ルールベースの決定
        if text_length < 100:
            return 3, f"超短文（{text_length}文字）"
        elif text_length < 300:
            return 5, f"短文（{text_length}文字）"
        elif text_length < 500:
            base = 7
            if technical_density > 2:
                base += 2  # 専門用語が多い場合は増やす
            return base, f"中文（{text_length}文字、専門用語密度: {technical_density:.1f}）"
        elif text_length < 1000:
            base = 10
            if sentences > 10:
                base += 2  # 文が多い場合は増やす
            return base, f"標準文（{text_length}文字、{sentences}文）"
        elif text_length < 2000:
            return 15, f"長文（{text_length}文字）"
        else:
            # 非常に長い文書は対数的に増加
            extra = int(math.log(text_length / 2000, 2))
            return min(20 + extra, 30), f"超長文（{text_length}文字）"

    def find_optimal_by_coverage(
        self,
        text: str,
        target_coverage: float = 0.7,
        min_n: int = 3,
        max_n: int = 20
    ) -> Tuple[int, float]:
        """
        目標カバレッジを達成する最小のtop_nを探索

        Returns:
            (最適なtop_n, 達成カバレッジ率)
        """
        best_n = min_n
        best_coverage = 0

        for n in range(min_n, max_n + 1):
            # 3手法で抽出して最良を選択
            result = self.extract_best(text, n, return_details=False)
            keywords = result['keywords']

            # カバレッジ計算（キーワードがカバーする文字数の割合）
            covered_chars = 0
            for keyword in keywords:
                occurrences = text.count(keyword)
                covered_chars += len(keyword) * occurrences

            coverage = covered_chars / len(text) if len(text) > 0 else 0

            if coverage >= target_coverage:
                return n, coverage

            best_coverage = coverage
            best_n = n

        # 目標に届かない場合は最大値を返す
        return best_n, best_coverage

    def find_optimal_by_diminishing_returns(
        self,
        text: str,
        min_n: int = 3,
        max_n: int = 20,
        threshold: float = 0.05
    ) -> Tuple[int, float, List[float]]:
        """
        収穫逓減の法則に基づいて最適なtop_nを決定

        Args:
            threshold: 改善率がこの値以下になったら停止

        Returns:
            (最適なtop_n, 最終スコア, 各nでのスコアリスト)
        """
        scores = []
        previous_score = 0
        optimal_n = min_n

        for n in range(min_n, max_n + 1):
            result = self.extract_best(text, n, return_details=False)
            current_score = result['total_score']
            scores.append(current_score)

            if n > min_n:
                improvement = current_score - previous_score
                improvement_rate = improvement / previous_score if previous_score > 0 else 1

                # 改善が閾値以下なら前の値が最適
                if improvement_rate < threshold:
                    optimal_n = n - 1
                    break

            previous_score = current_score
            optimal_n = n

        return optimal_n, scores[optimal_n - min_n], scores

    def extract_best_auto(
        self,
        text: str,
        mode: str = "auto",
        min_keywords: int = 3,
        max_keywords: int = 20,
        target_coverage: float = 0.7,
        return_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        自動的に最適なtop_nを決定してキーワード抽出

        Args:
            text: 分析対象テキスト
            mode: 抽出モード
                - "auto": テキスト長に基づく自動決定
                - "summary": 要約用（5個）
                - "standard": 標準（10個）
                - "detailed": 詳細（15個）
                - "coverage": カバレッジベース
                - "diminishing": 収穫逓減ベース
            min_keywords: 最小キーワード数
            max_keywords: 最大キーワード数
            target_coverage: カバレッジモードでの目標率
            return_analysis: 分析詳細を返すか

        Returns:
            抽出結果と決定根拠
        """
        analysis = {
            "mode": mode,
            "text_length": len(text),
            "sentence_count": len(re.split(r'[。！？\.\!\?]+', text))
        }

        # モードに応じてtop_nを決定
        if mode == "auto":
            top_n, reason = self.calculate_auto_top_n(text)
            analysis["decision_reason"] = reason

        elif mode in self.mode_defaults:
            top_n = self.mode_defaults[mode]
            analysis["decision_reason"] = f"固定値モード: {mode}"

        elif mode == "coverage":
            top_n, achieved_coverage = self.find_optimal_by_coverage(
                text, target_coverage, min_keywords, max_keywords
            )
            analysis["decision_reason"] = f"カバレッジ {achieved_coverage:.1%} 達成"
            analysis["target_coverage"] = target_coverage
            analysis["achieved_coverage"] = achieved_coverage

        elif mode == "diminishing":
            top_n, final_score, score_progression = self.find_optimal_by_diminishing_returns(
                text, min_keywords, max_keywords
            )
            analysis["decision_reason"] = f"収穫逓減点: n={top_n}"
            analysis["score_progression"] = score_progression
            analysis["final_score"] = final_score

        else:
            top_n = 10
            analysis["decision_reason"] = "デフォルト値"

        # 範囲制限
        top_n = max(min_keywords, min(max_keywords, top_n))
        analysis["selected_top_n"] = top_n

        # 最良の手法でキーワード抽出
        result = self.extract_best(text, top_n, return_details=True)

        # 結果に分析情報を追加
        result["optimization"] = analysis

        if not return_analysis:
            # 簡潔な結果のみ返す
            return {
                "keywords": result["keywords"],
                "method": result["best_method"],
                "top_n": top_n,
                "mode": mode,
                "reason": analysis["decision_reason"]
            }

        return result


# エクスポート用の関数
def get_best_keywords(text: str, top_n: int = 10, prefer_mecab: bool = True) -> List[str]:
    """
    テキストから最良のキーワードを抽出する簡易関数

    Args:
        text: 抽出対象のテキスト
        top_n: 抽出するキーワード数
        prefer_mecab: MeCabを優先するか

    Returns:
        キーワードのリスト
    """
    selector = BestKeywordSelector(prefer_mecab=prefer_mecab)
    result = selector.extract_best(text, top_n)
    return result['keywords']


def get_smart_keywords(text: str, mode: str = "auto", prefer_mecab: bool = True) -> Dict[str, Any]:
    """
    スマート選択でキーワードを抽出する簡易関数

    Args:
        text: 抽出対象のテキスト
        mode: 抽出モード（"auto", "summary", "detailed"等）
        prefer_mecab: MeCabを優先するか

    Returns:
        キーワードと選択情報を含む辞書
    """
    selector = SmartKeywordSelector(prefer_mecab=prefer_mecab)
    return selector.extract_best_auto(text, mode=mode)


class QACountOptimizer:
    """Q/Aペア数の最適化を行うクラス"""

    def __init__(self, llm_model: str = "gemini-2.0-flash"):
        self.llm_model_for_token_count = llm_model
        self.unified_client = create_llm_client(provider="gemini", default_model=self.llm_model_for_token_count)

    def calculate_optimal_qa_count(self, document: str, mode: str = "auto") -> Dict[str, Any]:
        """
        文書特性から最適なQ/A数を算出

        Args:
            document: 対象文書
            mode: 決定モード ("auto", "evaluation", "00_learning.md", "search_test", "faq")

        Returns:
            最適なQ/A数と決定根拠を含む辞書
        """
        # 基本メトリクスの計算
        metrics = self._analyze_document_metrics(document)

        # モード別の決定
        if mode == "evaluation":
            # 評価用：網羅性重視（文書長の5-10%）
            base_count = int(metrics['sentence_count'] * 0.075)
        elif mode == "00_learning.md":
            # 学習用：主要概念をカバー（10-20個）
            base_count = min(20, max(10, metrics['keyword_count'] // 2))
        elif mode == "search_test":
            # 検索テスト：多様な質問パターン（20-30個）
            base_count = min(30, max(20, metrics['sentence_count'] // 3))
        elif mode == "faq":
            # FAQ生成：ユーザーニーズベース（5-15個）
            base_count = min(15, max(5, metrics['keyword_count'] // 3))
        else:  # auto
            # 文書長ベースの自動決定
            base_count = self._calculate_by_document_length(metrics)

        # 情報密度による調整
        adjusted_count = self._adjust_by_information_density(document, base_count, metrics)

        # カバレッジ目標による調整
        final_count = self._adjust_by_coverage_target(document, adjusted_count, metrics)

        return {
            'optimal_count': final_count,
            'base_count': base_count,
            'adjusted_count': adjusted_count,
            'metrics': metrics,
            'mode': mode,
            'reasoning': self._generate_reasoning(metrics, base_count, adjusted_count, final_count, mode)
        }

    def _analyze_document_metrics(self, document: str) -> Dict:
        """文書の基本メトリクスを分析"""
        sentences = re.split(r'[。！？\.\!\?]+', document)
        sentences = [s.strip() for s in sentences if s.strip()]

        # トークン数の計算
        token_count = self.unified_client.count_tokens(document, model=self.llm_model_for_token_count)

        # キーワード候補の抽出
        technical_terms = re.findall(r'[ァ-ヴー]{3,}|[A-Z]{2,}[A-Z0-9]*|[一-龥]{4,}', document)

        # 段落数の計算
        paragraphs = document.split('\n\n')
        paragraphs = [p for p in paragraphs if p.strip()]

        return {
            'doc_length': len(document),
            'token_count': token_count,
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_sentence_length': len(document) / max(1, len(sentences)),
            'keyword_count': len(set(technical_terms)),
            'keyword_density': len(technical_terms) / max(1, len(document) / 100),
            'complexity_score': self._calculate_complexity_score(document, sentences, technical_terms)
        }

    def _calculate_complexity_score(self, document: str, sentences: List[str], technical_terms: List[str]) -> float:
        """文書の複雑さスコアを計算（0.0-1.0）"""
        score = 0.0

        # 文の長さの変動性
        if len(sentences) > 1:
            lengths = [len(s) for s in sentences]
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
            std_dev = variance ** 0.5
            length_complexity = min(1.0, std_dev / avg_len)
            score += length_complexity * 0.3

        # 専門用語の密度
        term_density = len(technical_terms) / max(1, len(document) / 100)
        term_complexity = min(1.0, term_density / 5)
        score += term_complexity * 0.4

        # 句読点の複雑さ
        complex_punctuation = len(re.findall(r'[、；：「」『』（）\[\]]', document))
        punct_complexity = min(1.0, complex_punctuation / max(1, len(sentences)))
        score += punct_complexity * 0.3

        return min(1.0, score)

    def _calculate_by_document_length(self, metrics: Dict) -> int:
        """文書長に基づく基本Q/A数の計算"""
        doc_length = metrics['doc_length']

        if doc_length < 500:
            return 3
        elif doc_length < 1000:
            return 5
        elif doc_length < 2000:
            return 8
        elif doc_length < 5000:
            return 12
        elif doc_length < 10000:
            return 18
        else:
            # 対数的増加
            extra = int(math.log(doc_length / 10000, 2) * 3)
            return min(30, 18 + extra)

    def _adjust_by_information_density(self, text: str, base_count: int, metrics: Dict) -> int:
        """情報密度に応じて調整"""
        keyword_density = metrics['keyword_density']
        complexity = metrics['complexity_score']

        # 情報密度による調整係数
        if keyword_density > 4 and complexity > 0.7:
            # 高密度・高複雑度
            multiplier = 1.5
        elif keyword_density > 2.5 or complexity > 0.5:
            # 中密度または中複雑度
            multiplier = 1.2
        elif keyword_density < 1 and complexity < 0.3:
            # 低密度・低複雑度
            multiplier = 0.7
        else:
            multiplier = 1.0

        return max(3, int(base_count * multiplier))

    def _adjust_by_coverage_target(self, text: str, current_count: int, metrics: Dict, target_coverage: float = 0.7) -> int:
        """カバレッジ目標による調整"""
        # セマンティックチャンク数の推定
        estimated_chunks = metrics['token_count'] // 150  # 150トークン/チャンク

        # 1つのQ/Aが平均2-3チャンクをカバーすると仮定
        coverage_per_qa = 2.5
        required_qa = int(estimated_chunks * target_coverage / coverage_per_qa)

        # 現在のカウントと必要数の中間を取る
        final_count = int((current_count + required_qa) / 2)

        # 範囲制限（3-50）
        return max(3, min(50, final_count))

    def _generate_reasoning(self, metrics: Dict, base_count: int, adjusted_count: int, final_count: int, mode: str) -> str:
        """決定理由の生成"""
        reasons = []

        # モードの説明
        mode_descriptions = {
            'auto': '自動決定モード',
            'evaluation': '評価用（網羅性重視）',
            '00_learning.md': '学習用（主要概念重視）',
            'search_test': '検索テスト用（多様性重視）',
            'faq': 'FAQ生成用（実用性重視）'
        }
        reasons.append(f"モード: {mode_descriptions.get(mode, mode)}")

        # 文書特性
        reasons.append(f"文書長: {metrics['doc_length']}文字（{metrics['sentence_count']}文）")
        reasons.append(f"複雑度: {metrics['complexity_score']:.2f}")
        reasons.append(f"キーワード密度: {metrics['keyword_density']:.1f}")

        # 調整内容
        if adjusted_count != base_count:
            if adjusted_count > base_count:
                reasons.append(f"情報密度が高いため {base_count}→{adjusted_count} に増加")
            else:
                reasons.append(f"情報密度が低いため {base_count}→{adjusted_count} に減少")

        if final_count != adjusted_count:
            reasons.append(f"カバレッジ目標により {adjusted_count}→{final_count} に調整")

        return " / ".join(reasons)

    def determine_by_coverage_improvement(self, text: str, existing_qa: List[Dict] = None) -> Dict:
        """カバレッジ改善率に基づく決定"""
        if existing_qa is None:
            existing_qa = []

        # 現在のカバレッジを推定
        current_coverage = len(existing_qa) * 2.5 / max(1, len(text) // 200)
        current_coverage = min(1.0, current_coverage)

        # 目標カバレッジまでの差分
        target_coverage = 0.7
        coverage_gap = max(0, target_coverage - current_coverage)

        # 必要な追加Q/A数
        additional_qa = int(coverage_gap * len(text) / 500)

        return {
            'additional_count': additional_qa,
            'current_coverage': current_coverage,
            'target_coverage': target_coverage,
            'total_count': len(existing_qa) + additional_qa
        }


class QAOptimizedExtractor(SmartKeywordSelector):
    """Q&Aペア生成に最適化されたキーワード抽出クラス"""

    def __init__(self, prefer_mecab: bool = True):
        super().__init__(prefer_mecab=prefer_mecab)
        self.qa_optimizer = QACountOptimizer()

        # Q&A用の追加ストップワード
        self.qa_stopwords = {
            '最新', '問題', '実現', '可能', '場合', '結果',
            '方法', '技術', '今後', '現在', '将来', '重要',
            '必要', '状況', '対象', '目的', '効果', '影響',
            'これ', 'それ', 'あれ', 'この', 'その', 'あの'
        }

        # 関係性パターンの定義
        self.relation_patterns = [
            (r'(.+?)は(.+?)である', 'is_a'),
            (r'(.+?)は(.+?)を(.+?)する', 'uses'),
            (r'(.+?)により(.+?)が(.+?)', 'enables'),
            (r'(.+?)のため(.+?)を', 'for'),
            (r'(.+?)を用いて(.+?)を', 'uses'),
            (r'(.+?)から(.+?)へ', 'transforms'),
            (r'(.+?)における(.+?)', 'in_context'),
            (r'(.+?)による(.+?)', 'by_means_of')
        ]

        # カテゴリ分類用パターン
        self.category_patterns = {
            'core_concept': ['技術', '手法', '方式', '理論', 'モデル'],
            'technical_term': r'^[A-Z]{2,}[A-Z0-9]*$|^[ァ-ヴー]{4,}$',
            'specific_name': ['BERT', 'GPT', 'CNN', 'RNN', 'LSTM'],
            'general_term': ['データ', 'システム', '処理', '分析']
        }

    def filter_for_qa(self, keywords: List[str]) -> List[str]:
        """Q&A生成に適したキーワードのみを残す"""
        filtered = []

        # 親クラスのstopwordsを取得
        parent_stopwords = self.extractor.stopwords if hasattr(self.extractor, 'stopwords') else set()

        for kw in keywords:
            # 基本フィルタリング
            if (kw not in self.qa_stopwords and
                kw not in parent_stopwords and
                len(kw) >= 2 and
                not kw.isdigit() and
                not kw.startswith('第') and  # 第1、第2などを除外
                not kw.endswith('等') and     # 〜等を除外
                not kw.endswith('的')):       # 〜的を除外
                filtered.append(kw)
        return filtered

    def remove_redundant_keywords(self, keywords: List[str]) -> List[str]:
        """包含関係にあるキーワードを整理"""
        # 長い順にソート
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        filtered = []

        for kw in sorted_keywords:
            # 既存のキーワードに部分的に含まれていないかチェック
            is_redundant = False
            for existing in filtered:
                # 完全一致は除外しない
                if kw != existing:
                    # より長いキーワードに完全に含まれている場合は冗長
                    if kw in existing:
                        is_redundant = True
                        break
                    # 意味的に重複している場合（例：AI と 人工知能）
                    if self._are_synonyms(kw, existing):
                        is_redundant = True
                        break

            if not is_redundant:
                filtered.append(kw)

        return filtered

    def _are_synonyms(self, word1: str, word2: str) -> bool:
        """同義語判定（簡易版）"""
        synonym_pairs = [
            ('AI', '人工知能'),
            ('ML', '機械学習'),
            ('DL', '深層学習'),
            ('NLP', '自然言語処理'),
            ('DX', 'デジタルトランスフォーメーション')
        ]

        for pair in synonym_pairs:
            if (word1 in pair and word2 in pair):
                return True
        return False

    def classify_difficulty(self, keyword: str, text: str) -> str:
        """キーワードの難易度を判定"""
        frequency = text.count(keyword)
        is_acronym = bool(re.match(r'^[A-Z]{2,}[A-Z0-9]*$', keyword))
        is_complex = len(keyword) >= 8
        has_explanation = any([
            f"{keyword}とは" in text,
            f"{keyword}は" in text and "である" in text,
            f"{keyword}（" in text  # 括弧で説明がある
        ])

        # 難易度判定ロジック
        if has_explanation and frequency >= 3:
            return "basic"
        elif is_acronym and not has_explanation:
            return "advanced"
        elif is_complex or frequency == 1:
            return "advanced"
        elif frequency >= 5:
            return "basic"
        else:
            return "intermediate"

    def categorize_keyword(self, keyword: str) -> str:
        """キーワードをカテゴリに分類"""
        # 特定の名前
        if keyword in self.category_patterns['specific_name']:
            return 'specific_name'

        # 技術用語（英略語やカタカナ専門語）
        if re.match(self.category_patterns['technical_term'], keyword):
            return 'technical_term'

        # 中核概念
        for term in self.category_patterns['core_concept']:
            if term in keyword:
                return 'core_concept'

        # 一般用語
        for term in self.category_patterns['general_term']:
            if term in keyword:
                return 'general_term'

        return 'other'

    def extract_keyword_relations(self, text: str, keywords: List[str]) -> List[Dict]:
        """キーワード間の関係を抽出"""
        relations = []
        sentences = re.split(r'[。！？\.\!\?]', text)

        for sent in sentences:
            if not sent.strip():
                continue

            # 文内に含まれるキーワードを特定
            keywords_in_sent = [kw for kw in keywords if kw in sent]

            if len(keywords_in_sent) >= 2:
                # パターンマッチングで関係性を判定
                for pattern, rel_type in self.relation_patterns:
                    match = re.search(pattern, sent)
                    if match:
                        # マッチしたパターンからキーワードペアを抽出
                        for i, kw1 in enumerate(keywords_in_sent):
                            for kw2 in keywords_in_sent[i+1:]:
                                if kw1 in sent and kw2 in sent:
                                    # 出現順序を確認
                                    pos1 = sent.index(kw1)
                                    pos2 = sent.index(kw2)

                                    if pos1 < pos2:
                                        from_kw, to_kw = kw1, kw2
                                    else:
                                        from_kw, to_kw = kw2, kw1

                                    relations.append({
                                        'from': from_kw,
                                        'to': to_kw,
                                        'context': sent.strip(),
                                        'relation_type': rel_type
                                    })
                                    break
                        break
                else:
                    # パターンにマッチしない場合は共起関係として記録
                    if len(keywords_in_sent) >= 2:
                        for i, kw1 in enumerate(keywords_in_sent):
                            for kw2 in keywords_in_sent[i+1:]:
                                relations.append({
                                    'from': kw1,
                                    'to': kw2,
                                    'context': sent.strip(),
                                    'relation_type': 'co_occurs'
                                })

        # 重複を除去
        unique_relations = []
        seen = set()
        for rel in relations:
            key = (rel['from'], rel['to'], rel['relation_type'])
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)

        return unique_relations

    def extract_with_context(self, text: str, keyword: str) -> Dict:
        """キーワードと周辺文脈を抽出"""
        sentences = re.split(r'[。！？\.\!\?]', text)
        contexts = []

        for i, sent in enumerate(sentences):
            if keyword in sent:
                # 文の重要度を計算
                importance = self._calculate_sentence_importance(sent, keyword, text)

                contexts.append({
                    'sentence': sent.strip(),
                    'position': sent.index(keyword),
                    'sentence_index': i,
                    'importance': importance
                })

        # 最も重要な文脈を選択
        if contexts:
            best_context = max(contexts, key=lambda x: x['importance'])
            best_context_sentence = best_context['sentence']
        else:
            best_context_sentence = ""

        # 難易度とカテゴリを判定
        difficulty = self.classify_difficulty(keyword, text)
        category = self.categorize_keyword(keyword)

        return {
            'keyword': keyword,
            'contexts': contexts,
            'best_context': best_context_sentence,
            'difficulty': difficulty,
            'category': category,
            'frequency': text.count(keyword)
        }

    def _calculate_sentence_importance(self, sentence: str, keyword: str, full_text: str) -> float:
        """文の重要度を計算"""
        importance = 0.0

        # 基準1: キーワードの位置（文頭に近いほど高スコア）
        position = sentence.index(keyword) if keyword in sentence else len(sentence)
        position_score = 1.0 - (position / len(sentence)) if len(sentence) > 0 else 0
        importance += position_score * 0.2

        # 基準2: 他のキーワードとの共起
        other_keywords = re.findall(r'[ァ-ヴー]{3,}|[A-Z]{2,}|[一-龥]{4,}', sentence)
        co_occurrence_score = min(len(other_keywords) / 5.0, 1.0)
        importance += co_occurrence_score * 0.3

        # 基準3: 説明的表現の存在
        explanatory_patterns = ['とは', 'である', 'という', 'により', 'ため']
        explanation_score = sum(1 for p in explanatory_patterns if p in sentence) / len(explanatory_patterns)
        importance += explanation_score * 0.3

        # 基準4: 文の長さ（適度な長さが理想）
        ideal_length = 50
        length_diff = abs(len(sentence) - ideal_length)
        length_score = max(0.0, 1.0 - (length_diff / ideal_length))
        importance += length_score * 0.2

        return min(importance, 1.0)

    def suggest_qa_templates(self, keywords_with_context: List[Dict]) -> List[Dict]:
        """キーワードに基づくQ&Aテンプレートを生成"""
        qa_templates = []

        for item in keywords_with_context:
            keyword = item['keyword']
            context = item['best_context']
            difficulty = item['difficulty']
            category = item['category']

            templates = []

            # 難易度とカテゴリに応じた質問テンプレート
            if difficulty == 'basic':
                templates.extend([
                    f"{keyword}とは何ですか？",
                    f"{keyword}について説明してください。"
                ])
            elif difficulty == 'intermediate':
                if category == 'core_concept':
                    templates.extend([
                        f"{keyword}の仕組みを説明してください。",
                        f"{keyword}の特徴は何ですか？"
                    ])
                else:
                    templates.extend([
                        f"{keyword}はどのように使われますか？",
                        f"{keyword}の利点は何ですか？"
                    ])
            else:  # advanced
                if category == 'technical_term':
                    templates.extend([
                        f"{keyword}の技術的な詳細を説明してください。",
                        f"{keyword}と他の手法との違いは何ですか？"
                    ])
                else:
                    templates.extend([
                        f"{keyword}の応用例を挙げてください。",
                        f"{keyword}の課題と解決策は何ですか？"
                    ])

            qa_templates.append({
                'keyword': keyword,
                'difficulty': difficulty,
                'category': category,
                'question_templates': templates,
                'answer_hint': context,
                'suggested_answer_length': {
                    'basic': '1-2文',
                    'intermediate': '2-3文',
                    'advanced': '3-5文'
                }.get(difficulty, '2-3文')
            })

        return qa_templates

    def extract_for_qa_generation(
        self,
        text: str,
        qa_count: Optional[int] = None,
        mode: str = "auto",
        difficulty_distribution: Optional[Dict[str, float]] = None,
        return_details: bool = True,
        use_progressive: bool = False
    ) -> Dict:
        """
        Q&Aペア生成に最適化されたキーワード抽出（自動Q/A数決定機能付き）

        Args:
            text: 分析対象テキスト
            qa_count: 生成するQ&Aペアの目標数（Noneで自動決定）
            mode: Q/A数決定モード ("auto", "evaluation", "00_learning.md", "search_test", "faq")
            difficulty_distribution: 難易度の分布
            return_details: 詳細情報を返すか
            use_progressive: 段階的生成を使用するか

        Returns:
            Q&A生成に必要な全情報を含む辞書
        """

        # Q/A数の自動決定
        if qa_count is None:
            optimization_result = self.qa_optimizer.calculate_optimal_qa_count(text, mode)
            qa_count = optimization_result['optimal_count']
            optimization_info = optimization_result
        else:
            optimization_info = None

        # デフォルトの難易度分布
        if difficulty_distribution is None:
            difficulty_distribution = {
                'basic': 0.3,
                'intermediate': 0.5,
                'advanced': 0.2
            }

        # 段階的生成を使用する場合
        if use_progressive:
            return self._progressive_qa_generation(
                text, qa_count, difficulty_distribution,
                optimization_info, return_details
            )

        # STEP 1: 基本キーワード抽出（多めに抽出）
        base_result = self.extract_best_auto(
            text,
            mode="auto",
            min_keywords=qa_count,
            max_keywords=qa_count * 3,
            return_analysis=False
        )

        # STEP 2: Q&A用フィルタリング
        qa_keywords = self.filter_for_qa(base_result['keywords'])

        # STEP 3: 重複除去
        qa_keywords = self.remove_redundant_keywords(qa_keywords)

        # STEP 4: 関係性抽出
        relations = self.extract_keyword_relations(text, qa_keywords)

        # STEP 5: 各キーワードの詳細情報を取得
        keywords_with_context = []
        keyword_difficulty_map = {}

        for kw in qa_keywords[:qa_count * 2]:  # 余裕を持って処理
            context_info = self.extract_with_context(text, kw)
            keywords_with_context.append(context_info)
            keyword_difficulty_map[kw] = context_info['difficulty']

        # STEP 6: 難易度分布に基づいて選択
        selected_keywords = self._select_by_difficulty_distribution(
            keywords_with_context,
            qa_count,
            difficulty_distribution
        )

        # STEP 7: Q&Aテンプレート生成
        qa_templates = self.suggest_qa_templates(selected_keywords)

        # 結果の構築
        result = {
            'keywords': selected_keywords,
            'relations': relations,
            'difficulty_map': keyword_difficulty_map,
            'suggested_qa_pairs': qa_templates,
            'metadata': {
                'total_keywords_extracted': len(qa_keywords),
                'total_relations_found': len(relations),
                'text_length': len(text),
                'extraction_method': base_result['method'],
                'qa_count': qa_count,
                'optimization_info': optimization_info
            }
        }

        if not return_details:
            # 簡潔版を返す
            return {
                'keywords': [kw['keyword'] for kw in selected_keywords],
                'qa_templates': [
                    {
                        'keyword': qa['keyword'],
                        'question': qa['question_templates'][0] if qa['question_templates'] else '',
                        'difficulty': qa['difficulty']
                    }
                    for qa in qa_templates
                ],
                'qa_count': qa_count,
                'mode': mode
            }

        return result

    def _select_by_difficulty_distribution(
        self,
        keywords_with_context: List[Dict],
        target_count: int,
        distribution: Dict[str, float]
    ) -> List[Dict]:
        """難易度分布に基づいてキーワードを選択"""

        # 難易度別にグループ化
        by_difficulty = defaultdict(list)
        for item in keywords_with_context:
            by_difficulty[item['difficulty']].append(item)

        # 各難易度から選択する数を計算
        selected = []
        for difficulty, ratio in distribution.items():
            count = int(target_count * ratio)
            if difficulty in by_difficulty:
                # 重要度順にソート
                candidates = sorted(
                    by_difficulty[difficulty],
                    key=lambda x: x['contexts'][0]['importance'] if x['contexts'] else 0,
                    reverse=True
                )
                selected.extend(candidates[:count])

        # 目標数に満たない場合は残りから補充
        if len(selected) < target_count:
            remaining = []
            for items in by_difficulty.values():
                remaining.extend([
                    item for item in items
                    if item not in selected
                ])

            # 重要度順にソート
            remaining.sort(
                key=lambda x: x['contexts'][0]['importance'] if x['contexts'] else 0,
                reverse=True
            )

            needed = target_count - len(selected)
            selected.extend(remaining[:needed])

        return selected[:target_count]

    def _progressive_qa_generation(
        self,
        text: str,
        target_count: int,
        difficulty_distribution: Dict[str, float],
        optimization_info: Optional[Dict],
        return_details: bool
    ) -> Dict:
        """
        段階的にQ/Aを生成し、収穫逓減を検出して停止

        Args:
            text: 分析対象テキスト
            target_count: 目標Q/A数
            difficulty_distribution: 難易度分布
            optimization_info: 最適化情報
            return_details: 詳細情報を返すか

        Returns:
            生成結果の辞書
        """
        all_keywords = []
        all_relations = []
        all_qa_templates = []
        coverage_scores = []

        # バッチサイズの定義（段階的に生成）
        batch_sizes = [5, 5, 10, 10, 10]
        diminishing_threshold = 0.05

        previous_coverage = 0
        total_generated = 0

        for batch_idx, batch_size in enumerate(batch_sizes):
            if total_generated >= target_count:
                break

            # 現在のバッチを生成
            current_batch_size = min(batch_size, target_count - total_generated)

            # キーワード抽出
            batch_result = self.extract_best_auto(
                text,
                mode="auto",
                min_keywords=current_batch_size,
                max_keywords=current_batch_size * 2,
                return_analysis=False
            )

            # フィルタリングと重複除去
            batch_keywords = self.filter_for_qa(batch_result['keywords'])
            batch_keywords = self.remove_redundant_keywords(batch_keywords)

            # 既存キーワードとの重複をチェック
            new_keywords = self._remove_duplicate_keywords(batch_keywords, all_keywords)

            if not new_keywords:
                # 新しいキーワードが見つからない場合は終了
                break

            # 詳細情報を取得
            keywords_with_context = []
            for kw in new_keywords[:current_batch_size]:
                context_info = self.extract_with_context(text, kw)
                keywords_with_context.append(context_info)

            # カバレッジを計算
            current_coverage = self._calculate_coverage(text, all_keywords + new_keywords)
            coverage_improvement = current_coverage - previous_coverage
            coverage_scores.append({
                'batch': batch_idx + 1,
                'coverage': current_coverage,
                'improvement': coverage_improvement,
                'keywords_added': len(new_keywords)
            })

            # 収穫逓減をチェック
            if batch_idx > 0 and coverage_improvement < diminishing_threshold:
                print(f"収穫逓減検出: 改善率 {coverage_improvement:.3f} < 閾値 {diminishing_threshold}")
                break

            # 結果を追加
            all_keywords.extend(new_keywords)
            total_generated += len(new_keywords)
            previous_coverage = current_coverage

            # 関係性を抽出
            batch_relations = self.extract_keyword_relations(text, new_keywords)
            all_relations.extend(batch_relations)

            # Q&Aテンプレートを生成
            batch_templates = self.suggest_qa_templates(keywords_with_context)
            all_qa_templates.extend(batch_templates)

        # 最終的な難易度分布に基づいて選択
        selected_keywords = self._select_by_difficulty_distribution(
            [self.extract_with_context(text, kw) for kw in all_keywords[:target_count * 2]],
            min(target_count, len(all_keywords)),
            difficulty_distribution
        )

        # 結果の構築
        result = {
            'keywords': selected_keywords,
            'relations': all_relations,
            'suggested_qa_pairs': all_qa_templates[:target_count],
            'metadata': {
                'total_keywords_extracted': len(all_keywords),
                'total_relations_found': len(all_relations),
                'text_length': len(text),
                'qa_count': min(target_count, len(all_qa_templates)),
                'optimization_info': optimization_info,
                'progressive_generation': {
                    'batches_processed': len(coverage_scores),
                    'final_coverage': previous_coverage,
                    'coverage_history': coverage_scores
                }
            }
        }

        if not return_details:
            return {
                'keywords': [kw['keyword'] for kw in selected_keywords],
                'qa_templates': [
                    {
                        'keyword': qa['keyword'],
                        'question': qa['question_templates'][0] if qa['question_templates'] else '',
                        'difficulty': qa['difficulty']
                    }
                    for qa in all_qa_templates[:target_count]
                ],
                'qa_count': min(target_count, len(all_qa_templates)),
                'final_coverage': previous_coverage
            }

        return result

    def _remove_duplicate_keywords(self, new_keywords: List[str], existing_keywords: List[str]) -> List[str]:
        """既存のキーワードと重複しない新しいキーワードを返す"""
        existing_set = set(existing_keywords)
        unique_new = []

        for kw in new_keywords:
            if kw not in existing_set:
                # 同義語チェック
                is_synonym = False
                for existing in existing_set:
                    if self._are_synonyms(kw, existing):
                        is_synonym = True
                        break

                if not is_synonym:
                    unique_new.append(kw)

        return unique_new

    def _calculate_coverage(self, text: str, keywords: List[str]) -> float:
        """キーワードのテキストカバレッジを計算"""
        if not keywords or not text:
            return 0.0

        # キーワードがカバーする文字数を計算
        covered_positions = set()

        for keyword in keywords:
            # すべての出現位置を見つける
            start = 0
            while True:
                pos = text.find(keyword, start)
                if pos == -1:
                    break
                # この位置の文字をマーク
                for i in range(pos, pos + len(keyword)):
                    covered_positions.add(i)
                start = pos + 1

        # カバレッジ率を計算
        coverage = len(covered_positions) / len(text) if len(text) > 0 else 0
        return min(1.0, coverage)

    def calculate_qa_quality_score(self, qa_pair: Dict) -> float:
        """Q/Aペアの品質スコアを計算"""
        score = 0.0

        # 質問の明確さ（疑問詞の存在）
        question_words = ['何', 'なぜ', 'どのように', 'いつ', 'どこ', 'だれ', 'どちら']
        has_question_word = any(word in qa_pair.get('question', '') for word in question_words)
        score += 0.2 if has_question_word else 0.0

        # 回答の適切な長さ
        answer_length = len(qa_pair.get('answer', ''))
        if 20 <= answer_length <= 200:
            score += 0.2
        elif 10 <= answer_length < 20 or 200 < answer_length <= 300:
            score += 0.1

        # 難易度の適切性
        difficulty = qa_pair.get('difficulty', 'intermediate')
        if difficulty in ['basic', 'intermediate', 'advanced']:
            score += 0.2

        # キーワードの含有
        keywords = qa_pair.get('keywords', [])
        if keywords:
            score += min(0.2, len(keywords) * 0.1)

        # カテゴリの明確さ
        if qa_pair.get('category'):
            score += 0.1

        # 提案された長さとの一致
        suggested_length = qa_pair.get('suggested_length', '')
        if suggested_length and self._check_length_compliance(answer_length, suggested_length):
            score += 0.1

        return min(1.0, score)

    def _check_length_compliance(self, actual_length: int, suggested_length: str) -> bool:
        """実際の長さが提案された長さに準拠しているかチェック"""
        if '1-2文' in suggested_length:
            return actual_length <= 100
        elif '2-3文' in suggested_length:
            return 50 <= actual_length <= 150
        elif '3-5文' in suggested_length:
            return 100 <= actual_length <= 250
        return True

    def detect_duplicate_qa(self, qa_pairs: List[Dict]) -> List[Tuple[int, int]]:
        """重複するQ/Aペアを検出"""
        duplicates = []

        for i in range(len(qa_pairs)):
            for j in range(i + 1, len(qa_pairs)):
                # 質問の類似度をチェック
                q1 = qa_pairs[i].get('question', '')
                q2 = qa_pairs[j].get('question', '')

                if self._are_similar_questions(q1, q2):
                    duplicates.append((i, j))

        return duplicates

    def _are_similar_questions(self, q1: str, q2: str) -> bool:
        """2つの質問が類似しているかチェック"""
        # 完全一致
        if q1 == q2:
            return True

        # キーワードの重複率
        words1 = set(q1.split())
        words2 = set(q2.split())

        if len(words1) == 0 or len(words2) == 0:
            return False

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return False

        jaccard_similarity = overlap / union
        return jaccard_similarity > 0.7

    def generate_qa_pairs(self, extraction_output: Dict) -> List[Dict]:
        """抽出結果から実際のQ&Aペアを生成"""
        qa_pairs = []

        for qa_template in extraction_output['suggested_qa_pairs']:
            keyword = qa_template['keyword']

            # 関連するキーワードを探す
            related_keywords = []
            for rel in extraction_output['relations']:
                if rel['from'] == keyword:
                    related_keywords.append(rel['to'])
                elif rel['to'] == keyword:
                    related_keywords.append(rel['from'])

            # 各質問テンプレートに対してQ&Aペアを生成
            for question in qa_template['question_templates']:
                # 回答のヒントから回答を構築
                answer_hint = qa_template['answer_hint']

                # 関連キーワードを含む回答を生成
                if related_keywords and answer_hint:
                    answer = f"{answer_hint} "
                    if qa_template['difficulty'] != 'basic':
                        answer += f"関連する概念として、{'、'.join(related_keywords[:2])}があります。"
                else:
                    answer = answer_hint if answer_hint else f"{keyword}に関する説明"

                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'difficulty': qa_template['difficulty'],
                    'category': qa_template['category'],
                    'keywords': [keyword] + related_keywords[:2],
                    'suggested_length': qa_template['suggested_answer_length']
                })

        return qa_pairs


# -----------------------------
# Classes moved from a03_rag_qa_coverage.py
# -----------------------------

# SemanticCoverage class is now imported from qa_generation.semantic for backward compatibility.


class QAGenerationConsiderations:
    """Q/A生成前のチェックリスト"""

    def analyze_document_characteristics(self, document):
        """文書の特性分析"""
        return {
            "document_type   ": self.detect_document_type(document),  # 技術文書、物語、レポート等
            "complexity_level": self.assess_complexity(document),   # 専門性のレベル
            "factual_density ": self.measure_factual_content(document),  # 事実情報の密度
            "structure       ": self.analyze_structure(document),  # 構造化の度合い
            "language        ": self.detect_language(document),     # 言語と文体
            "domain          ": self.identify_domain(document),       # ドメイン特定
            "length          ": len(document.split()),               # 文書長
            "ambiguity_level ": self.assess_ambiguity(document)  # 曖昧さの度合い
        }

    def define_qa_requirements(self):
        """Q/A生成の要件定義"""
        return {
            "purpose          ": ["評価", "学習", "検索テスト", "ユーザー支援"],
            "question_types   ": [
                "事実確認型",    # What/Who/When/Where
                "理解確認型",    # Why/How
                "推論型",       # What if/影響は
                "要約型",       # 要点は何か
                "比較型"        # 違いは何か
            ],
            "difficulty_levels": ["基礎", "中級", "上級", "専門家"],
            "answer_formats   ": ["短答", "説明", "リスト", "段落"],
            "coverage_targets ": {
                "minimum      ": 0.3,  # 最低限のカバレッジ
                "optimal      ": 0.6,  # 最適なカバレッジ
                "comprehensive": 0.8  # 包括的なカバレッジ
            }
        }


class QAPair(BaseModel):
    """Q/Aペアのデータモデル"""
    question: str
    answer: str
    question_type: str
    difficulty: str
    source_span: str


class QAPairsList(BaseModel):
    """Q/Aペアのリスト"""
    qa_pairs: List[QAPair]


class ChainOfThoughtAnalysis(BaseModel):
    """ChainOfThought分析結果モデル"""
    main_topics: List[str]
    key_concepts: List[str]
    information_density: str


class ChainOfThoughtQAPair(BaseModel):
    """ChainOfThought用Q/Aペアモデル"""
    question: str
    answer: str
    reasoning: str
    confidence: float


class ChainOfThoughtResponse(BaseModel):
    """ChainOfThought生成結果モデル"""
    analysis: ChainOfThoughtAnalysis
    qa_pairs: List[ChainOfThoughtQAPair]


class EnhancedQAPair(BaseModel):
    """LLM品質向上用のシンプルなQ/Aペアモデル"""
    question: str
    answer: str


class EnhancedQAPairsList(BaseModel):
    """LLM品質向上用のQ/Aペアリスト"""
    qa_pairs: List[EnhancedQAPair]


class LLMBasedQAGenerator:
    """LLMを使用したQ/A生成（Gemini API使用）"""

    def __init__(self, model="gemini-2.0-flash"):
        self.client = create_llm_client(provider="gemini")
        self.model = model

    def generate_basic_qa(self, text: str, num_pairs: int = 5) -> List[Dict]:
        """基本的なQ/A生成（Gemini API使用）"""

        prompt = f"""
        以下のテキストから{num_pairs}個の質問と回答のペアを生成してください。

        要件：
        1. 質問は具体的で明確にする
        2. 回答はテキストから直接答えられるものにする
        3. 質問の種類を多様にする（What/Why/How/When/Where）
        4. 回答は簡潔かつ正確にする

        テキスト：
        {text[:3000]}

        出力形式（JSON）：
        {{
            "qa_pairs": [
                {{
                    "question": "質問文",
                    "answer": "回答文",
                    "question_type": "種類（factual/reasoning/summary等）",
                    "difficulty": "難易度（easy/medium/hard）",
                    "source_span": "回答の根拠となる元テキストの一部"
                }}
            ]
        }}
        """

        try:
            # Gemini構造化出力APIを使用
            response = self.client.generate_structured(
                prompt=prompt,
                response_schema=QAPairsList,
                model=self.model
            )
            return [qa.model_dump() for qa in response.qa_pairs]
        except Exception as e:
            print(f"Q/A生成エラー: {e}")
            return []

    def generate_diverse_qa(self, text: str) -> List[Dict]:
        """多様な種類のQ/A生成（Gemini API使用）"""

        qa_types = {
            "factual"    : "事実確認の質問（Who/What/When/Where）",
            "causal"     : "因果関係の質問（Why/How）",
            "comparative": "比較の質問（違い、類似点）",
            "inferential": "推論が必要な質問",
            "summary"    : "要約を求める質問",
            "application": "応用・活用に関する質問"
        }

        all_qa_pairs = []

        for qa_type, description in qa_types.items():
            prompt = f"""
            以下のテキストから「{description}」を2個生成してください。

            テキスト：{text[:2000]}

            JSON形式で出力：
            {{"qa_pairs": [...]}}
            """

            try:
                # Gemini構造化出力APIを使用
                response = self.client.generate_structured(
                    prompt=prompt,
                    response_schema=QAPairsList,
                    model=self.model
                )
                qa_pairs = [qa.model_dump() for qa in response.qa_pairs]
                for qa in qa_pairs:
                    qa["question_type"] = qa_type
                all_qa_pairs.extend(qa_pairs)
            except Exception as e:
                print(f"Q/A生成エラー ({qa_type}): {e}")
                continue

        return all_qa_pairs


class ChainOfThoughtQAGenerator:
    """思考の連鎖を使った高品質Q/A生成（Gemini API使用）"""

    def __init__(self, model: str = "gemini-2.0-flash"):
        """
        Args:
            model: 使用するGeminiモデル（デフォルト: gemini-2.0-flash）
        """
        self.model = model
        self.client = create_llm_client(provider="gemini")

    def generate_with_reasoning(self, text: str) -> Dict:
        """推論過程付きのQ/A生成（Gemini API使用）"""

        prompt = f"""
        以下のテキストから質の高いQ/Aペアを生成します。
        各ステップを踏んで考えてください。

        ステップ1: テキストの主要なトピックと概念を抽出
        ステップ2: 各トピックについて重要な情報を特定
        ステップ3: その情報を問う質問を設計
        ステップ4: テキストから正確な回答を抽出
        ステップ5: 質問と回答の妥当性を検証

        テキスト：
        {text}
        """

        try:
            # Gemini構造化出力APIを使用
            response = self.client.generate_structured(
                prompt=prompt,
                response_schema=ChainOfThoughtResponse,
                model=self.model
            )

            # レスポンスを辞書形式に変換
            return {
                "analysis": response.analysis.model_dump(),
                "qa_pairs": [qa.model_dump() for qa in response.qa_pairs]
            }
        except Exception as e:
            print(f"ChainOfThought Q/A生成エラー: {e}")
            return {"analysis": {}, "qa_pairs": []}


class RuleBasedQAGenerator:
    """ルールベースのQ/A生成"""

    def __init__(self):
        self.nlp = spacy.load("ja_core_news_lg")

    def extract_definition_qa(self, text: str) -> List[Dict]:
        """定義文からQ/A生成"""

        qa_pairs = []

        # パターン1: 「〜とは〜である」
        pattern1 = r'([^。]+)とは([^。]+)(?:である|です)'
        matches = re.findall(pattern1, text)

        for term, definition in matches:
            qa_pairs.append({
                "question"  : f"{term.strip()}とは何ですか？",
                "answer"    : f"{term.strip()}とは{definition.strip()}です。",
                "type"      : "definition",
                "confidence": 0.9
            })

        # パターン2: 「〜は〜と呼ばれる」
        pattern2 = r'([^。]+)は([^。]+)と呼ばれ'
        matches = re.findall(pattern2, text)

        for subject, name in matches:
            qa_pairs.append({
                "question"  : f"{subject.strip()}は何と呼ばれますか？",
                "answer"    : f"{name.strip()}と呼ばれます。",
                "type"      : "terminology",
                "confidence": 0.85
            })

        return qa_pairs

    def extract_fact_qa(self, text: str) -> List[Dict]:
        """事実情報からQ/A生成"""

        doc = self.nlp(text)
        qa_pairs = []

        for sent in doc.sents:
            # 主語と動詞を含む文を対象
            subjects = [token for token in sent if token.dep_ == "nsubj"]
            verbs = [token for token in sent if token.pos_ == "VERB"]

            if subjects and verbs:
                # 日付を含む文
                dates = [ent for ent in sent.ents if ent.label_ == "DATE"]
                if dates:
                    qa_pairs.append({
                        "question"  : f"{subjects[0].text}はいつ{verbs[0].text}ましたか？",
                        "answer"    : f"{dates[0].text}です。",
                        "type"      : "temporal",
                        "confidence": 0.7
                    })

                # 場所を含む文
                locations = [ent for ent in sent.ents if ent.label_ in ["GPE", "LOC"]]
                if locations:
                    qa_pairs.append({
                        "question"  : f"{subjects[0].text}はどこで{verbs[0].text}ますか？",
                        "answer"    : f"{locations[0].text}です。",
                        "type"      : "location",
                        "confidence": 0.7
                    })

        return qa_pairs

    def extract_list_qa(self, text: str) -> List[Dict]:
        """列挙からQ/A生成"""

        qa_pairs = []

        # 「〜には、A、B、Cがある」パターン
        pattern = r'([^。]+)(?:には|に|では)、([^。]+(?:、[^。]+)+)が(?:ある|あります|含まれ|存在)'
        matches = re.findall(pattern, text)

        for topic, items in matches:
            item_list = [item.strip() for item in items.split('、')]
            qa_pairs.append({
                "question"  : f"{topic.strip()}には何がありますか？",
                "answer"    : f"{topic.strip()}には、{'、'.join(item_list)}があります。",
                "type"      : "enumeration",
                "items"     : item_list,
                "confidence": 0.8
            })

        return qa_pairs


class TemplateBasedQAGenerator:
    """テンプレートを使用したQ/A生成"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self):
        """質問テンプレートの定義"""
        return {
            "comparison"     : [
                "{A}と{B}の違いは何ですか？",
                "{A}と{B}のどちらが{property}ですか？",
                "{A}と{B}の共通点は何ですか？"
            ],
            "process"        : [
                "{process}のプロセスを説明してください。",
                "{process}にはどのようなステップがありますか？",
                "{process}の最初のステップは何ですか？"
            ],
            "cause_effect"   : [
                "{event}の原因は何ですか？",
                "{cause}の結果として何が起こりますか？",
                "なぜ{phenomenon}が発生するのですか？"
            ],
            "characteristics": [
                "{entity}の特徴は何ですか？",
                "{entity}の主な機能は何ですか？",
                "{entity}はどのように使用されますか？"
            ]
        }

    def generate_from_entities(self, text: str, entities: List[Dict]) -> List[Dict]:
        """抽出されたエンティティからQ/A生成"""

        qa_pairs = []

        for entity in entities:
            entity_type = entity['type']
            entity_text = entity['text']

            # エンティティタイプに応じたテンプレート選択
            if entity_type == "PERSON":
                questions = [
                    f"{entity_text}は誰ですか？",
                    f"{entity_text}は何をしましたか？",
                    f"{entity_text}の役割は何ですか？"
                ]
            elif entity_type == "ORG":
                questions = [
                    f"{entity_text}はどのような組織ですか？",
                    f"{entity_text}の使命は何ですか？",
                    f"{entity_text}は何を提供していますか？"
                ]
            elif entity_type in ["PRODUCT", "WORK_OF_ART"]:
                questions = [
                    f"{entity_text}の特徴は何ですか？",
                    f"{entity_text}はどのように使われますか？",
                    f"{entity_text}の利点は何ですか？"
                ]
            else:
                # 一般的なテンプレート
                questions = [
                    f"{entity_text}とは何ですか？",
                    f"{entity_text}の重要性は何ですか？",
                    f"{entity_text}の例を挙げてください。"
                ]

            # テキストから回答を探索
            for question in questions:
                answer = self.find_answer_in_text(text, entity_text, question)
                if answer:
                    qa_pairs.append({
                        "question"  : question,
                        "answer"    : answer,
                        "entity"    : entity_text,
                        "type"      : "entity_based",
                        "confidence": 0.75
                    })

        return qa_pairs


class HybridQAGenerator:
    """複数の手法を組み合わせた高度なQ/A生成"""

    def __init__(self):
        self.llm_generator = LLMBasedQAGenerator()
        self.rule_generator = RuleBasedQAGenerator()
        self.template_generator = TemplateBasedQAGenerator()

    def generate_comprehensive_qa(self, text: str,
                                  target_count: int = 20,
                                  quality_threshold: float = 0.7) -> List[Dict]:
        """包括的なQ/A生成パイプライン"""

        all_qa_pairs = []

        # Phase 1: ルールベースで確実なQ/Aを生成
        print("Phase 1: ルールベース生成...")
        rule_based_qa = []
        rule_based_qa.extend(self.rule_generator.extract_definition_qa(text))
        rule_based_qa.extend(self.rule_generator.extract_fact_qa(text))
        rule_based_qa.extend(self.rule_generator.extract_list_qa(text))

        # 高信頼度のものだけを選択
        rule_based_qa = [qa for qa in rule_based_qa
                         if qa.get('confidence', 0) >= quality_threshold]
        all_qa_pairs.extend(rule_based_qa)
        print(f"  生成数: {len(rule_based_qa)}")

        # Phase 2: テンプレートベースで補完
        print("Phase 2: テンプレートベース生成...")
        entities = self.extract_entities(text)
        template_qa = self.template_generator.generate_from_entities(text, entities)

        # 重複を除去
        template_qa = self.remove_duplicates(template_qa, all_qa_pairs)
        all_qa_pairs.extend(template_qa)
        print(f"  生成数: {len(template_qa)}")

        # Phase 3: LLMで不足分を補完
        remaining_count = target_count - len(all_qa_pairs)
        if remaining_count > 0:
            print(f"Phase 3: LLM生成（残り{remaining_count}個）...")

            # カバーされていない領域を特定
            uncovered_text = self.identify_uncovered_sections(text, all_qa_pairs)

            llm_qa = self.llm_generator.generate_diverse_qa(uncovered_text)
            llm_qa = llm_qa[:remaining_count]

            all_qa_pairs.extend(llm_qa)
            print(f"  生成数: {len(llm_qa)}")

        # Phase 4: 品質検証と改善
        print("Phase 4: 品質検証...")
        validated_qa = self.validate_and_improve_qa(all_qa_pairs, text)

        return validated_qa

    def validate_and_improve_qa(self, qa_pairs: List[Dict],
                                source_text: str) -> List[Dict]:
        """Q/Aペアの品質検証と改善"""

        validated_qa = []

        for qa in qa_pairs:
            # 検証項目
            validations = {
                "answer_found"      : self.verify_answer_in_text(
                    qa['answer'], source_text
                ),
                "question_clear"    : self.check_question_clarity(
                    qa['question']
                ),
                "no_contradiction"  : self.check_no_contradiction(
                    qa, validated_qa
                ),
                "appropriate_length": self.check_length_appropriateness(qa)
            }

            # すべての検証をパスしたものだけを採用
            if all(validations.values()):
                qa['validations'] = validations
                qa['quality_score'] = self.calculate_quality_score(qa)
                validated_qa.append(qa)

        # 品質スコアでソート
        validated_qa.sort(key=lambda x: x['quality_score'], reverse=True)

        return validated_qa


class AdvancedQAGenerationTechniques:
    """高度なQ/A生成技術"""

    def generate_adversarial_qa(self, text: str, existing_qa: List[Dict]) -> List[Dict]:
        """敵対的Q/A（システムを混乱させる質問）の生成"""

        adversarial_qa = []

        # 1. 否定質問
        for qa in existing_qa[:5]:
            adversarial_qa.append({
                "question": qa['question'].replace("何ですか", "何ではありませんか"),
                "answer"  : f"{qa['answer']}ではないものを指します。",
                "type"    : "adversarial_negation"
            })

        # 2. 文脈外質問
        adversarial_qa.append({
            "question": "この文書に書かれていない情報は何ですか？",
            "answer"  : "文書には含まれていない情報です。",
            "type"    : "out_of_context"
        })

        # 3. 曖昧な参照
        adversarial_qa.append({
            "question": "それは何を指していますか？",
            "answer"  : "文脈により異なります。",
            "type"    : "ambiguous_reference"
        })

        return adversarial_qa

    def generate_multi_hop_qa(self, text: str) -> List[Dict]:
        """マルチホップ推論が必要なQ/A生成"""


        # LLM呼び出し（実装略）
        return []

    def generate_counterfactual_qa(self, text: str) -> List[Dict]:
        """反事実的Q/A（もし〜だったら）の生成"""


        # テキストから条件と結果を抽出して適用
        return []


class QAGenerationOptimizer:
    """Q/A生成の最適化"""

    def optimize_for_coverage(self, text: str, budget: int) -> Dict:
        """カバレッジを最大化する生成戦略"""

        strategy = {
            "phase1": {
                "method"     : "rule_based",
                "target"     : "high_confidence_facts",
                "cost"       : 0,
                "expected_qa": 10
            },
            "phase2": {
                "method"     : "template_based",
                "target"     : "entities_and_concepts",
                "cost"       : 0,
                "expected_qa": 15
            },
            "phase3": {
                "method"     : "llm_cheap",
                "model"      : "gpt-3.5-turbo",
                "target"     : "gap_filling",
                "cost"       : budget * 0.3,
                "expected_qa": 20
            },
            "phase4": {
                "method"     : "llm_quality",
                "model"      : "gpt-4o",
                "target"     : "complex_reasoning",
                "cost"       : budget * 0.5,
                "expected_qa": 10
            },
            "phase5": {
                "method"     : "human_validation",
                "target"     : "quality_assurance",
                "cost"       : budget * 0.2,
                "expected_qa": "validation_only"
            }
        }

        return strategy

    def adaptive_generation(self, text: str,
                            initial_qa: List[Dict]) -> List[Dict]:
        """既存Q/Aを分析して適応的に生成"""

        # カバレッジ分析
        self.analyze_coverage(text, initial_qa)

        # 不足している質問タイプを特定
        missing_types = self.identify_missing_question_types(initial_qa)

        # ギャップを埋める新しいQ/A生成
        new_qa = []
        for missing_type in missing_types:
            new_qa.extend(
                self.generate_specific_type(text, missing_type, count=3)
            )

        return new_qa


class OptimizedHybridQAGenerator:
    """
    ハイブリッドアプローチによる最適化されたQ/A生成クラス（Gemini API使用）
    ルールベース抽出 + LLM品質向上 + 埋め込みベースカバレージ計算
    """

    def __init__(self, model: str = "gemini-2.0-flash", embedding_model: str = "gemini-embedding-001"):
        """
        Args:
            model: 使用するLLMモデル（デフォルト: gemini-2.0-flash）
            embedding_model: 埋め込みモデル（デフォルト: gemini-embedding-001）
        """
        self.client = create_llm_client(provider="gemini")
        self.embedding_client = create_embedding_client(provider="gemini")
        self.model = model
        self.embedding_model = embedding_model
        self.qa_extractor = QAOptimizedExtractor()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # サポートモデルリスト（Gemini）
        self.supported_models = [
            "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro",
            "gemini-2.5-flash", "gemini-2.5-flash-lite-preview-06-17"
        ]

        if model not in self.supported_models:
            print(f"警告: {model}は未検証です。利用可能モデル: {', '.join(self.supported_models)}")

    def generate_hybrid_qa(
        self,
        text: str,
        qa_count: int = None,
        use_llm: bool = True,
        calculate_coverage: bool = True,
        document_type: str = "auto"
    ) -> Dict:
        """
        ハイブリッドアプローチでQ/Aペアを生成

        Args:
            text: 入力テキスト
            qa_count: 生成するQ/A数（Noneで自動決定）
            use_llm: LLMによる品質向上を行うか
            calculate_coverage: カバレージ計算を行うか
            document_type: 文書タイプ（news/technical/academic/auto）

        Returns:
            生成結果の辞書
        """
        results = {
            "qa_pairs": [],
            "metadata": {},
            "coverage": {},
            "api_usage": {"calls": 0, "tokens": 0, "cost": 0.0}
        }

        # Step 1: ルールベースでキーワードとテンプレート生成
        rule_result = self.qa_extractor.extract_for_qa_generation(
            text, qa_count=qa_count, mode=document_type
        )

        # Step 2: LLMによる品質向上（オプション）
        if use_llm:
            enhanced_qa = self._enhance_with_llm(
                text, rule_result, document_type
            )
            results["qa_pairs"] = enhanced_qa["qa_pairs"]
            results["api_usage"]["calls"] += 1
            results["api_usage"]["tokens"] = enhanced_qa.get("tokens_used", 0)
            results["api_usage"]["cost"] = self._calculate_cost(
                enhanced_qa.get("tokens_used", 0)
            )
        else:
            # テンプレートからQ/Aペアを生成
            results["qa_pairs"] = self._template_to_qa(rule_result)

        # Step 3: セマンティックカバレージ計算（オプション）
        if calculate_coverage:
            coverage_result = self._calculate_semantic_coverage(
                text, results["qa_pairs"]
            )
            results["coverage"] = coverage_result
            results["api_usage"]["calls"] += coverage_result.get("embedding_calls", 0)

        # メタデータ追加
        results["metadata"] = {
            "document_type": document_type,
            "keywords_extracted": len(rule_result.get("keywords", [])),
            "qa_generated": len(results["qa_pairs"]),
            "model_used": self.model if use_llm else "rule-based",
            "hybrid_mode": use_llm
        }

        return results

    def _enhance_with_llm(self, text: str, rule_result: Dict, doc_type: str) -> Dict:
        """LLMでQ/A品質を向上（Gemini API使用）"""
        # 文書タイプ別のプロンプト調整
        type_instructions = {
            "news": "Focus on 5W1H questions (Who, What, When, Where, Why, How)",
            "technical": "Focus on How-to questions and technical details",
            "academic": "Focus on Why and What-if questions for deeper understanding",
            "auto": "Generate diverse question types appropriate for the content"
        }

        prompt = f"""Given the following text and extracted keywords, generate high-quality Q&A pairs.

Text: {text[:2000]}

Keywords and Templates:
{json.dumps(rule_result.get('suggested_qa_pairs', [])[:10], ensure_ascii=False, indent=2)}

Instructions:
1. {type_instructions.get(doc_type, type_instructions['auto'])}
2. Make questions specific and answers comprehensive
3. Ensure factual accuracy based on the text
4. Generate {len(rule_result.get('suggested_qa_pairs', [])[:5])} Q&A pairs
"""

        try:
            # Gemini構造化出力APIを使用
            response = self.client.generate_structured(
                prompt=prompt,
                response_schema=EnhancedQAPairsList,
                model=self.model
            )

            qa_pairs = [qa.model_dump() for qa in response.qa_pairs]

            return {
                "qa_pairs": qa_pairs,
                "tokens_used": 0  # Gemini APIはトークン数を直接返さない
            }
        except Exception as e:
            print(f"LLM enhancement failed: {e}")
            return {"qa_pairs": self._template_to_qa(rule_result), "tokens_used": 0}

    def _template_to_qa(self, rule_result: Dict) -> List[Dict]:
        """テンプレートからQ/Aペアを生成"""
        qa_pairs = []
        for item in rule_result.get("suggested_qa_pairs", [])[:5]:
            if item.get("question_templates"):
                # より詳細な回答を生成
                keyword = item['keyword']
                context = item.get('answer_hint', '')
                difficulty = item.get('difficulty', 'intermediate')

                # 難易度に応じた回答を生成
                if difficulty == 'basic':
                    answer = f"{keyword}は、{context[:100] if context else '文書内で言及されている重要な概念です。'}"
                elif difficulty == 'intermediate':
                    answer = f"{keyword}について、{context[:150] if context else '文書では詳細に説明されており、その特徴や使用方法が示されています。'}"
                else:  # advanced
                    answer = f"{keyword}の{context[:200] if context else '技術的な詳細や応用例が文書内で議論されています。'}"

                qa_pairs.append({
                    "question": item["question_templates"][0],
                    "answer": answer
                })
        return qa_pairs

    def _calculate_semantic_coverage(self, text: str, qa_pairs: List[Dict]) -> Dict:
        """
        セマンティックカバレージを多段階閾値で計算（TODO-B3実装）

        3つの閾値レベルでカバレージを評価:
        - Strict (0.85): 専門用語レベルの完全一致
        - Standard (0.80): 現行デフォルト
        - Lenient (0.75): 関連性重視
        """
        try:
            # テキストのチャンク化
            chunks = self._create_semantic_chunks(text, chunk_size=200)

            # 埋め込み生成
            chunk_embeddings = self._get_embeddings([c["text"] for c in chunks])
            qa_texts = [f"{qa['question']} {qa['answer']}" for qa in qa_pairs]
            qa_embeddings = self._get_embeddings(qa_texts)

            # カバレージ計算
            coverage_scores = []
            for chunk_emb in chunk_embeddings:
                max_similarity = 0
                for qa_emb in qa_embeddings:
                    similarity = self._cosine_similarity(chunk_emb, qa_emb)
                    max_similarity = max(max_similarity, similarity)
                coverage_scores.append(max_similarity)

            # 多段階閾値評価（TODO-B3）
            thresholds = {
                "strict": 0.85,    # 厳格（専門用語完全一致レベル）
                "standard": 0.80,  # 標準（現行）
                "lenient": 0.75    # 緩和（関連性あり）
            }

            # ルールベースQ/Aの場合は閾値を調整
            is_rule_based = any("文書内で" in qa.get("answer", "") or
                               "文書では" in qa.get("answer", "") or
                               "重要な概念" in qa.get("answer", "") for qa in qa_pairs)

            if is_rule_based:
                # ルールベースの場合は全体的に閾値を下げる
                thresholds = {
                    "strict": 0.50,
                    "standard": 0.40,
                    "lenient": 0.30
                }

            # 各閾値でカバレージを計算
            multi_threshold_results = {}
            for threshold_name, threshold_value in thresholds.items():
                covered = sum(1 for score in coverage_scores if score >= threshold_value)
                multi_threshold_results[threshold_name] = {
                    "threshold": threshold_value,
                    "covered_chunks": covered,
                    "coverage_percentage": (covered / len(chunks)) * 100 if chunks else 0
                }

            # 標準閾値での結果を主要な結果とする
            primary_threshold = "standard"
            covered_chunks = multi_threshold_results[primary_threshold]["covered_chunks"]
            coverage_percentage = multi_threshold_results[primary_threshold]["coverage_percentage"]

            # カバレージ分布統計
            coverage_distribution = {
                "excellent": sum(1 for s in coverage_scores if s >= 0.90),  # 90%以上
                "good": sum(1 for s in coverage_scores if 0.80 <= s < 0.90),  # 80-90%
                "fair": sum(1 for s in coverage_scores if 0.70 <= s < 0.80),  # 70-80%
                "poor": sum(1 for s in coverage_scores if 0.60 <= s < 0.70),  # 60-70%
                "uncovered": sum(1 for s in coverage_scores if s < 0.60)     # 60%未満
            }

            return {
                "total_chunks": len(chunks),
                "covered_chunks": covered_chunks,
                "coverage_percentage": coverage_percentage,
                "average_similarity": float(np.mean(coverage_scores)) if coverage_scores else 0,
                "median_similarity": float(np.median(coverage_scores)) if coverage_scores else 0,
                "min_similarity": float(np.min(coverage_scores)) if coverage_scores else 0,
                "max_similarity": float(np.max(coverage_scores)) if coverage_scores else 0,
                "multi_threshold_coverage": multi_threshold_results,
                "coverage_distribution": coverage_distribution,
                "quality_score": self._calculate_coverage_quality_score(
                    coverage_scores, multi_threshold_results
                ),
                "embedding_calls": 2,  # chunks + qa_pairs
                "is_rule_based": is_rule_based
            }
        except Exception as e:
            print(f"Coverage calculation failed: {e}")
            return {"coverage_percentage": 0, "embedding_calls": 0}

    def _calculate_coverage_quality_score(
        self,
        coverage_scores: List[float],
        multi_threshold_results: Dict
    ) -> float:
        """
        カバレージ品質スコアを計算（0.0-1.0）

        考慮要素:
        - 平均類似度（40%）
        - Strictカバレージ率（30%）
        - カバレージの均一性（30%）
        """
        if not coverage_scores:
            return 0.0

        # 平均類似度スコア（0.0-1.0）
        avg_similarity = np.mean(coverage_scores)
        avg_score = avg_similarity * 0.4

        # Strictカバレージ率スコア
        strict_coverage = multi_threshold_results["strict"]["coverage_percentage"] / 100
        strict_score = strict_coverage * 0.3

        # カバレージの均一性（標準偏差が小さいほど良い）
        std_dev = np.std(coverage_scores)
        uniformity = max(0.0, 1.0 - std_dev)  # 標準偏差1.0で0点
        uniformity_score = uniformity * 0.3

        total_score = avg_score + strict_score + uniformity_score
        return min(1.0, total_score)

    def _create_semantic_chunks(self, text: str, chunk_size: int = 200) -> List[Dict]:
        """
        テキストをセマンティックチャンクに分割（段落優先・MeCab対応）

        SemanticCoverage.create_semantic_chunks()を使用して、
        段落境界を最優先したセマンティック分割を実行。
        日本語の場合はMeCabによる高精度な文境界検出を自動適用。
        """
        # SemanticCoverageインスタンスを使用
        semantic_analyzer = SemanticCoverage(embedding_model=self.embedding_model)

        # 段落優先のセマンティック分割を実行
        semantic_chunks = semantic_analyzer.create_semantic_chunks(
            document=text,
            max_tokens=chunk_size,
            min_tokens=50,  # 最小トークン数
            prefer_paragraphs=True,  # 段落優先モード
            verbose=False
        )

        # SemanticCoverageの出力形式をBatchHybridQAGeneratorの形式に変換
        chunks = []
        for semantic_chunk in semantic_chunks:
            chunk_text = semantic_chunk['text']
            chunk_tokens = len(self.tokenizer.encode(chunk_text))

            chunks.append({
                "text": chunk_text,
                "tokens": chunk_tokens,
                "type": semantic_chunk.get('type', 'unknown'),  # paragraph/sentence_group/forced_split
                "sentences": semantic_chunk.get('sentences', [])
            })

        return chunks

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """テキストの埋め込みを取得（Gemini API使用）"""
        try:
            embeddings = self.embedding_client.embed_texts(texts)
            return embeddings
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """コサイン類似度を計算"""
        if not vec1 or not vec2:
            return 0.0

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _calculate_cost(self, tokens: int) -> float:
        """API使用コストを計算（Gemini料金）"""
        # Geminiモデル別の料金（1Mトークンあたり、USD）
        # https://ai.google.dev/pricing
        pricing = {
            "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
            "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
            "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
            "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
            "gemini-2.5-flash-lite-preview-06-17": {"input": 0.075, "output": 0.30},
        }

        model_pricing = pricing.get(self.model, pricing["gemini-2.0-flash"])
        # 簡易計算（入力:出力 = 7:3の仮定）
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)

        cost = (input_tokens * model_pricing["input"] +
                output_tokens * model_pricing["output"]) / 1_000_000

        return round(cost, 4)


class BatchHybridQAGenerator(OptimizedHybridQAGenerator):
    """
    バッチ処理に最適化されたハイブリッドQ/A生成クラス（Gemini API使用）
    API呼び出しを大幅に削減し、処理を高速化
    品質重視モード追加（カバレージ95%目標）
    """

    def __init__(self,
                 model: str = "gemini-2.0-flash",
                 embedding_model: str = "gemini-embedding-001",
                 batch_size: int = 10,
                 embedding_batch_size: int = 100,
                 quality_mode: bool = False,
                 target_coverage: float = 0.95):
        """
        Args:
            model: 使用するLLMモデル（デフォルト: gemini-2.0-flash）
            embedding_model: 埋め込みモデル（デフォルト: gemini-embedding-001）
            batch_size: LLM処理のバッチサイズ
            embedding_batch_size: 埋め込み処理のバッチサイズ
            quality_mode: 品質重視モード（True: カバレージ優先、False: 効率優先）
            target_coverage: 目標カバレージ率（デフォルト95%）
        """
        super().__init__(model, embedding_model)
        self.batch_size = batch_size if not quality_mode else min(5, batch_size)  # 品質モードでは最大5
        self.embedding_batch_size = embedding_batch_size
        self.quality_mode = quality_mode
        self.target_coverage = target_coverage

        # バッチ処理統計
        self.batch_stats = {
            "llm_batches": 0,
            "embedding_batches": 0,
            "total_llm_calls": 0,
            "total_embedding_calls": 0,
            "coverage_iterations": 0
        }

    def calculate_optimal_qa_count(self, text: str, target_coverage: float = 0.95) -> int:
        """
        文書の複雑度に基づいて最適なQ/A数を動的に決定

        Args:
            text: 対象文書
            target_coverage: 目標カバレージ率

        Returns:
            最適なQ/A数
        """
        # 文書の特徴分析
        text_length = len(text)
        re.split(r'[。！？\.\!\?]+', text)
        unique_concepts = len(set(re.findall(r'[ァ-ヴー]{3,}|[A-Z]{2,}|[一-龥]{4,}', text)))
        information_density = unique_concepts / (text_length / 1000) if text_length > 0 else 0

        # 基本Q/A数の計算
        if information_density > 10:  # 高密度
            base_qa_count = 8
        elif information_density > 5:  # 中密度
            base_qa_count = 6
        else:  # 低密度
            base_qa_count = 4

        # チャンク数に基づく調整
        chunks = self._create_semantic_chunks(text)
        chunk_adjustment = len(chunks) * 0.5

        # 品質モードでの調整
        if self.quality_mode:
            base_qa_count = int(base_qa_count * 1.5)  # 50%増加

        optimal_count = int(base_qa_count + chunk_adjustment)
        return min(optimal_count, 15)  # 上限15個

    def generate_hierarchical_qa(self, text: str, lang: str = "en") -> List[Dict]:
        """
        階層的アプローチでQ/Aを生成（3段階）

        Args:
            text: 対象テキスト
            lang: 言語コード

        Returns:
            階層的に生成されたQ/Aペア
        """
        all_qa_pairs = []

        # 第1層: 文書全体の包括的質問（1-2個）
        if lang == "ja":
            comprehensive_qa = [
                {
                    "question": "この文書の主要な内容を要約してください",
                    "answer": text[:500] if len(text) > 500 else text,
                    "type": "comprehensive",
                    "layer": 1
                }
            ]
        else:
            comprehensive_qa = [
                {
                    "question": "What is the main content of this document?",
                    "answer": text[:500] if len(text) > 500 else text,
                    "type": "comprehensive",
                    "layer": 1
                }
            ]
        all_qa_pairs.extend(comprehensive_qa)

        # 第2層: パラグラフレベルの詳細質問（3-4個）
        paragraphs = text.split('\n\n') if '\n\n' in text else [text[i:i+500] for i in range(0, len(text), 400)]
        paragraph_qa = []

        for i, para in enumerate(paragraphs[:4]):
            if len(para.strip()) > 50:
                if lang == "ja":
                    qa = {
                        "question": f"「{para[:30]}...」について説明してください",
                        "answer": para,
                        "type": "paragraph_detail",
                        "layer": 2
                    }
                else:
                    qa = {
                        "question": f"Explain the following: '{para[:30]}...'",
                        "answer": para,
                        "type": "paragraph_detail",
                        "layer": 2
                    }
                paragraph_qa.append(qa)
        all_qa_pairs.extend(paragraph_qa)

        # 第3層: キーワード/エンティティ特化質問（5-6個）
        # キーワード抽出
        if lang == "ja":
            pattern = r'[ァ-ヴー]{3,}|[一-龥]{4,}'
        else:
            pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|[A-Z]{2,}'

        entities = list(set(re.findall(pattern, text)))[:6]
        entity_qa = []

        for entity in entities:
            # エンティティの文脈を抽出
            context_match = re.search(r'[^。.!?]*' + re.escape(entity) + r'[^。.!?]*', text)
            if context_match:
                context = context_match.group(0)
                if lang == "ja":
                    qa = {
                        "question": f"「{entity}」について何が述べられていますか？",
                        "answer": context,
                        "type": "entity_specific",
                        "layer": 3
                    }
                else:
                    qa = {
                        "question": f"What is mentioned about {entity}?",
                        "answer": context,
                        "type": "entity_specific",
                        "layer": 3
                    }
                entity_qa.append(qa)
        all_qa_pairs.extend(entity_qa)

        return all_qa_pairs

    def generate_with_coverage_feedback(
        self,
        text: str,
        target_coverage: float = 0.95,
        max_iterations: int = 3,
        lang: str = "en"
    ) -> Dict:
        """
        カバレージフィードバックループでQ/Aを生成

        Args:
            text: 入力テキスト
            target_coverage: 目標カバレージ率
            max_iterations: 最大反復回数
            lang: 言語コード

        Returns:
            生成結果の辞書
        """
        qa_pairs = []
        current_coverage = 0
        iteration = 0
        uncovered_chunks = []

        # 初回は階層的生成
        if self.quality_mode:
            qa_pairs = self.generate_hierarchical_qa(text, lang)

        while current_coverage < target_coverage and iteration < max_iterations:
            iteration += 1
            self.batch_stats["coverage_iterations"] = iteration

            # カバレージ計算
            if qa_pairs:
                coverage_result = self._calculate_semantic_coverage(text, qa_pairs)
                current_coverage = coverage_result['coverage_percentage'] / 100

                # 未カバーチャンクを特定
                if current_coverage < target_coverage:
                    uncovered_chunks = self._identify_uncovered_chunks(
                        text, qa_pairs, coverage_result
                    )
            else:
                # 初回は通常の生成
                optimal_count = self.calculate_optimal_qa_count(text, target_coverage)
                rule_result = self.qa_extractor.extract_for_qa_generation(
                    text, qa_count=optimal_count, mode="auto"
                )
                qa_pairs = self._template_to_qa(rule_result)
                continue

            if current_coverage >= target_coverage:
                break

            # 未カバー領域に対してQ/A生成
            additional_qa = self._generate_targeted_qa(
                uncovered_chunks, text, lang
            )
            qa_pairs.extend(additional_qa)

            print(f"反復 {iteration}: カバレージ = {current_coverage:.1%}, "
                  f"Q/A総数 = {len(qa_pairs)}")

        return {
            "qa_pairs": qa_pairs,
            "final_coverage": current_coverage,
            "iterations": iteration,
            "total_qa": len(qa_pairs)
        }

    def _identify_uncovered_chunks(
        self,
        text: str,
        qa_pairs: List[Dict],
        coverage_result: Dict
    ) -> List[str]:
        """未カバーのチャンクを特定"""
        chunks = self._create_semantic_chunks(text)

        # 各チャンクのカバレージスコアを取得
        chunk_embeddings = self._get_embeddings([c["text"] for c in chunks])
        qa_texts = [f"{qa['question']} {qa['answer']}" for qa in qa_pairs]
        qa_embeddings = self._get_embeddings(qa_texts)

        uncovered = []
        threshold = 0.65 if self.quality_mode else 0.7

        for i, chunk in enumerate(chunks):
            max_similarity = 0
            for qa_emb in qa_embeddings:
                similarity = self._cosine_similarity(chunk_embeddings[i], qa_emb)
                max_similarity = max(max_similarity, similarity)

            if max_similarity < threshold:
                uncovered.append(chunk["text"])

        return uncovered

    def _generate_targeted_qa(
        self,
        uncovered_chunks: List[str],
        full_text: str,
        lang: str
    ) -> List[Dict]:
        """未カバー領域に焦点を当てたQ/A生成"""
        targeted_qa = []

        for chunk in uncovered_chunks[:5]:  # 最大5チャンク
            # チャンクから重要なキーワードを抽出
            if lang == "ja":
                pattern = r'[ァ-ヴー]{3,}|[一-龥]{4,}'
                question_template = "「{}」について詳しく説明してください"
            else:
                pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|[A-Z]{2,}'
                question_template = "Please explain in detail about {}"

            keywords = re.findall(pattern, chunk)
            if keywords:
                keyword = keywords[0]
                qa = {
                    "question": question_template.format(keyword),
                    "answer": chunk,
                    "type": "targeted_coverage"
                }
                targeted_qa.append(qa)

        return targeted_qa

    def generate_batch_hybrid_qa(
        self,
        texts: List[str],
        qa_count: int = None,
        use_llm: bool = True,
        calculate_coverage: bool = True,
        document_type: str = "auto",
        show_progress: bool = True,
        lang: str = "en"
    ) -> List[Dict]:
        """
        複数文書をバッチ処理でQ/A生成

        Args:
            texts: 入力テキストのリスト
            qa_count: 各文書のQ/A数
            use_llm: LLMを使用するか
            calculate_coverage: カバレージ計算するか
            document_type: 文書タイプ
            show_progress: 進捗表示
            lang: 言語コード（"en" または "ja"）

        Returns:
            各文書の生成結果リスト
        """
        from tqdm import tqdm

        all_results = []
        total_docs = len(texts)

        # Step 1: ルールベースでバッチ処理（既に高速）
        if show_progress:
            print("Step 1: ルールベース抽出...")

        rule_results = []
        for text in tqdm(texts, desc="ルールベース", disable=not show_progress):
            rule_result = self.qa_extractor.extract_for_qa_generation(
                text, qa_count=qa_count, mode=document_type
            )
            rule_results.append(rule_result)

        # Step 2: LLMバッチ処理（オプション）
        if use_llm:
            if show_progress:
                print(f"\nStep 2: LLM品質向上（バッチサイズ: {self.batch_size}）...")

            enhanced_qa_results = self._batch_enhance_with_llm(
                texts, rule_results, document_type, show_progress, lang
            )
        else:
            # テンプレートからQ/A生成
            enhanced_qa_results = []
            for rule_result in rule_results:
                qa_pairs = self._template_to_qa(rule_result)
                enhanced_qa_results.append({"qa_pairs": qa_pairs, "tokens_used": 0})

        # Step 3: カバレージ計算（バッチ処理）
        coverage_results = []
        if calculate_coverage:
            if show_progress:
                print(f"\nStep 3: カバレージ計算（埋め込みバッチサイズ: {self.embedding_batch_size}）...")

            coverage_results = self._batch_calculate_coverage(
                texts, [r["qa_pairs"] for r in enhanced_qa_results], show_progress
            )

        # 結果の統合
        for i, text in enumerate(texts):
            result = {
                "qa_pairs": enhanced_qa_results[i]["qa_pairs"],
                "metadata": {
                    "document_type": document_type,
                    "keywords_extracted": len(rule_results[i].get("keywords", [])),
                    "qa_generated": len(enhanced_qa_results[i]["qa_pairs"]),
                    "model_used": self.model if use_llm else "rule-based",
                    "hybrid_mode": use_llm
                },
                "coverage": coverage_results[i] if calculate_coverage else {},
                "api_usage": {
                    "calls": 0,  # バッチ処理後に更新
                    "tokens": enhanced_qa_results[i].get("tokens_used", 0),
                    "cost": self._calculate_cost(enhanced_qa_results[i].get("tokens_used", 0))
                }
            }
            all_results.append(result)

        # バッチ処理統計を追加
        if show_progress:
            self._print_batch_statistics(total_docs)

        return all_results

    def _batch_enhance_with_llm(
        self,
        texts: List[str],
        rule_results: List[Dict],
        doc_type: str,
        show_progress: bool,
        lang: str = "en"
    ) -> List[Dict]:
        """LLMでバッチ処理によるQ/A品質向上（Gemini API使用）"""
        from tqdm import tqdm

        enhanced_results = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        progress_bar = tqdm(total=len(texts), desc="LLM処理", disable=not show_progress)

        for batch_idx in range(0, len(texts), self.batch_size):
            batch_end = min(batch_idx + self.batch_size, len(texts))
            batch_texts = texts[batch_idx:batch_end]
            batch_rules = rule_results[batch_idx:batch_end]

            # バッチプロンプト作成（言語情報を渡す）
            batch_prompt = self._create_batch_prompt(batch_texts, batch_rules, doc_type, lang)

            try:
                # Gemini API で generate_content を使用（JSON出力）
                system_instruction = "You are a Q&A generation expert. Process multiple documents. Always respond with valid JSON."

                response_text = self.client.generate_content(
                    prompt=batch_prompt,
                    model=self.model,
                    system_instruction=system_instruction,
                    temperature=0.7
                )

                self.batch_stats["llm_batches"] += 1
                self.batch_stats["total_llm_calls"] += 1

                # バッチ応答のパース（Gemini形式）
                batch_results = self._parse_batch_response_gemini(response_text, len(batch_texts))

                # 結果が足りない場合は個別処理にフォールバック
                while len(batch_results) < len(batch_texts):
                    batch_results.append({
                        "qa_pairs": self._template_to_qa(batch_rules[len(batch_results)]),
                        "tokens_used": 0
                    })

                enhanced_results.extend(batch_results)

            except Exception as e:
                print(f"バッチ {batch_idx//self.batch_size + 1}/{total_batches} でエラー: {e}")
                # エラー時は個別処理にフォールバック
                for i in range(len(batch_texts)):
                    qa_pairs = self._template_to_qa(batch_rules[i])
                    enhanced_results.append({"qa_pairs": qa_pairs, "tokens_used": 0})

            progress_bar.update(len(batch_texts))

        progress_bar.close()
        return enhanced_results

    def _create_batch_prompt(
        self,
        texts: List[str],
        rule_results: List[Dict],
        doc_type: str,
        lang: str = "en",
        uncovered_chunks: List[List[str]] = None
    ) -> str:
        """
        バッチ処理用の高品質プロンプト作成

        Args:
            texts: 文書リスト
            rule_results: ルールベース抽出結果
            doc_type: 文書タイプ
            lang: 言語コード
            uncovered_chunks: 未カバーチャンク（マルチパス時）
        """
        # 言語別の指示文（高品質化）
        if lang == "ja":
            type_instructions = {
                "news": """
5W1H（いつ、どこで、誰が、何を、なぜ、どのように）に焦点を当てた質問を生成してください。
特に重要な事実、統計データ、固有名詞、結論を必ず含めること。""",
                "technical": """
How-to（やり方・方法）と技術的詳細に焦点を当てた質問を生成してください。
専門用語、プロセス、因果関係、制約条件を明確に含めること。""",
                "academic": """
Why（理由・原因）、What-if（仮定）、概念間の関係性に焦点を当てた質問を生成してください。
理論、根拠、研究結果、意義を必ず含めること。""",
                "auto": """
多様な種類の質問（事実、理由、方法、影響、比較、評価）を生成してください。
文書の主要概念と重要な詳細情報をバランスよく網羅すること。"""
            }
            language_instruction = "**重要**: 質問と回答は必ず日本語で生成してください。"
            process_text = f"以下の{len(texts)}件の文書を処理し、高品質なQ&Aペアを生成してください。"
            instruction_text = "指示"
            output_format_text = "出力形式（JSON）"
            quality_guidelines = """
【品質ガイドライン】
- 回答は文書から直接引用または要約し、ハルシネーションを避けること
- 質問は明確で具体的にすること（曖昧な「これ」「それ」を避ける）
- 重要度の高い情報を優先的に質問化すること
- 文書の異なる部分を網羅するよう質問を分散させること
- 回答が文書内に明確に存在する質問のみを生成すること"""
        else:
            type_instructions = {
                "news": """
Focus on 5W1H questions (When, Where, Who, What, Why, How).
Must include important facts, statistics, named entities, and conclusions.""",
                "technical": """
Focus on How-to and technical details.
Clearly include technical terms, processes, causal relationships, and constraints.""",
                "academic": """
Focus on Why, What-if, and relationships between concepts.
Must include theories, rationale, research findings, and significance.""",
                "auto": """
Generate diverse question types (factual, reasoning, method, impact, comparison, evaluation).
Balance coverage of main concepts and important details."""
            }
            language_instruction = "**IMPORTANT**: Generate questions and answers in English."
            process_text = f"Process these {len(texts)} documents and generate high-quality Q&A pairs."
            instruction_text = "Instructions"
            output_format_text = "Output format (JSON)"
            quality_guidelines = """
【Quality Guidelines】
- Answers must be directly quoted or summarized from the document (avoid hallucination)
- Questions should be clear and specific (avoid vague "this", "that")
- Prioritize high-importance information
- Distribute questions across different parts of the document
- Generate only questions whose answers clearly exist in the document"""

        documents = []
        for i, (text, rule_result) in enumerate(zip(texts, rule_results)):
            doc_info = {
                "document_id": i,
                "text": text[:1500],  # より長いコンテキスト（1000→1500）
                "keywords": rule_result.get("suggested_qa_pairs", [])[:8],  # より多くのキーワード（5→8）
                "key_entities": self._extract_key_entities(text, lang)[:5]
            }

            # マルチパス時は未カバー領域を明示
            if uncovered_chunks and i < len(uncovered_chunks) and uncovered_chunks[i]:
                doc_info["uncovered_areas"] = uncovered_chunks[i][:3]

            documents.append(doc_info)

        # Few-shot examples追加
        if lang == "ja":
            examples = """
【良い例】
Q: 「この研究の主要な発見は何ですか？」
A: 「研究の主要な発見は、X因子がY結果に対して統計的に有意な影響（p<0.05）を示したことです。」

Q: 「記事で言及されている3つの課題は何ですか？」
A: 「記事では以下の3つの課題が言及されています：1) コスト削減、2) 品質向上、3) 納期短縮。」

【避けるべき例】
Q: 「これについて説明してください」（曖昧）
Q: 「何が重要ですか？」（一般的すぎる）
"""
        else:
            examples = """
【Good Examples】
Q: "What are the main findings of this research?"
A: "The main findings show that factor X had a statistically significant impact (p<0.05) on outcome Y."

Q: "What are the three challenges mentioned in the article?"
A: "The article mentions three challenges: 1) cost reduction, 2) quality improvement, 3) shorter delivery time."

【Examples to Avoid】
Q: "Explain this" (too vague)
Q: "What is important?" (too general)
"""

        prompt = f"""{process_text}

Documents:
{json.dumps(documents, ensure_ascii=False, indent=2)}

{instruction_text}:
1. {type_instructions.get(doc_type, type_instructions['auto'])}
2. Generate 4-6 Q&A pairs per document (adjust based on document complexity)
3. Ensure factual accuracy and avoid hallucination
4. {language_instruction}

{quality_guidelines}

{examples}

IMPORTANT: Return your response in JSON format.

{output_format_text}:
{{
    "results": [
        {{
            "document_id": 0,
            "qa_pairs": [
                {{"question": "...", "answer": "...", "importance": "high|medium|low"}}
            ]
        }},
        ...
    ]
}}"""

        return prompt

    def _extract_key_entities(self, text: str, lang: str) -> List[str]:
        """重要なエンティティを抽出"""
        if lang == "ja":
            # 日本語: カタカナ3文字以上、漢字4文字以上
            pattern = r'[ァ-ヴー]{3,}|[一-龥]{4,}'
        else:
            # 英語: 大文字始まりの固有名詞、頭字語
            pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|[A-Z]{2,}'

        entities = re.findall(pattern, text)
        # 頻出エンティティを優先
        from collections import Counter
        entity_counts = Counter(entities)
        return [entity for entity, count in entity_counts.most_common(10)]

    def _parse_batch_response_gemini(self, response_text: str, expected_count: int) -> List[Dict]:
        """Gemini APIのバッチ応答をパース"""
        try:
            # JSONブロックを抽出（```json ... ``` または直接JSON）
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 直接JSONの場合
                json_str = response_text.strip()

            parsed = json.loads(json_str)

            results = []
            for doc_result in parsed.get("results", []):
                results.append({
                    "qa_pairs": doc_result.get("qa_pairs", []),
                    "tokens_used": 0  # Gemini APIはトークン数を直接返さない
                })

            return results

        except Exception as e:
            print(f"Geminiバッチ応答のパースエラー: {e}")
            return []

    def _batch_calculate_coverage(
        self,
        texts: List[str],
        qa_pairs_list: List[List[Dict]],
        show_progress: bool
    ) -> List[Dict]:
        """バッチ処理でカバレージ計算"""

        all_coverages = []

        # すべてのチャンクとQ/Aペアを収集
        all_chunks = []
        chunk_boundaries = []  # 各文書のチャンク境界を記録

        for text in texts:
            chunks = self._create_semantic_chunks(text, chunk_size=200)
            chunk_boundaries.append((len(all_chunks), len(all_chunks) + len(chunks)))
            all_chunks.extend(chunks)

        all_qa_texts = []
        qa_boundaries = []  # 各文書のQ/A境界を記録

        for qa_pairs in qa_pairs_list:
            qa_texts = [f"{qa['question']} {qa['answer']}" for qa in qa_pairs]
            qa_boundaries.append((len(all_qa_texts), len(all_qa_texts) + len(qa_texts)))
            all_qa_texts.extend(qa_texts)

        # バッチ処理で埋め込み生成
        if show_progress:
            print(f"埋め込み生成中... (チャンク: {len(all_chunks)}, Q/A: {len(all_qa_texts)})")

        chunk_embeddings = self._batch_get_embeddings(
            [c["text"] for c in all_chunks], "チャンク", show_progress
        )

        qa_embeddings = self._batch_get_embeddings(
            all_qa_texts, "Q/A", show_progress
        )

        # 各文書のカバレージ計算
        for i, text in enumerate(texts):
            chunk_start, chunk_end = chunk_boundaries[i]
            qa_start, qa_end = qa_boundaries[i]

            doc_chunk_embs = chunk_embeddings[chunk_start:chunk_end]
            doc_qa_embs = qa_embeddings[qa_start:qa_end]

            # カバレージ計算
            coverage_scores = []
            for chunk_emb in doc_chunk_embs:
                max_similarity = 0
                for qa_emb in doc_qa_embs:
                    similarity = self._cosine_similarity(chunk_emb, qa_emb)
                    max_similarity = max(max_similarity, similarity)
                coverage_scores.append(max_similarity)

            # 閾値判定
            is_rule_based = any("文書内で" in qa.get("answer", "") or
                               "文書では" in qa.get("answer", "") or
                               "重要な概念" in qa.get("answer", "")
                               for qa in qa_pairs_list[i])
            threshold = 0.4 if is_rule_based else 0.7

            covered_chunks = sum(1 for score in coverage_scores if score >= threshold)

            all_coverages.append({
                "total_chunks": len(doc_chunk_embs),
                "covered_chunks": covered_chunks,
                "coverage_percentage": (covered_chunks / len(doc_chunk_embs)) * 100 if doc_chunk_embs else 0,
                "average_similarity": np.mean(coverage_scores) if coverage_scores else 0,
                "embedding_calls": 0  # バッチ処理のため個別カウントなし
            })

        return all_coverages

    def _batch_get_embeddings(
        self,
        texts: List[str],
        desc: str,
        show_progress: bool
    ) -> List[List[float]]:
        """バッチ処理で埋め込みを取得（Gemini API使用）"""
        from tqdm import tqdm

        embeddings = []

        progress_bar = tqdm(
            total=len(texts),
            desc=f"{desc}埋め込み",
            disable=not show_progress
        )

        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i:i + self.embedding_batch_size]

            try:
                # Gemini Embedding APIを使用
                batch_embeddings = self.embedding_client.embed_texts(batch)

                self.batch_stats["embedding_batches"] += 1
                self.batch_stats["total_embedding_calls"] += 1

                embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"埋め込み生成エラー: {e}")
                # エラー時はゼロベクトル（Gemini embeddingは3072次元）
                embedding_dims = get_embedding_dimensions("gemini")
                for _ in batch:
                    embeddings.append([0.0] * embedding_dims)

            progress_bar.update(len(batch))

        progress_bar.close()
        return embeddings

    def _print_batch_statistics(self, total_docs: int):
        """バッチ処理統計を表示"""
        print("\n" + "=" * 80)
        print("📊 バッチ処理統計")
        print("=" * 80)
        print(f"処理文書数: {total_docs}")
        print("\nLLM処理:")
        print(f"  - バッチ数: {self.batch_stats['llm_batches']}")
        print(f"  - API呼び出し: {self.batch_stats['total_llm_calls']}回")
        print(f"  - 削減率: {(1 - self.batch_stats['total_llm_calls']/max(1, total_docs)) * 100:.1f}%")

        print("\n埋め込み処理:")
        print(f"  - バッチ数: {self.batch_stats['embedding_batches']}")
        print(f"  - API呼び出し: {self.batch_stats['total_embedding_calls']}回")

        total_calls = self.batch_stats['total_llm_calls'] + self.batch_stats['total_embedding_calls']
        original_calls = total_docs + total_docs * 2  # 個別処理の場合

        print("\n総合:")
        print(f"  - 総API呼び出し: {total_calls}回")
        print(f"  - 従来方式: {original_calls}回")
        print(f"  - 削減率: {(1 - total_calls/max(1, original_calls)) * 100:.1f}%")
        print("=" * 80)


# テスト用コード
if __name__ == "__main__":
    # サンプルテキスト
    sample_text = """
    人工知能（AI）は、機械学習と深層学習を基盤として急速に発展しています。
    特に自然言語処理（NLP）の分野では、トランスフォーマーモデルが革命的な成果を上げました。
    BERTやGPTなどの大規模言語モデルは、文脈理解能力を大幅に向上させています。
    """

    print("=" * 80)
    print("util_rag_qa.py - キーワード抽出ユーティリティのテスト")
    print("=" * 80)

    # BestKeywordSelectorのテスト
    print("\n【BestKeywordSelectorのテスト】")
    best_keywords = get_best_keywords(sample_text, top_n=5)
    print(f"最良のキーワード: {', '.join(best_keywords)}")

    # SmartKeywordSelectorのテスト
    print("\n【SmartKeywordSelectorのテスト】")
    smart_result = get_smart_keywords(sample_text, mode="auto")
    print("スマート選択結果:")
    print(f"  キーワード: {', '.join(smart_result['keywords'][:5])}")
    print(f"  選択手法: {smart_result['method']}")
    print(f"  決定理由: {smart_result['reason']}")
