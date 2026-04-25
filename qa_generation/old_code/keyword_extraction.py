#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/keyword_extraction.py - キーワード抽出モジュール
=============================================================
Q/A生成のためのキーワード抽出機能

統合元:
- helper_rag_qa.py::BestKeywordSelector
- helper_rag_qa.py::SmartKeywordSelector
- helper_rag_qa.py::get_best_keywords
- helper_rag_qa.py::get_smart_keywords
"""

import re
import math
from typing import List, Dict, Tuple, Any
from regex_mecab import KeywordExtractor


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
                    if any(sub in kw2 for sub in [kw1[j:j+2] for j in range(len(kw1)-1)]):
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


# ===================================================================
# ユーティリティ関数
# ===================================================================

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


# ===================================================================
# エクスポート
# ===================================================================

__all__ = [
    # クラス
    "BestKeywordSelector",
    "SmartKeywordSelector",
    # ユーティリティ関数
    "get_best_keywords",
    "get_smart_keywords",
]