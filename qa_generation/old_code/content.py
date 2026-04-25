#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/content.py - コンテンツ分析・キーワード抽出モジュール
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import tiktoken

class KeywordExtractor:
    """
    MeCabと正規表現を統合したキーワード抽出クラス

    MeCabが利用可能な場合は複合名詞抽出を優先し、
    利用不可の場合は正規表現版に自動フォールバック
    """

    def __init__(self, prefer_mecab: bool = True):
        """
        Args:
            prefer_mecab: MeCabを優先的に使用するか（デフォルト: True）
        """
        self.prefer_mecab = prefer_mecab
        self.mecab_available = self._check_mecab_availability()

        # ストップワード定義
        self.stopwords = {
            'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん',
            'ます', 'です', 'ある', 'いる', 'する', 'なる', 'できる',
            'いう', '的', 'な', 'に', 'を', 'は', 'が', 'で', 'と',
            'の', 'から', 'まで', '等', 'など', 'よる', 'おく', 'くる'
        }

        # 重要キーワードの定義（スコアブースト用）
        self.important_keywords = {
            'AI', '人工知能', '機械学習', '深層学習', 'ディープラーニング',
            '自然言語処理', 'NLP', 'トランスフォーマー', 'BERT', 'GPT',
            'CNN', 'Vision', 'Transformer', '医療', '診断', '自動運転',
            '倫理', 'バイアス', '課題', '問題', 'モデル', 'データ'
        }

        if self.mecab_available:
            # print("✅ MeCabが利用可能です（複合名詞抽出モード）")
            pass
        else:
            # print("⚠️ MeCabが利用できません（正規表現モード）")
            pass

    def _check_mecab_availability(self) -> bool:
        """MeCabの利用可能性をチェック"""
        try:
            import MeCab
            # 実際にインスタンス化して動作確認
            tagger = MeCab.Tagger()
            tagger.parse("テスト")
            return True
        except (ImportError, RuntimeError):
            return False

    def extract(self, text: str, top_n: int = 5,
                use_scoring: bool = True) -> List[str]:
        """
        テキストからキーワードを抽出（自動フォールバック対応）

        Args:
            text: 分析対象テキスト
            top_n: 抽出するキーワード数
            use_scoring: スコアリングを使用するか

        Returns:
            キーワードリスト
        """
        if self.mecab_available and self.prefer_mecab:
            try:
                keywords = self._extract_with_mecab(text, top_n, use_scoring)
                if keywords:  # 空でなければ成功
                    return keywords
            except Exception as e:
                # print(f"⚠️ MeCab抽出エラー: {e}")
                pass

        # フォールバック: 正規表現版
        return self._extract_with_regex(text, top_n, use_scoring)

    def _extract_with_mecab(self, text: str, top_n: int,
                           use_scoring: bool) -> List[str]:
        """MeCabを使用した複合名詞抽出"""
        import MeCab

        tagger = MeCab.Tagger()
        node = tagger.parseToNode(text)

        # 複合名詞の抽出
        compound_buffer = []
        compound_nouns = []

        while node:
            features = node.feature.split(',')
            pos = features[0]  # 品詞

            if pos == '名詞':
                compound_buffer.append(node.surface)
            else:
                # 名詞以外が来たらバッファをフラッシュ
                if compound_buffer:
                    compound_noun = ''.join(compound_buffer)
                    if len(compound_noun) > 0:
                        compound_nouns.append(compound_noun)
                    compound_buffer = []

            node = node.next

        # 最後のバッファをフラッシュ
        if compound_buffer:
            compound_noun = ''.join(compound_buffer)
            if len(compound_noun) > 0:
                compound_nouns.append(compound_noun)

        # フィルタリングとスコアリング
        if use_scoring:
            return self._score_and_rank(compound_nouns, top_n)
        else:
            return self._filter_and_count(compound_nouns, top_n)

    def _extract_with_regex(self, text: str, top_n: int,
                           use_scoring: bool) -> List[str]:
        """正規表現を使用したキーワード抽出"""
        # カタカナ語、漢字複合語、英数字を抽出
        pattern = r'[ァ-ヴー]{2,}|[一-龥]{2,}|[A-Za-z]{2,}[A-Za-z0-9]*'
        words = re.findall(pattern, text)

        # フィルタリングとスコアリング
        if use_scoring:
            return self._score_and_rank(words, top_n)
        else:
            return self._filter_and_count(words, top_n)

    def _filter_and_count(self, words: List[str], top_n: int) -> List[str]:
        """頻度ベースのフィルタリング（シンプル版）"""
        # ストップワード除外
        filtered = [w for w in words if w not in self.stopwords and len(w) > 1]

        # 頻度カウント
        word_freq = Counter(filtered)

        # 上位N件を返す
        return [word for word, freq in word_freq.most_common(top_n)]

    def _score_and_rank(self, words: List[str], top_n: int) -> List[str]:
        """スコアリングベースのランキング（高度版）"""
        word_scores = {}
        word_freq = Counter(words)

        for word, freq in word_freq.items():
            # ストップワード除外
            if word in self.stopwords or len(word) <= 1:
                continue

            score = 0.0

            # 1. 頻度スコア（正規化: 最大3回まで）
            freq_score = min(freq / 3.0, 1.0) * 0.3
            score += freq_score

            # 2. 長さスコア（複合語優遇）
            length_score = min(len(word) / 8.0, 1.0) * 0.3
            score += length_score

            # 3. 重要キーワードブースト
            if word in self.important_keywords:
                score += 0.5

            # 4. 文字種スコア
            # カタカナ3文字以上
            if re.match(r'^[ァ-ヴー]{3,}$', word):
                score += 0.2
            # 英大文字2文字以上
            elif re.match(r'^[A-Z]{2,}$', word):
                score += 0.3
            # 漢字4文字以上
            elif re.match(r'^[一-龥]{4,}$', word):
                score += 0.2

            word_scores[word] = score

        # スコア降順でソート
        ranked = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

        return [word for word, score in ranked[:top_n]]

    def extract_with_details(self, text: str, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        詳細情報付きでキーワードを抽出（比較分析用）

        Returns:
            各手法での抽出結果と詳細スコア
        """
        results = {}

        # MeCab複合名詞版
        if self.mecab_available:
            try:
                mecab_keywords = self._extract_with_mecab_scored(text, top_n)
                results['MeCab複合名詞'] = mecab_keywords
            except Exception as e:
                results['MeCab複合名詞'] = [(f"エラー: {e}", 0.0)]

        # 正規表現版
        regex_keywords = self._extract_with_regex_scored(text, top_n)
        results['正規表現'] = regex_keywords

        # 統合版（デフォルト動作）
        integrated_keywords = self._extract_integrated(text, top_n)
        results['統合版'] = integrated_keywords

        return results

    def _extract_with_mecab_scored(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """MeCab版（スコア付き）"""
        keywords = self._extract_with_mecab(text, top_n, use_scoring=True)
        # スコアを再計算
        scored = []
        for kw in keywords:
            score = self._calculate_keyword_score(kw, text)
            scored.append((kw, score))
        return scored

    def _extract_with_regex_scored(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """正規表現版（スコア付き）"""
        keywords = self._extract_with_regex(text, top_n, use_scoring=True)
        scored = []
        for kw in keywords:
            score = self._calculate_keyword_score(kw, text)
            scored.append((kw, score))
        return scored

    def _extract_integrated(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """統合版: MeCabと正規表現の結果をマージ"""
        all_keywords = set()

        # MeCabから抽出
        if self.mecab_available:
            try:
                mecab_kws = self._extract_with_mecab(text, top_n * 2, use_scoring=False)
                all_keywords.update(mecab_kws)
            except Exception:
                pass

        # 正規表現から抽出
        regex_kws = self._extract_with_regex(text, top_n * 2, use_scoring=False)
        all_keywords.update(regex_kws)

        # 統合スコアリング
        scored = []
        for kw in all_keywords:
            if kw in self.stopwords or len(kw) <= 1:
                continue
            score = self._calculate_keyword_score(kw, text)
            scored.append((kw, score))

        # スコア降順でソート
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def _calculate_keyword_score(self, keyword: str, text: str) -> float:
        """キーワードの総合スコアを計算"""
        score = 0.0

        # 出現頻度
        freq = text.count(keyword)
        freq_score = min(freq / 3.0, 1.0) * 0.3
        score += freq_score

        # 長さ
        length_score = min(len(keyword) / 8.0, 1.0) * 0.2
        score += length_score

        # 重要キーワード
        if keyword in self.important_keywords:
            score += 0.4

        # 文字種
        if re.match(r'^[ァ-ヴー]{3,}$', keyword):
            score += 0.15
        elif re.match(r'^[A-Z]{2,}$', keyword):
            score += 0.2
        elif re.match(r'^[一-龥]{4,}$', keyword):
            score += 0.15

        return min(score, 1.0)


# グローバルなKeywordExtractorインスタンス（一度だけ初期化）
_keyword_extractor = None

def get_keyword_extractor() -> KeywordExtractor:
    """KeywordExtractorのシングルトンインスタンスを取得"""
    global _keyword_extractor
    if _keyword_extractor is None:
        _keyword_extractor = KeywordExtractor()
    return _keyword_extractor


def analyze_chunk_complexity(chunk_text: str, lang: str = "ja") -> Dict:
    """チャンクの複雑度を分析

    Args:
        chunk_text: 分析対象テキスト
        lang: 言語

    Returns:
        複雑度指標の辞書
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # 基本メトリクス
    sentences = chunk_text.split('。') if lang == 'ja' else chunk_text.split('.')
    tokens = tokenizer.encode(chunk_text)

    # 専門用語の検出（簡易版）
    if lang == 'ja':
        # カタカナ語、漢字複合語を専門用語候補とする
        technical_pattern = r'[ァ-ヴー]{4,}|[一-龥]{4,}'
        technical_terms = re.findall(technical_pattern, chunk_text)
    else:
        # 大文字で始まる複合語、長い単語を専門用語候補とする
        technical_pattern = r'[A-Z][a-z]+(?:[A-Z][a-z]+)+|\b\w{10,}\b'
        technical_terms = re.findall(technical_pattern, chunk_text)

    # 文の複雑度（平均文長）
    avg_sentence_length = len(tokens) / max(len(sentences), 1)

    # 概念密度（専門用語の頻度）
    concept_density = len(technical_terms) / max(len(tokens), 1) * 100

    # 複雑度レベルの判定
    if concept_density > 5 or avg_sentence_length > 30:
        complexity_level = "high"
    elif concept_density > 2 or avg_sentence_length > 20:
        complexity_level = "medium"
    else:
        complexity_level = "low"

    return {
        "complexity_level": complexity_level,
        "technical_terms": list(set(technical_terms))[:10],  # 上位10個
        "avg_sentence_length": avg_sentence_length,
        "concept_density": concept_density,
        "sentence_count": len(sentences),
        "token_count": len(tokens)
    }

def extract_key_concepts(chunk_text: str, lang: str = "ja", top_n: int = 5) -> List[str]:
    """チャンクから主要概念を抽出

    Args:
        chunk_text: テキスト
        lang: 言語
        top_n: 抽出する概念数

    Returns:
        主要概念のリスト
    """
    # KeywordExtractorを使用
    extractor = get_keyword_extractor()
    keywords = extractor.extract(chunk_text, top_n=top_n)

    # 複雑度分析から専門用語も追加
    complexity = analyze_chunk_complexity(chunk_text, lang)
    technical_terms = complexity.get("technical_terms", [])

    # 重複を除いて統合
    all_concepts = list(set(keywords + technical_terms[:3]))

    return all_concepts[:top_n]
