"""
GRACE 実行メモリ層（P4）

実行ログから「(質問キーワード, 当たったコレクション, 成否, confidence)」を
蓄積し、Planner のコレクション優先順位・rag/web 振り分けの事前分布に反映する。

- 永続化: JSONL（既定 logs/grace_memory.jsonl）。1行=1実行レコード。
- 集計: コレクションごとに success_rate × mean_confidence を
  Laplace スムージング付きで算出し、優先順位を返す。
- キーワード一致でフィルタして「この種の質問で当たりやすいコレクション」を推定する。

外部依存なし・決定的。ユニットテスト可能。

[MIGRATION anthropic→openai] 本モジュールは provider 非依存（標準ライブラリのみ）。
anthropic 版から verbatim 移植（LLM/Embedding への依存なし）。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = "logs/grace_memory.jsonl"

# 日本語（漢字/かな/カナ）または英数字の2文字以上をキーワードとして抽出
_KEYWORD_RE = re.compile(r"[A-Za-z0-9]{2,}|[一-鿿゠-ヿ぀-ゟ]{2,}")


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """軽量なキーワード抽出（形態素解析非依存・決定的）。"""
    if not text:
        return []
    seen: list[str] = []
    for m in _KEYWORD_RE.findall(text):
        t = m.lower()
        if t not in seen:
            seen.append(t)
        if len(seen) >= top_n:
            break
    return seen


@dataclass
class MemoryRecord:
    query: str
    keywords: list[str]
    collection: Optional[str]
    success: bool
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "keywords": self.keywords,
            "collection": self.collection,
            "success": self.success,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        return cls(
            query=d.get("query", ""),
            keywords=list(d.get("keywords", []) or []),
            collection=d.get("collection"),
            success=bool(d.get("success", False)),
            confidence=float(d.get("confidence", 0.0) or 0.0),
            timestamp=float(d.get("timestamp", 0.0) or 0.0),
        )


@dataclass
class CollectionStat:
    collection: str
    count: int
    success_count: int
    mean_confidence: float

    @property
    def success_rate(self) -> float:
        return self.success_count / self.count if self.count else 0.0

    def score(self, alpha: float = 1.0, beta: float = 1.0) -> float:
        """success_rate（Laplace 平滑化）× mean_confidence。"""
        smoothed_sr = (self.success_count + alpha) / (self.count + alpha + beta)
        return smoothed_sr * self.mean_confidence


class ExecutionMemory:
    """実行レコードの蓄積と、コレクション事前分布の算出。"""

    def __init__(self, path: str = DEFAULT_MEMORY_PATH):
        self.path = Path(path)

    # --- 記録 ---
    def record(self, query: str, collection: Optional[str], success: bool,
               confidence: float, keywords: Optional[list[str]] = None) -> None:
        """1実行レコードを JSONL へ追記する（best-effort）。"""
        rec = MemoryRecord(
            query=query or "",
            keywords=keywords if keywords is not None else extract_keywords(query or ""),
            collection=collection,
            success=bool(success),
            confidence=float(confidence or 0.0),
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:  # 記録失敗は実行を止めない
            logger.warning(f"ExecutionMemory.record failed: {e}")

    def record_many(self, query: str, collections: list[Optional[str]], success: bool,
                    confidence: float, keywords: Optional[list[str]] = None) -> None:
        """複数コレクションを使った実行をまとめて記録する。"""
        kw = keywords if keywords is not None else extract_keywords(query or "")
        seen = set()
        for c in collections:
            if c in seen:
                continue
            seen.add(c)
            self.record(query, c, success, confidence, keywords=kw)

    # --- 読み込み ---
    def load(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        records: list[MemoryRecord] = []
        try:
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(MemoryRecord.from_dict(json.loads(line)))
        except Exception as e:
            logger.warning(f"ExecutionMemory.load failed: {e}")
        return records

    # --- 集計 ---
    def collection_priors(
        self,
        query: Optional[str] = None,
        min_keyword_overlap: int = 1,
    ) -> list[CollectionStat]:
        """コレクション事前分布を score 降順で返す。

        query 指定時はキーワードが overlap するレコードのみを対象にする
        （その種の質問に対する分布）。overlap レコードが無ければ全体集計に
        フォールバックする。
        """
        records = self.load()
        if not records:
            return []

        target_kw = set(extract_keywords(query)) if query else set()
        if target_kw:
            filtered = [
                r for r in records
                if r.collection and len(target_kw & set(r.keywords)) >= min_keyword_overlap
            ]
            if not filtered:
                filtered = [r for r in records if r.collection]
        else:
            filtered = [r for r in records if r.collection]

        agg: dict[str, dict] = {}
        for r in filtered:
            a = agg.setdefault(r.collection, {"count": 0, "succ": 0, "conf_sum": 0.0})
            a["count"] += 1
            a["succ"] += 1 if r.success else 0
            a["conf_sum"] += r.confidence

        stats = [
            CollectionStat(
                collection=c,
                count=a["count"],
                success_count=a["succ"],
                mean_confidence=(a["conf_sum"] / a["count"] if a["count"] else 0.0),
            )
            for c, a in agg.items()
        ]
        stats.sort(key=lambda s: s.score(), reverse=True)
        return stats

    def best_collection(
        self,
        query: Optional[str] = None,
        min_count: int = 3,
        min_score: float = 0.6,
    ) -> Optional[str]:
        """十分な実績（count>=min_count かつ score>=min_score）がある
        最良コレクションを返す。条件を満たすものが無ければ None（=全コレクション検索）。
        """
        for stat in self.collection_priors(query=query):
            if stat.count >= min_count and stat.score() >= min_score:
                return stat.collection
        return None


def create_execution_memory(path: str = DEFAULT_MEMORY_PATH) -> ExecutionMemory:
    return ExecutionMemory(path=path)
