"""
GRACE Confidence 較正（Calibration）— S1

confidence（自己申告の信頼度）と実正解率のズレ（ECE）を、事後較正で縮小する。
温度スケーリング（Temperature Scaling）を採用：

    z  = logit(p) = ln(p / (1 - p))
    p' = sigmoid(z / T)

- T = 1.0 : 無変換（恒等）
- T > 1.0 : 自信過剰を緩和（高すぎる confidence を引き下げる）
- T < 1.0 : 自信不足を補正（低すぎる confidence を引き上げる）

T は (confidence, 正誤) のペア集合に対して二値 NLL を最小化して推定する。
scipy 非依存（1次元探索）で実装し、ユニットテストで ECE 縮小を検証可能にする。

較正パラメータは JSON（既定 config/calibration.json）に保存／読込でき、
GRACE 本体（executor）が実行時に overall_confidence へ適用する。

[MIGRATION anthropic→openai] 本モジュールは provider 非依存（標準ライブラリのみ）。
anthropic 版から verbatim 移植（LLM/Embedding への依存なし）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_EPS = 1e-6
DEFAULT_CALIBRATION_PATH = "config/calibration.json"


def _clip01(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, float(p)))


def _logit(p: float) -> float:
    p = _clip01(p)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def apply_temperature(p: float, temperature: float) -> float:
    """confidence p に温度 T を適用して較正後の確率を返す。"""
    t = float(temperature)
    if t <= 0:
        t = 1.0
    return _sigmoid(_logit(p) / t)


def _nll(confidences: Sequence[float], correctness: Sequence[bool], t: float) -> float:
    """温度 t における二値負対数尤度（平均）。"""
    total = 0.0
    n = 0
    for p, y in zip(confidences, correctness):
        q = _clip01(apply_temperature(p, t))
        total += -(math.log(q) if y else math.log(1.0 - q))
        n += 1
    return total / n if n else 0.0


def fit_temperature(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    t_min: float = 0.05,
    t_max: float = 10.0,
) -> float:
    """(confidence, 正誤) から NLL 最小の温度 T を 1 次元探索で推定する。

    粗いグリッド → 近傍を細分化、の2段で十分な精度を得る。
    データが退化（全問正解 / 全問不正解 / 件数0）の場合は T=1.0 を返す。
    """
    n = len(confidences)
    if n == 0:
        return 1.0
    n_correct = sum(1 for y in correctness if y)
    if n_correct == 0 or n_correct == n:
        # 較正対象として退化（正誤が一定）。恒等を返す。
        return 1.0

    def _search(lo: float, hi: float, steps: int) -> tuple[float, float]:
        best_t, best_nll = 1.0, float("inf")
        for i in range(steps + 1):
            t = lo + (hi - lo) * i / steps
            if t <= 0:
                continue
            val = _nll(confidences, correctness, t)
            if val < best_nll:
                best_nll, best_t = val, t
        return best_t, best_nll

    coarse_t, _ = _search(t_min, t_max, 200)
    span = (t_max - t_min) / 200
    fine_t, _ = _search(max(t_min, coarse_t - span), min(t_max, coarse_t + span), 200)
    return round(fine_t, 4)


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """ECE（等幅ビン）。eval/metrics.py と整合する定義。"""
    n = len(confidences)
    if n == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        idx = [
            j for j, c in enumerate(confidences)
            if (c > lo or (i == 0 and c >= lo)) and (c <= hi if i < n_bins - 1 else c <= hi + 1e-9)
        ]
        if not idx:
            continue
        conf_b = sum(confidences[j] for j in idx) / len(idx)
        acc_b = sum(1 for j in idx if correctness[j]) / len(idx)
        ece += (len(idx) / n) * abs(acc_b - conf_b)
    return ece


@dataclass
class Calibrator:
    """温度スケーリングによる confidence 較正器。"""

    temperature: float = 1.0

    def transform(self, p: float) -> float:
        return apply_temperature(p, self.temperature)

    def is_identity(self) -> bool:
        return abs(self.temperature - 1.0) < 1e-9

    # --- 永続化 ---
    def save(self, path: str = DEFAULT_CALIBRATION_PATH) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump({"method": "temperature_scaling",
                       "temperature": self.temperature}, f,
                      ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str = DEFAULT_CALIBRATION_PATH) -> "Calibrator":
        """較正ファイルを読み込む。存在しなければ恒等較正器（T=1.0）を返す。"""
        p = Path(path)
        if not p.exists():
            return cls(temperature=1.0)
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
            t = float(data.get("temperature", 1.0))
            return cls(temperature=t if t > 0 else 1.0)
        except Exception:
            return cls(temperature=1.0)

    @classmethod
    def fit(cls, confidences: Sequence[float], correctness: Sequence[bool]) -> "Calibrator":
        return cls(temperature=fit_temperature(confidences, correctness))
