"""
CAPD 데이터 분석 모듈 (Task 1~4)
문서(proj1_ANALYSE HISTORICAL HEALTH DATA) 정의 기반

Task 1: Trend Analysis    -- 7d/30d rolling 비교, 5단계 분류
Task 2: Anomaly Detection -- rolling z-score + robust z-score (MAD 기반)
Task 3: Attribute Correlation -- Spearman 상관계수 (|r| >= 0.5 쌍만 선별)
Task 4: Exploratory Data Analysis -- min/max/mean/std + today vs 7d/30d

순수 Python 구현 (외부 라이브러리 의존 없음)
"""

import math
from typing import Optional


# ================================================================
# 공통 수학 유틸
# ================================================================

def _valid(series: list) -> list[float]:
    return [float(v) for v in series if v is not None]


def _mean(vals: list[float]) -> Optional[float]:
    return round(sum(vals) / len(vals), 2) if vals else None


def _std(vals: list[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return round(math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)), 2)


def _median(vals: list[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2.0 if n % 2 == 0 else float(s[mid])


def _mad(vals: list[float]) -> Optional[float]:
    """Median Absolute Deviation"""
    if len(vals) < 2:
        return None
    med = _median(vals)
    return _median([abs(v - med) for v in vals])


def _rank(vals: list[float]) -> list[float]:
    """평균 순위 (동점 처리 포함)"""
    n = len(vals)
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> Optional[float]:
    """scipy 없이 순수 Python으로 Spearman 상관계수 계산"""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return None
    return round(num / (den_x * den_y), 3)


# ================================================================
# Task 1: Trend Analysis
# ================================================================

_TREND_THRESH = {
    "body_weight_kg":         {"much": 1.5,  "normal": 0.5},
    "systolic_bp":            {"much": 15,   "normal": 5},
    "diastolic_bp":           {"much": 10,   "normal": 4},
    "mean_arterial_pressure": {"much": 10,   "normal": 4},
    "fasting_blood_sugar":    {"much": 30,   "normal": 10},
    "urination_count":        {"much": 5,    "normal": 2},
    "exchange_count":         {"much": 2,    "normal": 1},
    "dwell_mean_minutes":     {"much": 60,   "normal": 30},
    "concentration_max":      {"much": 1.0,  "normal": 0.5},
    "calculated_uf_sum_g":    {"pct_much": 20, "pct_normal": 10},
    "recorded_uf_sum_g":      {"pct_much": 20, "pct_normal": 10},
    "infused_sum_g":          {"pct_much": 20, "pct_normal": 10},
}

_UNITS = {
    "body_weight_kg":         "kg",
    "systolic_bp":            "mmHg",
    "diastolic_bp":           "mmHg",
    "mean_arterial_pressure": "mmHg",
    "fasting_blood_sugar":    "mg/dL",
    "urination_count":        "회",
    "exchange_count":         "회",
    "dwell_mean_minutes":     "분",
    "concentration_max":      "%",
    "calculated_uf_sum_g":    "g",
    "recorded_uf_sum_g":      "g",
    "infused_sum_g":          "g",
}

TREND_ATTRS = list(_TREND_THRESH.keys())


def _classify_trend(diff: float, thresh: dict, baseline: float = None) -> str:
    if "pct_much" in thresh:
        if baseline and baseline != 0:
            pct = abs(diff) / abs(baseline) * 100
            if pct >= thresh["pct_much"]:
                return "much_higher_than_baseline" if diff > 0 else "much_lower_than_baseline"
            if pct >= thresh["pct_normal"]:
                return "higher_than_baseline" if diff > 0 else "lower_than_baseline"
        return "stable"
    much, normal = thresh["much"], thresh["normal"]
    if abs(diff) >= much:
        return "much_higher_than_baseline" if diff > 0 else "much_lower_than_baseline"
    if abs(diff) >= normal:
        return "higher_than_baseline" if diff > 0 else "lower_than_baseline"
    return "stable"


def task1_trend_analysis(today_row: dict, historical_rows: list[dict]) -> dict:
    """
    오늘 값 vs 7일/30일 baseline 비교

    Args:
        today_row:       오늘 Daily Model Row
        historical_rows: 최신->과거 순 (오늘 제외)
    """
    results = {}
    for attr in TREND_ATTRS:
        raw = today_row.get(attr)
        if raw is None:
            continue
        today_val = float(raw)

        hist = _valid([r.get(attr) for r in historical_rows])
        last_7d  = hist[:7]
        last_30d = hist[:30]

        thresh = _TREND_THRESH[attr]
        unit   = _UNITS.get(attr, "")
        entry  = {"today_value": today_val, "unit": unit}

        if last_30d:
            m30 = sum(last_30d) / len(last_30d)
            d30 = round(today_val - m30, 2)
            entry["previous_30d_mean"]        = round(m30, 2)
            entry["difference_from_30d_mean"] = d30
            entry["trend_30d"] = _classify_trend(d30, thresh, m30)

        if last_7d:
            m7  = sum(last_7d) / len(last_7d)
            d7  = round(today_val - m7, 2)
            pct = round((today_val - m7) / m7 * 100, 1) if m7 != 0 else None
            entry["previous_7d_mean"]               = round(m7, 2)
            entry["difference_from_7d_mean"]        = d7
            entry["percentage_change_from_7d_mean"] = pct
            entry["trend_7d"] = _classify_trend(d7, thresh, m7)

        entry["trend_summary"] = entry.get("trend_30d") or entry.get("trend_7d") or "insufficient_data"

        parts = [f"오늘 값 {today_val} {unit}."]
        if "trend_30d" in entry:
            parts.append(
                f"30일 평균 {entry['previous_30d_mean']} {unit} 대비 "
                f"{entry['difference_from_30d_mean']:+.2f} {unit} ({entry['trend_30d']})."
            )
        if "trend_7d" in entry:
            parts.append(
                f"7일 평균 {entry['previous_7d_mean']} {unit} 대비 "
                f"{entry['difference_from_7d_mean']:+.2f} {unit} ({entry['trend_7d']})."
            )
        if len(parts) == 1:
            parts.append("(과거 데이터 없음)")
        entry["statement"] = " ".join(parts)

        results[attr] = entry

    return {"task": "trend_analysis", "results": results}


# ================================================================
# Task 2: Anomaly Detection
# ================================================================

ANOMALY_ATTRS = [
    "body_weight_kg",
    "reported_total_uf_g",
    "calculated_uf_sum_g",
    "recorded_uf_sum_g",
    "systolic_bp",
    "diastolic_bp",
    "mean_arterial_pressure",
    "fasting_blood_sugar",
    "infused_sum_g",
]

_Z_LEVELS = [(3.0, "strong_anomaly"), (2.0, "mild_anomaly")]


def _z_label(z: float) -> str:
    for thresh, label in _Z_LEVELS:
        if abs(z) >= thresh:
            return label
    return "normal"


def task2_anomaly_detection(today_row: dict, historical_rows: list[dict]) -> dict:
    """
    오늘 값 vs 30일 window:
    - rolling z-score = (today - 30d_mean) / 30d_std
    - robust z-score  = 0.6745 * (today - 30d_median) / MAD
    """
    results = {}
    for attr in ANOMALY_ATTRS:
        raw = today_row.get(attr)
        if raw is None:
            continue
        today_val = float(raw)

        hist = _valid([r.get(attr) for r in historical_rows[:30]])
        unit = _UNITS.get(attr, "")

        if len(hist) < 3:
            results[attr] = {
                "today_value":    today_val,
                "sufficient_data": False,
                "statement": f"오늘 값 {today_val} {unit} -- 과거 데이터 부족 ({len(hist)}개, 최소 3개 필요)",
            }
            continue

        mean_30 = sum(hist) / len(hist)
        std_30  = _std(hist) or 0.001
        med_30  = _median(hist) or mean_30
        mad_30  = _mad(hist) or 0.001

        z_score  = round((today_val - mean_30) / std_30, 3)
        robust_z = round(0.6745 * (today_val - med_30) / mad_30, 3)

        z_label      = _z_label(z_score)
        robust_label = _z_label(robust_z)

        statement = (
            f"오늘 값 {today_val} {unit}, 30일 평균 {round(mean_30, 2)} {unit}, "
            f"표준편차 {round(std_30, 2)} {unit}. "
            f"Rolling z-score: {z_score} -> {z_label}. "
            f"Robust z-score: {robust_z} -> {robust_label}."
        )

        results[attr] = {
            "today_value":           today_val,
            "baseline_mean":         round(mean_30, 2),
            "baseline_std":          round(std_30, 2),
            "z_score_30d":           z_score,
            "z_interpretation":      z_label,
            "robust_z_score":        robust_z,
            "robust_interpretation": robust_label,
            "is_anomaly":            z_label != "normal" or robust_label != "normal",
            "sufficient_data":       True,
            "statement":             statement,
        }

    return {"task": "anomaly_detection", "results": results}


# ================================================================
# Task 3: Attribute Correlation (Spearman)
# ================================================================

CORR_ATTRS = [
    "body_weight_kg",
    "reported_total_uf_g",
    "calculated_uf_sum_g",
    "recorded_uf_sum_g",
    "systolic_bp",
    "diastolic_bp",
    "mean_arterial_pressure",
    "fasting_blood_sugar",
    "urination_count",
    "exchange_count",
    "infused_sum_g",
    "dwell_mean_minutes",
    "concentration_max",
]

_CORR_LEVELS = [(0.9, "very strong"), (0.7, "strong"), (0.5, "moderate")]


def _corr_label(r: float) -> str:
    for thresh, label in _CORR_LEVELS:
        if abs(r) >= thresh:
            return label
    return "weak"


def task3_attribute_correlation(historical_rows: list[dict], window: int = 30) -> dict:
    """
    최근 window일치 Spearman 상관관계 -- |r| >= 0.5 쌍만 반환
    """
    rows = historical_rows[:window]
    if len(rows) < 7:
        return {
            "task":        "attribute_correlation",
            "method":      "spearman_correlation",
            "window_days": len(rows),
            "results":     [],
            "note":        f"데이터 부족 ({len(rows)}일) -- 최소 7일 필요",
        }

    series: dict[str, list[float]] = {}
    for attr in CORR_ATTRS:
        vals = _valid([r.get(attr) for r in rows])
        if len(vals) >= 7:
            series[attr] = vals

    attrs = list(series.keys())
    pairs = []

    for i in range(len(attrs)):
        for j in range(i + 1, len(attrs)):
            a1, a2 = attrs[i], attrs[j]
            x_list, y_list = [], []
            for r in rows:
                v1, v2 = r.get(a1), r.get(a2)
                if v1 is not None and v2 is not None:
                    x_list.append(float(v1))
                    y_list.append(float(v2))
            if len(x_list) < 7:
                continue

            corr = _spearman(x_list, y_list)
            if corr is None or abs(corr) < 0.5:
                continue

            direction = "positive" if corr > 0 else "negative"
            label     = _corr_label(corr)
            pairs.append({
                "attr1":          a1,
                "attr2":          a2,
                "correlation":    corr,
                "direction":      direction,
                "interpretation": label,
                "statement": (
                    f"{a1} and {a2} has a correlation of {corr} "
                    f"showing a {label} {direction} correlation."
                ),
            })

    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

    return {
        "task":        "attribute_correlation",
        "method":      "spearman_correlation",
        "window_days": len(rows),
        "results":     pairs,
    }


# ================================================================
# Task 4: Exploratory Data Analysis
# ================================================================

EDA_ATTRS = CORR_ATTRS


def task4_eda(today_row: dict, historical_rows: list[dict]) -> dict:
    """기초 통계 (30d) + today vs 7d/30d 비교"""
    results = {}
    for attr in EDA_ATTRS:
        today_raw = today_row.get(attr)
        hist_30   = _valid([r.get(attr) for r in historical_rows[:30]])
        hist_7    = _valid([r.get(attr) for r in historical_rows[:7]])

        entry: dict = {}
        if today_raw is not None:
            entry["today_value"] = float(today_raw)
        if hist_30:
            entry["recent_30d_mean"] = _mean(hist_30)
            entry["recent_30d_std"]  = _std(hist_30)
            entry["recent_30d_min"]  = round(min(hist_30), 2)
            entry["recent_30d_max"]  = round(max(hist_30), 2)
        if hist_7:
            entry["recent_7d_mean"] = _mean(hist_7)
            entry["recent_7d_min"]  = round(min(hist_7), 2)
            entry["recent_7d_max"]  = round(max(hist_7), 2)

        if entry:
            results[attr] = entry

    return {"task": "exploratory_data_analysis", "results": results}


# ================================================================
# 통합 실행
# ================================================================

def run_all_tasks(today_row: dict, historical_rows: list[dict], window: int = 30) -> dict:
    """
    4가지 분석 Task 모두 실행

    Args:
        today_row:       build_daily_model_row()로 생성한 오늘 Daily Model Row
        historical_rows: 최신->과거 순 Daily Model Row 리스트 (오늘 제외)
        window:          task3(상관관계)에 쓸 최근 며칠치 기준(기본 30일).
                         task1/2/4는 "오늘 vs 7일/30일 평균"이라는 고정된 통계 정의라
                         window와 무관하게 항상 7일·30일 기준 그대로 계산함(의도된 동작).

    Returns:
        {
            "trend_analysis":        {...},
            "anomaly_detection":     {...},
            "attribute_correlation": {...},
            "eda":                   {...},
            "has_anomaly":           bool,
            "anomaly_attrs":         [str],
        }
    """
    trend   = task1_trend_analysis(today_row, historical_rows)
    anomaly = task2_anomaly_detection(today_row, historical_rows)
    corr    = task3_attribute_correlation(historical_rows, window=window)
    eda     = task4_eda(today_row, historical_rows)

    anomaly_attrs = [
        attr
        for attr, res in anomaly["results"].items()
        if isinstance(res, dict) and res.get("is_anomaly")
    ]

    return {
        "trend_analysis":        trend,
        "anomaly_detection":     anomaly,
        "attribute_correlation": corr,
        "eda":                   eda,
        "has_anomaly":           bool(anomaly_attrs),
        "anomaly_attrs":         anomaly_attrs,
    }
