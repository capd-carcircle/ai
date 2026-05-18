"""
CAPD 데이터 엔지니어링 모듈
문서(proj1_ANALYSE HISTORICAL HEALTH DATA) 정의 기반:
  Exchange Event Table → Exchange Aggregate Table → Daily Table → Daily Model Row

build_daily_model_row(daily_data, exchange_records) → Daily Model Row dict (23개 컬럼)
"""

import statistics
from typing import Optional


# ── 유틸 ──────────────────────────────────────────────────────────

def _time_to_minutes(time_str: Optional[str]) -> Optional[int]:
    """HH:MM → 자정 이후 분 변환. None이면 None 반환."""
    if not time_str:
        return None
    try:
        h, m = time_str.strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _parse_bp(bp_str: Optional[str]) -> dict:
    """
    혈압 문자열 → 파생 지표
    Returns: {systolic_bp, diastolic_bp, pulse_pressure, mean_arterial_pressure}
    """
    if not bp_str:
        return {}
    try:
        parts = bp_str.strip().split("/")
        sys_bp = int(parts[0])
        dia_bp = int(parts[1])
        return {
            "systolic_bp":           sys_bp,
            "diastolic_bp":          dia_bp,
            "pulse_pressure":        sys_bp - dia_bp,
            "mean_arterial_pressure": round(dia_bp + (sys_bp - dia_bp) / 3.0, 1),
        }
    except Exception:
        return {}


# ── Exchange Event Table ───────────────────────────────────────────

def _build_exchange_events(exchange_records: list[dict]) -> list[dict]:
    """
    raw exchange_records → Exchange Event Table (슬롯별 파생 속성 계산)

    파생 속성:
    - exchange_time_minutes: 자정 이후 분
    - observed_flag:         실제 교환 데이터 있으면 1
    - dwell_minutes:         이전 교환 이후 경과 시간 (분)
    - calculated_uf_g:       drainage_volume - infusion_weight
    - uf_error_g:            ultrafiltration(보고) - calculated_uf_g
    """
    sorted_recs = sorted(exchange_records, key=lambda x: x.get("session_number", 0))

    events = []
    prev_minutes: Optional[int] = None

    for ex in sorted_recs:
        drainage   = ex.get("drainage_volume")
        infused    = ex.get("infusion_weight")
        reported_uf = ex.get("ultrafiltration")
        time_str   = ex.get("exchange_time")
        time_min   = _time_to_minutes(time_str)
        conc       = ex.get("infusion_concentration")

        # observed_flag
        observed = 1 if (drainage is not None and infused is not None) else 0

        # dwell_minutes
        dwell = None
        if time_min is not None and prev_minutes is not None:
            diff = time_min - prev_minutes
            if diff < 0:          # 자정 넘어감
                diff += 24 * 60
            dwell = diff

        # calculated_uf_g
        calc_uf = None
        if drainage is not None and infused is not None:
            calc_uf = float(drainage) - float(infused)

        # uf_error_g
        uf_error = None
        if reported_uf is not None and calc_uf is not None:
            uf_error = float(reported_uf) - calc_uf

        events.append({
            "session_number":         ex.get("session_number"),
            "exchange_time":          time_str,
            "exchange_time_minutes":  time_min,
            "drainage_volume":        float(drainage) if drainage is not None else None,
            "infusion_concentration": float(conc) if conc is not None else None,
            "infusion_weight":        float(infused) if infused is not None else None,
            "ultrafiltration":        float(reported_uf) if reported_uf is not None else None,
            "observed_flag":          observed,
            "dwell_minutes":          dwell,
            "calculated_uf_g":        round(calc_uf, 1) if calc_uf is not None else None,
            "uf_error_g":             round(uf_error, 1) if uf_error is not None else None,
        })

        if time_min is not None:
            prev_minutes = time_min

    return events


# ── Exchange Aggregate Table ──────────────────────────────────────

def _aggregate_exchanges(events: list[dict]) -> dict:
    """
    Exchange Event Table → Exchange Aggregate (일 단위 집계)

    집계 속성:
    - exchange_count, missing_exchange_slots
    - drain_sum_g, infused_sum_g
    - calculated_uf_sum_g, uf_min_g, uf_std_g
    - dwell_mean_minutes, dwell_std_minutes
    - concentration_max
    """
    observed = [e for e in events if e["observed_flag"] == 1]
    exchange_count  = len(observed)
    missing_slots   = 5 - exchange_count

    drain_sum   = sum(e["drainage_volume"] for e in observed if e["drainage_volume"] is not None)
    infused_sum = sum(e["infusion_weight"] for e in observed if e["infusion_weight"] is not None)

    calc_ufs    = [e["calculated_uf_g"] for e in observed if e["calculated_uf_g"] is not None]
    calc_uf_sum = round(sum(calc_ufs), 1) if calc_ufs else None
    uf_min      = round(min(calc_ufs), 1) if calc_ufs else None
    uf_std      = round(statistics.stdev(calc_ufs), 1) if len(calc_ufs) >= 2 else None

    dwells      = [e["dwell_minutes"] for e in events if e["dwell_minutes"] is not None]
    dwell_mean  = round(sum(dwells) / len(dwells), 1) if dwells else None
    dwell_std   = round(statistics.stdev(dwells), 1) if len(dwells) >= 2 else None

    concs       = [e["infusion_concentration"] for e in observed if e["infusion_concentration"] is not None]
    conc_max    = max(concs) if concs else None

    return {
        "exchange_count":       exchange_count,
        "missing_exchange_slots": missing_slots,
        "drain_sum_g":          round(drain_sum, 1) if drain_sum else None,
        "infused_sum_g":        round(infused_sum, 1) if infused_sum else None,
        "calculated_uf_sum_g":  calc_uf_sum,
        "uf_min_g":             uf_min,
        "uf_std_g":             uf_std,
        "dwell_mean_minutes":   dwell_mean,
        "dwell_std_minutes":    dwell_std,
        "concentration_max":    conc_max,
    }


# ── Daily Model Row ───────────────────────────────────────────────

def build_daily_model_row(daily_data: dict, exchange_records: list[dict]) -> dict:
    """
    하루치 기록 → Daily Model Row (23개 컬럼)

    Args:
        daily_data:       {date/record_date, weight, blood_pressure,
                           total_ultrafiltration, turbid_peritoneal,
                           fasting_blood_glucose, urine_count}
        exchange_records: [{session_number, exchange_time, drainage_volume,
                            infusion_concentration, infusion_weight, ultrafiltration}]

    Returns:
        Daily Model Row dict — analytics.py에 바로 입력 가능
    """
    events    = _build_exchange_events(exchange_records or [])
    agg       = _aggregate_exchanges(events)
    bp        = _parse_bp(daily_data.get("blood_pressure"))

    reported_uf  = daily_data.get("total_ultrafiltration")
    calc_uf_sum  = agg.get("calculated_uf_sum_g")
    uf_discrepancy = (
        round(float(reported_uf) - float(calc_uf_sum), 1)
        if reported_uf is not None and calc_uf_sum is not None
        else None
    )

    row = {
        "date": daily_data.get("date") or daily_data.get("record_date"),
        # Exchange 집계
        **agg,
        "reported_total_uf_g": float(reported_uf) if reported_uf is not None else None,
        "uf_discrepancy_g":    uf_discrepancy,
        # Daily 기본
        "body_weight_kg":       float(daily_data["weight"]) if daily_data.get("weight") is not None else None,
        "fasting_blood_sugar":  float(daily_data["fasting_blood_glucose"]) if daily_data.get("fasting_blood_glucose") is not None else None,
        "urination_count":      daily_data.get("urine_count"),
        "cloudy_dialysate":     1 if daily_data.get("turbid_peritoneal") else 0,
        # 혈압 파생
        **bp,
    }

    return row
