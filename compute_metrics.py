#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_metrics.py — K-ART LENS 평가 결과 자동 산출기 (프로토콜 v2.0)

목적
----
보고서 v2.0에 사전 등록된 측정식·판정 규칙을 코드로 고정한다.
원시 데이터가 확보되면 이 스크립트만 실행하여 결과표를 생성하며,
수기 계산·수기 기입을 전면 금지한다.

설계 원칙
---------
1. 판정은 점추정이 아니라 신뢰구간으로 한다.
2. 1차 종점 3개는 Bonferroni 보정(z=2.394), 2차 종점은 z=1.96.
3. 데이터가 없으면 수치를 만들지 않고 '미측정'을 출력한다.
4. 외부 의존성 없음 (Python 3.8+ 표준 라이브러리만 사용).

사용법
------
    python3 compute_metrics.py --selftest
    python3 compute_metrics.py --data ./raw --out ./results

입력 (--data 디렉터리)
    gold_labels.csv, predictions.jsonl, latency_log.csv,
    schema_validation_log.csv, top5_relevance_eval.csv,
    practitioner_survey.csv, synonym_map.csv(선택), run_manifest.json(선택)

출력 (--out 디렉터리)
    results_summary.csv, results_table.md, results_detail.json
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import unicodedata
from collections import defaultdict

# ---------------------------------------------------------------------------
# 사전 등록된 판정 파라미터 — 데이터 수집 전 동결. 실행 중 변경 금지.
# ---------------------------------------------------------------------------
Z_PRIMARY = 2.394      # Bonferroni: alpha = 0.05/3, 양측
Z_SECONDARY = 1.960    # alpha = 0.05, 양측
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260819

TARGETS = {
    # key            : (목표치, 종점등급, 방향)  방향 'ge'=이상이면 좋음, 'le'=이하면 좋음
    "color_match":     (0.90, "secondary", "ge"),
    "materiality_match": (0.80, "primary", "ge"),
    "technique_match": (0.65, "secondary", "ge"),
    "iconography_match": (0.55, "secondary", "ge"),
    "latency_p95":     (15.0, "primary", "le"),
    "json_first_pass": (0.95, "primary", "ge"),
    "top5_relevance":  (0.60, "secondary", "ge"),
}

LAYERS = ["iconography", "technique", "materiality", "color"]
LAYER_KO = {"iconography": "도상", "technique": "기법",
            "materiality": "물성", "color": "색채"}

# 층위별 채점 방식 (보고서 §4.5.2)
MATCH_RULE = {
    "color": "jaccard>=0.5",
    "materiality": "exact_primary",
    "technique": "top1",
    "iconography": "any_overlap",
}


# ---------------------------------------------------------------------------
# 통계 유틸
# ---------------------------------------------------------------------------
def wilson_ci(k, n, z=Z_SECONDARY):
    """이항 비율의 Wilson 점수 신뢰구간. 반환 (하한, 점추정, 상한)."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), p, min(1.0, center + half))


def required_n(p, e, z=Z_SECONDARY):
    """반폭 e를 얻기 위한 필요 표본 수 (항상 올림)."""
    return math.ceil(z * z * p * (1 - p) / (e * e))


def min_count_for_pass(n, target, z):
    """N에서 '달성' 판정이 가능한 최소 관측 건수. 불가능하면 None."""
    for k in range(n + 1):
        if wilson_ci(k, n, z)[0] >= target:
            return k
    return None


def percentile_linear(values, q):
    """선형보간 백분위수 (numpy.percentile 기본 방식과 동일)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return s[int(idx)]
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def bootstrap_ci(values, stat_fn, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """비모수 부트스트랩 백분위 신뢰구간."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    reps = []
    for _ in range(b):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        reps.append(stat_fn(sample))
    reps.sort()
    return (reps[int(0.025 * b)], reps[min(b - 1, int(0.975 * b))])


def cluster_bootstrap_ci(clusters, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """
    질의 단위 클러스터 부트스트랩.
    clusters: [[0/1, 0/1, ...], ...]  질의별 관련 판정 리스트.
    Top-5 항목은 질의 내부에서 상관되므로 항목 단위 Wilson은 구간을 과소 추정한다.
    """
    if not clusters:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = len(clusters)
    reps = []
    for _ in range(b):
        num = den = 0
        for _ in range(m):
            c = clusters[rng.randrange(m)]
            num += sum(c)
            den += len(c)
        reps.append(num / den if den else float("nan"))
    reps.sort()
    return (reps[int(0.025 * b)], reps[min(b - 1, int(0.975 * b))])


def mcnemar_exact(b_count, c_count):
    """
    대응표본 이진 결과의 McNemar 정확 검정 (양측).
    b: 개입 전 정답 -> 후 오답, c: 전 오답 -> 후 정답.
    """
    n = b_count + c_count
    if n == 0:
        return 1.0
    k = min(b_count, c_count)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def krippendorff_alpha_nominal(units):
    """
    명목 척도 Krippendorff alpha.
    units: [[라벨, 라벨, ...], ...] 단위별 평가자 라벨 (결측은 제외한 상태).
    """
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    n_total = sum(len(u) for u in units)
    if n_total < 2:
        return float("nan")

    do_num = 0.0
    for u in units:
        m = len(u)
        disagree = sum(1 for i in range(m) for j in range(m) if i != j and u[i] != u[j])
        do_num += disagree / (m - 1)
    d_o = do_num / n_total

    counts = defaultdict(int)
    for u in units:
        for lab in u:
            counts[lab] += 1
    d_e_num = sum(counts[a] * counts[b] for a in counts for b in counts if a != b)
    d_e = d_e_num / (n_total * (n_total - 1))

    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e


def decide(lo, hi, target, direction, measured):
    """사전 등록된 3분 판정 규칙."""
    if not measured:
        return "미측정"
    if any(math.isnan(x) for x in (lo, hi)):
        return "판정 불가(표본 부족)"
    if direction == "ge":
        if lo >= target:
            return "달성"
        if hi < target:
            return "미달"
        return "보류"
    else:  # 'le'
        if hi <= target:
            return "달성"
        if lo > target:
            return "미달"
        return "보류"


# ---------------------------------------------------------------------------
# 라벨 정규화 및 채점 (보고서 §4.5)
# ---------------------------------------------------------------------------
def normalize_label(text, synonyms=None):
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text)).strip().lower()
    # 괄호 주석 제거
    out, depth = [], 0
    for ch in s:
        if ch in "(（[":
            depth += 1
        elif ch in ")）]":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    s = "".join(out)
    for suffix in ["계열", "풍", "로 추정", "추정", "등"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    s = " ".join(s.split())
    if synonyms:
        s = synonyms.get(s, s)
    return s


def split_labels(cell, synonyms=None):
    if cell is None or str(cell).strip() == "":
        return []
    parts = str(cell).replace(";", "|").replace(",", "|").split("|")
    return [x for x in (normalize_label(p, synonyms) for p in parts) if x]


def is_match(layer, gold_cell, pred_cell, synonyms=None):
    """층위별 사전 확정된 채점 규칙. 유보(빈 예측)는 불일치로 보수적 처리."""
    g = split_labels(gold_cell, synonyms)
    p = split_labels(pred_cell, synonyms)
    if not g:
        return None          # 골드 라벨 없음 -> 평가 대상에서 제외
    if not p:
        return False         # 모델 유보 -> 불일치 (보수적)

    rule = MATCH_RULE[layer]
    if rule == "exact_primary":
        return g[0] == p[0]
    if rule == "top1":
        return g[0] == p[0]
    if rule == "any_overlap":
        return len(set(g) & set(p)) > 0
    if rule == "jaccard>=0.5":
        inter = len(set(g) & set(p))
        union = len(set(g) | set(p))
        return (inter / union) >= 0.5 if union else False
    raise ValueError("unknown rule: " + rule)


# ---------------------------------------------------------------------------
# 파일 로더
# ---------------------------------------------------------------------------
def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_synonyms(path):
    rows = read_csv(path)
    if not rows:
        return {}
    return {normalize_label(r.get("from", "")): normalize_label(r.get("to", ""))
            for r in rows if r.get("from")}


def truthy(v):
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "예")


# ---------------------------------------------------------------------------
# 지표 산출
# ---------------------------------------------------------------------------
def metric_stub(key, label):
    target, tier, direction = TARGETS[key]
    return {
        "key": key, "label": label, "target": target, "tier": tier,
        "direction": direction, "measured": False, "n": 0, "k": None,
        "point": None, "ci_lo": None, "ci_hi": None,
        "decision": "미측정", "note": "원시 데이터 없음",
    }


def finalize_ratio(m, k, n, z):
    lo, p, hi = wilson_ci(k, n, z)
    m.update(measured=True, n=n, k=k, point=p, ci_lo=lo, ci_hi=hi,
             decision=decide(lo, hi, m["target"], m["direction"], True), note="")
    return m


def compute_layer_metrics(data_dir, results, detail):
    gold = read_csv(os.path.join(data_dir, "gold_labels.csv"))
    preds = read_jsonl(os.path.join(data_dir, "predictions.jsonl"))
    syn = load_synonyms(os.path.join(data_dir, "synonym_map.csv"))

    for layer in LAYERS:
        results[layer + "_match"] = metric_stub(
            layer + "_match", f"{LAYER_KO[layer]} 일치율")

    if not gold or not preds:
        return

    pmap = {str(r.get("artwork_id")): r for r in preds}

    # 이중 라벨링 대비: artwork_id당 최종 골드 라벨 1행만 채점에 사용한다.
    # 우선순위 = adjudicated/is_final 플래그가 참인 행 > 최초 등장 행.
    # (중복 계상하면 이중 라벨링한 작품의 가중치가 2배가 되어 일치율이 왜곡된다)
    final_gold, dup = {}, 0
    for g in gold:
        aid = str(g.get("artwork_id"))
        is_final = truthy(g.get("adjudicated", "")) or truthy(g.get("is_final", ""))
        if aid not in final_gold:
            final_gold[aid] = g
        else:
            dup += 1
            if is_final:
                final_gold[aid] = g
    detail["goldset_rows"] = len(gold)
    detail["goldset_unique_artworks"] = len(final_gold)
    detail["goldset_duplicate_rows"] = dup

    per_layer = {l: {"k": 0, "n": 0} for l in LAYERS}
    fallback_used = 0
    matched_ids = 0

    for g in final_gold.values():
        aid = str(g.get("artwork_id"))
        p = pmap.get(aid)
        if p is None:
            continue
        matched_ids += 1
        if truthy(p.get("fallback_used", False)):
            fallback_used += 1
        for layer in LAYERS:
            r = is_match(layer, g.get(layer + "_gold"),
                         p.get(layer + "_pred"), syn)
            if r is None:
                continue
            per_layer[layer]["n"] += 1
            per_layer[layer]["k"] += 1 if r else 0

    detail["goldset_n"] = len(final_gold)
    detail["prediction_n"] = len(preds)
    detail["matched_n"] = matched_ids
    detail["fallback_rate"] = (fallback_used / matched_ids) if matched_ids else None

    for layer in LAYERS:
        key = layer + "_match"
        z = Z_PRIMARY if TARGETS[key][1] == "primary" else Z_SECONDARY
        c = per_layer[layer]
        if c["n"] > 0:
            finalize_ratio(results[key], c["k"], c["n"], z)
            results[key]["note"] = f"채점규칙: {MATCH_RULE[layer]}"


def compute_latency(data_dir, results, detail):
    results["latency_p95"] = metric_stub("latency_p95", "P95 처리시간(초)")
    rows = read_csv(os.path.join(data_dir, "latency_log.csv"))
    if not rows:
        return

    durations, failures = [], 0
    for r in rows:
        if str(r.get("status", "")).strip().lower() not in ("success", "성공", "ok", "1", "true"):
            failures += 1
            continue
        try:
            d = float(r.get("duration_sec"))
        except (TypeError, ValueError):
            continue
        if d >= 0:
            durations.append(d)

    if not durations:
        return

    p95 = percentile_linear(durations, 0.95)
    lo, hi = bootstrap_ci(durations, lambda s: percentile_linear(s, 0.95))
    m = results["latency_p95"]
    m.update(measured=True, n=len(durations), point=p95, ci_lo=lo, ci_hi=hi,
             decision=decide(lo, hi, m["target"], m["direction"], True), note="")

    detail["latency"] = {
        "n_success": len(durations), "n_failure": failures,
        "failure_rate": failures / len(rows) if rows else None,
        "p50": percentile_linear(durations, 0.50),
        "p90": percentile_linear(durations, 0.90),
        "p95": p95,
        "p99": percentile_linear(durations, 0.99),
        "max": max(durations),
    }
    if len(durations) < 59:
        m["note"] = "표본 부족(n<59): P95 추정 신뢰 불가"
        m["decision"] = "판정 불가(표본 부족)"
    elif len(durations) < 300:
        m["note"] = "표본 300 미만: 자동 로그이므로 확대 권고"


def compute_schema(data_dir, results, detail):
    results["json_first_pass"] = metric_stub("json_first_pass", "JSON 스키마 1차 통과율")
    rows = read_csv(os.path.join(data_dir, "schema_validation_log.csv"))
    if not rows:
        return
    n = len(rows)
    k = sum(1 for r in rows if truthy(r.get("first_pass_valid")))
    final_ok = sum(1 for r in rows if truthy(r.get("final_valid")))
    finalize_ratio(results["json_first_pass"], k, n, Z_PRIMARY)

    detail["schema"] = {
        "n": n, "first_pass": k, "final_valid": final_ok,
        "final_pass_rate": final_ok / n,
        "retry_rate": sum(1 for r in rows
                          if str(r.get("retry_count", "0")).strip() not in ("", "0")) / n,
        "error_types": dict(_count_by(rows, "validation_error")),
    }
    if n < 300:
        need = min_count_for_pass(n, 0.95, Z_PRIMARY)
        results["json_first_pass"]["note"] = (
            "표본 부족: N=%d에서 달성 최소 건수 %s" % (n, need if need is not None else "없음(판정 불가)"))


def compute_top5(data_dir, results, detail):
    results["top5_relevance"] = metric_stub("top5_relevance", "유사작 Top-5 관련성")
    rows = read_csv(os.path.join(data_dir, "top5_relevance_eval.csv"))
    if not rows:
        return

    strict, lenient = defaultdict(list), defaultdict(list)
    by_axis = defaultdict(lambda: [0, 0])
    for r in rows:
        try:
            s = int(float(r.get("curator_score_0_2")))
        except (TypeError, ValueError):
            continue
        q = str(r.get("query_artwork_id")) + "|" + str(r.get("search_axis", ""))
        strict[q].append(1 if s >= 2 else 0)
        lenient[q].append(1 if s >= 1 else 0)
        ax = by_axis[str(r.get("search_axis", "미지정"))]
        ax[0] += 1 if s >= 2 else 0
        ax[1] += 1

    clusters = list(strict.values())
    k = sum(sum(c) for c in clusters)
    n = sum(len(c) for c in clusters)
    if n == 0:
        return

    finalize_ratio(results["top5_relevance"], k, n, Z_SECONDARY)
    clo, chi = cluster_bootstrap_ci(clusters)
    m = results["top5_relevance"]
    # 클러스터 구조를 반영한 구간을 최종 판정 근거로 채택 (더 보수적)
    m["ci_lo"], m["ci_hi"] = clo, chi
    m["decision"] = decide(clo, chi, m["target"], m["direction"], True)
    m["note"] = "엄격기준(2점). 질의 단위 클러스터 부트스트랩 CI 적용"

    lk = sum(sum(c) for c in lenient.values())
    detail["top5"] = {
        "n_queries": len(clusters), "n_items": n,
        "strict_rate": k / n, "lenient_rate": lk / n,
        "by_axis": {a: {"k": v[0], "n": v[1], "rate": v[0] / v[1]}
                    for a, v in by_axis.items() if v[1]},
    }


def compute_survey(data_dir, results, detail):
    rows = read_csv(os.path.join(data_dir, "practitioner_survey.csv"))
    stub = {"key": "practitioner_likert", "label": "실무자 리커트 평균",
            "target": 3.5, "tier": "reference", "direction": "ge",
            "measured": False, "n": 0, "k": None, "point": None,
            "ci_lo": None, "ci_hi": None, "decision": "판정 없음",
            "note": "n=5 소표본. 기술통계만 제시 (§4.10)"}
    results["practitioner_likert"] = stub
    if not rows:
        stub["note"] = "원시 데이터 없음"
        return
    scores = []
    for r in rows:
        try:
            scores.append(float(r.get("likert_score_1_5")))
        except (TypeError, ValueError):
            pass
    if not scores:
        return
    stub.update(measured=True, n=len(scores), point=statistics.mean(scores))
    detail["survey"] = {
        "n": len(scores), "mean": statistics.mean(scores),
        "median": statistics.median(scores), "min": min(scores), "max": max(scores),
        "sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "scores": scores,
    }
    if len(scores) < 13:
        stub["note"] = ("n=%d: 95%% CI 반폭이 약 ±1.0점 수준이므로 3.5 기준 판정 불가. "
                        "판정 대상 아님(기술통계만)." % len(scores))


def _count_by(rows, field):
    c = defaultdict(int)
    for r in rows:
        v = (r.get(field) or "").strip()
        if v:
            c[v] += 1
    return c


def compute_agreement(data_dir, detail):
    """이중 라벨링 골드셋의 라벨러 간 신뢰도 (인간 상한)."""
    rows = read_csv(os.path.join(data_dir, "gold_labels.csv"))
    if not rows:
        return
    out = {}
    for layer in LAYERS:
        units = defaultdict(list)
        for r in rows:
            aid = str(r.get("artwork_id"))
            lab = normalize_label(r.get(layer + "_gold"))
            if lab:
                units[aid].append(lab)
        multi = [v for v in units.values() if len(v) >= 2]
        if multi:
            agree = sum(1 for v in multi if len(set(v)) == 1)
            out[layer] = {
                "double_labeled_n": len(multi),
                "pairwise_agreement": agree / len(multi),
                "krippendorff_alpha": krippendorff_alpha_nominal(multi),
            }
    if out:
        detail["annotator_agreement"] = out


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def fmt_ratio(m):
    if not m["measured"] or m["point"] is None:
        if m["key"] == "latency_p95":
            return "`__초`"
        if m["key"] == "practitioner_likert":
            return "`평균 __ / 5.0, n=__`"
        return "`__ / __ = __%`"
    if m["key"] == "latency_p95":
        return "%.1f초" % m["point"]
    if m["k"] is None:
        return "평균 %.2f / 5.0, n=%d" % (m["point"], m["n"])
    return "%d / %d = %.1f%%" % (m["k"], m["n"], m["point"] * 100)


def fmt_ci(m):
    if m["key"] == "practitioner_likert":
        return "산출 안 함"
    if not m["measured"] or m["ci_lo"] is None or math.isnan(m["ci_lo"]):
        return "`[__, __]`"
    if m["key"] == "latency_p95":
        return "[%.1f, %.1f]초" % (m["ci_lo"], m["ci_hi"])
    if m["k"] is None:
        return "산출 안 함"
    return "[%.1f%%, %.1f%%]" % (m["ci_lo"] * 100, m["ci_hi"] * 100)


def fmt_target(m):
    if m["key"] == "latency_p95":
        return "15초 이내"
    return "%.0f%% 이상" % (m["target"] * 100) if m["target"] <= 1 else "%.1f점 이상" % m["target"]


ORDER = ["color_match", "materiality_match", "technique_match", "iconography_match",
         "latency_p95", "json_first_pass", "top5_relevance", "practitioner_likert"]

GROUP = {"color_match": "4-Layer 라벨 정확도", "materiality_match": "4-Layer 라벨 정확도",
         "technique_match": "4-Layer 라벨 정확도", "iconography_match": "4-Layer 라벨 정확도",
         "latency_p95": "시스템 안정성", "json_first_pass": "시스템 안정성",
         "top5_relevance": "검색 품질", "practitioner_likert": "실무 수용성"}

TIER_KO = {"primary": "1차", "secondary": "2차", "reference": "참고"}


def write_outputs(results, detail, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "results_summary.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["평가구분", "지표", "종점", "목표기준", "분자", "분모",
                    "점추정", "CI하한", "CI상한", "판정", "비고"])
        for key in ORDER:
            m = results.get(key)
            if not m:
                continue
            w.writerow([GROUP[key], m["label"], TIER_KO[m["tier"]], fmt_target(m),
                        m["k"] if m["k"] is not None else "",
                        m["n"] if m["measured"] else "",
                        "" if m["point"] is None else round(m["point"], 4),
                        "" if m["ci_lo"] is None else round(m["ci_lo"], 4),
                        "" if m["ci_hi"] is None else round(m["ci_hi"], 4),
                        m["decision"], m["note"]])

    lines = ["# K-ART LENS 평가 결과 (자동 산출)", "",
             "> 본 표는 `compute_metrics.py`가 사전 등록된 판정 규칙에 따라 생성했다. 수기 수정 금지.", "",
             "| 평가 구분 | 지표 | 종점 | 목표 기준 | 실측 결과 | 95% CI | 판정 | 비고 |",
             "|---|---|:---:|---:|---|---|---|---|"]
    for key in ORDER:
        m = results.get(key)
        if not m:
            continue
        lines.append("| %s | %s | %s | %s | %s | %s | **%s** | %s |" % (
            GROUP[key], m["label"], TIER_KO[m["tier"]], fmt_target(m),
            fmt_ratio(m), fmt_ci(m), m["decision"], m["note"] or "—"))

    prim = [results[k] for k in ORDER if results.get(k, {}).get("tier") == "primary"]
    n_pass = sum(1 for m in prim if m["decision"] == "달성")
    lines += ["", "## 1차 종점 요약", "",
              "- 1차 종점 %d개 중 **달성 %d개**" % (len(prim), n_pass),
              "- 미측정 지표: %s" % (", ".join(m["label"] for m in results.values()
                                            if not m["measured"]) or "없음"), ""]
    if detail:
        lines += ["## 부가 통계", "", "```json",
                  json.dumps(detail, ensure_ascii=False, indent=2, default=str), "```", ""]

    with open(os.path.join(out_dir, "results_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(os.path.join(out_dir, "results_detail.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "detail": detail}, f,
                  ensure_ascii=False, indent=2, default=str)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 자체 검증
# ---------------------------------------------------------------------------
def selftest():
    ok = True

    def check(name, got, exp, tol=1e-4):
        nonlocal ok
        good = abs(got - exp) <= tol if isinstance(exp, float) else got == exp
        print("  [%s] %-42s got=%s exp=%s" % ("OK" if good else "FAIL", name, got, exp))
        ok = ok and good

    print("== Wilson 신뢰구간 ==")
    lo, p, hi = wilson_ci(92, 100, 1.96)
    check("wilson(92/100) 하한", round(lo, 4), 0.8500, 5e-4)
    check("wilson(92/100) 상한", round(hi, 4), 0.9589, 5e-4)

    print("== 판정 임계 카운트 (보고서 §4.6.1) ==")
    check("색채 N=100 (목표.90,z=1.96)", min_count_for_pass(100, 0.90, 1.96), 96)
    check("물성 N=150 (목표.80,z=2.394)", min_count_for_pass(150, 0.80, 2.394), 132)
    check("도상 N=100 (목표.55,z=1.96)", min_count_for_pass(100, 0.55, 1.96), 65)
    check("JSON N=100 (목표.95,z=2.394)", min_count_for_pass(100, 0.95, 2.394), None)
    check("JSON N=150 (목표.95,z=2.394)", min_count_for_pass(150, 0.95, 2.394), 149)

    print("== 필요 표본 수 (보고서 §4.4.2) ==")
    check("색채 e=5%p", required_n(0.90, 0.05), 139)
    check("물성 e=5%p", required_n(0.80, 0.05), 246)
    check("기법 e=5%p", required_n(0.65, 0.05), 350)
    check("도상 e=5%p", required_n(0.55, 0.05), 381)

    print("== 백분위수 ==")
    check("P95 of 1..100", round(percentile_linear(list(range(1, 101)), 0.95), 2), 95.05, 0.01)

    print("== McNemar ==")
    check("McNemar(b=2,c=12) p<0.05", mcnemar_exact(2, 12) < 0.05, True)
    check("McNemar(b=5,c=5) p=1.0", round(mcnemar_exact(5, 5), 4), 1.0)

    print("== Krippendorff alpha ==")
    check("완전 일치 alpha=1", round(krippendorff_alpha_nominal([["a", "a"], ["b", "b"]]), 4), 1.0)

    print("== 채점 규칙 ==")
    check("물성 주라벨 일치", is_match("materiality", "장지에 분채", "장지에 분채(부분 금박)"), True)
    check("물성 불일치", is_match("materiality", "장지에 분채", "캔버스에 아크릴"), False)
    check("도상 부분 겹침", is_match("iconography", "산수|수목", "수목|인물"), True)
    check("색채 Jaccard", is_match("color", "청색|백색", "청색|백색|회색"), True)
    check("유보는 불일치", is_match("technique", "수묵", ""), False)

    print("== 판정 규칙 ==")
    check("CI하한>=목표 -> 달성", decide(0.91, 0.97, 0.90, "ge", True), "달성")
    check("CI가 목표 포함 -> 보류", decide(0.85, 0.96, 0.90, "ge", True), "보류")
    check("CI상한<목표 -> 미달", decide(0.70, 0.84, 0.90, "ge", True), "미달")
    check("지연 CI상한<=15 -> 달성", decide(11.0, 13.5, 15.0, "le", True), "달성")
    check("데이터 없음 -> 미측정", decide(0, 0, 0.9, "ge", False), "미측정")

    print("\n결과: %s" % ("전체 통과" if ok else "실패 항목 있음"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="K-ART LENS 평가 결과 자동 산출기 (프로토콜 v2.0)")
    ap.add_argument("--data", help="원시 데이터 디렉터리")
    ap.add_argument("--out", default="./results", help="결과 출력 디렉터리")
    ap.add_argument("--selftest", action="store_true", help="통계 함수 자체 검증")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if not args.data:
        ap.error("--data 또는 --selftest 중 하나가 필요합니다.")

    results, detail = {}, {}
    manifest = os.path.join(args.data, "run_manifest.json")
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            detail["run_manifest"] = json.load(f)
    else:
        detail["run_manifest"] = "없음 — 재현성 고정 정보 누락 (§3.7 위반)"

    compute_layer_metrics(args.data, results, detail)
    compute_latency(args.data, results, detail)
    compute_schema(args.data, results, detail)
    compute_top5(args.data, results, detail)
    compute_survey(args.data, results, detail)
    compute_agreement(args.data, detail)

    table = write_outputs(results, detail, args.out)
    print(table)
    missing = [m["label"] for m in results.values() if not m["measured"]]
    if missing:
        print("\n[경고] 미측정 지표가 있습니다: %s" % ", ".join(missing))
        print("       해당 항목은 발표 자료에서 공란으로 유지하십시오.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
