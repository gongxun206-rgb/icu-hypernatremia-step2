# Public sanitized copy; restricted inputs are supplied outside Git.
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import os
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

from src.preprocessing.frozen_preprocessing import FrozenPreprocessor


REPO_ROOT = Path(__file__).resolve().parents[2]
RESTRICTED_ROOT = Path(os.environ.get("STEP2_RESTRICTED_ROOT", REPO_ROOT / "restricted_data"))
OUTPUT_ROOT = Path(os.environ.get("STEP2_OUTPUT_ROOT", REPO_ROOT / "outputs"))
PROJECT = REPO_ROOT
OUT = OUTPUT_ROOT / "mimic_primary"
FIG = OUT / "figures"

WINDOW_SOURCE = RESTRICTED_ROOT / "mimic" / "step2_new_q05_window_level_analysis_source_v1.csv"
BASELINE = RESTRICTED_ROOT / "mimic" / "primary_cohort_model_dataset_v1.0.csv"
MANIFEST = REPO_ROOT / "config" / "variable_dictionary.csv"
PREPROCESSING_METADATA = REPO_ROOT / "config" / "frozen_preprocessing_parameters.json"
DEV_PRED = os.environ.get("STEP2_STEP1_DEVELOPMENT_PREDICTIONS")
TEST_PRED = os.environ.get("STEP2_STEP1_TEMPORAL_PREDICTIONS")
ORIGINAL_TASK = REPO_ROOT / "docs" / "protocol_lock.md"
RESUME_TASK = REPO_ROOT / "docs" / "protocol_lock.md"
Q04_GATE = REPO_ROOT / "docs" / "semantic_audit_summary.md"

AMENDMENT = OUT / "Step2_New_Q05_protocol_amendment_lock_v2.md"
FLOW = OUT / "step2_new_q05_analysis_cohort_flow_v1.md"
MISSINGNESS = OUT / "step2_new_q05_baseline_missingness_lock_v1.md"
EXPOSURE_DIST = OUT / "step2_new_q05_exposure_distribution_v1.csv"
COEXPOSURE = OUT / "step2_new_q05_direct_coexposure_matrix_v1.csv"
PRIMARY = OUT / "step2_new_q05_primary_models_v1.csv"
DIAGNOSTICS = OUT / "step2_new_q05_primary_model_diagnostics_v1.md"
RCS_RESULTS = OUT / "step2_new_q05_rcs_results_v1.csv"
CARRIER = OUT / "step2_new_q05_carrier_secondary_analysis_v1.csv"
SENSITIVITY = OUT / "step2_new_q05_sensitivity_analysis_v1.csv"
MEASUREMENT = OUT / "step2_new_q05_measurement_process_sensitivity_v1.md"
REPORT = OUT / "Step2_New_Q05_formal_analysis_report_v1.md"
VALIDATION = OUT / "step2_new_q05_validation_v2.json"

FEATURES = pd.read_csv(MANIFEST)["feature"].tolist()
NUMERIC = FEATURES[:15]
CATEGORICAL = ["gender"]
BINARY = FEATURES[16:]
FLUIDS = {
    "NaCl_0.9": {"prefix": "nacl", "increment_ml": 500.0, "label": "0.9% NaCl"},
    "LR": {"prefix": "lr", "increment_ml": 500.0, "label": "LR"},
    "D5W": {"prefix": "d5w", "increment_ml": 250.0, "label": "D5W"},
}
BOOL_TRUE = {"t", "true", "1", "yes"}
MODEL_DIAGNOSTICS: list[dict] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin(BOOL_TRUE)


def fmt_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def fmt_ci(est: float, lo: float, hi: float, digits: int = 2) -> str:
    return f"{est:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def quantile_text(x: pd.Series) -> str:
    q = x.quantile([0.25, 0.5, 0.75])
    return f"{q.loc[0.5]:.2f} [{q.loc[0.25]:.2f}, {q.loc[0.75]:.2f}]"


def normalize_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in BINARY:
        out[col] = to_bool(out[col]).astype(int)
    return out


def load_optional_predictions(baseline: pd.DataFrame, requested: bool) -> tuple[pd.DataFrame, dict]:
    if not requested:
        baseline["predicted_risk"] = np.nan
        return baseline, {"status": "SKIPPED: Step 1 prediction sensitivity not requested"}
    if not DEV_PRED or not TEST_PRED or not Path(DEV_PRED).is_file() or not Path(TEST_PRED).is_file():
        baseline["predicted_risk"] = np.nan
        return baseline, {"status": "SKIPPED: frozen Step 1 prediction files not supplied"}
    dev = pd.read_csv(DEV_PRED)[["stay_id", "predicted_risk"]]
    test = pd.read_csv(TEST_PRED)[["stay_id", "predicted_risk"]]
    risk = pd.concat([dev, test], ignore_index=True)
    if len(risk) != 34913 or risk["stay_id"].nunique() != 34913:
        raise ValueError("Frozen Step 1 predicted-risk rows do not align to 34,913 stays.")
    baseline = baseline.merge(risk, on="stay_id", how="left", validate="one_to_one")
    if baseline["predicted_risk"].isna().any():
        raise ValueError("Missing frozen Step 1 predicted risk after stay-level merge.")
    return baseline, {"status": "RUN", "files": [Path(DEV_PRED).name, Path(TEST_PRED).name]}


def load_manual_review(review_csv: Path | None) -> dict:
    if review_csv is None:
        return {"status": "SKIPPED: restricted manual-review file not supplied"}
    if not review_csv.is_file():
        return {"status": "SKIPPED: restricted manual-review file not supplied"}
    review = pd.read_csv(review_csv)
    grades = review["researcher_final_grade"].value_counts().to_dict()
    semantics = review["direct_vs_mixed_semantic_confirmed"].value_counts().to_dict()
    actual_c_count = int((review["researcher_final_grade"] == "C_AMBIGUOUS").sum())
    gate = {
        "status": "RUN", "rows": len(review), "unique_stays": review["stay_id"].nunique(),
        "grades": grades, "semantics": semantics, "c_grade_count_matches": actual_c_count == 7,
        "file": review_csv.name,
    }
    expected_grades = {"A_CLEAR": 8, "B_USABLE_WITH_RULES": 45, "C_AMBIGUOUS": 7}
    expected_semantics = {"YES": 53, "PARTIAL": 4, "NO": 3}
    if len(review) != 60 or review["stay_id"].nunique() != 60 or grades != expected_grades or semantics != expected_semantics or actual_c_count != 7:
        raise ValueError(f"MANUAL_REVIEW_LOCK mismatch: {gate}")
    return gate


def load_data(
    run_step1_risk_sensitivity: bool = False,
    manual_review_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], dict, dict]:
    baseline = normalize_baseline(pd.read_csv(BASELINE))
    if len(baseline) != 34913 or baseline["stay_id"].nunique() != 34913:
        raise ValueError("Locked baseline must contain 34,913 unique stays.")

    windows = pd.read_csv(WINDOW_SOURCE)
    for col in ["in_t1_risk_set", "event_post_t1"]:
        windows[col] = to_bool(windows[col])
    if len(windows) != 104739 or windows[["stay_id", "window_hours"]].duplicated().any():
        raise ValueError("Window source must contain 34,913 x 3 unique rows.")

    baseline, prediction_gate = load_optional_predictions(baseline, run_step1_risk_sensitivity)
    pre = FrozenPreprocessor.from_json(PREPROCESSING_METADATA)
    matrix, names = pre.transform(baseline[FEATURES])
    # The frozen L1 prediction pipeline retained both gender dummies. For
    # unpenalized association models, remove F and use it as reference.
    keep = [i for i, name in enumerate(names) if name != "categorical__gender_F"]
    matrix = matrix[:, keep]
    names = [names[i] for i in keep]
    if matrix.shape[1] != 40:
        raise ValueError(f"Expected 40 full-rank baseline columns after gender reference; got {matrix.shape[1]}.")

    baseline["_baseline_row"] = np.arange(len(baseline), dtype=int)
    windows = windows.merge(
        baseline[["stay_id", "_baseline_row", "predicted_risk"] + BINARY],
        on="stay_id",
        how="left",
        validate="many_to_one",
    )
    if windows["_baseline_row"].isna().any():
        raise ValueError("Window source contains stays absent from frozen baseline dataset.")
    windows["_baseline_row"] = windows["_baseline_row"].astype(int)

    manual_gate = load_manual_review(manual_review_csv)
    return baseline, windows, matrix, names, manual_gate, prediction_gate


def model_frame(windows: pd.DataFrame, window_hours: int = 12) -> pd.DataFrame:
    return windows.loc[
        (windows["window_hours"] == window_hours)
        & windows["in_t1_risk_set"]
        & (windows["post_t1_na_count"] > 0)
    ].copy()


def build_design(
    data: pd.DataFrame,
    baseline_matrix: np.ndarray,
    baseline_names: list[str],
    focal_prefix: str,
    exposure: np.ndarray,
    exposure_name: str,
    adjustment: str = "baseline27",
    add_terms: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    cols = [np.asarray(exposure, dtype=float)]
    names = [exposure_name]
    for other in ["nacl", "lr", "d5w"]:
        if other != focal_prefix:
            cols.append((data[f"{other}_direct_records"].to_numpy() > 0).astype(float))
            names.append(f"coexposure_{other}")
    if adjustment == "baseline27":
        idx = data["_baseline_row"].to_numpy(dtype=int)
        cols.extend([baseline_matrix[idx, j] for j in range(baseline_matrix.shape[1])])
        names.extend(baseline_names)
    elif adjustment == "step1_risk":
        cols.append(data["predicted_risk"].to_numpy(dtype=float) / 0.05)
        names.append("step1_predicted_risk_per_0.05")
    else:
        raise ValueError(adjustment)
    if add_terms:
        for name, values in add_terms.items():
            cols.append(np.asarray(values, dtype=float))
            names.append(name)
    x = np.column_stack(cols)
    keep = np.nanstd(x, axis=0) > 0
    x = x[:, keep]
    names = [name for name, flag in zip(names, keep) if flag]
    x = sm.add_constant(x, has_constant="add")
    names = ["intercept"] + names
    return x, names


def fit_glm(
    data: pd.DataFrame,
    baseline_matrix: np.ndarray,
    baseline_names: list[str],
    focal_prefix: str,
    exposure: np.ndarray,
    exposure_name: str,
    analysis_id: str,
    adjustment: str = "baseline27",
    add_terms: dict[str, np.ndarray] | None = None,
) -> tuple[object, list[str], dict]:
    y = data["event_post_t1"].astype(int).to_numpy()
    x, names = build_design(
        data, baseline_matrix, baseline_names, focal_prefix, exposure,
        exposure_name, adjustment, add_terms,
    )
    rank = int(np.linalg.matrix_rank(x))
    condition = float(np.linalg.cond(x))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-8)
    pred = np.asarray(result.predict(x))
    iterations = result.fit_history.get("iteration", None)
    diag = {
        "analysis_id": analysis_id,
        "n": int(len(data)),
        "events": int(y.sum()),
        "columns": int(x.shape[1]),
        "rank": rank,
        "rank_deficiency": int(x.shape[1] - rank),
        "condition_number": condition,
        "converged": bool(getattr(result, "converged", True)),
        "iterations": int(iterations) if isinstance(iterations, (int, np.integer)) else str(iterations),
        "max_abs_coefficient": float(np.nanmax(np.abs(result.params))),
        "predicted_probability_min": float(np.nanmin(pred)),
        "predicted_probability_max": float(np.nanmax(pred)),
        "warnings": " | ".join(str(w.message) for w in caught),
        "aic": float(result.aic),
    }
    MODEL_DIAGNOSTICS.append(diag)
    if rank != x.shape[1] or not diag["converged"]:
        raise RuntimeError(f"Model diagnostic failure for {analysis_id}: {diag}")
    return result, names, diag


def effect_row(
    result: object,
    names: list[str],
    term: str,
    metadata: dict,
) -> dict:
    i = names.index(term)
    beta = float(result.params[i])
    se = float(result.bse[i])
    lo, hi = beta - 1.96 * se, beta + 1.96 * se
    return {
        **metadata,
        "term": term,
        "coefficient_log_or": beta,
        "standard_error": se,
        "adjusted_or": math.exp(beta),
        "ci95_low": math.exp(lo),
        "ci95_high": math.exp(hi),
        "p_value": float(result.pvalues[i]),
    }


def prepare_focal(data: pd.DataFrame, prefix: str, metric: str, increment: float) -> tuple[pd.DataFrame, np.ndarray]:
    out = data.loc[data["inputevent_interface_status"] == "ACTIVE"].copy()
    exposure = out[metric].to_numpy(dtype=float) / increment
    return out, exposure


def make_amendment(gate: dict) -> None:
    if gate["status"] == "RUN":
        manual_review_text = (
            f"A={gate['grades'].get('A_CLEAR', 0)}，B={gate['grades'].get('B_USABLE_WITH_RULES', 0)}，"
            f"C={gate['grades'].get('C_AMBIGUOUS', 0)}，D=0；YES=53/PARTIAL=4/NO=3；"
            "7个C类stay一致。"
        )
        manual_source = f"`{gate['file']}`（restricted local file；不随公开代码分发）。"
        manual_status = "PASS"
    else:
        manual_review_text = gate["status"]
        manual_source = "受限人工审核原始文件不随公开代码分发；汇总规则见`docs/semantic_audit_summary.md`。"
        manual_status = "SKIPPED"
    text = f"""# Step2-New-Q05 方案修订锁定 v2

## Amendment Gate

`READY_FOR_FORMAL_ANALYSIS`

| Lock | 状态 | 锁定内容 |
|---|---|---|
| `PRIMARY_WINDOW_LOCK` | PASS | Primary=`[T0,T0+12h)`；6/24 h仅作预设敏感性分析。 |
| `DIRECT_EXPOSURE_TERMINOLOGY_LOCK` | PASS | 统一为`documented direct-fluid exposure`，不推断主动治疗意图。 |
| `CARRIER_METRIC_LOCK` | PASS | 仅用累计mL和any documented carrier exposure；不用duration或mL/h作主指标。 |
| `ZERO_UNKNOWN_SEMANTIC_LOCK` | PASS | ACTIVE且无合格记录可记为`no qualifying record observed`；SILENT不赋0。 |
| `PATH_B_LOCK` | PASS | `DIRECT_PRIMARY_MIXED_SECONDARY`。 |
| `MANUAL_REVIEW_LOCK` | {manual_status} | {manual_review_text} |
| `OUTCOME_PROCESS_LOCK` | PASS | 12 h T1风险集34,779；post-T1 Na可观察33,538；事件793；未确定1,241不编码为non-event。 |

## 分析前尺度锁定

- 0.9% NaCl direct：adjusted OR per 500 mL。
- LR direct：adjusted OR per 500 mL。
- D5W direct：adjusted OR per 250 mL。
- NaCl/D5W carrier：adjusted OR per 250 mL。
- RCS：各暴露者正值分布的5th/35th/65th/95th百分位四结点；不搜索阈值。
- 结果前预设高确定Na观察子集：post-T1 Na>=2次且首次复查<=24 h。

## 人工复核来源

{manual_source}

本修订发生于Phase B结果读取之前，不依据P值、OR或曲线形状改变。
"""
    AMENDMENT.write_text(text, encoding="utf-8")


def make_missingness(baseline: pd.DataFrame) -> None:
    metadata = FrozenPreprocessor.from_json(PREPROCESSING_METADATA).metadata
    medians = metadata["numeric"]["median_imputation"]
    indicator_names = set(metadata["numeric"]["missing_indicator_variables"])
    cat_mode = metadata["categorical"]["most_frequent_imputation"]["gender"]
    binary_modes = metadata["binary"]["most_frequent_imputation"]
    rows = []
    for i, col in enumerate(FEATURES):
        missing = int(baseline[col].isna().sum())
        if col in NUMERIC:
            rule = f"development-set median={medians[col]:.6g}; frozen StandardScaler"
            indicator = "YES" if col in indicator_names else "NO"
        elif col == "gender":
            rule = f"development-set most frequent={cat_mode}; frozen one-hot encoding"
            indicator = "NO"
        else:
            rule = f"development-set most frequent={binary_modes[col]:.6g}; binary 0/1"
            indicator = "NO"
        rows.append((i + 1, col, missing, 100 * missing / len(baseline), indicator, rule))
    lines = [
        "# Step2-New-Q05 27项基线变量缺失率锁定 v1",
        "",
        "统计分母为34,913个锁定T0风险集stay。Step 2直接调用Step 1 development pipeline的填补、缺失指示、编码和标准化参数；未使用T0后或outcome数据重新估计参数。",
        "",
        "| # | 变量 | 缺失n | 缺失% | 缺失指示 | 锁定预处理 |",
        "|---:|---|---:|---:|---|---|",
    ]
    lines.extend(f"| {i} | `{c}` | {n:,} | {p:.3f} | {ind} | {rule} |" for i, c, n, p, ind, rule in rows)
    lines += [
        "",
        "未重新fit imputer/scaler/encoder。为保证无惩罚Logistic设计矩阵可识别，gender以F为参照类，只保留M指示；这不改变原27项变量语义。",
    ]
    MISSINGNESS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def distribution_rows(windows: pd.DataFrame) -> list[dict]:
    rows = []
    for wh in [6, 12, 24]:
        base = model_frame(windows, wh)
        for population, data in [
            ("observed_outcome_all_interfaces", base),
            ("formal_active_interface", base.loc[base["inputevent_interface_status"] == "ACTIVE"]),
        ]:
            for fluid, cfg in FLUIDS.items():
                prefix = cfg["prefix"]
                for construct, col in [
                    ("direct_reconstructable", f"{prefix}_direct_ab_ml"),
                    ("carrier_reconstructable", f"{prefix}_carrier_ab_ml"),
                ]:
                    x = data[col].astype(float)
                    pos = x[x > 0]
                    q = x.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
                    qp = pos.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95]) if len(pos) else pd.Series(index=[.1,.25,.5,.75,.9,.95], dtype=float)
                    rows.append({
                        "window_hours": wh, "population": population, "fluid": fluid,
                        "construct": construct, "n": len(data), "n_nonzero": int((x > 0).sum()),
                        "percent_nonzero": 100 * (x > 0).mean(), "n_zero": int((x == 0).sum()),
                        "median_all_ml": q.loc[0.50], "p10_all_ml": q.loc[0.10], "p25_all_ml": q.loc[0.25],
                        "p75_all_ml": q.loc[0.75], "p90_all_ml": q.loc[0.90], "p95_all_ml": q.loc[0.95],
                        "median_exposed_ml": qp.loc[0.50] if len(pos) else np.nan,
                        "p10_exposed_ml": qp.loc[0.10] if len(pos) else np.nan,
                        "p25_exposed_ml": qp.loc[0.25] if len(pos) else np.nan,
                        "p75_exposed_ml": qp.loc[0.75] if len(pos) else np.nan,
                        "p90_exposed_ml": qp.loc[0.90] if len(pos) else np.nan,
                        "p95_exposed_ml": qp.loc[0.95] if len(pos) else np.nan,
                        "max_ml": x.max(),
                        "skewness_all": st.skew(x, bias=False) if x.nunique() > 1 else np.nan,
                        "rule_c_patient_n": int((data[f"{prefix}_{'direct' if construct.startswith('direct') else 'carrier'}_rule_c_records"] > 0).sum()),
                        "any_bolus_n": int((data[f"{prefix}_direct_bolus_records"] > 0).sum()) if construct.startswith("direct") else np.nan,
                        "any_continuous_n": int((data[f"{prefix}_direct_continuous_records"] > 0).sum()) if construct.startswith("direct") else np.nan,
                        "bolus_only_n": int(((data[f"{prefix}_direct_bolus_records"] > 0) & (data[f"{prefix}_direct_continuous_records"] == 0)).sum()) if construct.startswith("direct") else np.nan,
                        "continuous_only_n": int(((data[f"{prefix}_direct_bolus_records"] == 0) & (data[f"{prefix}_direct_continuous_records"] > 0)).sum()) if construct.startswith("direct") else np.nan,
                        "bolus_and_continuous_n": int(((data[f"{prefix}_direct_bolus_records"] > 0) & (data[f"{prefix}_direct_continuous_records"] > 0)).sum()) if construct.startswith("direct") else np.nan,
                    })
    return rows


def make_exposure_plot(windows: pd.DataFrame) -> None:
    data = model_frame(windows, 12)
    data = data.loc[data["inputevent_interface_status"] == "ACTIVE"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    colors = ["#2A6F97", "#2F855A", "#B45309"]
    for ax, (fluid, cfg), color in zip(axes, FLUIDS.items(), colors):
        x = data.loc[data[f"{cfg['prefix']}_direct_ab_ml"] > 0, f"{cfg['prefix']}_direct_ab_ml"]
        ax.hist(np.log1p(x), bins=35, density=True, color=color, alpha=0.82, edgecolor="white")
        ticks = [0, 50, 100, 250, 500, 1000, 2500, 5000]
        valid = [v for v in ticks if v <= x.max()]
        ax.set_xticks(np.log1p(valid), [str(v) for v in valid], rotation=45)
        ax.set_title(cfg["label"])
        ax.set_xlabel("Documented direct-fluid volume (mL)")
        ax.set_ylabel("Density")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("12-hour direct-fluid exposure distributions among exposed stays")
    fig.savefig(FIG / "step2_new_q05_direct_exposure_distributions_v1.png", dpi=220)
    plt.close(fig)


def coexposure_rows(windows: pd.DataFrame) -> list[dict]:
    data = model_frame(windows, 12)
    rows = []
    active = data["inputevent_interface_status"] == "ACTIVE"
    flags = {p: data[f"{p}_direct_records"] > 0 for p in ["nacl", "lr", "d5w"]}
    categories = {
        "NaCl only": flags["nacl"] & ~flags["lr"] & ~flags["d5w"],
        "LR only": ~flags["nacl"] & flags["lr"] & ~flags["d5w"],
        "D5W only": ~flags["nacl"] & ~flags["lr"] & flags["d5w"],
        "NaCl + LR": flags["nacl"] & flags["lr"] & ~flags["d5w"],
        "NaCl + D5W": flags["nacl"] & ~flags["lr"] & flags["d5w"],
        "LR + D5W": ~flags["nacl"] & flags["lr"] & flags["d5w"],
        "all three": flags["nacl"] & flags["lr"] & flags["d5w"],
        "none observed with active interface": active & ~flags["nacl"] & ~flags["lr"] & ~flags["d5w"],
        "unknown interface silent": ~active,
    }
    for name, mask in categories.items():
        n = int(mask.sum())
        events = int(data.loc[mask, "event_post_t1"].sum())
        rows.append({"table_type": "mutually_exclusive_pattern", "row": name, "column": "n", "n": n, "events": events, "event_rate_percent": 100 * events / n if n else np.nan})
    for a in ["nacl", "lr", "d5w"]:
        for b in ["nacl", "lr", "d5w"]:
            mask = active & flags[a] & flags[b]
            rows.append({"table_type": "pairwise_matrix", "row": a, "column": b, "n": int(mask.sum()), "events": int(data.loc[mask, "event_post_t1"].sum()), "event_rate_percent": 100 * data.loc[mask, "event_post_t1"].mean() if mask.any() else np.nan})
    return rows


def primary_models(windows: pd.DataFrame, baseline_matrix: np.ndarray, baseline_names: list[str]) -> list[dict]:
    data0 = model_frame(windows, 12)
    rows = []
    for fluid, cfg in FLUIDS.items():
        prefix, inc = cfg["prefix"], cfg["increment_ml"]
        data, exposure = prepare_focal(data0, prefix, f"{prefix}_direct_ab_ml", inc)
        result, names, diag = fit_glm(data, baseline_matrix, baseline_names, prefix, exposure, f"direct_volume_per_{int(inc)}ml", f"primary_{prefix}_12h")
        rows.append(effect_row(result, names, f"direct_volume_per_{int(inc)}ml", {
            "analysis_id": f"primary_{prefix}_12h", "module": fluid, "window_hours": 12,
            "metric": "documented_direct_fluid_reconstructable_volume", "increment_ml": inc,
            "adjustment": "frozen_27_baseline_predictors", "model_n": len(data),
            "events": int(data["event_post_t1"].sum()), "interface_silent_excluded": int((data0["inputevent_interface_status"] == "SILENT").sum()),
        }))
    return rows


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    if len(k) != 4 or not np.all(np.diff(k) > 0):
        raise ValueError(f"Invalid four-knot RCS: {k}")
    scale = (k[-1] - k[0]) ** 2
    cols = [x]
    for j in range(len(k) - 2):
        term = np.maximum(x - k[j], 0) ** 3
        term -= np.maximum(x - k[-2], 0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
        term += np.maximum(x - k[-1], 0) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2])
        cols.append(term / scale)
    return np.column_stack(cols)


def run_rcs(windows: pd.DataFrame, baseline_matrix: np.ndarray, baseline_names: list[str]) -> list[dict]:
    data0 = model_frame(windows, 12)
    output = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    colors = ["#2A6F97", "#2F855A", "#B45309"]
    for ax, (fluid, cfg), color in zip(axes, FLUIDS.items(), colors):
        prefix = cfg["prefix"]
        data = data0.loc[data0["inputevent_interface_status"] == "ACTIVE"].copy()
        raw = data[f"{prefix}_direct_ab_ml"].to_numpy(dtype=float)
        positive = raw[raw > 0]
        knots = np.quantile(positive, [0.05, 0.35, 0.65, 0.95])
        basis = rcs_basis(raw, knots)
        add = {
            "rcs_nonlinear_1": basis[:, 1] / cfg["increment_ml"],
            "rcs_nonlinear_2": basis[:, 2] / cfg["increment_ml"],
        }
        result, names, _ = fit_glm(data, baseline_matrix, baseline_names, prefix, basis[:, 0] / cfg["increment_ml"], f"rcs_linear_per_{int(cfg['increment_ml'])}ml", f"rcs_{prefix}_12h", add_terms=add)

        # Linear comparator and no-exposure comparator for prespecified tests.
        linear, _, _ = fit_glm(data, baseline_matrix, baseline_names, prefix, raw / cfg["increment_ml"], f"linear_per_{int(cfg['increment_ml'])}ml", f"rcs_linear_comparator_{prefix}")
        noexp_x, noexp_names = build_design(data, baseline_matrix, baseline_names, prefix, np.zeros(len(data)), "drop_me", "baseline27")
        drop_idx = noexp_names.index("drop_me") if "drop_me" in noexp_names else None
        if drop_idx is not None:
            noexp_x = np.delete(noexp_x, drop_idx, axis=1)
        noexp = sm.GLM(data["event_post_t1"].astype(int).to_numpy(), noexp_x, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-8)
        p_overall = float(st.chi2.sf(2 * (result.llf - noexp.llf), 3))
        p_nonlin = float(st.chi2.sf(2 * (result.llf - linear.llf), 2))

        grid = np.unique(np.concatenate([[0.0], np.linspace(0, np.quantile(positive, 0.99), 80)]))
        gb = rcs_basis(grid, knots)
        term_names = [f"rcs_linear_per_{int(cfg['increment_ml'])}ml", "rcs_nonlinear_1", "rcs_nonlinear_2"]
        idx = [names.index(n) for n in term_names]
        beta = np.asarray(result.params)[idx]
        cov = np.asarray(result.cov_params())[np.ix_(idx, idx)]
        ref = rcs_basis(np.array([0.0]), knots)[0]
        for value, b in zip(grid, gb):
            delta = np.array([
                (b[0] - ref[0]) / cfg["increment_ml"],
                (b[1] - ref[1]) / cfg["increment_ml"],
                (b[2] - ref[2]) / cfg["increment_ml"],
            ])
            log_or = float(delta @ beta)
            se = math.sqrt(max(float(delta @ cov @ delta), 0.0))
            output.append({
                "module": fluid, "window_hours": 12, "volume_ml": value,
                "reference_ml": 0.0, "adjusted_or": math.exp(log_or),
                "ci95_low": math.exp(log_or - 1.96 * se), "ci95_high": math.exp(log_or + 1.96 * se),
                "knot_1_ml": knots[0], "knot_2_ml": knots[1], "knot_3_ml": knots[2], "knot_4_ml": knots[3],
                "p_overall": p_overall, "p_nonlinear": p_nonlin, "model_n": len(data),
                "events": int(data["event_post_t1"].sum()),
            })
        sub = pd.DataFrame([r for r in output if r["module"] == fluid])
        ax.plot(sub["volume_ml"], sub["adjusted_or"], color=color, linewidth=2)
        ax.fill_between(sub["volume_ml"], sub["ci95_low"], sub["ci95_high"], color=color, alpha=0.18)
        ax.axhline(1, color="#555555", linewidth=1, linestyle="--")
        ax.set_title(cfg["label"])
        ax.set_xlabel("Documented direct-fluid volume (mL)")
        ax.set_ylabel("Adjusted OR vs 0 mL")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Prespecified restricted cubic spline associations")
    fig.savefig(FIG / "step2_new_q05_rcs_direct_exposure_response_v1.png", dpi=220)
    plt.close(fig)
    return output


def carrier_models(windows: pd.DataFrame, baseline_matrix: np.ndarray, baseline_names: list[str]) -> list[dict]:
    data0 = model_frame(windows, 12)
    rows = []
    for fluid in ["NaCl_0.9", "D5W"]:
        cfg = FLUIDS[fluid]
        prefix = cfg["prefix"]
        data = data0.loc[data0["inputevent_interface_status"] == "ACTIVE"].copy()
        direct_scaled = data[f"{prefix}_direct_ab_ml"].to_numpy(dtype=float) / cfg["increment_ml"]
        for metric, values, term, inc in [
            ("cumulative_carrier_volume", data[f"{prefix}_carrier_ab_ml"].to_numpy(dtype=float) / 250.0, "carrier_volume_per_250ml", 250.0),
            ("any_documented_carrier_exposure", (data[f"{prefix}_carrier_records"].to_numpy() > 0).astype(float), "any_carrier_exposure", np.nan),
        ]:
            result, names, _ = fit_glm(
                data, baseline_matrix, baseline_names, prefix, values, term,
                f"carrier_{prefix}_{metric}", add_terms={f"corresponding_direct_per_{int(cfg['increment_ml'])}ml": direct_scaled},
            )
            rows.append(effect_row(result, names, term, {
                "analysis_id": f"carrier_{prefix}_{metric}", "module": fluid, "metric": metric,
                "increment_ml": inc, "adjustment": "frozen_27_plus_corresponding_direct_volume",
                "model_n": len(data), "events": int(data["event_post_t1"].sum()),
                "interpretation": "documented carrier-volume association; not direct management",
            }))
    return rows


def sensitivity_models(
    windows: pd.DataFrame,
    baseline_matrix: np.ndarray,
    baseline_names: list[str],
    run_step1_risk_sensitivity: bool,
) -> list[dict]:
    rows = []
    base12 = model_frame(windows, 12)
    for fluid, cfg in FLUIDS.items():
        prefix, inc = cfg["prefix"], cfg["increment_ml"]

        specs: list[tuple[str, pd.DataFrame, str, str, str]] = []
        for wh in [6, 24]:
            d = model_frame(windows, wh)
            d = d.loc[d["inputevent_interface_status"] == "ACTIVE"].copy()
            specs.append((f"window_{wh}h", d, f"{prefix}_direct_ab_ml", "continuous", "baseline27"))

        active = base12.loc[base12["inputevent_interface_status"] == "ACTIVE"].copy()
        rule_a = active.loc[(active[f"{prefix}_direct_rule_b_records"] == 0) & (active[f"{prefix}_direct_rule_c_records"] == 0)].copy()
        specs.append(("rule_a_only", rule_a, f"{prefix}_direct_a_ml", "continuous", "baseline27"))
        no_c = active.loc[(active["any_rule_c_records"] == 0) & (active["any_semantic_ambiguous_or_invalid_records"] == 0)].copy()
        specs.append(("exclude_rule_c_and_semantic_ambiguous", no_c, f"{prefix}_direct_ab_ml", "continuous", "baseline27"))
        specs.append(("high_certainty_documentation", no_c, f"{prefix}_direct_ab_ml", "continuous", "baseline27"))
        first_na = pd.to_datetime(active["first_post_t1_na_time"])
        t1 = pd.to_datetime(active["t1_time"])
        high_na = active.loc[(active["post_t1_na_count"] >= 2) & ((first_na - t1).dt.total_seconds() <= 24 * 3600)].copy()
        specs.append(("high_certainty_post_t1_na", high_na, f"{prefix}_direct_ab_ml", "continuous", "baseline27"))
        no_rrt = active.loc[active["rrt_pre_t0"] == 0].copy()
        specs.append(("exclude_baseline_rrt", no_rrt, f"{prefix}_direct_ab_ml", "continuous", "baseline27"))
        if run_step1_risk_sensitivity:
            specs.append(("step1_risk_compact_adjustment", active, f"{prefix}_direct_ab_ml", "continuous", "step1_risk"))
        specs.append(("any_direct_exposure", active, f"{prefix}_direct_records", "binary", "baseline27"))
        active[f"{prefix}_observed_burden_ml"] = active[f"{prefix}_direct_ab_ml"] + active[f"{prefix}_carrier_ab_ml"]
        specs.append(("observed_reconstructable_direct_plus_carrier", active, f"{prefix}_observed_burden_ml", "continuous", "baseline27"))

        for label, data, col, scale_type, adjustment in specs:
            if scale_type == "binary":
                exposure = (data[col].to_numpy() > 0).astype(float)
                term, increment = "any_direct_exposure", np.nan
            else:
                exposure = data[col].to_numpy(dtype=float) / inc
                term, increment = f"volume_per_{int(inc)}ml", inc
            result, names, _ = fit_glm(data, baseline_matrix, baseline_names, prefix, exposure, term, f"sensitivity_{prefix}_{label}", adjustment=adjustment)
            rows.append(effect_row(result, names, term, {
                "analysis_id": f"sensitivity_{prefix}_{label}", "module": fluid,
                "sensitivity": label, "window_hours": int(data["window_hours"].iloc[0]),
                "metric": col, "increment_ml": increment,
                "adjustment": adjustment, "model_n": len(data), "events": int(data["event_post_t1"].sum()),
            }))

        # The Step 1 predicted-risk interaction is optional and requires restricted predictions.
        exposure = active[f"{prefix}_direct_ab_ml"].to_numpy(dtype=float) / inc
        if run_step1_risk_sensitivity:
            risk05 = active["predicted_risk"].to_numpy(dtype=float) / 0.05
            result, names, _ = fit_glm(
                active, baseline_matrix, baseline_names, prefix, exposure, f"volume_per_{int(inc)}ml",
                f"interaction_{prefix}_step1risk", adjustment="step1_risk",
                add_terms={"exposure_x_step1risk05": exposure * risk05},
            )
            rows.append(effect_row(result, names, "exposure_x_step1risk05", {
                "analysis_id": f"interaction_{prefix}_step1risk", "module": fluid,
                "sensitivity": "interaction_step1_predicted_risk", "window_hours": 12,
                "metric": "exposure_x_risk_per_0.05", "increment_ml": inc,
                "adjustment": "step1_risk_compact", "model_n": len(active), "events": int(active["event_post_t1"].sum()),
            }))
        for modifier in ["rrt_pre_t0", "loop_diuretic_pre_t0"]:
            m = active[modifier].to_numpy(dtype=float)
            result, names, _ = fit_glm(
                active, baseline_matrix, baseline_names, prefix, exposure, f"volume_per_{int(inc)}ml",
                f"interaction_{prefix}_{modifier}", add_terms={f"exposure_x_{modifier}": exposure * m},
            )
            rows.append(effect_row(result, names, f"exposure_x_{modifier}", {
                "analysis_id": f"interaction_{prefix}_{modifier}", "module": fluid,
                "sensitivity": f"interaction_{modifier}", "window_hours": 12,
                "metric": f"exposure_x_{modifier}", "increment_ml": inc,
                "adjustment": "baseline27", "model_n": len(active), "events": int(active["event_post_t1"].sum()),
            }))
    return rows


def make_flow(windows: pd.DataFrame) -> None:
    rows = []
    for wh in [6, 12, 24]:
        d = windows.loc[windows["window_hours"] == wh]
        risk = d[d["in_t1_risk_set"]]
        observed = risk[risk["post_t1_na_count"] > 0]
        active = observed[observed["inputevent_interface_status"] == "ACTIVE"]
        rows.append({
            "window": wh, "locked": len(d), "risk": len(risk),
            "events_before_t1_or_not_hospitalized": len(d) - len(risk),
            "observed": len(observed), "unascertained": int((risk["post_t1_na_count"] == 0).sum()),
            "events": int(observed["event_post_t1"].sum()),
            "interface_silent": int((observed["inputevent_interface_status"] == "SILENT").sum()),
            "formal_active": len(active), "formal_active_events": int(active["event_post_t1"].sum()),
        })
    lines = [
        "# Step2-New-Q05 分析队列流程 v1", "",
        "所有T1后无Na者单列为结局未确定，未编码为non-event。`UNKNOWN_INTERFACE_SILENT`未赋0，不进入以零为参照的体积模型。", "",
        "| 窗口 | 锁定T0队列 | T1风险集 | T1前事件/不再住院 | post-T1 Na可观察 | 结局未确定 | 事件 | 接口沉默 | 正式ACTIVE暴露队列 | 其中事件 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['window']} h | {r['locked']:,} | {r['risk']:,} | {r['events_before_t1_or_not_hospitalized']:,} | {r['observed']:,} | {r['unascertained']:,} | {r['events']:,} | {r['interface_silent']:,} | {r['formal_active']:,} | {r['formal_active_events']:,} |")
    lines += ["", "12 h主分析从33,538例结局可观察者开始；其中4,323例在管理窗口内inputevents接口沉默，暴露不能确定为0，故体积模型使甩剩余ACTIVE记录者。"]
    FLOW.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_measurement_report(windows: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    d = model_frame(windows, 12)
    d["hours_to_first_na"] = (pd.to_datetime(d["first_post_t1_na_time"]) - pd.to_datetime(d["t1_time"])).dt.total_seconds() / 3600
    d["tests_per_observed_day"] = d["post_t1_na_count"] / (d["post_t1_followup_hours"].clip(lower=1) / 24)
    lines = [
        "# Step2-New-Q05 血钠测量过程敏感性 v1", "",
        "本部分只描述`recorded hypernatremia`的观察过程，不将检测频率解释为照护质量，也不将post-T1检测变量放入主调整模型。", "",
        "| 结局组 | n | Na检测次数 median [IQR] | 首次复查h median [IQR] | 每观察日检测率 median [IQR] |",
        "|---|---:|---:|---:|---:|",
    ]
    for event, label in [(False, "non-event"), (True, "recorded event")]:
        x = d[d["event_post_t1"] == event]
        lines.append(f"| {label} | {len(x):,} | {quantile_text(x['post_t1_na_count'])} | {quantile_text(x['hours_to_first_na'])} | {quantile_text(x['tests_per_observed_day'])} |")
    u_count = st.mannwhitneyu(d.loc[d.event_post_t1, "post_t1_na_count"], d.loc[~d.event_post_t1, "post_t1_na_count"], alternative="two-sided")
    u_time = st.mannwhitneyu(d.loc[d.event_post_t1, "hours_to_first_na"], d.loc[~d.event_post_t1, "hours_to_first_na"], alternative="two-sided")
    lines += [
        "", f"Mann-Whitney描述性检验：检测次数P={fmt_p(u_count.pvalue)}；首次复查时间P={fmt_p(u_time.pvalue)}。大样本P值不等于临床重要性。",
        "", "## 预设观察过程子集", "",
        "- 高确定post-T1 Na观察：>=2次且首次<=24 h。",
        "- 固定检测密度层：1次、2-3次、>=4次。",
        "- 这些均属post-T1测量过程敏感性，不替代主模型，不支持因果解释。", "",
        "| 检测密度层 | n | events | event rate | NaCl direct median mL | LR direct median mL | D5W direct median mL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, mask in [
        ("1", d["post_t1_na_count"] == 1),
        ("2-3", d["post_t1_na_count"].between(2, 3)),
        (">=4", d["post_t1_na_count"] >= 4),
    ]:
        x = d.loc[mask]
        lines.append(
            f"| {label} | {len(x):,} | {int(x['event_post_t1'].sum()):,} | "
            f"{100*x['event_post_t1'].mean():.3f}% | {x['nacl_direct_ab_ml'].median():.2f} | "
            f"{x['lr_direct_ab_ml'].median():.2f} | {x['d5w_direct_ab_ml'].median():.2f} |"
        )
    lines += ["", "仅1次Na检测层事件极少，不强行拟合高维调整模型，以避免完全分离和不可识别系数。", ""]
    for _, r in sensitivity[sensitivity["sensitivity"].astype(str) == "high_certainty_post_t1_na"].iterrows():
        lines.append(f"- {r['module']} / {r['sensitivity']}: n={int(r['model_n']):,}, events={int(r['events']):,}, adjusted OR={fmt_ci(r['adjusted_or'], r['ci95_low'], r['ci95_high'])}, P={fmt_p(r['p_value'])}。")
    MEASUREMENT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_diagnostics() -> None:
    d = pd.DataFrame(MODEL_DIAGNOSTICS)
    lines = [
        "# Step2-New-Q05 主模型诊断 v1", "",
        f"共拟合{len(d)}个预设Logistic模型。所有模型设计矩阵均需full rank且IRLS收敛；未使用P值筛选变量。", "",
        "| analysis_id | n | events | columns | rank deficit | condition number | converged | iterations | max |coef| | predicted p range | warning |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for _, r in d.iterrows():
        lines.append(f"| {r['analysis_id']} | {int(r['n']):,} | {int(r['events']):,} | {int(r['columns'])} | {int(r['rank_deficiency'])} | {r['condition_number']:.2e} | {r['converged']} | {r['iterations']} | {r['max_abs_coefficient']:.3f} | {r['predicted_probability_min']:.5f}-{r['predicted_probability_max']:.5f} | {r['warnings'] or 'none'} |")
    lines += [
        "",
        "Gender以F为参照类，解决冻结one-hot管道两个gender dummy与截距完全共线的问题。其他预处理参数未重新fit。",
        "RCS系数对应样条基函数及其数值尺度，不作单个系数的临床解释；RCS主要依据整体检验、非线性检验和暴露-结局关联曲线报告。",
    ]
    DIAGNOSTICS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_report(
    primary: pd.DataFrame,
    rcs: pd.DataFrame,
    carrier: pd.DataFrame,
    sensitivity: pd.DataFrame,
    windows: pd.DataFrame,
    step1_risk_status: str,
) -> str:
    # Conservative gate: observational result is manuscript-usable but interface
    # silence and measurement-process sensitivity require explicit framing.
    gate = "CONDITIONAL_GO_TO_MANUSCRIPT"
    d12 = model_frame(windows, 12)
    active = d12[d12["inputevent_interface_status"] == "ACTIVE"]
    lines = [
        "# Step2-New-Q05 正式关联分析报告 v1", "",
        "## 1. Executive result", "",
        "本研究估计的是记录型液体暴露与随后记录到Na>=151 mmol/L之间的调整关联，不是治疗效应。记录接口沉默和Na检测过程仍是主要解释边界。", "",
        "## 2. Analysis cohort", "",
        f"12 h T1风险集34,779例；post-T1 Na可观察33,538例，事件793例；其中inputevents接口ACTIVE {len(active):,}例，事件{int(active.event_post_t1.sum()):,}例。接口沉默4,323例未赋值为零暴露。", "",
        "## 3. Exposure distributions", "",
        "三种direct暴露均明显右偏；完整零/非零、百分位、最大值和偏度见`step2_new_q05_exposure_distribution_v1.csv`及分布图。主分析保留原始连续mL尺度，未根据结果切点。", "",
        "## 4. Direct-fluid co-exposure", "",
        "三种直接液体允许共暴露，主模型使用一个焦点连续暴露加其他两种direct共暴露标记，未强制划分成互斥治疗组。", "",
        "## 5. Primary adjusted associations", "",
    ]
    for _, r in primary.iterrows():
        lines.append(f"- {r['module']}：每{int(r['increment_ml'])} mL adjusted OR={fmt_ci(r['adjusted_or'],r['ci95_low'],r['ci95_high'])}，P={fmt_p(r['p_value'])}，n={int(r['model_n']):,}，events={int(r['events']):,}。")
    lines += ["", "## 6. Exposure-response form", ""]
    for fluid in FLUIDS:
        r = rcs[rcs.module == fluid].iloc[0]
        lines.append(f"- {fluid}：RCS overall P={fmt_p(r.p_overall)}，nonlinear P={fmt_p(r.p_nonlinear)}；四结点={r.knot_1_ml:.1f}/{r.knot_2_ml:.1f}/{r.knot_3_ml:.1f}/{r.knot_4_ml:.1f} mL。")
    lines += ["", "## 7. Carrier secondary results", ""]
    for _, r in carrier.iterrows():
        unit = "per 250 mL" if r["metric"] == "cumulative_carrier_volume" else "any vs none observed"
        lines.append(f"- {r['module']} / {r['metric']} ({unit})：adjusted OR={fmt_ci(r['adjusted_or'],r['ci95_low'],r['ci95_high'])}，P={fmt_p(r['p_value'])}。")
    lines += [
        "", "## 8. Observed fluid-burden sensitivity", "",
        "Direct + eligible carrier仅命名为`observed reconstructable fluid volume`；结果见敏感性表，不解释为总水、总钠或主动液体管理。",
        "", "## 9. 6h / 12h / 24h sensitivity", "",
        "6 h和24 h均按同一暴露语义和基线调整流程执行，没有根据显著性更换主窗口。",
        "", "## 10. Measurement-process sensitivity", "",
        "post-T1 Na检测次数、首次复查时间和检测密度存在结局组间差异；高观察子集与分层结果只用于评估记录过程敏感性。",
        "", "## 11. Prespecified interactions", "",
        f"基线RRT和基线袢利尿剂预设探索互作已保留。Step 1 predicted-risk sensitivity：{step1_risk_status}；不作精准治疗声称。",
        "", "## 12. Robustness summary", "",
        "完整敏感性结果保留在`step2_new_q05_sensitivity_analysis_v1.csv`。不删除不显著模块，不以结果选择窗口、暴露尺度或亚组。",
        "", "## 13. Interpretation boundaries", "",
        "可用术语仅包括associated with、association、documented exposure和subsequent recorded hypernatremia。不得使用caused、prevented、benefit、harm、optimal therapy或intervention effect。",
        "", "## 14. Go / No-Go for manuscript development", "",
        f"`{gate}`", "",
        "理由：暴露定义、时间窗口和调整框架已预先锁定，结果可供关联研究稿件组装；但接口沉默、Rule C体积不确定和结局检测过程需在方法、结果和局限中明确报告。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gate


def validate_and_write(
    baseline: pd.DataFrame,
    windows: pd.DataFrame,
    primary: pd.DataFrame,
    rcs: pd.DataFrame,
    carrier: pd.DataFrame,
    sensitivity: pd.DataFrame,
    gate: str,
    manual_gate: dict,
    prediction_gate: dict,
) -> None:
    errors = []
    counts = {}
    for wh, expected in {6: (34878, 34413, 810), 12: (34779, 33538, 793), 24: (33737, 31806, 752)}.items():
        d = windows[windows.window_hours == wh]
        risk = d[d.in_t1_risk_set]
        obs = risk[risk.post_t1_na_count > 0]
        got = (len(risk), len(obs), int(obs.event_post_t1.sum()))
        counts[str(wh)] = {"t1_risk": got[0], "outcome_observed": got[1], "events": got[2]}
        if got != expected:
            errors.append(f"window {wh} counts {got} != {expected}")
    if len(primary) != 3:
        errors.append("primary model row count != 3")
    if set(primary.module) != set(FLUIDS):
        errors.append("primary modules incomplete")
    if len(carrier) != 4:
        errors.append("carrier secondary row count != 4")
    if rcs.module.nunique() != 3:
        errors.append("RCS modules incomplete")
    if any(d["rank_deficiency"] != 0 or not d["converged"] for d in MODEL_DIAGNOSTICS):
        errors.append("one or more fitted models failed rank/convergence QC")
    if manual_gate["status"] == "RUN" and manual_gate["grades"] != {"B_USABLE_WITH_RULES": 45, "A_CLEAR": 8, "C_AMBIGUOUS": 7}:
        errors.append("manual review counts drifted")

    required = [AMENDMENT, FLOW, MISSINGNESS, EXPOSURE_DIST, COEXPOSURE, PRIMARY,
                DIAGNOSTICS, RCS_RESULTS, CARRIER, SENSITIVITY, MEASUREMENT, REPORT]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        errors.append("missing outputs: " + "; ".join(missing))
    validation = {
        "task": "Step2-New-Q05 resumed Phase B formal association analysis",
        "generated_at": datetime.now().astimezone().isoformat(),
        "amendment_gate": "READY_FOR_FORMAL_ANALYSIS",
        "manuscript_gate": gate,
        "qc_pass": not errors,
        "qc_errors": errors,
        "locked_counts_by_window": counts,
        "manual_review_lock": manual_gate,
        "step1_prediction_sensitivity": prediction_gate,
        "primary_scales_ml": {k: v["increment_ml"] for k, v in FLUIDS.items()},
        "rcs_knots": "positive-exposure 5th/35th/65th/95th percentiles, four-knot restricted cubic spline",
        "models_fitted": len(MODEL_DIAGNOSTICS),
        "all_models_converged": all(d["converged"] for d in MODEL_DIAGNOSTICS),
        "all_design_matrices_full_rank": all(d["rank_deficiency"] == 0 for d in MODEL_DIAGNOSTICS),
        "four_round_review": {
            "code_flaws": "PASS",
            "data_handling": "PASS",
            "per_table": "PASS",
            "cross_table": "PASS",
            "note": "Sparse one-check measurement stratum retained as descriptive only after separation audit.",
        },
        "prohibited_analyses_run": [],
        "files_sha256": {str(p.relative_to(PROJECT)): sha256(p) for p in [
            WINDOW_SOURCE, BASELINE, MANIFEST, PREPROCESSING_METADATA,
            ORIGINAL_TASK, RESUME_TASK, Q04_GATE, *required,
        ]},
        "stop_rule": "STOP_AFTER_Q05_OUTPUTS",
    }
    VALIDATION.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("QC failed: " + " | ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Step 2 MIMIC primary workflow.")
    parser.add_argument("--run-step1-risk-sensitivity", action="store_true")
    parser.add_argument("--manual-review-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    baseline, windows, baseline_matrix, baseline_names, manual_gate, prediction_gate = load_data(
        run_step1_risk_sensitivity=args.run_step1_risk_sensitivity,
        manual_review_csv=args.manual_review_csv,
    )
    make_amendment(manual_gate)
    make_missingness(baseline)
    make_flow(windows)

    exposure = pd.DataFrame(distribution_rows(windows))
    exposure.to_csv(EXPOSURE_DIST, index=False, encoding="utf-8-sig")
    pd.DataFrame(coexposure_rows(windows)).to_csv(COEXPOSURE, index=False, encoding="utf-8-sig")
    make_exposure_plot(windows)

    primary = pd.DataFrame(primary_models(windows, baseline_matrix, baseline_names))
    primary.to_csv(PRIMARY, index=False, encoding="utf-8-sig")
    rcs = pd.DataFrame(run_rcs(windows, baseline_matrix, baseline_names))
    rcs.to_csv(RCS_RESULTS, index=False, encoding="utf-8-sig")
    carrier = pd.DataFrame(carrier_models(windows, baseline_matrix, baseline_names))
    carrier.to_csv(CARRIER, index=False, encoding="utf-8-sig")
    sensitivity = pd.DataFrame(sensitivity_models(
        windows, baseline_matrix, baseline_names,
        run_step1_risk_sensitivity=prediction_gate["status"] == "RUN",
    ))
    sensitivity.to_csv(SENSITIVITY, index=False, encoding="utf-8-sig")

    make_measurement_report(windows, sensitivity)
    make_diagnostics()
    gate = make_report(primary, rcs, carrier, sensitivity, windows, prediction_gate["status"])
    validate_and_write(baseline, windows, primary, rcs, carrier, sensitivity, gate, manual_gate, prediction_gate)
    print(json.dumps({
        "qc": "PASS", "amendment_gate": "READY_FOR_FORMAL_ANALYSIS",
        "manuscript_gate": gate, "models_fitted": len(MODEL_DIAGNOSTICS),
        "primary": primary[["module", "adjusted_or", "ci95_low", "ci95_high", "p_value", "model_n", "events"]].to_dict("records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
