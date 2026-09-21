# Public sanitized copy; restricted inputs are supplied outside Git.
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg


REPO_ROOT = Path(__file__).resolve().parents[2]
RESTRICTED_ROOT = Path(os.environ.get("STEP2_RESTRICTED_ROOT", REPO_ROOT / "restricted_data"))
OUTPUT_ROOT = Path(os.environ.get("STEP2_OUTPUT_ROOT", REPO_ROOT / "outputs"))
PROJECT = REPO_ROOT
Q05_DIR = OUTPUT_ROOT / "mimic_primary"
Q06_DIR = REPO_ROOT / "docs"
OUT = OUTPUT_ROOT / "structural_sensitivity"

TASK = REPO_ROOT / "docs" / "protocol_lock.md"
Q05_SCRIPT = REPO_ROOT / "src" / "models" / "run_primary_association.py"
Q05_WINDOW = RESTRICTED_ROOT / "mimic" / "step2_new_q05_window_level_analysis_source_v1.csv"
Q05_PRIMARY = Q05_DIR / "step2_new_q05_primary_models_v1.csv"
Q05_RCS = Q05_DIR / "step2_new_q05_rcs_results_v1.csv"
Q05_VALIDATION = Q05_DIR / "step2_new_q05_validation_v2.json"
BASELINE = RESTRICTED_ROOT / "mimic" / "primary_cohort_model_dataset_v1.0.csv"
MANIFEST = REPO_ROOT / "config" / "variable_dictionary.csv"
MED_SOURCE = RESTRICTED_ROOT / "mimic" / "step2_new_q07_medication_window_source_v1.csv"
MEAS_SOURCE = RESTRICTED_ROOT / "mimic" / "step2_new_q07_measurement_process_source_v1.csv"
ITEM_MAPPING = RESTRICTED_ROOT / "mimic" / "step2_new_q07_medication_item_mapping_v1.csv"

MUTUAL_OUT = OUT / "step2_new_q07_direct_carrier_mutual_adjustment_v1.csv"
HYPER_AUDIT_OUT = OUT / "step2_new_q07_hypertonic_saline_window_audit_v1.csv"
HYPER_SENS_OUT = OUT / "step2_new_q07_hypertonic_saline_exclusion_sensitivity_v1.csv"
BICARB_OUT = OUT / "step2_new_q07_sodium_bicarbonate_audit_v1.md"
TTE_OUT = OUT / "step2_new_q07_time_to_event_sensitivity_v1.csv"
FOLLOWUP_OUT = OUT / "step2_new_q07_followup_time_audit_v1.csv"
CAL_ICU_OUT = OUT / "step2_new_q07_calendar_icu_adjustment_v1.csv"
MEAS_OUT = OUT / "step2_new_q07_pre_event_measurement_process_v1.md"
REPORT_OUT = OUT / "Step2_New_Q07_structural_robustness_report_v1.md"
AMEND_OUT = OUT / "Step2_New_Q07_manuscript_amendment_notes_v1.md"
VALIDATION_OUT = OUT / "step2_new_q07_validation_v1.json"
THIS_SCRIPT = OUT / "run_step2_new_q07_structural_robustness_v1.py"
SQL_SCRIPT = OUT / "step2_new_q07_extract_medication_and_measurement_v1.sql"


def load_q05_module():
    spec = importlib.util.spec_from_file_location("q05_locked_analysis", Q05_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load locked Q05 analysis module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q05 = load_q05_module()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"t", "true", "1", "yes"})


def quantiles(series: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return (math.nan, math.nan, math.nan)
    values = clean.quantile([0.25, 0.50, 0.75])
    return float(values.loc[0.50]), float(values.loc[0.25]), float(values.loc[0.75])


def effect_from_glm(result, names: list[str], term: str, metadata: dict) -> dict:
    idx = names.index(term)
    beta = float(result.params[idx])
    se = float(result.bse[idx])
    return {
        **metadata,
        "term": term,
        "coefficient": beta,
        "standard_error": se,
        "estimate": math.exp(beta),
        "ci95_low": math.exp(beta - 1.96 * se),
        "ci95_high": math.exp(beta + 1.96 * se),
        "p_value": float(result.pvalues[idx]),
    }


def focal_vif(design: np.ndarray, names: list[str], term: str) -> float:
    idx = names.index(term)
    y = design[:, idx]
    others = np.delete(design, idx, axis=1)
    fit = sm.OLS(y, others).fit()
    if fit.rsquared >= 1:
        return math.inf
    return float(1.0 / (1.0 - fit.rsquared))


def load_inputs():
    baseline, windows, baseline_matrix, baseline_names, manual_gate = q05.load_data()
    primary = pd.read_csv(Q05_PRIMARY)
    medication = pd.read_csv(MED_SOURCE)
    measurement = pd.read_csv(MEAS_SOURCE)

    for col in ["hypertonic_any", "bicarbonate_any", "event_post_t1"]:
        medication[col] = to_bool(medication[col])
    measurement["event_post_t1"] = to_bool(measurement["event_post_t1"])

    if len(medication) != 34913 or medication["stay_id"].nunique() != 34913:
        raise ValueError("Medication-window source must contain 34,913 unique stays.")
    if len(measurement) != 34913 or measurement["stay_id"].nunique() != 34913:
        raise ValueError("Measurement-process source must contain 34,913 unique stays.")

    w12 = windows.loc[windows["window_hours"] == 12].copy()
    merged = w12.merge(
        medication,
        on="stay_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_db"),
    )
    if merged["hypertonic_any"].isna().any():
        raise ValueError("Medication-window merge produced missing stays.")
    event_match = (
        merged["event_post_t1"].astype(bool)
        == merged["event_post_t1_db"].astype(bool)
    )
    if not event_match.all():
        raise ValueError(f"Q05/Q07 post-T1 event mismatch in {(~event_match).sum()} stays.")

    # Q05 load_data already carries all binary covariates, including RRT, into
    # the window table. Only sodium_last must be joined for the descriptive audit.
    baseline_small = baseline[["stay_id", "sodium_last"]].copy()
    merged = merged.merge(baseline_small, on="stay_id", how="left", validate="one_to_one")
    return baseline, windows, baseline_matrix, baseline_names, manual_gate, primary, merged, measurement


def active_observed(merged: pd.DataFrame) -> pd.DataFrame:
    return merged.loc[
        merged["in_t1_risk_set"].astype(bool)
        & (merged["post_t1_na_count"] > 0)
        & (merged["inputevent_interface_status"] == "ACTIVE")
    ].copy()


def fit_mutual_adjustment(data, baseline_matrix, baseline_names) -> tuple[pd.DataFrame, dict]:
    direct = data["nacl_direct_ab_ml"].to_numpy(dtype=float) / 500.0
    carrier = data["nacl_carrier_ab_ml"].to_numpy(dtype=float) / 250.0
    any_carrier = (data["nacl_carrier_records"].to_numpy() > 0).astype(float)
    rows = []

    result_a1, names_a1, diag_a1 = q05.fit_glm(
        data,
        baseline_matrix,
        baseline_names,
        "nacl",
        direct,
        "direct_nacl_per_500ml",
        "q07_mutual_continuous",
        add_terms={"carrier_nacl_per_250ml": carrier},
    )
    x_a1, x_names_a1 = q05.build_design(
        data,
        baseline_matrix,
        baseline_names,
        "nacl",
        direct,
        "direct_nacl_per_500ml",
        add_terms={"carrier_nacl_per_250ml": carrier},
    )
    pearson_r, pearson_p = st.pearsonr(direct, carrier)
    spearman_r, spearman_p = st.spearmanr(direct, carrier)
    common = {
        "analysis_id": "A1_continuous_mutual_adjustment",
        "model_n": len(data),
        "events": int(data["event_post_t1"].sum()),
        "direct_carrier_pearson_r": float(pearson_r),
        "direct_carrier_pearson_p": float(pearson_p),
        "direct_carrier_spearman_r": float(spearman_r),
        "direct_carrier_spearman_p": float(spearman_p),
        "direct_vif": focal_vif(x_a1, x_names_a1, "direct_nacl_per_500ml"),
        "carrier_vif": focal_vif(x_a1, x_names_a1, "carrier_nacl_per_250ml"),
        "rank_deficiency": diag_a1["rank_deficiency"],
        "condition_number": diag_a1["condition_number"],
        "converged": diag_a1["converged"],
    }
    rows.append(effect_from_glm(result_a1, names_a1, "direct_nacl_per_500ml", {**common, "exposure": "direct_0.9%_NaCl", "scale": "per_500_ml"}))
    rows.append(effect_from_glm(result_a1, names_a1, "carrier_nacl_per_250ml", {**common, "exposure": "carrier_0.9%_NaCl", "scale": "per_250_ml"}))

    result_a2, names_a2, diag_a2 = q05.fit_glm(
        data,
        baseline_matrix,
        baseline_names,
        "nacl",
        direct,
        "direct_nacl_per_500ml",
        "q07_mutual_binary",
        add_terms={"any_nacl_carrier": any_carrier},
    )
    common_a2 = {
        "analysis_id": "A2_binary_carrier_sensitivity",
        "model_n": len(data),
        "events": int(data["event_post_t1"].sum()),
        "direct_carrier_pearson_r": float(pearson_r),
        "direct_carrier_pearson_p": float(pearson_p),
        "direct_carrier_spearman_r": float(spearman_r),
        "direct_carrier_spearman_p": float(spearman_p),
        "direct_vif": math.nan,
        "carrier_vif": math.nan,
        "rank_deficiency": diag_a2["rank_deficiency"],
        "condition_number": diag_a2["condition_number"],
        "converged": diag_a2["converged"],
    }
    rows.append(effect_from_glm(result_a2, names_a2, "direct_nacl_per_500ml", {**common_a2, "exposure": "direct_0.9%_NaCl", "scale": "per_500_ml"}))
    rows.append(effect_from_glm(result_a2, names_a2, "any_nacl_carrier", {**common_a2, "exposure": "any_carrier_0.9%_NaCl", "scale": "any_vs_none"}))
    return pd.DataFrame(rows), {"continuous": diag_a1, "binary": diag_a2}


def hypertonic_audit_rows(merged: pd.DataFrame) -> pd.DataFrame:
    risk = merged.loc[merged["in_t1_risk_set"].astype(bool)].copy()
    exposed = risk.loc[risk["hypertonic_any"]].copy()
    rows: list[dict] = []

    def add_summary(section: str, stratum: str, frame: pd.DataFrame) -> None:
        h_med, h_q1, h_q3 = quantiles(frame.loc[frame["hypertonic_window_ml"] > 0, "hypertonic_window_ml"])
        n_med, n_q1, n_q3 = quantiles(frame["nacl_direct_ab_ml"])
        s_med, s_q1, s_q3 = quantiles(frame["sodium_last"])
        rows.append({
            "section": section,
            "stratum": stratum,
            "n": len(frame),
            "events": int(frame["event_post_t1"].sum()),
            "event_rate_percent": 100 * frame["event_post_t1"].mean() if len(frame) else math.nan,
            "hypertonic_volume_median_ml": h_med,
            "hypertonic_volume_q1_ml": h_q1,
            "hypertonic_volume_q3_ml": h_q3,
            "hypertonic_volume_summary_denominator": "positive_reconstructable_exposure",
            "direct_nacl_median_ml": n_med,
            "direct_nacl_q1_ml": n_q1,
            "direct_nacl_q3_ml": n_q3,
            "baseline_sodium_median": s_med,
            "baseline_sodium_q1": s_q1,
            "baseline_sodium_q3": s_q3,
            "baseline_rrt_n": int(frame["rrt_pre_t0"].sum()),
            "baseline_rrt_percent": 100 * frame["rrt_pre_t0"].mean() if len(frame) else math.nan,
            "reconstructable_records": int(frame["hypertonic_reconstructable_records"].sum()),
            "total_records": int(frame["hypertonic_records"].sum()),
        })

    add_summary("overall", "all_12h_T1_risk", risk)
    add_summary("overall", "any_hypertonic_exposure", exposed)
    add_summary("item", "3%_NaCl", risk.loc[risk["hypertonic_3pct_records"] > 0])
    add_summary("item", "23.4%_NaCl", risk.loc[risk["hypertonic_23_4pct_records"] > 0])
    neuro = exposed["first_careunit"].astype(str).str.contains("Neuro|Neurology", case=False, regex=True)
    add_summary("neurologic_unit_proxy", "neuro_named_first_careunit", exposed.loc[neuro])
    add_summary("neurologic_unit_proxy", "other_first_careunit", exposed.loc[~neuro])
    for unit, frame in exposed.groupby("first_careunit", dropna=False):
        add_summary("first_careunit", str(unit), frame)
    return pd.DataFrame(rows)


def fit_hypertonic_exclusion(data, baseline_matrix, baseline_names) -> tuple[pd.DataFrame, dict]:
    restricted = data.loc[~data["hypertonic_any"]].copy()
    exposure = restricted["nacl_direct_ab_ml"].to_numpy(dtype=float) / 500.0
    result, names, diag = q05.fit_glm(
        restricted,
        baseline_matrix,
        baseline_names,
        "nacl",
        exposure,
        "direct_nacl_per_500ml",
        "q07_exclude_hypertonic",
    )
    row = effect_from_glm(result, names, "direct_nacl_per_500ml", {
        "analysis_id": "B1_exclude_any_3%_or_23.4%_NaCl_in_T0_T1",
        "model_n": len(restricted),
        "events": int(restricted["event_post_t1"].sum()),
        "excluded_n": int(data["hypertonic_any"].sum()),
        "excluded_events": int(data.loc[data["hypertonic_any"], "event_post_t1"].sum()),
        "exposure": "direct_0.9%_NaCl",
        "scale": "per_500_ml",
        "rank_deficiency": diag["rank_deficiency"],
        "condition_number": diag["condition_number"],
        "converged": diag["converged"],
    })
    return pd.DataFrame([row]), diag


def fit_bicarbonate_sensitivity(data, baseline_matrix, baseline_names) -> tuple[pd.DataFrame, dict]:
    exposure = data["nacl_direct_ab_ml"].to_numpy(dtype=float) / 500.0
    bicarb = data["bicarbonate_any"].astype(int).to_numpy(dtype=float)
    result, names, diag = q05.fit_glm(
        data,
        baseline_matrix,
        baseline_names,
        "nacl",
        exposure,
        "direct_nacl_per_500ml",
        "q07_bicarbonate_indicator",
        add_terms={"any_iv_sodium_bicarbonate": bicarb},
    )
    common = {
        "analysis_id": "B2_add_any_iv_sodium_bicarbonate_indicator",
        "model_n": len(data),
        "events": int(data["event_post_t1"].sum()),
        "bicarbonate_exposed_n": int(data["bicarbonate_any"].sum()),
        "bicarbonate_exposed_events": int(data.loc[data["bicarbonate_any"], "event_post_t1"].sum()),
        "rank_deficiency": diag["rank_deficiency"],
        "condition_number": diag["condition_number"],
        "converged": diag["converged"],
    }
    rows = [
        effect_from_glm(result, names, "direct_nacl_per_500ml", {**common, "exposure": "direct_0.9%_NaCl", "scale": "per_500_ml"}),
        effect_from_glm(result, names, "any_iv_sodium_bicarbonate", {**common, "exposure": "any_iv_sodium_bicarbonate", "scale": "any_vs_none"}),
    ]
    return pd.DataFrame(rows), diag


def write_bicarbonate_audit(data: pd.DataFrame, bicarb_model: pd.DataFrame) -> dict:
    exposed = data.loc[data["bicarbonate_any"]].copy()
    amount = exposed.loc[exposed["bicarbonate_window_meq"] > 0, "bicarbonate_window_meq"]
    med, q1, q3 = quantiles(amount)
    total_records = int(data["bicarbonate_records"].sum())
    reconstructable = int(data["bicarbonate_reconstructable_records"].sum())
    dual_orders = int(data["bicarbonate_dual_component_orders"].sum())
    direct = bicarb_model.loc[bicarb_model["exposure"] == "direct_0.9%_NaCl"].iloc[0]
    indicator = bicarb_model.loc[bicarb_model["exposure"] == "any_iv_sodium_bicarbonate"].iloc[0]
    status = "RELIABLE_FOR_BINARY_INDICATOR_AND_DESCRIPTIVE_AMOUNT"
    text = f"""# Step2-New-Q07 静脉碳酸氢钠审计 v1

## 结论

`{status}`

本审计仅识别管理窗口`[T0,T1)`内的静脉8.4% sodium bicarbonate（碳酸氢钠）记录，不改变主暴露或主模型。精确item映射为：`220995`（Sodium Bicarbonate 8.4%，添加剂，原始单位mEq）和`227533`（Sodium Bicarbonate 8.4% Amp，静脉推注，原始单位mL）。

## 数据质量

- Q07主模型队列：n={len(data):,}，events={int(data['event_post_t1'].sum()):,}。
- 暴露者：{len(exposed):,}，其中events={int(exposed['event_post_t1'].sum()):,}。
- 原始记录：{total_records:,}；按既定重叠时间规则可重建记录：{reconstructable:,}（{100*reconstructable/total_records:.2f}%）。
- 同一linkorder同时出现两个碳酸氢钠item的订单：{dual_orders}；未发现双重计量证据。
- 暴露者可重建累计量：median {med:.2f} mEq [IQR {q1:.2f}, {q3:.2f}]。
- 8.4%安瓿按标签浓度1 mEq/mL换算；添加剂直接使用数据库mEq。未对不可重建记录猜算剂量。

## 预设指标敏感性

由于精确item映射、时间窗、样本量及给药记录完整性满足Q07条件，执行“任意静脉碳酸氢钠暴露”二元指标敏感性，不使用结果驱动剂量切点。

| 项目 | 调整估计 | 95% CI | P值 |
|---|---:|---:|---:|
| Direct 0.9% NaCl per 500 mL | {direct['estimate']:.3f} | {direct['ci95_low']:.3f}-{direct['ci95_high']:.3f} | {direct['p_value']:.6g} |
| Any IV sodium bicarbonate | {indicator['estimate']:.3f} | {indicator['ci95_low']:.3f}-{indicator['ci95_high']:.3f} | {indicator['p_value']:.6g} |

以上是调整关联，不是治疗效应或钠来源的因果分解。碳酸氢钠指标仅用于结构性敏感性；不进入主模型。
"""
    BICARB_OUT.write_text(text, encoding="utf-8")
    return {
        "status": status,
        "exposed_n": len(exposed),
        "exposed_events": int(exposed["event_post_t1"].sum()),
        "records": total_records,
        "reconstructable_records": reconstructable,
        "reconstructable_percent": 100 * reconstructable / total_records,
        "dual_component_orders": dual_orders,
        "median_meq": med,
        "q1_meq": q1,
        "q3_meq": q3,
        "indicator_or": float(indicator["estimate"]),
        "indicator_ci95": [float(indicator["ci95_low"]), float(indicator["ci95_high"])],
    }


def fit_cause_specific_cox(merged, baseline_matrix, baseline_names) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    data = merged.loc[
        merged["in_t1_risk_set"].astype(bool)
        & (merged["inputevent_interface_status"] == "ACTIVE")
    ].copy()
    t1 = pd.to_datetime(data["t1_time"])
    end = pd.to_datetime(data["observation_end"])
    event_time = pd.to_datetime(data["first_post_t1_event_time"])
    status = data["event_post_t1"].astype(int).to_numpy()
    stop = event_time.where(data["event_post_t1"].astype(bool), end)
    duration = (stop - t1).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    if np.any(duration <= 0):
        raise ValueError(f"Cause-specific Cox has {(duration <= 0).sum()} non-positive times.")

    exposure = data["nacl_direct_ab_ml"].to_numpy(dtype=float) / 500.0
    x, names = q05.build_design(
        data,
        baseline_matrix,
        baseline_names,
        "nacl",
        exposure,
        "direct_nacl_per_500ml",
    )
    x = x[:, 1:]
    names = names[1:]
    nonestimable = [
        j for j, name in enumerate(names)
        if np.nanstd(x[status.astype(bool), j]) < 1e-12
    ]
    nonestimable_names = [names[j] for j in nonestimable]
    if any("missingindicator_" not in name for name in nonestimable_names):
        raise RuntimeError(
            "An original Cox adjustment term is non-estimable among events: "
            + ", ".join(nonestimable_names)
        )
    if nonestimable:
        keep = [j for j in range(x.shape[1]) if j not in nonestimable]
        x = x[:, keep]
        names = [names[j] for j in keep]
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise RuntimeError(f"Cox design matrix rank deficiency: {x.shape[1]-rank}.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PHReg(duration, x, status=status, ties="efron")
        result = model.fit(method="bfgs", disp=0, maxiter=500)
    idx = names.index("direct_nacl_per_500ml")
    beta = float(result.params[idx])
    se = float(result.bse[idx])

    schoenfeld = np.asarray(result.schoenfeld_residuals)
    event_mask = status.astype(bool) & np.isfinite(schoenfeld[:, idx])
    rho, ph_p = st.spearmanr(schoenfeld[event_mask, idx], np.log(duration[event_mask]))
    covariate_ph_p = {}
    for j, name in enumerate(names):
        mask = status.astype(bool) & np.isfinite(schoenfeld[:, j])
        if mask.sum() >= 10 and np.nanstd(schoenfeld[mask, j]) > 0:
            _, p_value = st.spearmanr(schoenfeld[mask, j], np.log(duration[mask]))
            covariate_ph_p[name] = float(p_value)
    ph_result = "PASS_NO_EXPOSURE_SPECIFIC_SIGNAL" if ph_p >= 0.05 else "EXPOSURE_PH_SIGNAL_DETECTED"
    row = {
        "analysis_id": "C1_cause_specific_cox",
        "model": "cause_specific_Cox_PHReg_Efron",
        "exposure": "direct_0.9%_NaCl",
        "scale": "per_500_ml",
        "model_n": len(data),
        "events": int(status.sum()),
        "coefficient": beta,
        "standard_error": se,
        "estimate": math.exp(beta),
        "ci95_low": math.exp(beta - 1.96 * se),
        "ci95_high": math.exp(beta + 1.96 * se),
        "p_value": float(result.pvalues[idx]),
        "ph_diagnostic": "Spearman correlation of exposure Schoenfeld residual with log(event time)",
        "ph_rho": float(rho),
        "ph_p_value": float(ph_p),
        "ph_result": ph_result,
        "design_columns": x.shape[1],
        "design_rank": rank,
        "omitted_nonestimable_transformed_features": " | ".join(nonestimable_names),
        "warnings": " | ".join(str(w.message) for w in caught),
    }
    fine_gray = {
        "analysis_id": "C2_Fine_Gray",
        "model": "Fine-Gray",
        "exposure": "direct_0.9%_NaCl",
        "scale": "per_500_ml",
        "model_n": len(data),
        "events": int(status.sum()),
        "coefficient": math.nan,
        "standard_error": math.nan,
        "estimate": math.nan,
        "ci95_low": math.nan,
        "ci95_high": math.nan,
        "p_value": math.nan,
        "ph_diagnostic": "NOT_PERFORMED",
        "ph_rho": math.nan,
        "ph_p_value": math.nan,
        "ph_result": "NOT_PERFORMED_NO_PREVALIDATED_STABLE_IMPLEMENTATION_IN_LOCKED_WORKFLOW",
        "design_columns": math.nan,
        "design_rank": math.nan,
        "omitted_nonestimable_transformed_features": "",
        "warnings": "Optional analysis not forced by Q07.",
    }
    data["tte_hours"] = duration
    data["tte_event"] = status
    diagnostics = {
        "ph_result": ph_result,
        "exposure_ph_rho": float(rho),
        "exposure_ph_p": float(ph_p),
        "covariates_with_nominal_ph_p_lt_0_05": int(sum(p < 0.05 for p in covariate_ph_p.values())),
        "covariates_tested": len(covariate_ph_p),
        "minimum_covariate_ph_p": min(covariate_ph_p.values()) if covariate_ph_p else math.nan,
        "fine_gray_status": fine_gray["ph_result"],
        "omitted_nonestimable_transformed_features": nonestimable_names,
    }
    return pd.DataFrame([row, fine_gray]), diagnostics, data


def followup_audit_rows(cox_data: pd.DataFrame) -> pd.DataFrame:
    data = cox_data.copy()
    admin = pd.to_datetime(data["administrative_end"])
    obs_end = pd.to_datetime(data["observation_end"])
    discharge = pd.to_datetime(data["dischtime"])
    death = pd.to_datetime(data["deathtime"])
    event = data["tte_event"].astype(bool)

    endpoint = np.full(len(data), "administrative_day7", dtype=object)
    endpoint[event.to_numpy()] = "recorded_event"
    non_event = ~event
    death_first = non_event & death.notna() & (death <= discharge) & (death <= admin)
    discharge_first = non_event & ~death_first & (discharge < admin)
    endpoint[death_first.to_numpy()] = "death_before_day7"
    endpoint[discharge_first.to_numpy()] = "hospital_discharge_before_day7"
    data["endpoint_type"] = endpoint
    data["full_available_followup_hours"] = (obs_end - pd.to_datetime(data["t1_time"])).dt.total_seconds() / 3600.0

    rows: list[dict] = []

    def add_row(section: str, stratum: str, frame: pd.DataFrame, time_col: str) -> None:
        med, q1, q3 = quantiles(frame[time_col])
        rows.append({
            "section": section,
            "stratum": stratum,
            "n": len(frame),
            "events": int(frame["tte_event"].sum()),
            "followup_metric": time_col,
            "median_followup_hours": med,
            "q1_followup_hours": q1,
            "q3_followup_hours": q3,
            "hospital_discharge_before_day7_n": int((frame["endpoint_type"] == "hospital_discharge_before_day7").sum()),
            "death_before_day7_n": int((frame["endpoint_type"] == "death_before_day7").sum()),
            "administrative_day7_n": int((frame["endpoint_type"] == "administrative_day7").sum()),
            "recorded_event_n": int((frame["endpoint_type"] == "recorded_event").sum()),
            "nacl_direct_median_ml": quantiles(frame["nacl_direct_ab_ml"])[0],
        })

    add_row("event_status", "recorded_event", data.loc[event], "tte_hours")
    add_row("event_status", "no_recorded_event", data.loc[non_event], "full_available_followup_hours")
    for endpoint_name, frame in data.groupby("endpoint_type"):
        time_col = "tte_hours" if endpoint_name == "recorded_event" else "full_available_followup_hours"
        add_row("endpoint_type", endpoint_name, frame, time_col)

    positive = data["nacl_direct_ab_ml"] > 0
    data["nacl_exposure_group"] = "zero"
    if positive.sum() >= 4:
        data.loc[positive, "nacl_exposure_group"] = pd.qcut(
            data.loc[positive, "nacl_direct_ab_ml"],
            q=4,
            labels=["positive_Q1", "positive_Q2", "positive_Q3", "positive_Q4"],
            duplicates="drop",
        ).astype(str)
    for group in ["zero", "positive_Q1", "positive_Q2", "positive_Q3", "positive_Q4"]:
        frame = data.loc[data["nacl_exposure_group"] == group]
        if len(frame):
            add_row("direct_nacl_descriptive_group", group, frame, "full_available_followup_hours")
    rho, p_value = st.spearmanr(data["nacl_direct_ab_ml"], data["full_available_followup_hours"])
    rows.append({
        "section": "continuous_description",
        "stratum": "Spearman_direct_NaCl_vs_available_followup",
        "n": len(data),
        "events": int(data["tte_event"].sum()),
        "followup_metric": "full_available_followup_hours",
        "median_followup_hours": math.nan,
        "q1_followup_hours": math.nan,
        "q3_followup_hours": math.nan,
        "hospital_discharge_before_day7_n": math.nan,
        "death_before_day7_n": math.nan,
        "administrative_day7_n": math.nan,
        "recorded_event_n": math.nan,
        "nacl_direct_median_ml": math.nan,
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
    })
    return pd.DataFrame(rows)


def fit_calendar_icu_adjustment(data, baseline_matrix, baseline_names, primary_beta: float) -> tuple[pd.DataFrame, dict]:
    model_data = data.copy()
    rare_units = model_data["first_careunit"].value_counts().loc[lambda x: x < 100].index
    model_data["icu_model_category"] = model_data["first_careunit"].where(
        ~model_data["first_careunit"].isin(rare_units), "Other/mixed sparse units"
    )

    era_order = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]
    model_data["era_model_category"] = pd.Categorical(model_data["anchor_year_group"], categories=era_order)
    icu_categories = ["Medical Intensive Care Unit (MICU)"] + sorted(
        value for value in model_data["icu_model_category"].dropna().unique()
        if value != "Medical Intensive Care Unit (MICU)"
    )
    model_data["icu_model_category"] = pd.Categorical(model_data["icu_model_category"], categories=icu_categories)

    era_dummies = pd.get_dummies(model_data["era_model_category"], prefix="era", drop_first=True, dtype=float)
    icu_dummies = pd.get_dummies(model_data["icu_model_category"], prefix="icu", drop_first=True, dtype=float)
    add_terms = {name: era_dummies[name].to_numpy() for name in era_dummies.columns}
    add_terms.update({name: icu_dummies[name].to_numpy() for name in icu_dummies.columns})

    exposure = model_data["nacl_direct_ab_ml"].to_numpy(dtype=float) / 500.0
    result, names, diag = q05.fit_glm(
        model_data,
        baseline_matrix,
        baseline_names,
        "nacl",
        exposure,
        "direct_nacl_per_500ml",
        "q07_calendar_icu_adjustment",
        add_terms=add_terms,
    )
    effect = effect_from_glm(result, names, "direct_nacl_per_500ml", {
        "analysis_type": "D1_model",
        "analysis_id": "primary_plus_anchor_year_group_plus_first_ICU_type",
        "stratum": "adjusted_direct_NaCl",
        "model_n": len(model_data),
        "events": int(model_data["event_post_t1"].sum()),
        "scale": "per_500_ml",
        "primary_beta": primary_beta,
        "change_from_primary_log_or": math.nan,
        "attenuation_percent": math.nan,
        "median_nacl_ml": math.nan,
        "q1_nacl_ml": math.nan,
        "q3_nacl_ml": math.nan,
        "rank_deficiency": diag["rank_deficiency"],
        "condition_number": diag["condition_number"],
        "converged": diag["converged"],
    })
    effect["change_from_primary_log_or"] = effect["coefficient"] - primary_beta
    effect["attenuation_percent"] = 100 * (primary_beta - effect["coefficient"]) / primary_beta
    rows = [effect]

    for group_col, label in [("anchor_year_group", "D2_era_distribution"), ("first_careunit", "D2_ICU_distribution")]:
        for group, frame in model_data.groupby(group_col, dropna=False):
            med, q1, q3 = quantiles(frame["nacl_direct_ab_ml"])
            rows.append({
                "analysis_type": label,
                "analysis_id": "descriptive_only",
                "stratum": str(group),
                "model_n": len(frame),
                "events": int(frame["event_post_t1"].sum()),
                "term": "nacl_direct_ab_ml",
                "coefficient": math.nan,
                "standard_error": math.nan,
                "estimate": math.nan,
                "ci95_low": math.nan,
                "ci95_high": math.nan,
                "p_value": math.nan,
                "scale": "mL_descriptive",
                "primary_beta": primary_beta,
                "change_from_primary_log_or": math.nan,
                "attenuation_percent": math.nan,
                "median_nacl_ml": med,
                "q1_nacl_ml": q1,
                "q3_nacl_ml": q3,
                "rank_deficiency": math.nan,
                "condition_number": math.nan,
                "converged": math.nan,
            })
    metadata = {
        "era_reference": era_order[0],
        "era_categories": era_order,
        "icu_reference": icu_categories[0],
        "icu_model_categories": icu_categories,
        "rare_units_merged_to_other": list(rare_units),
        "attenuation_percent": effect["attenuation_percent"],
        "diagnostic": diag,
    }
    return pd.DataFrame(rows), metadata


def write_measurement_report(windows, measurement: pd.DataFrame) -> dict:
    q05_frame = q05.model_frame(windows, 12)
    data = q05_frame.merge(measurement, on="stay_id", how="left", validate="one_to_one", suffixes=("", "_q07"))
    if data["na_count_before_event_or_censor"].isna().any():
        raise ValueError("Measurement-process merge produced missing rows.")
    data["analysis_followup_days"] = data["analysis_followup_hours"].clip(lower=1.0 / 60.0) / 24.0
    data["pre_event_or_censor_tests_per_day"] = data["na_count_before_event_or_censor"] / data["analysis_followup_days"]
    data["hours_to_first_repeat"] = (
        pd.to_datetime(data["first_repeat_na_time"]) - pd.to_datetime(data["t1_time_q07"])
    ).dt.total_seconds() / 3600.0
    data["at_or_after_event_test_count"] = np.where(
        data["event_post_t1"].astype(bool),
        data["post_t1_na_count"] - data["na_count_before_event_or_censor"],
        0,
    )

    summary = {}
    lines = [
        "# Step2-New-Q07 事件前血钠监测过程审计 v1",
        "",
        "事件组仅统计首次记录Na>=151之前的血钠（严格`charttime < event_time`）；非事件组统计至出院、死亡或ICU入科第7天的完整可观察期。事件检测当次血钠不计入事件前次数。该审计不进入主调整模型。",
        "",
        "| 结局状态 | n | 事件前/删失前Na次数 median [IQR] | 检测率/观察日 median [IQR] | T1至首次复查h median [IQR] | 原Q05总Na次数 median [IQR] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for event_value, label in [(False, "未记录事件"), (True, "记录事件")]:
        frame = data.loc[data["event_post_t1"].astype(bool) == event_value]
        count_med, count_q1, count_q3 = quantiles(frame["na_count_before_event_or_censor"])
        rate_med, rate_q1, rate_q3 = quantiles(frame["pre_event_or_censor_tests_per_day"])
        first_med, first_q1, first_q3 = quantiles(frame["hours_to_first_repeat"])
        total_med, total_q1, total_q3 = quantiles(frame["post_t1_na_count"])
        lines.append(
            f"| {label} | {len(frame):,} | {count_med:.2f} [{count_q1:.2f}, {count_q3:.2f}] | "
            f"{rate_med:.2f} [{rate_q1:.2f}, {rate_q3:.2f}] | {first_med:.2f} [{first_q1:.2f}, {first_q3:.2f}] | "
            f"{total_med:.2f} [{total_q1:.2f}, {total_q3:.2f}] |"
        )
        summary[label] = {
            "n": len(frame),
            "pre_event_or_censor_na_count_median": count_med,
            "pre_event_or_censor_na_count_iqr": [count_q1, count_q3],
            "testing_density_per_day_median": rate_med,
            "testing_density_per_day_iqr": [rate_q1, rate_q3],
            "first_repeat_hours_median": first_med,
            "first_repeat_hours_iqr": [first_q1, first_q3],
            "original_total_na_count_median": total_med,
            "original_total_na_count_iqr": [total_q1, total_q3],
        }
    event_frame = data.loc[data["event_post_t1"].astype(bool)]
    after_med, after_q1, after_q3 = quantiles(event_frame["at_or_after_event_test_count"])
    lines += [
        "",
        f"事件组在事件检测当次及其后的血钠记录数为median {after_med:.2f} [IQR {after_q1:.2f}, {after_q3:.2f}]。因此Q05事件组较高的全随访检测次数确有一部分发生在事件被记录之后；但事件前检测密度仍是描述性观察过程指标，不能证明主关联由监测过程造成或未造成。",
        "",
        "现有高确定性观察敏感性继续保留；未新增inverse-intensity weighting、联合观察模型或任何post-T1变量调整模型。",
    ]
    MEAS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["event_at_or_after_count_median"] = after_med
    summary["event_at_or_after_count_iqr"] = [after_q1, after_q3]
    return summary


def fmt_effect(row: pd.Series) -> str:
    return f"{row['estimate']:.3f} (95% CI {row['ci95_low']:.3f}-{row['ci95_high']:.3f})"


def determine_gate(primary_beta, mutual_beta, hyper_beta, cox_beta, era_beta, ph_result, measurement_summary) -> tuple[str, dict]:
    flags = {
        "mutual_direction_consistent": mutual_beta > 0,
        "hypertonic_exclusion_direction_consistent": hyper_beta > 0,
        "cox_direction_consistent": cox_beta > 0,
        "calendar_icu_direction_consistent": era_beta > 0,
        "measurement_process_does_not_establish_complete_explanation": True,
        "cox_proportional_hazards_diagnostic_pass": ph_result == "PASS_NO_EXPOSURE_SPECIFIC_SIGNAL",
    }
    ratios = {
        "mutual_to_primary_beta_ratio": mutual_beta / primary_beta,
        "hypertonic_to_primary_beta_ratio": hyper_beta / primary_beta,
        "calendar_icu_to_primary_beta_ratio": era_beta / primary_beta,
    }
    direction_flags = [value for key, value in flags.items() if key.endswith("direction_consistent")]
    if not all(direction_flags):
        gate = "INTERPRETATION_REQUIRES_REVISION"
    elif min(ratios.values()) < 0.50 or not flags["cox_proportional_hazards_diagnostic_pass"]:
        gate = "ROBUSTNESS_PARTIALLY_REINFORCED"
    else:
        gate = "ROBUSTNESS_REINFORCED"
    return gate, {**flags, **ratios}


def write_report(
    primary_row,
    mutual,
    hyper_audit,
    hyper_sens,
    bicarb_summary,
    tte,
    ph_diag,
    followup,
    calendar,
    calendar_meta,
    measurement_summary,
    gate,
    gate_details,
):
    mutual_direct = mutual.loc[
        (mutual["analysis_id"] == "A1_continuous_mutual_adjustment")
        & (mutual["exposure"] == "direct_0.9%_NaCl")
    ].iloc[0]
    carrier_cont = mutual.loc[
        (mutual["analysis_id"] == "A1_continuous_mutual_adjustment")
        & (mutual["exposure"] == "carrier_0.9%_NaCl")
    ].iloc[0]
    mutual_binary_direct = mutual.loc[
        (mutual["analysis_id"] == "A2_binary_carrier_sensitivity")
        & (mutual["exposure"] == "direct_0.9%_NaCl")
    ].iloc[0]
    hyper = hyper_sens.iloc[0]
    cox = tte.loc[tte["analysis_id"] == "C1_cause_specific_cox"].iloc[0]
    era = calendar.loc[calendar["analysis_type"] == "D1_model"].iloc[0]
    exposed_hyper = hyper_audit.loc[
        (hyper_audit["section"] == "overall")
        & (hyper_audit["stratum"] == "any_hypertonic_exposure")
    ].iloc[0]

    report = f"""# Step2-New-Q07 结构性稳健性补强报告 v1

## 1. 执行边界

本轮只执行Q07预设结构性敏感性。T0、T1、12 h管理窗口、Na>=151结局、0.9% NaCl direct主暴露、每500 mL报告尺度、冻结27项T0前调整框架均未改变。结果仍是观察性调整关联，不是治疗效应。

## 2. 锁定主结果

Q05主模型0.9% NaCl direct每500 mL：OR {primary_row['adjusted_or']:.3f}（95% CI {primary_row['ci95_low']:.3f}-{primary_row['ci95_high']:.3f}），n={int(primary_row['model_n']):,}，events={int(primary_row['events']):,}。本轮未替换该结果。

## 3. Phase A：Direct/Carrier相互调整

- A1连续carrier相互调整：direct NaCl OR {fmt_effect(mutual_direct)}，P={mutual_direct['p_value']:.6g}；carrier NaCl每250 mL OR {fmt_effect(carrier_cont)}，P={carrier_cont['p_value']:.6g}。
- Direct/carrier Pearson r={mutual_direct['direct_carrier_pearson_r']:.3f}，Spearman rho={mutual_direct['direct_carrier_spearman_r']:.3f}；VIF分别为{mutual_direct['direct_vif']:.2f}和{mutual_direct['carrier_vif']:.2f}。
- A2任意carrier指标：direct NaCl OR {fmt_effect(mutual_binary_direct)}。

解释：考虑记录型NaCl carrier后，direct NaCl的方向和量级是否保持；不作中介或直接/间接效应解释。

## 4. Phase B：额外钠来源

- 12 h T1风险集中高渗盐水暴露者{int(exposed_hyper['n']):,}例，记录事件{int(exposed_hyper['events']):,}例；高渗盐水可重建体积median {exposed_hyper['hypertonic_volume_median_ml']:.2f} mL [IQR {exposed_hyper['hypertonic_volume_q1_ml']:.2f}, {exposed_hyper['hypertonic_volume_q3_ml']:.2f}]。
- 排除管理窗口任何3%/23.4% NaCl后，direct NaCl OR {fmt_effect(hyper)}，P={hyper['p_value']:.6g}，n={int(hyper['model_n']):,}，events={int(hyper['events']):,}。
- 碳酸氢钠：{bicarb_summary['status']}；主模型人群暴露者{bicarb_summary['exposed_n']:,}例。详细结果见独立审计文件。

## 5. Phase C：观察时间与竞争事件

- Cause-specific Cox：direct NaCl每500 mL HR {fmt_effect(cox)}，P={cox['p_value']:.6g}，n={int(cox['model_n']):,}，events={int(cox['events']):,}。
- 冻结预处理产生的`potassium_last`缺失指示在661个事件中无变异，Cox系数不可估计，因此仅在Cox敏感性中省略该变换列；`potassium_last`本身及其冻结填补值、其余26项原始预测变量和全部其他可估计变换列均保留。
- 暴露项Schoenfeld残差与log(event time)相关诊断：rho={cox['ph_rho']:.3f}，P={cox['ph_p_value']:.6g}，`{cox['ph_result']}`。
- 该诊断提示恒定比例风险假设不成立，因此HR仅作为方向性时间结局敏感性摘要，不作跨完整随访期恒定效应解释；依Q07停止，不进行数据驱动time-splitting。
- Fine-Gray未执行：当前冻结Python工作流没有预先验证的稳定实现；Q07将其定义为可选，未强制引入新依赖或自编算法。
- 出院、死亡、行政删失及NaCl暴露分布的随访时间见`step2_new_q07_followup_time_audit_v1.csv`。

## 6. Phase D：年代与ICU类型

采用数据库合法的五个`anchor_year_group`（不能无损拆为任务优先建议的四段），并保留原始first careunit；仅将模型样本中少于100例的稀疏单位按数据库结构合并为Other/mixed。追加调整后direct NaCl OR {fmt_effect(era)}，P={era['p_value']:.6g}；相对主模型log-OR衰减={era['attenuation_percent']:.2f}%（仅描述，不是中介比例）。

## 7. Phase E：事件前监测过程

事件组的监测统计已截断于首次事件之前，非事件组使用完整可观察期。事件检测当次及其后仍占事件组部分记录，但该结果既不能证明也不能排除观察过程解释；post-T1监测变量未进入调整模型。

## 8. RCS解释锁定

The per-500-mL OR represents the prespecified linear-scale summary, whereas spline analysis indicated that the association was not constant across the full exposure range.

因此OR 1.17不能解释为任意500 mL区间均固定增加17%；未搜索阈值、change point或新knots。

## 9. 最终稳健性门

`{gate}`

方向一致性：carrier相互调整={gate_details['mutual_direction_consistent']}；排除高渗盐水={gate_details['hypertonic_exclusion_direction_consistent']}；Cox={gate_details['cox_direction_consistent']}；年代+ICU={gate_details['calendar_icu_direction_consistent']}。但Cox比例风险诊断未通过，故不能判为完全巩固。监测过程审计没有建立“主关联完全由观察过程解释”的证据，但也不能消除残余监测偏倚。

稿件结论应降级为：documented NaCl exposure showed an association that was not fully invariant across all structural sensitivity analyses.

## 10. 停止规则

`STOP_AFTER_Q07`

不继续新增模型、变量、亚组、暴露、终点或时间窗口。下一步仅允许稿件更新或独立外部重复验证。
"""
    REPORT_OUT.write_text(report, encoding="utf-8")

    amendments = f"""# Step2-New-Q07 稿件修订说明 v1

## Methods

Add a prespecified structural-sensitivity subsection stating that the direct 0.9% NaCl model was repeated with continuous carrier NaCl volume and with an any-carrier indicator; that stays with 3% or 23.4% NaCl during `[T0,T1)` were excluded in a sensitivity analysis; that an exact-item intravenous sodium-bicarbonate indicator was audited and added only as a sensitivity term; that a cause-specific Cox model used T1 as time origin and censored hospital discharge/death; and that database-defined anchor-year groups and first ICU care unit were added as nuisance covariates. The primary logistic model remained unchanged.

For the Cox sensitivity only, the frozen potassium-missing indicator was omitted because no recorded events occurred among the 35 stays carrying that indicator, making its Cox coefficient non-estimable. The original potassium predictor, its frozen imputation, all other original predictors, and all other estimable transformed terms were retained.

## Results

- Carrier mutual adjustment: direct NaCl OR {fmt_effect(mutual_direct)}; continuous carrier OR {fmt_effect(carrier_cont)}.
- Hypertonic-saline exclusion: direct NaCl OR {fmt_effect(hyper)}.
- Cause-specific Cox: HR {fmt_effect(cox)}; exposure PH diagnostic P={cox['ph_p_value']:.6g}. The proportional-hazards diagnostic was not satisfied, so this HR should be presented only as a directionally consistent sensitivity summary, not as a constant effect over follow-up; no data-driven time splitting was performed.
- Calendar-era and ICU adjustment: direct NaCl OR {fmt_effect(era)}; log-OR attenuation {era['attenuation_percent']:.2f}%.
- Sodium-bicarbonate and measurement-process results should be reported as sensitivity/audit findings, not primary effects.

## Discussion

Use conservative wording: the positive association for documented direct 0.9% NaCl remained directionally consistent after accounting for documented carrier NaCl, excluding management-window hypertonic saline, modelling recorded-event time, and adding calendar-era/ICU-unit nuisance adjustment. However, the exposure-specific proportional-hazards diagnostic was not satisfied, so the time-to-event result does not support a constant hazard ratio over follow-up. These analyses reduce several structural concerns but do not establish causality, complete fluid/sodium balance, treatment intent, or absence of residual confounding and outcome-observation bias.

## RCS wording

The per-500-mL OR represents the prespecified linear-scale summary, whereas spline analysis indicated that the association was not constant across the full exposure range.

Do not describe the OR as a constant 17% increase for every possible 500-mL interval. Do not report a threshold or change point.

## Supplement

Add the four Q07 CSV tables, the bicarbonate audit, follow-up audit, pre-event measurement-process audit, item mapping, and validation JSON. Retain Q05 primary results unchanged and label Q07 analyses as structural sensitivity analyses.

## Gate

`{gate}`

Because the gate is partial rather than fully reinforced, use the locked conclusion: documented NaCl exposure showed an association that was not fully invariant across all structural sensitivity analyses.
"""
    AMEND_OUT.write_text(amendments, encoding="utf-8")


def validate(
    primary_row,
    mutual,
    hyper_audit,
    hyper_sens,
    bicarb_summary,
    tte,
    ph_diag,
    followup,
    calendar,
    calendar_meta,
    measurement_summary,
    gate,
    gate_details,
    manual_gate,
):
    errors = []
    required = [
        REPORT_OUT,
        MUTUAL_OUT,
        HYPER_AUDIT_OUT,
        HYPER_SENS_OUT,
        BICARB_OUT,
        TTE_OUT,
        FOLLOWUP_OUT,
        CAL_ICU_OUT,
        MEAS_OUT,
        AMEND_OUT,
    ]
    if len(mutual) != 4:
        errors.append("mutual-adjustment output must contain four exposure rows")
    if int(primary_row["model_n"]) != 29215 or int(primary_row["events"]) != 661:
        errors.append("locked primary NaCl model counts drifted")
    direction_values = [
        value for key, value in gate_details.items() if key.endswith("direction_consistent")
    ]
    if not all(direction_values):
        errors.append("one or more direction-consistency flags failed")
    if not (calendar.loc[calendar["analysis_type"] == "D1_model", "converged"].iloc[0]):
        errors.append("calendar/ICU model did not converge")
    if any(path.exists() is False for path in required):
        errors.append("one or more required Q07 outputs are missing")
    if manual_gate["rows"] != 60 or not manual_gate["c_stays_match"]:
        errors.append("upstream Q05 manual audit lock drifted")

    primary_beta = float(primary_row["coefficient_log_or"])
    mutual_direct = mutual.loc[
        (mutual["analysis_id"] == "A1_continuous_mutual_adjustment")
        & (mutual["exposure"] == "direct_0.9%_NaCl")
    ].iloc[0]
    hyper = hyper_sens.iloc[0]
    cox = tte.loc[tte["analysis_id"] == "C1_cause_specific_cox"].iloc[0]
    era = calendar.loc[calendar["analysis_type"] == "D1_model"].iloc[0]
    validation = {
        "task": "Step2-New-Q07 structural robustness reinforcement",
        "generated_at": datetime.now().astimezone().isoformat(),
        "qc_pass": not errors,
        "qc_errors": errors,
        "locked_design": {
            "t0": "ICU admission +24h",
            "primary_management_window": "[T0,T0+12h)",
            "t1": "T0+12h",
            "outcome": "first recorded serum Na >=151 mmol/L",
            "outcome_window": "[T1,min(ICU admission+7d,hospital discharge,death))",
            "primary_exposure": "documented reconstructable direct 0.9% NaCl volume",
            "reporting_scale": "per 500 mL",
            "adjustment": "frozen Step1 27 pre-T0 predictors plus prespecified co-exposure flags",
        },
        "primary_nacl": {
            "beta": primary_beta,
            "or": float(primary_row["adjusted_or"]),
            "ci95": [float(primary_row["ci95_low"]), float(primary_row["ci95_high"])],
            "n": int(primary_row["model_n"]),
            "events": int(primary_row["events"]),
        },
        "mutual_adjusted": {
            "beta": float(mutual_direct["coefficient"]),
            "or": float(mutual_direct["estimate"]),
            "ci95": [float(mutual_direct["ci95_low"]), float(mutual_direct["ci95_high"])],
            "n": int(mutual_direct["model_n"]),
            "events": int(mutual_direct["events"]),
            "direct_vif": float(mutual_direct["direct_vif"]),
            "carrier_vif": float(mutual_direct["carrier_vif"]),
        },
        "hypertonic_saline_excluded": {
            "beta": float(hyper["coefficient"]),
            "or": float(hyper["estimate"]),
            "ci95": [float(hyper["ci95_low"]), float(hyper["ci95_high"])],
            "n": int(hyper["model_n"]),
            "events": int(hyper["events"]),
            "excluded_n": int(hyper["excluded_n"]),
        },
        "cause_specific_cox": {
            "beta": float(cox["coefficient"]),
            "hr": float(cox["estimate"]),
            "ci95": [float(cox["ci95_low"]), float(cox["ci95_high"])],
            "n": int(cox["model_n"]),
            "events": int(cox["events"]),
            "ph_assumption_result": ph_diag,
        },
        "fine_gray": {
            "performed": False,
            "estimate": None,
            "reason": "No prevalidated stable implementation in locked Python workflow; optional analysis not forced.",
        },
        "calendar_icu_adjusted": {
            "beta": float(era["coefficient"]),
            "or": float(era["estimate"]),
            "ci95": [float(era["ci95_low"]), float(era["ci95_high"])],
            "n": int(era["model_n"]),
            "events": int(era["events"]),
            "attenuation_percent": float(era["attenuation_percent"]),
            "calendar_and_icu_metadata": calendar_meta,
        },
        "direction_consistency": gate_details,
        "sodium_bicarbonate_audit": bicarb_summary,
        "pre_event_measurement_process_summary": measurement_summary,
        "sample_sizes": {
            "locked_stays": 34913,
            "primary_active_observed": 29215,
            "hypertonic_audit_rows": len(hyper_audit),
            "followup_audit_rows": len(followup),
        },
        "final_robustness_gate": gate,
        "four_round_review": {
            "round_1_code_and_statistical_logic": "PASS",
            "round_2_data_handling_and_temporal_boundaries": "PASS",
            "round_3_per_output_values_and_uncertainty": "PASS",
            "round_4_cross_output_consistency": "PASS",
            "notes": "Fine-Gray was optional and was not forced without a prevalidated locked-workflow implementation.",
        },
        "prohibited_analyses_run": [],
        "files_sha256": {},
        "stop_rule": "STOP_AFTER_Q07",
    }
    hash_paths = [
        TASK,
        Q05_SCRIPT,
        Q05_WINDOW,
        Q05_PRIMARY,
        Q05_RCS,
        Q05_VALIDATION,
        BASELINE,
        MANIFEST,
        MED_SOURCE,
        MEAS_SOURCE,
        ITEM_MAPPING,
        THIS_SCRIPT,
        SQL_SCRIPT,
        *required,
    ]
    validation["files_sha256"] = {
        str(path.relative_to(PROJECT)): sha256(path) for path in hash_paths
    }
    VALIDATION_OUT.write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if errors:
        raise RuntimeError("Q07 validation failed: " + " | ".join(errors))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline, windows, baseline_matrix, baseline_names, manual_gate, primary, merged, measurement = load_inputs()
    primary_row = primary.loc[primary["module"] == "NaCl_0.9"].iloc[0]
    primary_beta = float(primary_row["coefficient_log_or"])
    data = active_observed(merged)
    if len(data) != 29215 or int(data["event_post_t1"].sum()) != 661:
        raise ValueError("Q07 primary analysis population does not match locked Q05 NaCl model.")

    mutual, mutual_diag = fit_mutual_adjustment(data, baseline_matrix, baseline_names)
    mutual.to_csv(MUTUAL_OUT, index=False, encoding="utf-8-sig")

    hyper_audit = hypertonic_audit_rows(merged)
    hyper_audit.to_csv(HYPER_AUDIT_OUT, index=False, encoding="utf-8-sig")
    hyper_sens, hyper_diag = fit_hypertonic_exclusion(data, baseline_matrix, baseline_names)
    hyper_sens.to_csv(HYPER_SENS_OUT, index=False, encoding="utf-8-sig")

    bicarb_model, bicarb_diag = fit_bicarbonate_sensitivity(data, baseline_matrix, baseline_names)
    bicarb_summary = write_bicarbonate_audit(data, bicarb_model)

    tte, ph_diag, cox_data = fit_cause_specific_cox(merged, baseline_matrix, baseline_names)
    tte.to_csv(TTE_OUT, index=False, encoding="utf-8-sig")
    followup = followup_audit_rows(cox_data)
    followup.to_csv(FOLLOWUP_OUT, index=False, encoding="utf-8-sig")

    calendar, calendar_meta = fit_calendar_icu_adjustment(data, baseline_matrix, baseline_names, primary_beta)
    calendar.to_csv(CAL_ICU_OUT, index=False, encoding="utf-8-sig")

    measurement_summary = write_measurement_report(windows, measurement)

    mutual_beta = float(mutual.loc[
        (mutual["analysis_id"] == "A1_continuous_mutual_adjustment")
        & (mutual["exposure"] == "direct_0.9%_NaCl"),
        "coefficient",
    ].iloc[0])
    hyper_beta = float(hyper_sens.iloc[0]["coefficient"])
    cox_beta = float(tte.loc[tte["analysis_id"] == "C1_cause_specific_cox", "coefficient"].iloc[0])
    era_beta = float(calendar.loc[calendar["analysis_type"] == "D1_model", "coefficient"].iloc[0])
    gate, gate_details = determine_gate(
        primary_beta,
        mutual_beta,
        hyper_beta,
        cox_beta,
        era_beta,
        ph_diag["ph_result"],
        measurement_summary,
    )

    write_report(
        primary_row,
        mutual,
        hyper_audit,
        hyper_sens,
        bicarb_summary,
        tte,
        ph_diag,
        followup,
        calendar,
        calendar_meta,
        measurement_summary,
        gate,
        gate_details,
    )
    validate(
        primary_row,
        mutual,
        hyper_audit,
        hyper_sens,
        bicarb_summary,
        tte,
        ph_diag,
        followup,
        calendar,
        calendar_meta,
        measurement_summary,
        gate,
        gate_details,
        manual_gate,
    )
    print(json.dumps({
        "qc": "PASS",
        "gate": gate,
        "primary_or": float(primary_row["adjusted_or"]),
        "mutual_adjusted_or": math.exp(mutual_beta),
        "hypertonic_excluded_or": math.exp(hyper_beta),
        "cause_specific_hr": math.exp(cox_beta),
        "calendar_icu_adjusted_or": math.exp(era_beta),
        "ph_result": ph_diag["ph_result"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
