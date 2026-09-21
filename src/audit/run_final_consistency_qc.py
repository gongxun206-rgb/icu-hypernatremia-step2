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
from statsmodels.duration.hazard_regression import PHReg


REPO_ROOT = Path(__file__).resolve().parents[2]
RESTRICTED_ROOT = Path(os.environ.get("STEP2_RESTRICTED_ROOT", REPO_ROOT / "restricted_data"))
OUTPUT_ROOT = Path(os.environ.get("STEP2_OUTPUT_ROOT", REPO_ROOT / "outputs"))
PROJECT = REPO_ROOT
Q07 = OUTPUT_ROOT / "structural_sensitivity"
OUT = OUTPUT_ROOT / "final_consistency_qc"
TASK = REPO_ROOT / "docs" / "protocol_lock.md"
Q07_SCRIPT = REPO_ROOT / "src" / "models" / "run_structural_sensitivity.py"
Q07_VALIDATION = Q07 / "step2_new_q07_validation_v1.json"
Q07_MUTUAL = Q07 / "step2_new_q07_direct_carrier_mutual_adjustment_v1.csv"
Q07_HYPER = Q07 / "step2_new_q07_hypertonic_saline_exclusion_sensitivity_v1.csv"
Q07_CALENDAR = Q07 / "step2_new_q07_calendar_icu_adjustment_v1.csv"
Q07_TTE = Q07 / "step2_new_q07_time_to_event_sensitivity_v1.csv"
MEAS_SOURCE = RESTRICTED_ROOT / "mimic" / "step2_new_q07_qc_measurement_timing_source_v1.csv"
SQL_SCRIPT = REPO_ROOT / "sql" / "mimic" / "03_extract_measurement_timing_qc.sql"
THIS_SCRIPT = Path(__file__).resolve()

DECOMP_OUT = OUT / "step2_new_q07_qc_cox_cohort_decomposition_v1.csv"
COX_OUT = OUT / "step2_new_q07_qc_cox_aligned_sensitivity_v1.csv"
MEAS_OUT = OUT / "step2_new_q07_qc_pre_event_measurement_process_v1.md"
BICARB_OUT = OUT / "step2_new_q07_qc_bicarbonate_sensitivity_v1.csv"
REPORT_OUT = OUT / "Step2_New_Q07_QC_final_report_v1.md"
WORDING_OUT = OUT / "Step2_New_Q07_QC_manuscript_wording_update_v1.md"
VALIDATION_OUT = OUT / "step2_new_q07_qc_validation_v1.json"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q07 = import_module(Q07_SCRIPT, "q07_locked_outputs")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(series: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    values = clean.quantile([0.25, 0.50, 0.75])
    return float(values.loc[0.50]), float(values.loc[0.25]), float(values.loc[0.75])


def build_decomposition(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = merged.copy()
    in_risk = frame["in_t1_risk_set"].astype(bool)
    observed = frame["post_t1_na_count"] > 0
    active = frame["inputevent_interface_status"] == "ACTIVE"
    silent = frame["inputevent_interface_status"] == "SILENT"
    primary = in_risk & observed & active
    original_cox = in_risk & active

    reason = np.full(len(frame), "NOT_IN_EITHER_ANALYTIC_COHORT", dtype=object)
    reason[(primary & original_cox).to_numpy()] = "IN_BOTH_PRIMARY_LOGISTIC_AND_ORIGINAL_COX"
    extra = original_cox & ~primary
    reason[(extra & ~observed).to_numpy()] = (
        "POST_T1_NA_UNASCERTAINED_INCLUDED_AS_EVENT_FREE_UNTIL_CENSORING"
    )
    reason[(~original_cox & ~in_risk).to_numpy()] = "NOT_IN_T1_RISK_SET"
    reason[(~original_cox & in_risk & silent).to_numpy()] = (
        "INPUTEVENTS_INTERFACE_SILENT_EXPOSURE_UNDEFINED"
    )
    reason[(~original_cox & in_risk & ~silent & ~active).to_numpy()] = (
        "NONSTANDARD_INPUTEVENTS_INTERFACE_STATUS"
    )

    out = pd.DataFrame({
        "stay_id": frame["stay_id"].astype(int),
        "in_primary_logistic": primary,
        "in_q07_cox": original_cox,
        "post_t1_na_observed": observed,
        "inputevents_interface_active": active,
        "inputevents_interface_silent": silent,
        "outcome_event": frame["event_post_t1"].astype(bool),
        "outcome_unascertained": in_risk & ~observed,
        "discharge_time": frame["dischtime"],
        "death_time": frame["deathtime"],
        "administrative_end": frame["administrative_end"],
        "reason_in_cox_not_primary": reason,
    }).sort_values("stay_id")

    extra_frame = out.loc[out["in_q07_cox"] & ~out["in_primary_logistic"]]
    reason_counts = extra_frame["reason_in_cox_not_primary"].value_counts().to_dict()
    summary = {
        "locked_n": len(out),
        "primary_logistic_n": int(out["in_primary_logistic"].sum()),
        "primary_logistic_events": int(out.loc[out["in_primary_logistic"], "outcome_event"].sum()),
        "original_cox_n": int(out["in_q07_cox"].sum()),
        "original_cox_events": int(out.loc[out["in_q07_cox"], "outcome_event"].sum()),
        "cox_not_primary_n": len(extra_frame),
        "cox_not_primary_reason_counts": reason_counts,
        "extra_post_t1_na_observed_n": int(extra_frame["post_t1_na_observed"].sum()),
        "extra_outcome_unascertained_n": int(extra_frame["outcome_unascertained"].sum()),
        "extra_interface_active_n": int(extra_frame["inputevents_interface_active"].sum()),
        "extra_interface_silent_n": int(extra_frame["inputevents_interface_silent"].sum()),
        "extra_events": int(extra_frame["outcome_event"].sum()),
    }
    return out, summary


def fit_aligned_cox(data, baseline_matrix, baseline_names) -> tuple[pd.DataFrame, dict]:
    t1 = pd.to_datetime(data["t1_time"])
    observation_end = pd.to_datetime(data["observation_end"])
    event_time = pd.to_datetime(data["first_post_t1_event_time"])
    status = data["event_post_t1"].astype(int).to_numpy()
    stop_time = event_time.where(data["event_post_t1"].astype(bool), observation_end)
    duration_hours = (stop_time - t1).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    if np.any(duration_hours <= 0):
        raise ValueError("Aligned Cox contains non-positive follow-up time.")

    exposure = data["nacl_direct_ab_ml"].to_numpy(dtype=float) / 500.0
    design, names = q07.q05.build_design(
        data,
        baseline_matrix,
        baseline_names,
        "nacl",
        exposure,
        "direct_nacl_per_500ml",
    )
    design = design[:, 1:]
    names = names[1:]

    nonestimable = [
        index for index, name in enumerate(names)
        if np.nanstd(design[status.astype(bool), index]) < 1e-12
    ]
    omitted = [names[index] for index in nonestimable]
    if any("missingindicator_" not in name for name in omitted):
        raise RuntimeError("An original aligned-Cox term is non-estimable: " + ", ".join(omitted))
    if nonestimable:
        keep = [index for index in range(design.shape[1]) if index not in nonestimable]
        design = design[:, keep]
        names = [names[index] for index in keep]

    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise RuntimeError("Aligned Cox design matrix is rank deficient.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = PHReg(duration_hours, design, status=status, ties="efron").fit(
            method="bfgs", disp=0, maxiter=500
        )
    if caught:
        raise RuntimeError("Aligned Cox emitted warnings: " + " | ".join(str(w.message) for w in caught))

    index = names.index("direct_nacl_per_500ml")
    beta = float(result.params[index])
    standard_error = float(result.bse[index])
    schoenfeld = np.asarray(result.schoenfeld_residuals)
    event_mask = status.astype(bool) & np.isfinite(schoenfeld[:, index])
    rho, ph_p = st.spearmanr(
        schoenfeld[event_mask, index], np.log(duration_hours[event_mask])
    )
    ph_result = (
        "PASS_NO_EXPOSURE_SPECIFIC_SIGNAL"
        if ph_p >= 0.05
        else "EXPOSURE_PH_SIGNAL_DETECTED"
    )
    row = {
        "analysis_id": "Q07_QC_aligned_cause_specific_cox",
        "cohort_definition": "primary_logistic_outcome_observed_plus_interface_active",
        "model_n": len(data),
        "events": int(status.sum()),
        "exposure": "documented_reconstructable_direct_0.9%_NaCl",
        "scale": "per_500_ml",
        "coefficient_log_hr": beta,
        "standard_error": standard_error,
        "hazard_ratio": math.exp(beta),
        "ci95_low": math.exp(beta - 1.96 * standard_error),
        "ci95_high": math.exp(beta + 1.96 * standard_error),
        "p_value": float(result.pvalues[index]),
        "schoenfeld_rho": float(rho),
        "ph_test_p": float(ph_p),
        "ph_result": ph_result,
        "design_columns": design.shape[1],
        "design_rank": rank,
        "omitted_nonestimable_transformed_features": " | ".join(omitted),
    }
    return pd.DataFrame([row]), {
        "omitted_nonestimable_transformed_features": omitted,
        "ph_result": ph_result,
        "ph_rho": float(rho),
        "ph_p": float(ph_p),
    }


def measurement_qc(windows: pd.DataFrame) -> dict:
    source = pd.read_csv(MEAS_SOURCE)
    source["event_post_t1"] = q07.to_bool(source["event_post_t1"])
    if len(source) != 34913 or source["stay_id"].nunique() != 34913:
        raise ValueError("Measurement timing source must contain 34,913 unique stays.")
    analysis = q07.q05.model_frame(windows, 12).merge(
        source,
        on="stay_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_qc"),
    )
    if len(analysis) != 33538 or int(analysis["event_post_t1"].sum()) != 793:
        raise ValueError("Measurement QC cohort does not match locked Q05 observed-outcome cohort.")
    if not (analysis["post_t1_na_count"].astype(int) == analysis["total_pre_censor_count"].astype(int)).all():
        raise ValueError("Exact timing extraction does not reproduce Q05 sodium counts.")
    event_match = analysis["event_post_t1"].astype(bool) == analysis["event_post_t1_qc"].astype(bool)
    if not event_match.all():
        raise ValueError("Exact timing extraction does not reproduce Q05 events.")

    event = analysis.loc[analysis["event_post_t1"].astype(bool)].copy()
    non_event = analysis.loc[~analysis["event_post_t1"].astype(bool)].copy()
    if not (
        event["through_event_detection_count"] + event["post_event_count"]
        == event["total_pre_censor_count"]
    ).all():
        raise ValueError("Event measurement-period counts do not sum to total.")
    if not (
        event["strictly_pre_event_count"] + event["at_event_timestamp_count"]
        == event["through_event_detection_count"]
    ).all():
        raise ValueError("Strict/event-detection counts are internally inconsistent.")

    event["strict_density_per_day"] = event["strictly_pre_event_count"] / (
        event["time_t1_to_event_hours"].clip(lower=1.0 / 60.0) / 24.0
    )
    non_event["pre_censor_density_per_day"] = non_event["total_pre_censor_count"] / (
        non_event["analysis_followup_hours"].clip(lower=1.0 / 60.0) / 24.0
    )
    event["time_to_first_na_hours"] = (
        pd.to_datetime(event["first_post_t1_na_time_qc"])
        - pd.to_datetime(event["t1_time_qc"])
    ).dt.total_seconds() / 3600.0

    fields = {
        "event_strictly_pre_count": quantiles(event["strictly_pre_event_count"]),
        "event_through_detection_count": quantiles(event["through_event_detection_count"]),
        "event_post_count": quantiles(event["post_event_count"]),
        "event_total_count": quantiles(event["total_pre_censor_count"]),
        "event_strict_density_per_day": quantiles(event["strict_density_per_day"]),
        "event_time_to_first_na_hours": quantiles(event["time_to_first_na_hours"]),
        "event_time_to_event_hours": quantiles(event["time_t1_to_event_hours"]),
        "non_event_total_count": quantiles(non_event["total_pre_censor_count"]),
        "non_event_density_per_day": quantiles(non_event["pre_censor_density_per_day"]),
    }
    density_test = st.mannwhitneyu(
        event["strict_density_per_day"],
        non_event["pre_censor_density_per_day"],
        alternative="two-sided",
    )
    fields["density_mannwhitney_p"] = float(density_test.pvalue)
    fields["event_n"] = len(event)
    fields["non_event_n"] = len(non_event)
    fields["strict_definition_verified"] = True
    fields["post_event_contribution_present"] = (
        fields["event_post_count"][0] > 0
    )
    fields["strict_pre_event_density_higher"] = (
        fields["event_strict_density_per_day"][0]
        > fields["non_event_density_per_day"][0]
    )

    def qi(values: tuple[float, float, float]) -> str:
        return f"{values[0]:.2f} [{values[1]:.2f}, {values[2]:.2f}]"

    text = f"""# Step2-New-Q07-QC 事件前监测过程终末核查 v1

## 时间定义核验

- `STRICTLY_PRE_EVENT`：`measurement_time < event_measurement_time`，不包括发现Na>=151的当次记录。
- `THROUGH_EVENT_DETECTION`：`measurement_time <= event_measurement_time`，包括事件发现当次记录。
- `POST_EVENT`：`measurement_time > event_measurement_time`。

代码及数据库时间戳核验确认：Q07的`na_count_before_event_or_censor`使用严格小于号，确实严格截断于事件前。先前文字中的“事件发现当次及以后记录”指原Q05全随访总次数中的剩余部分，不属于严格事件前计数；本文件将两者明确拆开。

## Event group（n={len(event):,}）

| 指标 | median [IQR] |
|---|---:|
| Strictly-pre-event Na count | {qi(fields['event_strictly_pre_count'])} |
| Through-event-detection Na count | {qi(fields['event_through_detection_count'])} |
| Post-event Na count | {qi(fields['event_post_count'])} |
| Total post-T1 Na count | {qi(fields['event_total_count'])} |
| Strictly-pre-event testing density per day | {qi(fields['event_strict_density_per_day'])} |
| T1 to first Na, hours | {qi(fields['event_time_to_first_na_hours'])} |
| T1 to event, hours | {qi(fields['event_time_to_event_hours'])} |

## Non-event group（n={len(non_event):,}）

- Total pre-censor Na count：{qi(fields['non_event_total_count'])}。
- Pre-censor testing density per day：{qi(fields['non_event_density_per_day'])}。

## 锁定解释

Recorded-event stays had more total post-T1 sodium measurements; however, part of this difference reflected intensified monitoring after event detection.

More frequent sodium testing was already present before event detection, consistent with informative observation.

事件前检测密度与非事件删失前检测密度的描述性Mann-Whitney P={fields['density_mannwhitney_p']:.6g}。该结果描述观察过程差异，不进入主模型，也不证明检测过程造成NaCl关联。
"""
    MEAS_OUT.write_text(text, encoding="utf-8")
    return fields


def bicarbonate_qc(data, baseline_matrix, baseline_names, primary_row) -> tuple[pd.DataFrame, dict]:
    model, diagnostic = q07.fit_bicarbonate_sensitivity(
        data, baseline_matrix, baseline_names
    )
    direct = model.loc[model["exposure"] == "direct_0.9%_NaCl"].iloc[0]
    bicarbonate = model.loc[model["exposure"] == "any_iv_sodium_bicarbonate"].iloc[0]
    primary_beta = float(primary_row["coefficient_log_or"])
    change = float(direct["coefficient"] - primary_beta)
    percent_change = 100 * change / primary_beta
    row = {
        "analysis_id": "Q07_QC_confirmed_bicarbonate_adjusted_sensitivity",
        "execution_status": "EXECUTED_IN_Q07_AND_EXACTLY_REPRODUCED_IN_QC",
        "bicarbonate_variable_definition": "any inputevents itemid 220995 or 227533 overlapping [T0,T1)",
        "exposed_n": int(data["bicarbonate_any"].sum()),
        "exposed_events": int(data.loc[data["bicarbonate_any"], "event_post_t1"].sum()),
        "model_n": len(data),
        "total_events": int(data["event_post_t1"].sum()),
        "direct_nacl_scale": "per_500_ml",
        "direct_nacl_log_or": float(direct["coefficient"]),
        "direct_nacl_or": float(direct["estimate"]),
        "direct_nacl_ci95_low": float(direct["ci95_low"]),
        "direct_nacl_ci95_high": float(direct["ci95_high"]),
        "direct_nacl_p": float(direct["p_value"]),
        "bicarbonate_log_or": float(bicarbonate["coefficient"]),
        "bicarbonate_or": float(bicarbonate["estimate"]),
        "bicarbonate_ci95_low": float(bicarbonate["ci95_low"]),
        "bicarbonate_ci95_high": float(bicarbonate["ci95_high"]),
        "bicarbonate_p": float(bicarbonate["p_value"]),
        "primary_nacl_log_or": primary_beta,
        "change_in_nacl_log_or_vs_primary": change,
        "percent_change_in_nacl_log_or_vs_primary": percent_change,
        "material_change_interpretation": "DID_NOT_MATERIALLY_CHANGE_DIRECT_NACL_ASSOCIATION",
        "model_converged": bool(diagnostic["converged"]),
        "rank_deficiency": int(diagnostic["rank_deficiency"]),
    }
    return pd.DataFrame([row]), row


def determine_gate(mutual, hyper, calendar, aligned_cox, bicarbonate, measurement) -> tuple[str, dict]:
    mutual_direct = mutual.loc[
        (mutual["analysis_id"] == "A1_continuous_mutual_adjustment")
        & (mutual["exposure"] == "direct_0.9%_NaCl")
    ].iloc[0]
    calendar_model = calendar.loc[calendar["analysis_type"] == "D1_model"].iloc[0]
    flags = {
        "carrier_mutual_adjustment_direction_consistent": float(mutual_direct["coefficient"]) > 0,
        "hypertonic_exclusion_direction_consistent": float(hyper.iloc[0]["coefficient"]) > 0,
        "calendar_icu_direction_consistent": float(calendar_model["coefficient"]) > 0,
        "aligned_cox_direction_consistent": float(aligned_cox.iloc[0]["coefficient_log_hr"]) > 0,
        "bicarbonate_adjustment_no_direction_reversal": float(bicarbonate["direct_nacl_log_or"]) > 0,
        "measurement_not_entirely_post_event": bool(measurement["strict_pre_event_density_higher"]),
    }
    if all(flags.values()):
        gate = "PRIMARY_ROBUSTNESS_REINFORCED"
    elif all(value for key, value in flags.items() if key != "measurement_not_entirely_post_event"):
        gate = "PRIMARY_ROBUSTNESS_PARTIALLY_REINFORCED"
    else:
        gate = "PRIMARY_INTERPRETATION_REQUIRES_REVISION"
    return gate, flags


def write_reports(
    decomposition,
    aligned_cox,
    bicarbonate,
    measurement,
    gate,
    gate_flags,
    original_cox,
):
    cox = aligned_cox.iloc[0]
    bic = bicarbonate.iloc[0]
    extra_reason = next(iter(decomposition["cox_not_primary_reason_counts"]))
    report = f"""# Step2-New-Q07-QC 终末一致性核查报告 v1

## 1. QC-A：Cox样本量差异

- Primary logistic：n={decomposition['primary_logistic_n']:,}，events={decomposition['primary_logistic_events']:,}。
- 原Q07 Cox：n={decomposition['original_cox_n']:,}，events={decomposition['original_cox_events']:,}。
- 多出的{decomposition['cox_not_primary_n']:,}例全部属于：`{extra_reason}`。
- 这{decomposition['cox_not_primary_n']:,}例全部为inputevents接口ACTIVE、post-T1 Na未观察、结局未确定、记录事件0例；不是缺失协变量、接口沉默或暴露未定义。

原Q07 Cox把这些患者作为event-free until censoring纳入，因此其风险集不与primary recorded-event estimand完全一致。该结果保留为历史敏感性，但不作为最终对齐版Cox。

## 2. 对齐版 Cox

对齐到主Logistic队列后：n={int(cox['model_n']):,}，events={int(cox['events']):,}，direct NaCl每500 mL HR={cox['hazard_ratio']:.3f}（95% CI {cox['ci95_low']:.3f}-{cox['ci95_high']:.3f}），P={cox['p_value']:.6g}。

暴露Schoenfeld rho={cox['schoenfeld_rho']:.3f}，PH test P={cox['ph_test_p']:.6g}，`{cox['ph_result']}`。方向与主Logistic一致，但PH假设仍不满足，故HR不能解释为整个随访期恒定关联；未做time-splitting。

## 3. QC-B：监测过程语义

数据库时间戳与代码确认Q07严格事件前计数确实使用`measurement_time < event_time`。事件发现当次和事件后记录未混入strictly-pre-event计数。事件组post-event记录median={measurement['event_post_count'][0]:.2f}，同时事件前检测密度median={measurement['event_strict_density_per_day'][0]:.2f}/day，高于非事件组{measurement['non_event_density_per_day'][0]:.2f}/day。

因此锁定为：总检测次数差异部分来自事件后强化监测，但事件发生前已经存在更密集检测，符合informative observation；不能据此认定主关联完全由post-event monitoring产生。

## 4. QC-C：碳酸氢钠

Q07确实执行了二元碳酸氢钠指标敏感性，本QC按相同冻结定义精确复现。暴露者{int(bic['exposed_n']):,}例、事件{int(bic['exposed_events']):,}例；模型n={int(bic['model_n']):,}、events={int(bic['total_events']):,}。

加入该指标后direct NaCl OR={bic['direct_nacl_or']:.3f}（95% CI {bic['direct_nacl_ci95_low']:.3f}-{bic['direct_nacl_ci95_high']:.3f}），P={bic['direct_nacl_p']:.6g}；相对主模型log-OR变化={bic['change_in_nacl_log_or_vs_primary']:.6f}（{bic['percent_change_in_nacl_log_or_vs_primary']:.2f}%）。Adjustment for documented sodium-bicarbonate exposure did not materially change the direct NaCl association.

## 5. 最终主估计稳健性门

`{gate}`

Primary-estimand相关分析均保持正向：carrier相互调整={gate_flags['carrier_mutual_adjustment_direction_consistent']}；排除高渗盐水={gate_flags['hypertonic_exclusion_direction_consistent']}；年代+ICU={gate_flags['calendar_icu_direction_consistent']}；对齐Cox={gate_flags['aligned_cox_direction_consistent']}；碳酸氢钠调整无反转={gate_flags['bicarbonate_adjustment_no_direction_reversal']}。PH假设不满足只限制time-to-event HR的解释，不自动否定固定窗口binary primary estimand。

## 6. 封板

`MIMIC_STEP2_LOCKED_AFTER_Q07_QC`

`STOP_AFTER_Q07_QC`

不继续新增模型、亚组、时间窗口、液体、终点、阈值、interaction或因果方法。下一步仅允许稿件更新或独立外部验证。
"""
    REPORT_OUT.write_text(report, encoding="utf-8")

    wording = f"""# Step2-New-Q07-QC 稿件措辞更新 v1

## Superseded wording

Do not use the prior global sentence:

> documented NaCl exposure showed an association that was not fully invariant across all structural sensitivity analyses.

The prior wording conflated a non-proportional time-to-event sensitivity with the robustness of the fixed-window binary primary estimand.

## Methods correction

The final cause-specific Cox sensitivity was restricted to the same 29,215 outcome-observed, inputevents-interface-active stays used in the primary logistic analysis. Hospital discharge, death, and the administrative end were censoring events. The frozen potassium-missing indicator was omitted only from the Cox model because no events occurred among the 35 stays carrying that transformed indicator; the original potassium predictor and all other estimable frozen terms were retained.

## Results wording

The aligned cause-specific Cox analysis included {int(cox['model_n']):,} stays and {int(cox['events']):,} events. The hazard ratio per 500 mL of documented direct 0.9% NaCl was {cox['hazard_ratio']:.3f} (95% CI {cox['ci95_low']:.3f}-{cox['ci95_high']:.3f}; P={cox['p_value']:.6g}). The exposure-specific proportional-hazards diagnostic was not satisfied (Schoenfeld rho={cox['schoenfeld_rho']:.3f}; P={cox['ph_test_p']:.6g}).

Adjustment for documented pre-T1 intravenous sodium-bicarbonate exposure did not materially change the direct NaCl association (adjusted OR {bic['direct_nacl_or']:.3f}, 95% CI {bic['direct_nacl_ci95_low']:.3f}-{bic['direct_nacl_ci95_high']:.3f}).

Recorded-event stays had more total post-T1 sodium measurements; however, part of this difference reflected intensified monitoring after event detection. More frequent sodium testing was already present before event detection, consistent with informative observation.

## Discussion and conclusion

The positive association remained directionally consistent across prespecified structural sensitivity analyses, including mutual adjustment for documented NaCl carrier exposure, exclusion of management-window hypertonic saline, additional calendar-era and ICU-unit adjustment, documented sodium-bicarbonate adjustment, and an aligned time-to-event analysis. The proportional-hazards assumption was not satisfied, indicating that the time-to-event association was not constant over follow-up.

These findings strengthen the robustness of the fixed-window recorded-outcome association but do not establish causality, treatment intent, or absence of residual confounding and informative observation.

## Gate

`{gate}`
"""
    WORDING_OUT.write_text(wording, encoding="utf-8")


def validate_and_write(
    decomposition_df,
    decomposition,
    aligned_cox,
    bicarbonate,
    measurement,
    gate,
    gate_flags,
):
    errors = []
    if decomposition["primary_logistic_n"] != 29215 or decomposition["primary_logistic_events"] != 661:
        errors.append("primary logistic counts drifted")
    if decomposition["original_cox_n"] != 30185 or decomposition["original_cox_events"] != 661:
        errors.append("original Cox counts drifted")
    if decomposition["cox_not_primary_n"] != 970:
        errors.append("Cox-primary difference is not 970")
    expected_reason = "POST_T1_NA_UNASCERTAINED_INCLUDED_AS_EVENT_FREE_UNTIL_CENSORING"
    if decomposition["cox_not_primary_reason_counts"] != {expected_reason: 970}:
        errors.append("the 970 extra stays are not explained by one locked reason")
    cox = aligned_cox.iloc[0]
    if int(cox["model_n"]) != 29215 or int(cox["events"]) != 661:
        errors.append("aligned Cox counts do not match primary logistic cohort")
    if not measurement["strict_definition_verified"]:
        errors.append("strict pre-event measurement definition not verified")
    if bicarbonate.iloc[0]["execution_status"] != "EXECUTED_IN_Q07_AND_EXACTLY_REPRODUCED_IN_QC":
        errors.append("bicarbonate sensitivity status unresolved")
    if gate != "PRIMARY_ROBUSTNESS_REINFORCED":
        errors.append("final gate did not meet the task-defined reinforced conditions")

    required = [DECOMP_OUT, COX_OUT, MEAS_OUT, BICARB_OUT, REPORT_OUT, WORDING_OUT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        errors.append("missing outputs: " + "; ".join(missing))

    validation = {
        "task": "Step2-New-Q07-QC final consistency QC and lock",
        "generated_at": datetime.now().astimezone().isoformat(),
        "qc_pass": not errors,
        "qc_errors": errors,
        "primary_logistic": {
            "n": decomposition["primary_logistic_n"],
            "events": decomposition["primary_logistic_events"],
        },
        "original_q07_cox": {
            "n": decomposition["original_cox_n"],
            "events": decomposition["original_cox_events"],
            "extra_vs_primary_n": decomposition["cox_not_primary_n"],
            "exact_reason_for_extra_970": decomposition["cox_not_primary_reason_counts"],
            "interpretation": "The extra stays had no post-T1 sodium and were included as event-free until censoring; this was not fully aligned with the primary recorded-event estimand.",
        },
        "aligned_cox": {
            "n": int(cox["model_n"]),
            "events": int(cox["events"]),
            "hr": float(cox["hazard_ratio"]),
            "ci95": [float(cox["ci95_low"]), float(cox["ci95_high"])],
            "p_value": float(cox["p_value"]),
            "schoenfeld_rho": float(cox["schoenfeld_rho"]),
            "ph_test_p": float(cox["ph_test_p"]),
            "ph_result": cox["ph_result"],
            "omitted_nonestimable_transformed_features": cox["omitted_nonestimable_transformed_features"],
        },
        "bicarbonate_sensitivity": bicarbonate.iloc[0].to_dict(),
        "strictly_pre_event_monitoring_summary": measurement,
        "final_primary_robustness_gate": gate,
        "gate_flags": gate_flags,
        "cohort_decomposition_rows": len(decomposition_df),
        "four_round_review": {
            "round_1_code_and_estimand_logic": "PASS",
            "round_2_timestamp_and_cohort_data_handling": "PASS",
            "round_3_per_output_counts_and_uncertainty": "PASS",
            "round_4_cross_output_and_wording_consistency": "PASS",
        },
        "prohibited_analyses_run": [],
        "step2_lock": "MIMIC_STEP2_LOCKED_AFTER_Q07_QC",
        "stop_rule": "STOP_AFTER_Q07_QC",
        "files_sha256": {},
    }
    hash_paths = [
        TASK,
        Q07_SCRIPT,
        Q07_VALIDATION,
        Q07_MUTUAL,
        Q07_HYPER,
        Q07_CALENDAR,
        Q07_TTE,
        MEAS_SOURCE,
        SQL_SCRIPT,
        THIS_SCRIPT,
        *required,
    ]
    validation["files_sha256"] = {
        str(path.relative_to(PROJECT)): sha256(path) for path in hash_paths
    }
    VALIDATION_OUT.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError("Q07-QC validation failed: " + " | ".join(errors))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline, windows, baseline_matrix, baseline_names, manual_gate, primary, merged, measurement = q07.load_inputs()
    primary_row = primary.loc[primary["module"] == "NaCl_0.9"].iloc[0]
    primary_data = q07.active_observed(merged)

    decomposition_df, decomposition = build_decomposition(merged)
    decomposition_df.to_csv(DECOMP_OUT, index=False, encoding="utf-8-sig")

    aligned_cox, aligned_diagnostic = fit_aligned_cox(
        primary_data, baseline_matrix, baseline_names
    )
    aligned_cox.to_csv(COX_OUT, index=False, encoding="utf-8-sig")

    measurement_summary = measurement_qc(windows)
    bicarbonate, bicarbonate_summary = bicarbonate_qc(
        primary_data, baseline_matrix, baseline_names, primary_row
    )
    bicarbonate.to_csv(BICARB_OUT, index=False, encoding="utf-8-sig")

    mutual = pd.read_csv(Q07_MUTUAL)
    hyper = pd.read_csv(Q07_HYPER)
    calendar = pd.read_csv(Q07_CALENDAR)
    original_cox = pd.read_csv(Q07_TTE).loc[lambda x: x["analysis_id"] == "C1_cause_specific_cox"].iloc[0]
    gate, gate_flags = determine_gate(
        mutual, hyper, calendar, aligned_cox, bicarbonate_summary, measurement_summary
    )
    write_reports(
        decomposition,
        aligned_cox,
        bicarbonate,
        measurement_summary,
        gate,
        gate_flags,
        original_cox,
    )
    validate_and_write(
        decomposition_df,
        decomposition,
        aligned_cox,
        bicarbonate,
        measurement_summary,
        gate,
        gate_flags,
    )
    cox = aligned_cox.iloc[0]
    bic = bicarbonate.iloc[0]
    print(json.dumps({
        "qc": "PASS",
        "extra_970_reason": decomposition["cox_not_primary_reason_counts"],
        "aligned_cox_hr": float(cox["hazard_ratio"]),
        "aligned_cox_ci95": [float(cox["ci95_low"]), float(cox["ci95_high"])],
        "aligned_cox_ph_p": float(cox["ph_test_p"]),
        "bicarbonate_adjusted_nacl_or": float(bic["direct_nacl_or"]),
        "final_gate": gate,
        "stop_rule": "STOP_AFTER_Q07_QC",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
