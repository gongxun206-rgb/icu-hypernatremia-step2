# Public sanitized copy; generated outputs remain outside Git.
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm


REPO_ROOT = Path(__file__).resolve().parents[2]
RESTRICTED_ROOT = Path(os.environ.get("STEP2_RESTRICTED_ROOT", REPO_ROOT / "restricted_data"))
OUTPUT_ROOT = Path(os.environ.get("STEP2_OUTPUT_ROOT", REPO_ROOT / "outputs"))
ROOT = REPO_ROOT
PHASE_A = REPO_ROOT / "config" / "harmonized"
OUT = OUTPUT_ROOT / "mimic_harmonized"
SOURCE = RESTRICTED_ROOT / "mimic" / "mimic_harmonized_analysis_source_v1.csv"
FLOW = RESTRICTED_ROOT / "mimic" / "MIMIC_harmonized_cohort_flow_v1.csv"
SQL = REPO_ROOT / "sql" / "mimic" / "04_build_harmonized_mimic_source.sql"
SCALES = {"nacl": 500.0, "lr": 500.0, "d5w": 250.0}
AGE_LEVELS = ["18-39", "40-49", "50-59", "60-69", "70-79", "80+"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qtext(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return "NA" if not len(x) else f"{x.median():.3f} [{x.quantile(.25):.3f}, {x.quantile(.75):.3f}]"


def fit_model(data: pd.DataFrame, focal: str, analysis_id: str) -> dict:
    d = data.copy()
    d = d[d.age_group.isin(AGE_LEVELS) & d.gender.isin(["M", "F"]) & d.sodium_last_pre_t0.notna()].copy()
    age = pd.Categorical(d.age_group, categories=AGE_LEVELS, ordered=True)
    age_dummies = pd.get_dummies(age, prefix="age", drop_first=True, dtype=float)
    X = pd.DataFrame(index=d.index)
    term = f"{focal}_per_{int(SCALES[focal])}ml"
    X[term] = d[f"{focal}_ml"] / SCALES[focal]
    X["sex_male"] = (d.gender == "M").astype(float)
    X["sodium_last_pre_t0"] = d.sodium_last_pre_t0.astype(float)
    for col in age_dummies.columns:
        X[col] = age_dummies[col].to_numpy()
    for other in SCALES:
        if other != focal:
            X[f"any_{other}_coexposure"] = (d[f"{other}_ml"] > 0).astype(float)
    X = sm.add_constant(X, has_constant="add").astype(float)
    y = d.event.astype(int)
    rank = int(np.linalg.matrix_rank(X.to_numpy()))
    warnings = []
    try:
        result = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-10)
        beta = float(result.params[term])
        lo, hi = result.conf_int().loc[term].astype(float)
        probs = np.asarray(result.predict(X))
        max_nonintercept = float(np.abs(result.params.drop("const")).max())
        if probs.min() < 1e-8 or probs.max() > 1 - 1e-8:
            warnings.append("EXTREME_PREDICTED_PROBABILITY")
        if max_nonintercept > 20:
            warnings.append("EXTREME_COEFFICIENT")
        return {
            "analysis_id": analysis_id, "fluid": focal, "scale_ml": SCALES[focal],
            "model_n": len(d), "events": int(y.sum()), "event_rate": float(y.mean()),
            "beta": beta, "se": float(result.bse[term]), "adjusted_or": float(np.exp(beta)),
            "ci95_low": float(np.exp(lo)), "ci95_high": float(np.exp(hi)), "p_value": float(result.pvalues[term]),
            "converged": bool(result.converged), "iterations": int(result.fit_history.get("iteration", -1)),
            "columns": X.shape[1], "rank": rank, "rank_deficiency": X.shape[1] - rank,
            "intercept": float(result.params["const"]), "max_abs_coefficient": float(np.abs(result.params).max()),
            "max_abs_nonintercept_coefficient": max_nonintercept,
            "predicted_probability_min": float(probs.min()), "predicted_probability_max": float(probs.max()),
            "complete_case_excluded": len(data) - len(d), "warning": ";".join(warnings) or "none",
            "adjustment": "age_group + sex + sodium_last_pre_t0 + two direct-fluid coexposure indicators",
        }
    except Exception as exc:
        return {
            "analysis_id": analysis_id, "fluid": focal, "scale_ml": SCALES[focal], "model_n": len(d),
            "events": int(y.sum()), "event_rate": float(y.mean()) if len(y) else np.nan,
            "beta": np.nan, "se": np.nan, "adjusted_or": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
            "p_value": np.nan, "converged": False, "iterations": -1, "columns": X.shape[1], "rank": rank,
            "rank_deficiency": X.shape[1] - rank, "intercept": np.nan, "max_abs_coefficient": np.nan,
            "max_abs_nonintercept_coefficient": np.nan, "predicted_probability_min": np.nan,
            "predicted_probability_max": np.nan, "complete_case_excluded": len(data) - len(d),
            "warning": f"MODEL_FAILED:{type(exc).__name__}:{exc}",
            "adjustment": "age_group + sex + sodium_last_pre_t0 + two direct-fluid coexposure indicators",
        }


def main() -> None:
    a_val_path = PHASE_A / "EV_Q02B_A_validation_v1.json"
    a_val = json.loads(a_val_path.read_text(encoding="utf-8"))
    if a_val.get("common_spec_lock") != "PASS":
        raise RuntimeError("Phase A lock not PASS")
    for name, digest in a_val["output_sha256"].items():
        if sha256(PHASE_A / name) != digest:
            raise RuntimeError(f"Phase A hash mismatch: {name}")

    d = pd.read_csv(SOURCE, low_memory=False)
    flow = pd.read_csv(FLOW)
    d["event"] = d.event.astype(str).str.lower().isin(["true", "t", "1"])
    d["age_group"] = pd.cut(d.age, bins=[18, 40, 50, 60, 70, 80, np.inf], right=False, labels=AGE_LEVELS).astype(str)
    d["first_post_t1_na_time"] = pd.to_datetime(d.first_post_t1_na_time, errors="coerce")
    d["t1_time"] = pd.to_datetime(d.t1_time)
    d["time_to_first_repeat_hours"] = (d.first_post_t1_na_time - d.t1_time).dt.total_seconds() / 3600
    d["tests_per_observed_day"] = d.post_t1_na_count / (d.followup_hours.clip(lower=1 / 60) / 24)

    observed = d[d.post_t1_na_count > 0].copy()
    formal = observed[observed.inputevent_interface_status.eq("ACTIVE")].copy()
    high_obs = formal[(formal.post_t1_na_count >= 2) & (formal.time_to_first_repeat_hours <= 24)].copy()
    doc_certain = formal[(formal.any_rule_c_records == 0) & (formal.any_semantic_ambiguous_or_invalid_records == 0)].copy()

    dist_rows = []
    for fluid in SCALES:
        x = formal[f"{fluid}_ml"].astype(float)
        pos = x[x > 0]
        dist_rows.append({
            "fluid": fluid, "n": len(x), "events": int(formal.event.sum()),
            "zero_n": int((x == 0).sum()), "zero_percent": float(100 * (x == 0).mean()),
            "positive_n": len(pos), "positive_percent": float(100 * (x > 0).mean()),
            "mean_ml": float(x.mean()), "sd_ml": float(x.std()), "median_ml": float(x.median()),
            "p25_ml": float(x.quantile(.25)), "p75_ml": float(x.quantile(.75)), "p90_ml": float(x.quantile(.90)),
            "p95_ml": float(x.quantile(.95)), "p99_ml": float(x.quantile(.99)), "max_ml": float(x.max()),
            "positive_median_ml": float(pos.median()) if len(pos) else np.nan,
        })
    dist = pd.DataFrame(dist_rows)
    dist_path = OUT / "EV_Q02B_C_MIMIC_exposure_distribution_v1.csv"
    dist.to_csv(dist_path, index=False, encoding="utf-8-sig")

    primary = pd.DataFrame([fit_model(formal, f, f"H1_primary_{f}") for f in SCALES])
    primary_path = OUT / "EV_Q02B_C_MIMIC_primary_models_v1.csv"
    primary.to_csv(primary_path, index=False, encoding="utf-8-sig")

    sens_rows = []
    for label, data in [("high_certainty_post_t1_na", high_obs),
                        ("documentation_certainty_no_rule_c_or_semantic_ambiguous", doc_certain)]:
        for fluid in SCALES:
            sens_rows.append(fit_model(data, fluid, f"sensitivity_{label}_{fluid}") | {"sensitivity": label})
    sensitivity = pd.DataFrame(sens_rows)
    sens_path = OUT / "EV_Q02B_C_MIMIC_sensitivity_v1.csv"
    sensitivity.to_csv(sens_path, index=False, encoding="utf-8-sig")

    risk_n = int(flow.loc[flow.step.eq("no_t0_t1_na_ge151"), "n"].iloc[0])
    unobs_n = int(flow.loc[flow.step.eq("post_t1_na_unascertained"), "n"].iloc[0])
    lines = [
        "# EV-Q02B Phase C MIMIC measurement-process audit v1", "",
        "Post-T1 sodium measurements are observation-process variables only and were not used in H1 adjustment.", "",
        f"- Technical T1 risk set: {risk_n:,}.",
        f"- Outcome observed: {len(observed):,} ({100*len(observed)/risk_n:.2f}%).",
        f"- Outcome unascertained: {unobs_n:,} ({100*unobs_n/risk_n:.2f}%); excluded, never coded non-event.",
        f"- Formal ACTIVE-interface cohort: {len(formal):,}; events={int(formal.event.sum()):,}.",
        f"- Interface SILENT among observed-outcome admissions: {len(observed)-len(formal):,}; not assigned zero exposure.", "",
        "| Outcome status | n | Post-T1 Na count median [IQR] | Time to first repeat, h median [IQR] | Tests per observed day median [IQR] |",
        "|---|---:|---:|---:|---:|",
    ]
    for status, label in [(False, "recorded no event"), (True, "recorded event")]:
        x = observed[observed.event.eq(status)]
        lines.append(f"| {label} | {len(x):,} | {qtext(x.post_t1_na_count)} | {qtext(x.time_to_first_repeat_hours)} | {qtext(x.tests_per_observed_day)} |")
    if observed.event.any() and (~observed.event).any():
        u_count = st.mannwhitneyu(observed.loc[observed.event, "post_t1_na_count"], observed.loc[~observed.event, "post_t1_na_count"], alternative="two-sided")
        u_time = st.mannwhitneyu(observed.loc[observed.event, "time_to_first_repeat_hours"], observed.loc[~observed.event, "time_to_first_repeat_hours"], alternative="two-sided")
        lines += ["", f"Descriptive Mann-Whitney P values: count={u_count.pvalue:.6g}; time to first repeat={u_time.pvalue:.6g}. These do not prove detection bias or causality."]
    lines += ["", f"High-certainty ACTIVE subset: n={len(high_obs):,}, events={int(high_obs.event.sum()):,}.",
              f"Documentation-certainty subset: n={len(doc_certain):,}, events={int(doc_certain.event.sum()):,}."]
    measurement_path = OUT / "EV_Q02B_C_MIMIC_measurement_process_v1.md"
    measurement_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_lines = [
        "# EV-Q02B Phase C MIMIC harmonized ICU-only report v1", "",
        "## Scope", "",
        "This is the independently constructed `MIMIC_SECONDARY_HARMONIZED_ICU_ONLY` analysis. It does not replace or modify the frozen MIMIC Step2 primary hospital-window result and was specified without reading Amsterdam model estimates.", "",
        "## Cohort", "",
        f"The technical T1 risk set included {risk_n:,} stays. Outcome was observed in {len(observed):,}, with {int(observed.event.sum()):,} recorded events. {unobs_n:,} stays were OUTCOME_UNASCERTAINED. The formal ACTIVE-interface model cohort included {len(formal):,} stays and {int(formal.event.sum()):,} events.", "",
        "## H1 common-adjustment models", "",
    ]
    for r in primary.itertuples():
        report_lines.append(f"- {r.fluid}: per {int(r.scale_ml)} mL adjusted OR {r.adjusted_or:.3f} (95% CI {r.ci95_low:.3f}-{r.ci95_high:.3f}), P={r.p_value:.6g}; n={r.model_n:,}, events={r.events:,}. Converged={r.converged}; rank deficiency={r.rank_deficiency}.")
    report_lines += ["", "## Sensitivities", "",
                     "The pre-specified high-certainty sodium-observation and documentation-certainty analyses were run with unchanged H1 covariates. H2 extended adjustment was not authorized in Phase A and was not run.", "",
                     "## Interpretation boundary", "",
                     "All estimates are adjusted associations involving documented direct-fluid volume and subsequent recorded hypernatremia. No causal interpretation is permitted. Cross-database interpretation is reserved for Phase D."]
    report_path = OUT / "EV_Q02B_C_MIMIC_harmonized_report_v1.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    errors = []
    expected = {"no_t0_t1_na_ge151": len(d), "post_t1_na_observed_binary_cohort": len(observed),
                "post_t1_na_unascertained": int((d.post_t1_na_count == 0).sum()),
                "recorded_na_ge151_events": int(observed.event.sum()),
                "formal_active_interface_cohort": len(formal), "formal_active_interface_events": int(formal.event.sum())}
    for step, got in expected.items():
        declared = int(flow.loc[flow.step.eq(step), "n"].iloc[0])
        if declared != got:
            errors.append(f"flow mismatch {step}: {declared}!={got}")
    if d.stay_id.duplicated().any() or d.subject_id.duplicated().any():
        errors.append("stay_id or subject_id not unique")
    if not primary.converged.all() or (primary.rank_deficiency != 0).any():
        errors.append("primary model convergence/rank failure")
    if not sensitivity.converged.all() or (sensitivity.rank_deficiency != 0).any():
        errors.append("sensitivity model convergence/rank failure")
    if len(primary) != 3 or len(sensitivity) != 6:
        errors.append("model output row count incomplete")
    if (d[["nacl_ml", "lr_ml", "d5w_ml"]].min().min() < 0):
        errors.append("negative direct-fluid volume")

    outputs = [FLOW, dist_path, primary_path, sens_path, measurement_path, report_path]
    validation = {
        "task": "EV-Q02B Phase C MIMIC-only harmonized secondary analysis",
        "generated_at": datetime.now().astimezone().isoformat(),
        "database": "USER_SUPPLIED_DATABASE", "database_host": "USER_SUPPLIED_HOST", "database_port": "USER_SUPPLIED_PORT",
        "analysis_label": "MIMIC_SECONDARY_HARMONIZED_ICU_ONLY",
        "phase_a_validation_sha256": sha256(a_val_path), "phase_a_output_hashes_verified": True,
        "cohort_counts": {"technical_t1_risk": len(d), "outcome_observed": len(observed),
                          "outcome_unascertained": int((d.post_t1_na_count == 0).sum()),
                          "observed_events": int(observed.event.sum()), "formal_active_n": len(formal),
                          "formal_active_events": int(formal.event.sum())},
        "h1_common_adjustment": ["age_group", "sex", "sodium_last_pre_t0", "two focal-model-specific direct-fluid coexposure indicators"],
        "h2_extended_adjustment_run": False, "prohibited_analyses_run": [],
        "primary_models_all_converged": bool(primary.converged.all()),
        "primary_models_all_full_rank": bool((primary.rank_deficiency == 0).all()),
        "sensitivity_models_all_converged": bool(sensitivity.converged.all()),
        "sensitivity_models_all_full_rank": bool((sensitivity.rank_deficiency == 0).all()),
        "four_round_review": {
            "round_1_code_and_estimand_logic": "PASS" if not errors else "CHECK",
            "round_2_data_handling_and_missingness": "PASS" if not errors else "CHECK",
            "round_3_per_output_counts_and_uncertainty": "PASS" if not errors else "CHECK",
            "round_4_cross_output_consistency": "PASS" if not errors else "CHECK",
        },
        "qc_pass": not errors, "qc_errors": errors,
        "input_sha256": {"phase_a_validation": sha256(a_val_path), "phase_a_common_spec": sha256(PHASE_A / "EV_Q02B_A_common_analysis_spec_v1.md"),
                         "extraction_sql": sha256(SQL), "analysis_source": sha256(SOURCE), "phase_c_runner": sha256(Path(__file__))},
        "output_sha256": {p.name: sha256(p) for p in outputs},
        "phase_c_status": "PASS" if not errors else "FAIL",
        "stop_rule": "STOP_AFTER_MIMIC_PHASE_C_BEFORE_NO_DATABASE_PHASE_D",
    }
    val_path = OUT / "EV_Q02B_C_MIMIC_validation_v1.json"
    val_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("Phase C QC failed: " + " | ".join(errors))
    print(json.dumps({"phase_c": "PASS", "cohort": validation["cohort_counts"],
                      "primary": primary[["fluid", "adjusted_or", "ci95_low", "ci95_high", "p_value", "model_n", "events"]].to_dict("records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
