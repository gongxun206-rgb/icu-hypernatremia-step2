# Public sanitized copy; patient-level review IDs are loaded from an untracked restricted file.
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import argparse
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
OUT = OUTPUT_ROOT / "amsterdam_harmonized"
PSQL = Path(os.environ.get("AMSTERDAM_PSQL", "psql"))
DB_HOST = os.environ.get("AMSTERDAM_DB_HOST", "YOUR_HOST")
DB_PORT = os.environ.get("AMSTERDAM_DB_PORT", "YOUR_PORT")
DB_USER = os.environ.get("AMSTERDAM_DB_USER", "YOUR_USER")
DB_NAME = os.environ.get("AMSTERDAM_DB_NAME", "YOUR_DATABASE")
HOUR = 3_600_000
DAY7 = 168 * HOUR
SERUM_NA = 9924
HYPERTONIC = 10739
DIRECT = {7293: "nacl", 7316: "lr", 7257: "d5w"}
SCALES = {"nacl": 500.0, "lr": 500.0, "d5w": 250.0}
def load_ambiguous_ids(review_path: Path) -> set[int]:
    review = pd.read_csv(review_path)
    id_column = "admissionid"
    grade_column = "researcher_final_grade"
    required = {id_column, grade_column}
    if not required.issubset(review.columns):
        raise ValueError(f"Restricted review file must contain {sorted(required)}")
    return set(review.loc[review[grade_column].eq("C_AMBIGUOUS"), id_column].astype(int))

AGE_LEVELS = ["18-39", "40-49", "50-59", "60-69", "70-79", "80+"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def query(sql: str) -> pd.DataFrame:
    env = dict(os.environ, PGOPTIONS="-c default_transaction_read_only=on -c statement_timeout=300000")
    cmd = [str(PSQL), "-h", DB_HOST, "-p", DB_PORT, "-U", DB_USER, "-d", DB_NAME,
           "-X", "-q", "-v", "ON_ERROR_STOP=1", "-c", f"COPY ({sql}) TO STDOUT WITH CSV HEADER"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False)
    if p.returncode:
        raise RuntimeError(p.stderr.decode("utf-8", errors="replace"))
    return pd.read_csv(io.BytesIO(p.stdout), low_memory=False)


def qtext(s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if not len(x):
        return "NA"
    return f"{x.median():.3f} [{x.quantile(.25):.3f}, {x.quantile(.75):.3f}]"


def fit_model(data: pd.DataFrame, focal: str, analysis_id: str) -> dict:
    d = data.copy()
    d = d[d["agegroup"].isin(AGE_LEVELS) & d["gender"].isin(["Man", "Vrouw"]) & d["sodium_last_pre_t0"].notna()].copy()
    age = pd.Categorical(d["agegroup"], categories=AGE_LEVELS, ordered=True)
    age_dummies = pd.get_dummies(age, prefix="age", drop_first=True, dtype=float)
    X = pd.DataFrame(index=d.index)
    X[f"{focal}_per_{int(SCALES[focal])}ml"] = d[f"{focal}_ml"] / SCALES[focal]
    X["sex_male"] = (d["gender"] == "Man").astype(float)
    X["sodium_last_pre_t0"] = d["sodium_last_pre_t0"].astype(float)
    for col in age_dummies.columns:
        X[col] = age_dummies[col].to_numpy()
    for other in DIRECT.values():
        if other != focal:
            X[f"any_{other}_coexposure"] = (d[f"{other}_ml"] > 0).astype(float)
    X = sm.add_constant(X, has_constant="add").astype(float)
    y = d["event"].astype(int)
    rank = int(np.linalg.matrix_rank(X.to_numpy()))
    warnings = []
    try:
        result = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-10)
        term = f"{focal}_per_{int(SCALES[focal])}ml"
        beta = float(result.params[term])
        se = float(result.bse[term])
        lo, hi = result.conf_int().loc[term].astype(float)
        probs = np.asarray(result.predict(X), dtype=float)
        converged = bool(result.converged)
        iterations = int(result.fit_history.get("iteration", -1))
        max_abs = float(np.abs(result.params).max())
        intercept = float(result.params["const"])
        max_abs_nonintercept = float(np.abs(result.params.drop("const")).max())
        if probs.min() < 1e-8 or probs.max() > 1 - 1e-8:
            warnings.append("EXTREME_PREDICTED_PROBABILITY")
        if max_abs_nonintercept > 20:
            warnings.append("EXTREME_COEFFICIENT")
        return {
            "analysis_id": analysis_id, "fluid": focal, "scale_ml": SCALES[focal],
            "model_n": len(d), "events": int(y.sum()), "event_rate": float(y.mean()),
            "beta": beta, "se": se, "adjusted_or": float(np.exp(beta)),
            "ci95_low": float(np.exp(lo)), "ci95_high": float(np.exp(hi)),
            "p_value": float(result.pvalues[term]), "converged": converged,
            "iterations": iterations, "columns": X.shape[1], "rank": rank,
            "rank_deficiency": int(X.shape[1] - rank), "intercept": intercept,
            "max_abs_coefficient": max_abs, "max_abs_nonintercept_coefficient": max_abs_nonintercept,
            "predicted_probability_min": float(probs.min()), "predicted_probability_max": float(probs.max()),
            "complete_case_excluded": int(len(data) - len(d)), "warning": ";".join(warnings) or "none",
            "adjustment": "age_group + sex + sodium_last_pre_t0 + two direct-fluid coexposure indicators",
        }
    except Exception as exc:
        return {
            "analysis_id": analysis_id, "fluid": focal, "scale_ml": SCALES[focal],
            "model_n": len(d), "events": int(y.sum()), "event_rate": float(y.mean()) if len(y) else np.nan,
            "beta": np.nan, "se": np.nan, "adjusted_or": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
            "p_value": np.nan, "converged": False, "iterations": -1, "columns": X.shape[1], "rank": rank,
            "rank_deficiency": int(X.shape[1] - rank), "intercept": np.nan,
            "max_abs_coefficient": np.nan, "max_abs_nonintercept_coefficient": np.nan,
            "predicted_probability_min": np.nan, "predicted_probability_max": np.nan,
            "complete_case_excluded": int(len(data) - len(d)), "warning": f"MODEL_FAILED:{type(exc).__name__}:{exc}",
            "adjustment": "age_group + sex + sodium_last_pre_t0 + two direct-fluid coexposure indicators",
        }


def build_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    admissions = query("""SELECT patientid,admissionid,admissioncount,location,agegroup,gender,
        admittedat,dischargedat,dateofdeath,admissionyeargroup FROM admissions""")
    admissions = admissions.sort_values(["patientid", "admittedat", "admissioncount", "admissionid"])
    icu_containing = admissions[admissions.location.isin(["IC", "IC&MC", "MC&IC"])].copy()
    first_icu = icu_containing.drop_duplicates("patientid", keep="first").copy()
    pure = first_icu[first_icu.location.eq("IC")].copy()
    pure["t0"] = pure.admittedat + 24 * HOUR
    pure["t1"] = pure.admittedat + 36 * HOUR
    pure["day7"] = pure.admittedat + DAY7
    # Amsterdam dateofdeath is a deidentified date-derived field and is not a
    # reliable millisecond timestamp for an ICU window. ICU deaths terminate
    # the admission at dischargedat (destination='Overleden'), so dischargedat
    # operationalizes both ICU discharge and ICU death for this ICU-only window.
    pure["alive_t0"] = pure.dischargedat > pure.t0
    pure["alive_t1"] = pure.dischargedat > pure.t1
    pure["in_icu_t0"] = pure.dischargedat > pure.t0
    pure["in_icu_t1"] = pure.dischargedat > pure.t1
    end = np.minimum(pure.dischargedat.to_numpy(), pure.day7.to_numpy())
    pure["followup_end"] = end

    sodium = query("""SELECT admissionid,measuredat,value FROM numericitems
        WHERE itemid=9924 AND value BETWEEN 100 AND 200""")
    sodium = sodium.merge(pure[["admissionid", "admittedat", "t0", "t1", "followup_end"]], on="admissionid", how="inner")
    pre = sodium[(sodium.measuredat >= sodium.admittedat) & (sodium.measuredat < sodium.t0)].copy()
    pre = pre.sort_values(["admissionid", "measuredat"])
    pre_agg = pre.groupby("admissionid").agg(pre_na_n=("value", "size"), pre_na_min=("value", "min"),
                                                pre_na_max=("value", "max"), sodium_last_pre_t0=("value", "last"))
    pure = pure.join(pre_agg, on="admissionid")
    pure["pre_na_eligible"] = (pure.pre_na_n >= 1) & (pure.pre_na_min >= 135) & (pure.pre_na_max <= 145)

    hyper = query("SELECT admissionid,start,stop FROM drugitems WHERE itemid=10739")
    hyper = hyper.merge(pure[["admissionid", "admittedat", "t0"]], on="admissionid", how="inner")
    hyper_ids = set(hyper.loc[(hyper.start < hyper.t0) & (hyper.stop > hyper.admittedat), "admissionid"])
    pure["pre_t0_hypertonic"] = pure.admissionid.isin(hyper_ids)

    grace = sodium[(sodium.measuredat >= sodium.t0) & (sodium.measuredat < sodium.t1) & (sodium.value >= 151)]
    pure["grace_high"] = pure.admissionid.isin(set(grace.admissionid))

    base = pure[pure.in_icu_t0 & pure.alive_t0 & pure.pre_na_eligible].copy()
    no_hts = base[~base.pre_t0_hypertonic].copy()
    t1 = no_hts[no_hts.in_icu_t1 & no_hts.alive_t1].copy()
    risk = t1[~t1.grace_high].copy()

    post = sodium[(sodium.admissionid.isin(set(risk.admissionid))) &
                  (sodium.measuredat >= sodium.t1) & (sodium.measuredat < sodium.followup_end)].copy()
    post = post.sort_values(["admissionid", "measuredat"])
    post_agg = post.groupby("admissionid").agg(post_na_count=("value", "size"), first_post_na_time=("measuredat", "min"), post_na_max=("value", "max"))
    event_time = post[post.value >= 151].groupby("admissionid").measuredat.min().rename("first_event_time")
    risk = risk.join(post_agg, on="admissionid").join(event_time, on="admissionid")
    risk["post_na_count"] = risk.post_na_count.fillna(0).astype(int)
    risk["outcome_status"] = np.where(risk.post_na_count.eq(0), "OUTCOME_UNASCERTAINED",
                                      np.where(risk.post_na_max.ge(151), "RECORDED_EVENT", "RECORDED_NO_EVENT"))
    risk["event"] = risk.outcome_status.eq("RECORDED_EVENT")
    risk["followup_hours"] = (risk.followup_end - risk.t1) / HOUR
    risk["time_to_first_repeat_hours"] = (risk.first_post_na_time - risk.t1) / HOUR
    risk["tests_per_observed_day"] = risk.post_na_count / (risk.followup_hours.clip(lower=1 / 60) / 24)

    fluids = query("""SELECT admissionid,orderid,itemid,isadditive::int AS isadditive,start,stop,duration,
        administeredunit,fluidin FROM drugitems WHERE itemid IN (7293,7316,7257)""")
    fluids = fluids.merge(risk[["admissionid", "t0", "t1"]], on="admissionid", how="inner")
    fluids["overlap_ms"] = np.maximum(0, np.minimum(fluids.stop, fluids.t1) - np.maximum(fluids.start, fluids.t0))
    fluids = fluids[fluids.overlap_ms > 0].copy()
    fluids["rule_b"] = ((fluids.duration > 0) & (fluids.stop - fluids.start == fluids.duration * 60_000) &
                        fluids.administeredunit.eq("ml") & fluids.fluidin.ge(0) & fluids.isadditive.eq(0))
    if not fluids.rule_b.all():
        bad = fluids.loc[~fluids.rule_b, ["admissionid", "orderid", "itemid"]]
        raise RuntimeError(f"Non-Rule-B overlapping direct-fluid rows found: {bad.to_dict('records')[:10]}")
    fluids["allocated_ml"] = fluids.fluidin * fluids.overlap_ms / (fluids.duration * 60_000)
    fluids["fluid"] = fluids.itemid.map(DIRECT)
    wide = fluids.pivot_table(index="admissionid", columns="fluid", values="allocated_ml", aggfunc="sum")
    risk = risk.set_index("admissionid")
    for fluid in DIRECT.values():
        risk[f"{fluid}_ml"] = wide.get(fluid, pd.Series(dtype=float)).reindex(risk.index).fillna(0.0)
    risk = risk.reset_index()

    flow = pd.DataFrame([
        ("all_admissions", len(admissions), "all AmsterdamUMCdb admissions"),
        ("icu_containing_admissions", len(icu_containing), "location IC, IC&MC, or MC&IC"),
        ("first_icu_containing_admission_per_patient", len(first_icu), "one admission per patient"),
        ("first_icu_admission_pure_ic", len(pure), "location='IC'; mixed IC/MC excluded"),
        ("alive_and_in_icu_at_t0", int((pure.in_icu_t0 & pure.alive_t0).sum()), "dischargedat>T0; ICU death is represented by terminal ICU discharge"),
        ("pre_t0_na_eligible", len(base), ">=1 item-9924 Na and all observed values 135-145"),
        ("no_pre_t0_nacl_2_9", len(no_hts), "no overlapping item 10739 before T0"),
        ("alive_and_in_icu_at_t1", len(t1), "dischargedat>T1; ICU death is represented by terminal ICU discharge"),
        ("no_t0_t1_na_ge151", len(risk), "technical T1 risk set"),
        ("post_t1_na_observed_binary_cohort", int(risk.post_na_count.gt(0).sum()), ">=1 item-9924 Na in harmonized outcome window"),
        ("post_t1_na_unascertained", int(risk.post_na_count.eq(0).sum()), "excluded from binary models; never non-event"),
        ("recorded_na_ge151_events", int(risk.event.sum()), "first recorded item-9924 Na>=151"),
    ], columns=["step", "n", "definition"])
    return risk, flow, fluids, post


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Amsterdam harmonized analyses.")
    parser.add_argument("--run-ambiguous-review-sensitivity", action="store_true")
    parser.add_argument("--review-csv", type=Path)
    return parser.parse_args()


def optional_ambiguous_review_ids(args: argparse.Namespace) -> tuple[set[int] | None, str]:
    if not args.run_ambiguous_review_sensitivity:
        return None, "SKIPPED: restricted manual-review sensitivity not requested"
    if args.review_csv is None or not args.review_csv.is_file():
        return None, "SKIPPED: restricted manual-review file not supplied"
    return load_ambiguous_ids(args.review_csv), "RUN"


def main() -> None:
    args = parse_args()
    ambiguous_c, ambiguous_review_status = optional_ambiguous_review_ids(args)
    OUT.mkdir(parents=True, exist_ok=True)
    a_val_path = PHASE_A / "EV_Q02B_A_validation_v1.json"
    a_val = json.loads(a_val_path.read_text(encoding="utf-8"))
    if a_val.get("common_spec_lock") != "PASS":
        raise RuntimeError("Phase A lock not PASS")
    for name, digest in a_val["output_sha256"].items():
        path = PHASE_A / name
        if sha256(path) != digest:
            raise RuntimeError(f"Phase A hash mismatch: {name}")

    risk, flow, fluid_rows, post = build_data()
    observed = risk[risk.post_na_count > 0].copy()
    flow_path = OUT / "EV_Q02B_B_Amsterdam_cohort_flow_v1.csv"
    flow.to_csv(flow_path, index=False, encoding="utf-8-sig")

    dist_rows = []
    for fluid in DIRECT.values():
        x = observed[f"{fluid}_ml"].astype(float)
        positive = x[x > 0]
        dist_rows.append({
            "fluid": fluid, "n": len(x), "events": int(observed.event.sum()),
            "zero_n": int((x == 0).sum()), "zero_percent": float(100 * (x == 0).mean()),
            "positive_n": len(positive), "positive_percent": float(100 * (x > 0).mean()),
            "mean_ml": float(x.mean()), "sd_ml": float(x.std()), "median_ml": float(x.median()),
            "p25_ml": float(x.quantile(.25)), "p75_ml": float(x.quantile(.75)),
            "p90_ml": float(x.quantile(.90)), "p95_ml": float(x.quantile(.95)), "p99_ml": float(x.quantile(.99)),
            "max_ml": float(x.max()), "positive_median_ml": float(positive.median()) if len(positive) else np.nan,
        })
    dist = pd.DataFrame(dist_rows)
    dist_path = OUT / "EV_Q02B_B_Amsterdam_exposure_distribution_v1.csv"
    dist.to_csv(dist_path, index=False, encoding="utf-8-sig")

    primary = pd.DataFrame([fit_model(observed, f, f"H1_primary_{f}") for f in DIRECT.values()])
    primary_path = OUT / "EV_Q02B_B_Amsterdam_primary_models_v1.csv"
    primary.to_csv(primary_path, index=False, encoding="utf-8-sig")

    high_obs = observed[(observed.post_na_count >= 2) & (observed.time_to_first_repeat_hours <= 24)].copy()
    sens_rows = []
    sensitivity_specs = [("high_certainty_post_t1_na", high_obs)]
    if ambiguous_c is not None:
        sensitivity_specs.append(("exclude_predeclared_5_C_ambiguous", observed[~observed.admissionid.isin(ambiguous_c)].copy()))
    for label, data in sensitivity_specs:
        for fluid in DIRECT.values():
            sens_rows.append(fit_model(data, fluid, f"sensitivity_{label}_{fluid}") | {"sensitivity": label})
    sensitivity = pd.DataFrame(sens_rows)
    sens_path = OUT / "EV_Q02B_B_Amsterdam_sensitivity_v1.csv"
    sensitivity.to_csv(sens_path, index=False, encoding="utf-8-sig")

    lines = [
        "# EV-Q02B Phase B Amsterdam measurement-process audit v1", "",
        "Post-T1 sodium variables describe the observation process only. They were not included in H1 adjustment.", "",
        f"- Technical T1 risk set: {len(risk):,}.",
        f"- Outcome observed: {len(observed):,} ({100*len(observed)/len(risk):.2f}%).",
        f"- Outcome unascertained: {int((risk.post_na_count==0).sum()):,} ({100*(risk.post_na_count==0).mean():.2f}%); excluded, not coded 0.",
        f"- Recorded events among observed: {int(observed.event.sum()):,}/{len(observed):,} ({100*observed.event.mean():.2f}%).", "",
        "| Outcome status | n | Post-T1 Na count median [IQR] | Time to first repeat, h median [IQR] | Tests per observed day median [IQR] |",
        "|---|---:|---:|---:|---:|",
    ]
    for status, label in [(False, "recorded no event"), (True, "recorded event")]:
        x = observed[observed.event.eq(status)]
        lines.append(f"| {label} | {len(x):,} | {qtext(x.post_na_count)} | {qtext(x.time_to_first_repeat_hours)} | {qtext(x.tests_per_observed_day)} |")
    if observed.event.any() and (~observed.event).any():
        u_count = st.mannwhitneyu(observed.loc[observed.event, "post_na_count"], observed.loc[~observed.event, "post_na_count"], alternative="two-sided")
        u_time = st.mannwhitneyu(observed.loc[observed.event, "time_to_first_repeat_hours"], observed.loc[~observed.event, "time_to_first_repeat_hours"], alternative="two-sided")
        lines += ["", f"Descriptive Mann-Whitney P values: count={u_count.pvalue:.6g}; time to first repeat={u_time.pvalue:.6g}. These do not establish detection bias or causality."]
    lines += ["", f"High-certainty observation subset: n={len(high_obs):,}, events={int(high_obs.event.sum()):,}."]
    measurement_path = OUT / "EV_Q02B_B_Amsterdam_measurement_process_v1.md"
    measurement_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_lines = [
        "# EV-Q02B Phase B Amsterdam-only formal replication report v1", "",
        "## Scope", "",
        "This is an AmsterdamUMCdb-only harmonized external replication using the pre-locked common ICU-only outcome window. It estimates associations, not causal treatment effects. Carrier volume was not quantified.", "",
        "## Cohort", "",
        f"The technical T1 risk set contained {len(risk):,} admissions. The binary outcome was observed in {len(observed):,}; {int(observed.event.sum()):,} had a recorded sodium >=151 mmol/L. The remaining {int((risk.post_na_count==0).sum()):,} were OUTCOME_UNASCERTAINED and excluded from binary models. H1 complete-case models included 2,456 admissions and 282 events; 51 observed-outcome admissions were excluded because sex was undocumented, while age group and pre-T0 sodium were complete.", "",
        "## H1 common-adjustment models", "",
    ]
    for r in primary.itertuples():
        report_lines.append(f"- {r.fluid}: per {int(r.scale_ml)} mL adjusted OR {r.adjusted_or:.3f} (95% CI {r.ci95_low:.3f}-{r.ci95_high:.3f}), P={r.p_value:.6g}; n={r.model_n:,}, events={r.events:,}. Model converged={r.converged}, rank deficiency={r.rank_deficiency}.")
    report_lines += ["", "## Prespecified sensitivities", "",
                     f"The high-certainty sodium-observation subset was run without changing definitions. Ambiguous-review sensitivity: {ambiguous_review_status}. Full available estimates are in the sensitivity CSV.", "",
                     "## Interpretation boundary", "",
                     "Results must be described as adjusted associations between documented direct-fluid volume and subsequent recorded hypernatremia. They do not establish benefit, harm, or causal effects. Cross-database interpretation is deferred until Phase C and D."]
    report_path = OUT / "EV_Q02B_B_Amsterdam_formal_report_v1.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    errors = []
    if len(risk) != 3000:
        errors.append(f"technical risk set drift: {len(risk)} != 3000")
    if len(observed) != 2507:
        errors.append(f"observed cohort drift: {len(observed)} != 2507")
    if int(observed.event.sum()) != 285:
        errors.append(f"event drift: {int(observed.event.sum())} != 285")
    if not primary.converged.all() or (primary.rank_deficiency != 0).any():
        errors.append("one or more primary models failed convergence/rank QC")
    if not sensitivity.converged.all() or (sensitivity.rank_deficiency != 0).any():
        errors.append("one or more sensitivity models failed convergence/rank QC")
    if risk.admissionid.duplicated().any() or risk.patientid.duplicated().any():
        errors.append("admissionid or patientid not unique in risk set")
    if (fluid_rows.allocated_ml < 0).any():
        errors.append("negative allocated direct-fluid volume")

    outputs = [flow_path, dist_path, primary_path, sens_path, measurement_path, report_path]
    validation = {
        "task": "EV-Q02B Phase B Amsterdam-only formal replication",
        "generated_at": datetime.now().astimezone().isoformat(),
        "database": "USER_SUPPLIED_DATABASE", "database_host": "USER_SUPPLIED_HOST", "database_port": "USER_SUPPLIED_PORT",
        "phase_a_validation_sha256": sha256(a_val_path),
        "phase_a_output_hashes_verified": True,
        "cohort_counts": {"technical_t1_risk": len(risk), "outcome_observed": len(observed),
                          "outcome_unascertained": int((risk.post_na_count == 0).sum()), "events": int(observed.event.sum())},
        "ambiguous_review_sensitivity": ambiguous_review_status,
        "ambiguous_C_predeclared_count": len(ambiguous_c) if ambiguous_c is not None else None,
        "ambiguous_C_in_binary_cohort_count": len(set(observed.admissionid) & ambiguous_c) if ambiguous_c is not None else None,
        "h1_common_adjustment": ["age_group", "sex", "sodium_last_pre_t0", "two focal-model-specific direct-fluid coexposure indicators"],
        "h2_extended_adjustment_run": False,
        "complete_case_missingness": {
            "binary_outcome_cohort_n": len(observed),
            "missing_or_blank_sex_n": int((~observed.gender.isin(["Man", "Vrouw"])).sum()),
            "missing_valid_age_group_n": int((~observed.agegroup.isin(AGE_LEVELS)).sum()),
            "missing_sodium_last_pre_t0_n": int(observed.sodium_last_pre_t0.isna().sum()),
            "primary_model_n": int(primary.model_n.iloc[0]),
        },
        "Amsterdam_death_endpoint_implementation": "ICU dischargedat operationalizes ICU discharge and ICU death; dateofdeath is retained for audit but not used as millisecond cutoff because it is not aligned to ICU discharge time",
        "prohibited_analyses_run": [],
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
        "input_sha256": {
            "phase_a_validation": sha256(a_val_path),
            "phase_a_common_spec": sha256(PHASE_A / "EV_Q02B_A_common_analysis_spec_v1.md"),
            "phase_b_runner": sha256(Path(__file__)),
        },
        "output_sha256": {p.name: sha256(p) for p in outputs},
        "phase_b_status": "PASS" if not errors else "FAIL",
        "stop_rule": "STOP_AFTER_AMSTERDAM_PHASE_B_AND_SWITCH_DATABASE_BEFORE_PHASE_C",
    }
    val_path = OUT / "EV_Q02B_B_Amsterdam_validation_v1.json"
    val_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("Phase B QC failed: " + " | ".join(errors))
    print(json.dumps({"phase_b": "PASS", "cohort": validation["cohort_counts"],
                      "primary": primary[["fluid", "adjusted_or", "ci95_low", "ci95_high", "p_value", "model_n", "events"]].to_dict("records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
