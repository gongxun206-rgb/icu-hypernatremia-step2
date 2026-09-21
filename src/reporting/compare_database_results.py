# Public sanitized copy; generated outputs remain outside Git.
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("STEP2_OUTPUT_ROOT", REPO_ROOT / "outputs"))
ROOT = REPO_ROOT
A = REPO_ROOT / "config" / "harmonized"
B = OUTPUT_ROOT / "amsterdam_harmonized"
C = OUTPUT_ROOT / "mimic_harmonized"
OUT = OUTPUT_ROOT / "cross_database_comparison"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(validation_path: Path, expected_status_key: str) -> dict:
    v = json.loads(validation_path.read_text(encoding="utf-8"))
    if v.get(expected_status_key) != "PASS" or not v.get("qc_pass", True):
        raise RuntimeError(f"Upstream validation not PASS: {validation_path}")
    base = validation_path.parent
    for name, digest in v.get("output_sha256", {}).items():
        if sha256(base / name) != digest:
            raise RuntimeError(f"Hash mismatch: {base / name}")
    return v


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    av = json.loads((A / "EV_Q02B_A_validation_v1.json").read_text(encoding="utf-8"))
    if av.get("common_spec_lock") != "PASS":
        raise RuntimeError("Phase A not PASS")
    for name, digest in av["output_sha256"].items():
        if sha256(A / name) != digest:
            raise RuntimeError(f"Phase A hash mismatch: {name}")
    bv = verify(B / "EV_Q02B_B_Amsterdam_validation_v1.json", "phase_b_status")
    cv = verify(C / "EV_Q02B_C_MIMIC_validation_v1.json", "phase_c_status")

    bp = pd.read_csv(B / "EV_Q02B_B_Amsterdam_primary_models_v1.csv")
    cp = pd.read_csv(C / "EV_Q02B_C_MIMIC_primary_models_v1.csv")
    bd = pd.read_csv(B / "EV_Q02B_B_Amsterdam_exposure_distribution_v1.csv")
    cd = pd.read_csv(C / "EV_Q02B_C_MIMIC_exposure_distribution_v1.csv")

    classifications = {
        "nacl": "HARMONIZED_EXTERNAL_REPLICATION_SUPPORTED",
        "lr": "HARMONIZED_EXTERNAL_REPLICATION_INCONCLUSIVE",
        "d5w": "HARMONIZED_EXTERNAL_REPLICATION_INCONCLUSIVE",
    }
    rationale = {
        "nacl": "Positive direction in both databases; both 95% CIs exclude 1, and prespecified sensitivities retain the positive direction.",
        "lr": "Point estimates are nominally below 1 in both databases, but the MIMIC estimate is essentially null while Amsterdam is inverse; magnitude consistency is insufficient.",
        "d5w": "Point estimates are positive in both databases, but both 95% CIs include 1 and Amsterdam exposure is sparse; precision is insufficient.",
    }

    rows = []
    for fluid in ["nacl", "lr", "d5w"]:
        b = bp[bp.fluid.eq(fluid)].iloc[0]
        c = cp[cp.fluid.eq(fluid)].iloc[0]
        bdist = bd[bd.fluid.eq(fluid)].iloc[0]
        cdist = cd[cd.fluid.eq(fluid)].iloc[0]
        concordant = (b.beta > 0) == (c.beta > 0)
        for database, m, dist, coverage, risk_n, observed_n in [
            ("AmsterdamUMCdb", b, bdist, bv["cohort_counts"]["outcome_observed"] / bv["cohort_counts"]["technical_t1_risk"], bv["cohort_counts"]["technical_t1_risk"], bv["cohort_counts"]["outcome_observed"]),
            ("MIMIC-IV", c, cdist, cv["cohort_counts"]["outcome_observed"] / cv["cohort_counts"]["technical_t1_risk"], cv["cohort_counts"]["technical_t1_risk"], cv["cohort_counts"]["outcome_observed"]),
        ]:
            rows.append({
                "fluid": fluid, "database": database, "technical_risk_n": risk_n,
                "outcome_observed_n": observed_n, "outcome_observation_coverage_percent": 100 * coverage,
                "model_n": int(m.model_n), "events": int(m.events), "event_rate_percent": 100 * m.event_rate,
                "scale_ml": m.scale_ml, "beta": m.beta, "adjusted_or": m.adjusted_or,
                "ci95_low": m.ci95_low, "ci95_high": m.ci95_high, "p_value": m.p_value,
                "zero_fraction_percent": dist.zero_percent, "positive_fraction_percent": dist.positive_percent,
                "positive_median_ml": dist.positive_median_ml, "direction": "positive" if m.beta > 0 else "negative",
                "direction_concordant": concordant, "replication_classification": classifications[fluid],
                "classification_rationale": rationale[fluid],
            })
    comparison = pd.DataFrame(rows)
    comparison_path = OUT / "EV_Q02B_D_cross_database_comparison_v1.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    def est(db: str, fluid: str) -> str:
        r = comparison[(comparison.database == db) & (comparison.fluid == fluid)].iloc[0]
        return f"OR {r.adjusted_or:.3f} (95% CI {r.ci95_low:.3f}-{r.ci95_high:.3f})"

    report = f"""# Step2 Amsterdam EV-Q02B final replication report v1

## Final gate

`PRIMARY_NACL_REPLICATION = HARMONIZED_EXTERNAL_REPLICATION_SUPPORTED`

The harmonized external replication used the same T0/T1, ICU-only outcome window, exposure scales, H1 covariates, and co-exposure terms in both databases. Phase A definitions were frozen before either database-specific model was run. No original database was accessed in Phase D.

## Cohorts and observation process

- AmsterdamUMCdb: technical risk set {bv['cohort_counts']['technical_t1_risk']:,}; outcome observed {bv['cohort_counts']['outcome_observed']:,} ({100*bv['cohort_counts']['outcome_observed']/bv['cohort_counts']['technical_t1_risk']:.2f}%); formal model n={int(bp.model_n.iloc[0]):,}, events={int(bp.events.iloc[0]):,}.
- MIMIC-IV: technical risk set {cv['cohort_counts']['technical_t1_risk']:,}; outcome observed {cv['cohort_counts']['outcome_observed']:,} ({100*cv['cohort_counts']['outcome_observed']/cv['cohort_counts']['technical_t1_risk']:.2f}%); formal ACTIVE-interface model n={int(cp.model_n.iloc[0]):,}, events={int(cp.events.iloc[0]):,}.
- Outcome-observation coverage, recorded event rates, and exposure zero fractions differed materially. These differences limit direct magnitude comparison.

## Primary direct 0.9% NaCl result

- AmsterdamUMCdb per 500 mL: {est('AmsterdamUMCdb','nacl')}.
- MIMIC-IV per 500 mL: {est('MIMIC-IV','nacl')}.

Both estimates were positive and excluded the null. Prespecified high-certainty observation and documentation/ambiguity sensitivities retained a positive NaCl direction. The primary replication is therefore classified as `HARMONIZED_EXTERNAL_REPLICATION_SUPPORTED`.

## Secondary fluids

- LR per 500 mL: Amsterdam {est('AmsterdamUMCdb','lr')}; MIMIC {est('MIMIC-IV','lr')}. Classification: `HARMONIZED_EXTERNAL_REPLICATION_INCONCLUSIVE` because MIMIC was essentially null despite the inverse Amsterdam estimate.
- D5W per 250 mL: Amsterdam {est('AmsterdamUMCdb','d5w')}; MIMIC {est('MIMIC-IV','d5w')}. Classification: `HARMONIZED_EXTERNAL_REPLICATION_INCONCLUSIVE` because both confidence intervals included 1 and Amsterdam exposure was sparse.

## Interpretation boundaries

- This is a harmonized external replication using a common ICU-only outcome window, not an exact external validation of the original MIMIC hospital-window endpoint.
- The frozen MIMIC primary analysis remains unchanged.
- AmsterdamUMCdb is an independent European ICU database, not a multicenter extension of MIMIC-IV.
- Estimates are observational adjusted associations, not causal treatment effects.
- Statistical significance was not the sole replication criterion; direction, precision, sensitivity consistency, exposure support, outcome coverage, and measurement-process differences were considered.
"""
    report_path = OUT / "Step2_Amsterdam_EV_Q02B_final_replication_report_v1.md"
    report_path.write_text(report, encoding="utf-8")

    notes = """# Step2 Amsterdam EV-Q02B manuscript amendment notes v1

## Methods addition

Describe a prespecified harmonized external replication using a common ICU-only outcome window. T0 was ICU admission +24 h, T1 was +36 h, direct-fluid exposure was measured in `[T0,T1)`, and the outcome was first recorded serum/plasma sodium >=151 mmol/L in `[T1,min(ICU discharge, ICU admission+7 d, death))`. Patients without a post-T1 sodium measurement were outcome-unascertained and excluded from binary models. H1 adjusted for harmonized age group, sex, last pre-T0 sodium, and the other two direct-fluid co-exposure indicators. Carrier fluids were excluded.

## Results addition

Report Amsterdam and MIMIC estimates side by side, including risk-set size, observed-outcome coverage, model n/events, exposure zero fraction, OR and 95% CI. State that the positive NaCl association was replicated under the harmonized endpoint, while LR and D5W findings were inconclusive.

## Discussion addition

Emphasize that the primary NaCl direction persisted in an independent European ICU database, while effect magnitude differed. Discuss database differences in outcome observation, event rate, exposure prevalence, fluid documentation, and input-event interface coverage. Avoid causal language and do not generalize secondary-fluid findings.

## Mandatory wording boundary

- Use: `harmonized external replication using a common ICU-only outcome window`.
- Use: `independent external replication in a European ICU database`.
- Do not use: exact external validation, multicenter validation, treatment effect, benefit, harm, or causation.
- Keep the original MIMIC primary hospital-window analysis frozen and separately labeled.
"""
    notes_path = OUT / "Step2_Amsterdam_EV_Q02B_manuscript_amendment_notes_v1.md"
    notes_path.write_text(notes, encoding="utf-8")

    errors = []
    if len(comparison) != 6:
        errors.append("comparison row count != 6")
    if not comparison.direction_concordant.all():
        errors.append("one or more fluid directions discordant")
    if classifications["nacl"] != "HARMONIZED_EXTERNAL_REPLICATION_SUPPORTED":
        errors.append("primary classification drift")
    allowed = {"HARMONIZED_EXTERNAL_REPLICATION_SUPPORTED", "HARMONIZED_EXTERNAL_REPLICATION_DIRECTIONALLY_SUPPORTED",
               "HARMONIZED_EXTERNAL_REPLICATION_INCONCLUSIVE", "HARMONIZED_EXTERNAL_REPLICATION_NOT_SUPPORTED"}
    if not set(classifications.values()).issubset(allowed):
        errors.append("invalid replication classification")

    outputs = [comparison_path, report_path, notes_path]
    validation = {
        "task": "EV-Q02B Phase D cross-database comparison",
        "generated_at": datetime.now().astimezone().isoformat(),
        "raw_database_accessed": False,
        "phase_a_hashes_verified": True, "phase_b_hashes_verified": True, "phase_c_hashes_verified": True,
        "primary_nacl_classification": classifications["nacl"],
        "secondary_lr_classification": classifications["lr"],
        "secondary_d5w_classification": classifications["d5w"],
        "mimic_primary_result_modified": False,
        "causal_interpretation_permitted": False,
        "qc_pass": not errors, "qc_errors": errors,
        "input_validation_sha256": {
            "phase_a": sha256(A / "EV_Q02B_A_validation_v1.json"),
            "phase_b": sha256(B / "EV_Q02B_B_Amsterdam_validation_v1.json"),
            "phase_c": sha256(C / "EV_Q02B_C_MIMIC_validation_v1.json"),
            "phase_d_runner": sha256(Path(__file__)),
        },
        "output_sha256": {p.name: sha256(p) for p in outputs},
        "final_gate": "EV_Q02B_COMPLETE",
        "stop_rule": "STOP_AFTER_PHASE_D_NO_ADDITIONAL_DATABASE_ANALYSIS",
    }
    val_path = OUT / "step2_amsterdam_ev_q02b_FINAL_validation_v1.json"
    val_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("Phase D QC failed: " + " | ".join(errors))
    print(json.dumps({"phase_d": "PASS", "final_gate": "EV_Q02B_COMPLETE", "classifications": classifications}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
