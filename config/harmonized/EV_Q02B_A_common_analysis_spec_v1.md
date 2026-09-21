# EV-Q02B Phase A common analysis specification v1

## Lock status

`COMMON_SPEC_LOCK = PASS`

This specification was frozen before either EV-Q02B database-specific model was run. It governs only the harmonized external replication using a common ICU-only outcome window and does not replace the frozen MIMIC primary hospital-window analysis.

## Time zero, exposure, and outcome

- T0: ICU admission +24 h.
- T1: ICU admission +36 h.
- Direct-fluid exposure window: `[T0,T1)`.
- Outcome window: `[T1,min(ICU discharge, ICU admission+7 d, death))`, left-closed/right-open.
- Outcome: first recorded serum/plasma sodium >=151 mmol/L.
- Patients with no eligible post-T1 sodium are `OUTCOME_UNASCERTAINED`; they are excluded from binary models and never coded as non-events.

## Exposures and scales

- Primary: documented direct 0.9% NaCl cumulative volume, per 500 mL.
- Secondary: direct lactated Ringer's solution, per 500 mL; direct D5W, per 250 mL.
- Amsterdam allocation uses the locked `fluidin x interval overlap / documented duration` Rule B.
- Carrier/solution volume is excluded from formal harmonized replication (`CARRIER_CONTEXT_ONLY`).
- For each focal fluid model, binary indicators for any exposure to each of the other two direct fluids in `[T0,T1)` are pre-authorized co-exposure terms.

## H1 common adjustment set

The H1 set is restricted to variables that both databases can implement before T0 without outcome-guided mapping:

1. age group: 18-39, 40-49, 50-59, 60-69, 70-79, and >=80 years; categorical, with 18-39 as reference;
2. sex: female reference, male indicator;
3. last eligible serum/plasma sodium before T0; continuous mmol/L;
4. two binary direct-fluid co-exposure indicators specific to the focal exposure model.

The Amsterdam age source is the privacy-preserving `admissions.agegroup`; MIMIC exact age must later be categorized into the same bins. This harmonizes category meaning without pretending exact age is available in Amsterdam.

No post-T1 measurement-process variable is an adjustment variable. No variable is selected by P value or outcome association.

## H2 decision

`H2_DATABASE_SPECIFIC_EXTENDED_ADJUSTMENT = NOT_AUTHORIZED_IN_EV_Q02B_V1`

Q02A identified several candidate proxies but did not freeze their item-level definitions. H2 is therefore not run. This avoids outcome-guided or database-specific post hoc adjustment.

## Sensitivity locks

1. High-certainty observation: >=2 eligible post-T1 sodium measurements and first repeat <=24 h after T1.
2. Amsterdam ambiguity sensitivity: exclude the five predeclared C-grade admissions only.

No restricted cubic spline, threshold search, interaction, subgroup mining, causal weighting, or carrier quantification is authorized.
