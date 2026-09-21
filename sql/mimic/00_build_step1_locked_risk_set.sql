-- Public repository copy: paths are psql variables; no patient-level data are included.
-- Purpose: Recompute the v1.0 locked primary-cohort flow using the same eligibility rules.
-- No persistent database writes. A session-local temporary table supports an optional restricted export.
-- Mannitol remains a recorded baseline flag and is not a primary-cohort exclusion.
-- Optional restricted output: pass -v restricted_output_path=/authorized/local/path/step2_locked_cohort_key.csv.
-- The exported CSV is patient-level restricted data and is ignored by Git.

SET enable_nestloop = off;
SET work_mem = '256MB';

CREATE TEMP TABLE step1_locked_cohort_key (
    subject_id integer,
    hadm_id integer,
    stay_id integer,
    anchor_year_group text,
    temporal_split text,
    outcome_na_ge_151 boolean
) ON COMMIT DROP;

WITH ranked_icu AS (
    SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,
        a.dischtime,
        a.deathtime,
        p.anchor_year_group,
        p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year AS age,
        ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime, i.stay_id) AS icu_rank
    FROM mimiciv_icu.icustays AS i
    JOIN mimiciv_hosp.admissions AS a
        ON i.subject_id = a.subject_id AND i.hadm_id = a.hadm_id
    JOIN mimiciv_hosp.patients AS p
        ON i.subject_id = p.subject_id
), first_icu AS (
    SELECT *
    FROM ranked_icu
    WHERE icu_rank = 1
), adult_first_icu AS (
    SELECT *
    FROM first_icu
    WHERE age >= 18
), t0_risk_set AS (
    SELECT
        *,
        intime + INTERVAL '24 hour' AS t0_time,
        LEAST(intime + INTERVAL '7 day', dischtime, COALESCE(deathtime, dischtime)) AS observation_end
    FROM adult_first_icu
    WHERE outtime > intime + INTERVAL '24 hour'
      AND dischtime > intime + INTERVAL '24 hour'
      AND (deathtime IS NULL OR deathtime > intime + INTERVAL '24 hour')
), sodium_window AS MATERIALIZED (
    SELECT
        r.stay_id,
        COUNT(c.sodium) FILTER (WHERE c.charttime < r.t0_time) AS sodium_n_first24,
        MIN(c.sodium) FILTER (WHERE c.charttime < r.t0_time) AS sodium_min_first24,
        MAX(c.sodium) FILTER (WHERE c.charttime < r.t0_time) AS sodium_max_first24,
        COUNT(c.sodium) FILTER (WHERE c.charttime >= r.t0_time) AS sodium_n_followup,
        MIN(c.charttime) FILTER (WHERE c.charttime >= r.t0_time AND c.sodium >= 151) AS first_na_ge_151_time
    FROM t0_risk_set AS r
    LEFT JOIN mimiciv_derived.chemistry AS c
        ON c.hadm_id = r.hadm_id
       AND c.charttime >= r.intime
       AND c.charttime < r.observation_end
       AND c.sodium IS NOT NULL
    GROUP BY r.stay_id
), normal_sodium_risk_set AS (
    SELECT r.*, s.sodium_n_first24, s.sodium_min_first24, s.sodium_max_first24,
           s.sodium_n_followup, s.first_na_ge_151_time
    FROM t0_risk_set AS r
    JOIN sodium_window AS s USING (stay_id)
    WHERE s.sodium_n_first24 >= 1
      AND s.sodium_min_first24 >= 135
      AND s.sodium_max_first24 <= 145
), hypertonic_flag AS MATERIALIZED (
    SELECT
        r.stay_id,
        COALESCE(BOOL_OR(i.itemid IN (225161, 228341)), false) AS hypertonic_saline_pre_t0
    FROM normal_sodium_risk_set AS r
    LEFT JOIN mimiciv_icu.inputevents AS i
        ON i.stay_id = r.stay_id
       AND i.starttime < r.t0_time
       AND COALESCE(i.endtime, i.starttime) > r.intime
       AND i.itemid IN (225161, 228341)
    GROUP BY r.stay_id
), primary_risk_set AS (
    SELECT r.*
    FROM normal_sodium_risk_set AS r
    LEFT JOIN hypertonic_flag AS h USING (stay_id)
    WHERE NOT h.hypertonic_saline_pre_t0
), model_cohort AS (
    SELECT *
    FROM primary_risk_set
    WHERE sodium_n_followup > 0
), split_model_cohort AS (
    SELECT
        *,
        CASE
            WHEN substring(anchor_year_group FROM 1 FOR 4)::int <= 2016 THEN 'development_2008_2016'
            ELSE 'temporal_validation_2017_2022'
        END AS temporal_split
    FROM model_cohort
), exported_restricted_key AS (
    INSERT INTO step1_locked_cohort_key (
        subject_id, hadm_id, stay_id, anchor_year_group, temporal_split, outcome_na_ge_151
    )
    SELECT
        subject_id,
        hadm_id,
        stay_id,
        anchor_year_group,
        CASE
            WHEN substring(anchor_year_group FROM 1 FOR 4)::int <= 2016 THEN 'development_2008_2016'
            ELSE 'temporal_validation_2017_2022'
        END AS temporal_split,
        first_na_ge_151_time IS NOT NULL AS outcome_na_ge_151
    FROM model_cohort
    RETURNING stay_id
), rows AS (
    SELECT 1 AS node_order, 'All ICU stays'::text AS node, 'All rows in mimiciv_icu.icustays'::text AS definition,
           COUNT(*)::bigint AS n_stays, NULL::bigint AS recorded_events, NULL::bigint AS no_followup_sodium
    FROM ranked_icu
    UNION ALL
    SELECT 2, 'First ICU stay per patient', 'icu_rank = 1 before adult eligibility', COUNT(*)::bigint, NULL::bigint, NULL::bigint
    FROM first_icu
    UNION ALL
    SELECT 3, 'Adult first ICU stay', 'Admission age >= 18 using anchor_age, anchor_year, and admittime', COUNT(*)::bigint, NULL::bigint, NULL::bigint
    FROM adult_first_icu
    UNION ALL
    SELECT 4, 'At risk at T0', 'ICU stay >24 h, in hospital at T0, and alive at T0', COUNT(*)::bigint, NULL::bigint, NULL::bigint
    FROM t0_risk_set
    UNION ALL
    SELECT 5, 'Normal observed T0-pre sodium', 'At least 1 sodium before T0 and all observed values 135-145 mmol/L',
           COUNT(*)::bigint, COUNT(first_na_ge_151_time)::bigint, COUNT(*) FILTER (WHERE sodium_n_followup = 0)::bigint
    FROM normal_sodium_risk_set
    UNION ALL
    SELECT 6, 'After hypertonic saline exclusion', 'Exclude T0-pre IV 3% or 23.4% NaCl; mannitol not excluded',
           COUNT(*)::bigint, COUNT(first_na_ge_151_time)::bigint, COUNT(*) FILTER (WHERE sodium_n_followup = 0)::bigint
    FROM primary_risk_set
    UNION ALL
    SELECT 7, 'Final binary modeling cohort', 'At least 1 follow-up sodium within the locked observation window',
           COUNT(*)::bigint, COUNT(first_na_ge_151_time)::bigint, 0::bigint
    FROM model_cohort
    UNION ALL
    SELECT 8, 'Development cohort', 'anchor_year_group starts 2008-2016',
           COUNT(*)::bigint, COUNT(first_na_ge_151_time)::bigint, 0::bigint
    FROM split_model_cohort
    WHERE temporal_split = 'development_2008_2016'
    UNION ALL
    SELECT 9, 'Temporal validation cohort', 'anchor_year_group starts 2017-2022',
           COUNT(*)::bigint, COUNT(first_na_ge_151_time)::bigint, 0::bigint
    FROM split_model_cohort
    WHERE temporal_split = 'temporal_validation_2017_2022'
)
SELECT
    node_order,
    node,
    definition,
    n_stays,
    recorded_events,
    no_followup_sodium,
    CASE
        WHEN node_order < 5 THEN 'not assessed before outcome-risk eligibility'
        WHEN node_order IN (5, 6) THEN 'recorded events; event rate is not calculated because no-follow-up stays remain'
        ELSE 'recorded Na >=151 mmol/L events among binary-model eligible stays'
    END AS event_count_interpretation
FROM rows
ORDER BY node_order;

\if :{?restricted_output_path}
\copy (SELECT subject_id,hadm_id,stay_id,anchor_year_group,temporal_split,outcome_na_ge_151 FROM step1_locked_cohort_key ORDER BY stay_id) TO :'restricted_output_path' CSV HEADER
\endif
