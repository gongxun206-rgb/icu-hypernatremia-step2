-- Public repository copy: paths are psql variables; no patient-level data are included.
-- Step2-New-Q07-QC: exact sodium-measurement timing relative to the first
-- recorded post-T1 Na >=151 event. This script is read-only with respect to
-- permanent database objects and does not redefine the outcome.
\set ON_ERROR_STOP on
\pset pager off
SET enable_nestloop = off;
SET work_mem = '1GB';

CREATE TEMP TABLE q07qc_locked_key (
    subject_id integer,
    hadm_id integer,
    stay_id integer,
    anchor_year_group text,
    temporal_split text,
    outcome_na_ge_151 boolean
);
\copy q07qc_locked_key FROM :'locked_cohort_csv' CSV HEADER

CREATE TEMP TABLE q07qc_base AS
SELECT k.subject_id,k.hadm_id,k.stay_id,
       i.intime + INTERVAL '36 hour' AS t1_time,
       LEAST(i.intime + INTERVAL '7 day',a.dischtime,COALESCE(a.deathtime,a.dischtime)) AS observation_end
FROM q07qc_locked_key k
JOIN mimiciv_icu.icustays i USING(subject_id,hadm_id,stay_id)
JOIN mimiciv_hosp.admissions a USING(subject_id,hadm_id);

CREATE INDEX q07qc_base_hadm_idx ON q07qc_base(hadm_id);

CREATE TEMP TABLE q07qc_first_event AS
SELECT b.stay_id,
       MIN(c.charttime) FILTER(WHERE c.sodium>=151) AS first_event_time
FROM q07qc_base b
LEFT JOIN mimiciv_derived.chemistry c
  ON c.hadm_id=b.hadm_id
 AND c.charttime>=b.t1_time
 AND c.charttime<b.observation_end
 AND c.sodium IS NOT NULL
GROUP BY b.stay_id;

CREATE TEMP TABLE q07qc_measurement_timing AS
SELECT b.stay_id,b.t1_time,b.observation_end,e.first_event_time,
       (e.first_event_time IS NOT NULL) AS event_post_t1,
       COUNT(c.sodium)::integer AS total_pre_censor_count,
       COUNT(c.sodium) FILTER(
           WHERE e.first_event_time IS NOT NULL
             AND c.charttime<e.first_event_time
       )::integer AS strictly_pre_event_count,
       COUNT(c.sodium) FILTER(
           WHERE e.first_event_time IS NOT NULL
             AND c.charttime<=e.first_event_time
       )::integer AS through_event_detection_count,
       COUNT(c.sodium) FILTER(
           WHERE e.first_event_time IS NOT NULL
             AND c.charttime>e.first_event_time
       )::integer AS post_event_count,
       COUNT(c.sodium) FILTER(
           WHERE e.first_event_time IS NOT NULL
             AND c.charttime=e.first_event_time
       )::integer AS at_event_timestamp_count,
       MIN(c.charttime) AS first_post_t1_na_time,
       EXTRACT(EPOCH FROM (COALESCE(e.first_event_time,b.observation_end)-b.t1_time))/3600.0 AS analysis_followup_hours,
       CASE WHEN e.first_event_time IS NOT NULL
            THEN EXTRACT(EPOCH FROM (e.first_event_time-b.t1_time))/3600.0
            ELSE NULL END AS time_t1_to_event_hours
FROM q07qc_base b
JOIN q07qc_first_event e USING(stay_id)
LEFT JOIN mimiciv_derived.chemistry c
  ON c.hadm_id=b.hadm_id
 AND c.charttime>=b.t1_time
 AND c.charttime<b.observation_end
 AND c.sodium IS NOT NULL
GROUP BY b.stay_id,b.t1_time,b.observation_end,e.first_event_time;

\copy q07qc_measurement_timing TO :'measurement_timing_csv' CSV HEADER

SELECT COUNT(*) AS locked_stays,
       COUNT(*) FILTER(WHERE event_post_t1) AS events,
       COUNT(*) FILTER(WHERE total_pre_censor_count>0) AS post_t1_na_observed,
       SUM(strictly_pre_event_count) FILTER(WHERE event_post_t1) AS event_group_strict_count,
       SUM(through_event_detection_count) FILTER(WHERE event_post_t1) AS event_group_through_count,
       SUM(post_event_count) FILTER(WHERE event_post_t1) AS event_group_post_count
FROM q07qc_measurement_timing;

DROP TABLE q07qc_measurement_timing;
DROP TABLE q07qc_first_event;
DROP TABLE q07qc_base;
DROP TABLE q07qc_locked_key;
