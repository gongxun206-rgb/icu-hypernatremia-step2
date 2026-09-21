-- Public repository copy: paths are psql variables; no patient-level data are included.
-- Step2-New-Q07: extract prespecified 12-hour medication-window and
-- pre-event sodium-measurement process variables from the locked cohort.
-- This script does not change the cohort, outcome, exposure, or models.
\set ON_ERROR_STOP on
\pset pager off
SET enable_nestloop = off;
SET work_mem = '1GB';

CREATE TEMP TABLE q07_locked_key (
    subject_id integer,
    hadm_id integer,
    stay_id integer,
    anchor_year_group text,
    temporal_split text,
    outcome_na_ge_151 boolean
);
\copy q07_locked_key FROM :'locked_cohort_csv' CSV HEADER

CREATE TEMP TABLE q07_base AS
SELECT k.subject_id,k.hadm_id,k.stay_id,k.anchor_year_group,k.temporal_split,
       i.first_careunit,i.intime,a.dischtime,a.deathtime,
       i.intime + INTERVAL '24 hour' AS t0_time,
       i.intime + INTERVAL '36 hour' AS t1_time,
       i.intime + INTERVAL '7 day' AS administrative_end,
       LEAST(i.intime + INTERVAL '7 day',a.dischtime,COALESCE(a.deathtime,a.dischtime)) AS observation_end
FROM q07_locked_key k
JOIN mimiciv_icu.icustays i USING(subject_id,hadm_id,stay_id)
JOIN mimiciv_hosp.admissions a USING(subject_id,hadm_id);

CREATE INDEX q07_base_stay_idx ON q07_base(stay_id);
CREATE INDEX q07_base_hadm_idx ON q07_base(hadm_id);

CREATE TEMP TABLE q07_first_event AS
SELECT b.stay_id,
       MIN(c.charttime) FILTER(WHERE c.sodium>=151) AS first_event_time
FROM q07_base b
LEFT JOIN mimiciv_derived.chemistry c
  ON c.hadm_id=b.hadm_id
 AND c.charttime>=b.t1_time
 AND c.charttime<b.observation_end
 AND c.sodium IS NOT NULL
GROUP BY b.stay_id;

CREATE TEMP TABLE q07_med_raw AS
SELECT b.stay_id,i.itemid,d.label,i.orderid,i.linkorderid,
       i.starttime,i.endtime,i.amount,i.amountuom,i.rate,i.rateuom,
       i.ordercategoryname,i.ordercomponenttypedescription,i.statusdescription,
       EXTRACT(EPOCH FROM (i.endtime-i.starttime))/3600.0 AS duration_hours,
       EXTRACT(EPOCH FROM (LEAST(i.endtime,b.t1_time)-GREATEST(i.starttime,b.t0_time)))/3600.0 AS overlap_hours
FROM q07_base b
JOIN mimiciv_icu.inputevents i ON i.stay_id=b.stay_id
JOIN mimiciv_icu.d_items d USING(itemid)
WHERE i.itemid IN (225161,228341,220995,227533)
  AND i.starttime<b.t1_time
  AND i.endtime>b.t0_time;

CREATE INDEX q07_med_raw_stay_idx ON q07_med_raw(stay_id);

CREATE TEMP TABLE q07_med_classified AS
SELECT r.*,
       CASE
         WHEN itemid IN (225161,228341)
          AND duration_hours>0 AND duration_hours<=48
          AND rate>0 AND rateuom='mL/hour'
           THEN rate*overlap_hours
         WHEN itemid IN (225161,228341)
          AND duration_hours>0 AND duration_hours<=48
          AND rate>0 AND rateuom='mL/min'
           THEN rate*overlap_hours*60.0
         WHEN itemid IN (225161,228341)
          AND duration_hours>0 AND duration_hours<=24
          AND amount>0 AND amountuom='mL'
           THEN amount*overlap_hours/duration_hours
         ELSE NULL
       END AS hypertonic_window_ml,
       CASE
         WHEN itemid=220995
          AND duration_hours>0 AND duration_hours<=24
          AND amount>0 AND amountuom='mEq'
           THEN amount*overlap_hours/duration_hours
         WHEN itemid=227533
          AND duration_hours>0 AND duration_hours<=24
          AND amount>0 AND amountuom='mL'
           THEN amount*overlap_hours/duration_hours
         ELSE NULL
       END AS bicarbonate_window_meq
FROM q07_med_raw r;

CREATE TEMP TABLE q07_bicarb_dual_order AS
SELECT stay_id,linkorderid
FROM q07_med_raw
WHERE itemid IN (220995,227533)
GROUP BY stay_id,linkorderid
HAVING COUNT(DISTINCT itemid)=2;

CREATE TEMP TABLE q07_med_stay AS
SELECT b.stay_id,
       COUNT(m.itemid) FILTER(WHERE m.itemid IN (225161,228341))::integer AS hypertonic_records,
       COUNT(m.itemid) FILTER(WHERE m.itemid=225161)::integer AS hypertonic_3pct_records,
       COUNT(m.itemid) FILTER(WHERE m.itemid=228341)::integer AS hypertonic_23_4pct_records,
       COUNT(m.hypertonic_window_ml) FILTER(WHERE m.itemid IN (225161,228341))::integer AS hypertonic_reconstructable_records,
       COALESCE(SUM(m.hypertonic_window_ml) FILTER(WHERE m.itemid IN (225161,228341)),0) AS hypertonic_window_ml,
       COALESCE(SUM(m.hypertonic_window_ml) FILTER(WHERE m.itemid=225161),0) AS hypertonic_3pct_window_ml,
       COALESCE(SUM(m.hypertonic_window_ml) FILTER(WHERE m.itemid=228341),0) AS hypertonic_23_4pct_window_ml,
       COUNT(m.itemid) FILTER(WHERE m.itemid IN (220995,227533))::integer AS bicarbonate_records,
       COUNT(m.itemid) FILTER(WHERE m.itemid=220995)::integer AS bicarbonate_additive_records,
       COUNT(m.itemid) FILTER(WHERE m.itemid=227533)::integer AS bicarbonate_amp_records,
       COUNT(m.bicarbonate_window_meq) FILTER(WHERE m.itemid IN (220995,227533))::integer AS bicarbonate_reconstructable_records,
       COALESCE(SUM(m.bicarbonate_window_meq) FILTER(WHERE m.itemid IN (220995,227533)),0) AS bicarbonate_window_meq,
       COALESCE(SUM(m.bicarbonate_window_meq) FILTER(WHERE m.itemid=220995),0) AS bicarbonate_additive_window_meq,
       COALESCE(SUM(m.bicarbonate_window_meq) FILTER(WHERE m.itemid=227533),0) AS bicarbonate_amp_window_meq,
       COALESCE(d.dual_component_orders,0)::integer AS bicarbonate_dual_component_orders
FROM q07_base b
LEFT JOIN q07_med_classified m USING(stay_id)
LEFT JOIN (
    SELECT stay_id,COUNT(*)::integer AS dual_component_orders
    FROM q07_bicarb_dual_order GROUP BY stay_id
) d USING(stay_id)
GROUP BY b.stay_id,d.dual_component_orders;

CREATE TEMP TABLE q07_medication_window_source AS
SELECT b.subject_id,b.hadm_id,b.stay_id,b.anchor_year_group,b.temporal_split,
       b.first_careunit,b.t0_time,b.t1_time,b.administrative_end,b.observation_end,
       b.dischtime,b.deathtime,e.first_event_time,
       (e.first_event_time IS NOT NULL) AS event_post_t1,
       (m.hypertonic_records>0) AS hypertonic_any,
       m.hypertonic_records,m.hypertonic_3pct_records,m.hypertonic_23_4pct_records,
       m.hypertonic_reconstructable_records,m.hypertonic_window_ml,
       m.hypertonic_3pct_window_ml,m.hypertonic_23_4pct_window_ml,
       (m.bicarbonate_records>0) AS bicarbonate_any,
       m.bicarbonate_records,m.bicarbonate_additive_records,m.bicarbonate_amp_records,
       m.bicarbonate_reconstructable_records,m.bicarbonate_window_meq,
       m.bicarbonate_additive_window_meq,m.bicarbonate_amp_window_meq,
       m.bicarbonate_dual_component_orders
FROM q07_base b
JOIN q07_first_event e USING(stay_id)
JOIN q07_med_stay m USING(stay_id)
ORDER BY b.stay_id;

\copy q07_medication_window_source TO :'medication_source_csv' CSV HEADER

CREATE TEMP TABLE q07_measurement_process AS
SELECT b.stay_id,
       e.first_event_time,
       COALESCE(COUNT(c.sodium) FILTER(
           WHERE c.charttime < COALESCE(e.first_event_time,b.observation_end)
       ),0)::integer AS na_count_before_event_or_censor,
       MIN(c.charttime) AS first_repeat_na_time
FROM q07_base b
JOIN q07_first_event e USING(stay_id)
LEFT JOIN mimiciv_derived.chemistry c
  ON c.hadm_id=b.hadm_id
 AND c.charttime>=b.t1_time
 AND c.charttime<b.observation_end
 AND c.sodium IS NOT NULL
GROUP BY b.stay_id,e.first_event_time;

CREATE TEMP TABLE q07_measurement_process_source AS
SELECT b.stay_id,b.t1_time,b.observation_end,e.first_event_time,
       (e.first_event_time IS NOT NULL) AS event_post_t1,
       p.na_count_before_event_or_censor,p.first_repeat_na_time,
       EXTRACT(EPOCH FROM (COALESCE(e.first_event_time,b.observation_end)-b.t1_time))/3600.0 AS analysis_followup_hours
FROM q07_base b
JOIN q07_first_event e USING(stay_id)
JOIN q07_measurement_process p USING(stay_id)
ORDER BY b.stay_id;

\copy q07_measurement_process_source TO :'measurement_source_csv' CSV HEADER

CREATE TEMP TABLE q07_medication_item_mapping AS
SELECT itemid,label,abbreviation,category,unitname
FROM mimiciv_icu.d_items
WHERE itemid IN (225161,228341,220995,227533)
ORDER BY itemid;

\copy q07_medication_item_mapping TO :'medication_mapping_csv' CSV HEADER

SELECT COUNT(*) AS locked_stays,
       COUNT(*) FILTER(WHERE hypertonic_records>0) AS hypertonic_exposed,
       COUNT(*) FILTER(WHERE bicarbonate_records>0) AS bicarbonate_exposed,
       SUM(hypertonic_records) AS hypertonic_records,
       SUM(hypertonic_reconstructable_records) AS hypertonic_reconstructable_records,
       SUM(bicarbonate_records) AS bicarbonate_records,
       SUM(bicarbonate_reconstructable_records) AS bicarbonate_reconstructable_records,
       SUM(bicarbonate_dual_component_orders) AS bicarbonate_dual_component_orders
FROM q07_med_stay;

DROP TABLE q07_medication_item_mapping;
DROP TABLE q07_measurement_process_source;
DROP TABLE q07_measurement_process;
DROP TABLE q07_medication_window_source;
DROP TABLE q07_med_stay;
DROP TABLE q07_bicarb_dual_order;
DROP TABLE q07_med_classified;
DROP TABLE q07_med_raw;
DROP TABLE q07_first_event;
DROP TABLE q07_base;
DROP TABLE q07_locked_key;
