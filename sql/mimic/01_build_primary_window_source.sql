-- Public repository copy: paths are psql variables; no patient-level data are included.
-- Step2-New-Q05 Phase B: build locked 6/12/24 h exposure and outcome-process source.
-- This script only constructs prespecified variables. It does not fit models.
\set ON_ERROR_STOP on
\pset pager off
SET enable_nestloop = off;
SET work_mem = '1GB';

CREATE TEMP TABLE q05_locked_key (
    subject_id integer,
    hadm_id integer,
    stay_id integer,
    anchor_year_group text,
    temporal_split text,
    outcome_na_ge_151 boolean
);
\copy q05_locked_key FROM :'locked_cohort_csv' CSV HEADER

CREATE TEMP TABLE q05_base AS
SELECT k.subject_id,k.hadm_id,k.stay_id,k.anchor_year_group,k.temporal_split,
       i.intime,i.outtime,i.first_careunit,a.dischtime,a.deathtime,
       i.intime + INTERVAL '24 hour' AS t0_time,
       LEAST(i.intime + INTERVAL '7 day',a.dischtime,COALESCE(a.deathtime,a.dischtime)) AS observation_end
FROM q05_locked_key k
JOIN mimiciv_icu.icustays i USING(subject_id,hadm_id,stay_id)
JOIN mimiciv_hosp.admissions a USING(subject_id,hadm_id);

CREATE INDEX q05_base_stay_idx ON q05_base(stay_id);
CREATE INDEX q05_base_hadm_idx ON q05_base(hadm_id);

CREATE TEMP TABLE q05_first_event AS
SELECT b.stay_id,
       MIN(c.charttime) FILTER(WHERE c.sodium>=151) AS first_event_time
FROM q05_base b
LEFT JOIN mimiciv_derived.chemistry c
  ON c.hadm_id=b.hadm_id
 AND c.charttime>=b.t0_time
 AND c.charttime<b.observation_end
 AND c.sodium IS NOT NULL
GROUP BY b.stay_id;

CREATE TEMP TABLE q05_windows AS
SELECT b.*,w.window_hours,
       b.t0_time + make_interval(hours=>w.window_hours) AS t1_time,
       LEAST(b.t0_time + make_interval(hours=>w.window_hours),b.observation_end) AS management_end,
       e.first_event_time,
       (b.observation_end>b.t0_time + make_interval(hours=>w.window_hours)
        AND (e.first_event_time IS NULL OR e.first_event_time>=b.t0_time + make_interval(hours=>w.window_hours))) AS in_t1_risk_set
FROM q05_base b
JOIN q05_first_event e USING(stay_id)
CROSS JOIN (VALUES (6),(12),(24)) AS w(window_hours);

CREATE INDEX q05_windows_stay_idx ON q05_windows(stay_id,window_hours);
CREATE INDEX q05_windows_hadm_idx ON q05_windows(hadm_id,window_hours);

CREATE TEMP TABLE q05_post_t1_process AS
SELECT w.stay_id,w.window_hours,
       COUNT(c.sodium)::integer AS post_t1_na_count,
       MIN(c.charttime) AS first_post_t1_na_time,
       MIN(c.charttime) FILTER(WHERE c.sodium>=151) AS first_post_t1_event_time,
       BOOL_OR(c.sodium>=151) AS event_post_t1
FROM q05_windows w
LEFT JOIN mimiciv_derived.chemistry c
  ON c.hadm_id=w.hadm_id
 AND c.charttime>=w.t1_time
 AND c.charttime<w.observation_end
 AND c.sodium IS NOT NULL
GROUP BY w.stay_id,w.window_hours;

CREATE TEMP TABLE q05_interface AS
SELECT w.stay_id,w.window_hours,
       COUNT(i.itemid)::integer AS all_inputevent_records
FROM q05_windows w
LEFT JOIN mimiciv_icu.inputevents i
  ON i.stay_id=w.stay_id
 AND i.starttime<w.management_end
 AND i.endtime>w.t0_time
GROUP BY w.stay_id,w.window_hours;

CREATE TEMP TABLE q05_candidate_raw AS
SELECT w.subject_id,w.hadm_id,w.stay_id,w.anchor_year_group,w.temporal_split,
       w.window_hours,w.t0_time,w.t1_time,w.observation_end,w.in_t1_risk_set,
       i.itemid,d.label,d.category,d.unitname,i.orderid,i.linkorderid,
       i.starttime,i.endtime,i.storetime,i.amount,i.amountuom,i.rate,i.rateuom,
       i.originalamount,i.originalrate,i.totalamount,i.totalamountuom,
       i.ordercategoryname,i.secondaryordercategoryname,
       i.ordercomponenttypedescription,i.ordercategorydescription,
       i.statusdescription,i.patientweight,
       CASE i.itemid WHEN 225158 THEN 'NaCl_0.9'
                     WHEN 225828 THEN 'LR'
                     WHEN 220949 THEN 'D5W' END AS treatment,
       GREATEST(i.starttime,w.t0_time) AS overlap_start,
       LEAST(i.endtime,w.management_end) AS overlap_end,
       EXTRACT(EPOCH FROM (i.endtime-i.starttime))/3600.0 AS duration_hours,
       EXTRACT(EPOCH FROM (LEAST(i.endtime,w.management_end)-GREATEST(i.starttime,w.t0_time)))/3600.0 AS overlap_hours,
       (i.starttime<w.t0_time AND i.endtime>w.t0_time) AS cross_t0,
       (i.starttime<w.t1_time AND i.endtime>w.t1_time) AS cross_t1
FROM q05_windows w
JOIN mimiciv_icu.inputevents i ON i.stay_id=w.stay_id
JOIN mimiciv_icu.d_items d USING(itemid)
WHERE i.itemid IN (220949,225158,225828)
  AND i.starttime<w.management_end
  AND i.endtime>w.t0_time;

CREATE INDEX q05_candidate_order_idx ON q05_candidate_raw(stay_id,orderid);

CREATE TEMP TABLE q05_order_keys AS
SELECT DISTINCT stay_id,orderid FROM q05_candidate_raw;
CREATE INDEX q05_order_keys_idx ON q05_order_keys(stay_id,orderid);

CREATE TEMP TABLE q05_linked_main AS
SELECT k.stay_id,k.orderid,
       string_agg(DISTINCT d.label,' | ' ORDER BY d.label) AS linked_main_labels,
       string_agg(DISTINCT d.category,' | ' ORDER BY d.category) AS linked_main_categories,
       COUNT(DISTINCT j.itemid)::integer AS linked_main_item_count
FROM q05_order_keys k
JOIN mimiciv_icu.inputevents j USING(stay_id,orderid)
JOIN mimiciv_icu.d_items d USING(itemid)
WHERE j.itemid NOT IN (220949,225158,225828)
  AND j.ordercomponenttypedescription='Main order parameter'
GROUP BY k.stay_id,k.orderid;
CREATE INDEX q05_linked_main_idx ON q05_linked_main(stay_id,orderid);

CREATE TEMP TABLE q05_classified AS
WITH enriched AS (
    SELECT r.*,l.linked_main_labels,l.linked_main_categories,
           COALESCE(l.linked_main_item_count,0) AS linked_main_item_count,
           CASE
             WHEN r.endtime>r.starttime AND r.duration_hours<=48
              AND r.rate>0 AND r.rateuom IN ('mL/hour','mL/min')
               THEN 'RULE_A_RATE_OVERLAP'
             WHEN r.endtime>r.starttime AND r.duration_hours<=24
              AND r.amount>0 AND r.amountuom='mL'
               THEN 'RULE_B_TIME_PRORATED_AMOUNT'
             ELSE 'RULE_C_AMBIGUOUS'
           END AS volume_allocation_rule
    FROM q05_candidate_raw r
    LEFT JOIN q05_linked_main l USING(stay_id,orderid)
)
SELECT e.*,
       CASE
         WHEN endtime<=starttime OR overlap_hours<=0
           OR ((amount IS NULL OR amount<=0 OR amountuom<>'mL')
               AND (rate IS NULL OR rate<=0 OR rateuom NOT IN ('mL/hour','mL/min')))
           THEN 'EXCLUDE_NONCLINICAL_OR_INVALID'
         WHEN ordercomponenttypedescription='Main order parameter'
          AND ordercategoryname IN ('02-Fluids (Crystalloids)','03-IV Fluid Bolus')
           THEN 'DIRECT_FLUID_RECORD'
         WHEN ordercomponenttypedescription='Mixed solution' AND linked_main_item_count>0
           THEN 'MEDICATION_CARRIER_RECORD'
         WHEN ordercomponenttypedescription='Mixed solution'
           THEN 'MIXED_SOLUTION_RECORD'
         ELSE 'AMBIGUOUS_FLUID_RECORD'
       END AS semantic_class,
       CASE
         WHEN volume_allocation_rule='RULE_A_RATE_OVERLAP' AND rateuom='mL/hour'
           THEN rate*overlap_hours
         WHEN volume_allocation_rule='RULE_A_RATE_OVERLAP' AND rateuom='mL/min'
           THEN rate*overlap_hours*60.0
         WHEN volume_allocation_rule='RULE_B_TIME_PRORATED_AMOUNT'
           THEN amount*overlap_hours/duration_hours
         ELSE NULL
       END AS observable_window_ml
FROM enriched e;

CREATE INDEX q05_classified_stay_idx ON q05_classified(stay_id,window_hours);

CREATE TEMP TABLE q05_exposure_stay AS
SELECT w.stay_id,w.window_hours,
       COUNT(c.orderid) FILTER(WHERE treatment='NaCl_0.9' AND semantic_class='DIRECT_FLUID_RECORD')::integer AS nacl_direct_records,
       COUNT(c.orderid) FILTER(WHERE treatment='NaCl_0.9' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_A_RATE_OVERLAP')::integer AS nacl_direct_rule_a_records,
       COUNT(c.orderid) FILTER(WHERE treatment='NaCl_0.9' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_B_TIME_PRORATED_AMOUNT')::integer AS nacl_direct_rule_b_records,
       COUNT(c.orderid) FILTER(WHERE treatment='NaCl_0.9' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_C_AMBIGUOUS')::integer AS nacl_direct_rule_c_records,
       COALESCE(SUM(c.observable_window_ml) FILTER(WHERE treatment='NaCl_0.9' AND semantic_class='DIRECT_FLUID_RECORD'),0) AS nacl_direct_ab_ml,
       COALESCE(SUM(c.observable_window_ml) FILTER(WHERE treatment='NaCl_0.9' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_A_RATE_OVERLAP'),0) AS nacl_direct_a_ml,
       COUNT(c.orderid) FILTER(WHERE treatment='NaCl_0.9' AND semantic_class='DIRECT_FLUID_RECORD' AND ordercategoryname='03-IV Fluid Bolus')::integer AS nacl_direct_bolus_records,
       COUNT(c.orderid) FILTER(WHERE treatment='NaCl_0.9' AND semantic_class='DIRECT_FLUID_RECORD' AND ordercategoryname<>'03-IV Fluid Bolus')::integer AS nacl_direct_continuous_records,
       COUNT(c.orderid) FILTER(WHERE treatment='NaCl_0.9' AND semantic_class='MEDICATION_CARRIER_RECORD')::integer AS nacl_carrier_records,
       COUNT(c.orderid) FILTER(WHERE treatment='NaCl_0.9' AND semantic_class='MEDICATION_CARRIER_RECORD' AND volume_allocation_rule='RULE_C_AMBIGUOUS')::integer AS nacl_carrier_rule_c_records,
       COALESCE(SUM(c.observable_window_ml) FILTER(WHERE treatment='NaCl_0.9' AND semantic_class='MEDICATION_CARRIER_RECORD'),0) AS nacl_carrier_ab_ml,
       COUNT(c.orderid) FILTER(WHERE treatment='LR' AND semantic_class='DIRECT_FLUID_RECORD')::integer AS lr_direct_records,
       COUNT(c.orderid) FILTER(WHERE treatment='LR' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_A_RATE_OVERLAP')::integer AS lr_direct_rule_a_records,
       COUNT(c.orderid) FILTER(WHERE treatment='LR' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_B_TIME_PRORATED_AMOUNT')::integer AS lr_direct_rule_b_records,
       COUNT(c.orderid) FILTER(WHERE treatment='LR' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_C_AMBIGUOUS')::integer AS lr_direct_rule_c_records,
       COALESCE(SUM(c.observable_window_ml) FILTER(WHERE treatment='LR' AND semantic_class='DIRECT_FLUID_RECORD'),0) AS lr_direct_ab_ml,
       COALESCE(SUM(c.observable_window_ml) FILTER(WHERE treatment='LR' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_A_RATE_OVERLAP'),0) AS lr_direct_a_ml,
       COUNT(c.orderid) FILTER(WHERE treatment='LR' AND semantic_class='DIRECT_FLUID_RECORD' AND ordercategoryname='03-IV Fluid Bolus')::integer AS lr_direct_bolus_records,
       COUNT(c.orderid) FILTER(WHERE treatment='LR' AND semantic_class='DIRECT_FLUID_RECORD' AND ordercategoryname<>'03-IV Fluid Bolus')::integer AS lr_direct_continuous_records,
       COUNT(c.orderid) FILTER(WHERE treatment='LR' AND semantic_class='MEDICATION_CARRIER_RECORD')::integer AS lr_carrier_records,
       COUNT(c.orderid) FILTER(WHERE treatment='LR' AND semantic_class='MEDICATION_CARRIER_RECORD' AND volume_allocation_rule='RULE_C_AMBIGUOUS')::integer AS lr_carrier_rule_c_records,
       COALESCE(SUM(c.observable_window_ml) FILTER(WHERE treatment='LR' AND semantic_class='MEDICATION_CARRIER_RECORD'),0) AS lr_carrier_ab_ml,
       COUNT(c.orderid) FILTER(WHERE treatment='D5W' AND semantic_class='DIRECT_FLUID_RECORD')::integer AS d5w_direct_records,
       COUNT(c.orderid) FILTER(WHERE treatment='D5W' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_A_RATE_OVERLAP')::integer AS d5w_direct_rule_a_records,
       COUNT(c.orderid) FILTER(WHERE treatment='D5W' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_B_TIME_PRORATED_AMOUNT')::integer AS d5w_direct_rule_b_records,
       COUNT(c.orderid) FILTER(WHERE treatment='D5W' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_C_AMBIGUOUS')::integer AS d5w_direct_rule_c_records,
       COALESCE(SUM(c.observable_window_ml) FILTER(WHERE treatment='D5W' AND semantic_class='DIRECT_FLUID_RECORD'),0) AS d5w_direct_ab_ml,
       COALESCE(SUM(c.observable_window_ml) FILTER(WHERE treatment='D5W' AND semantic_class='DIRECT_FLUID_RECORD' AND volume_allocation_rule='RULE_A_RATE_OVERLAP'),0) AS d5w_direct_a_ml,
       COUNT(c.orderid) FILTER(WHERE treatment='D5W' AND semantic_class='DIRECT_FLUID_RECORD' AND ordercategoryname='03-IV Fluid Bolus')::integer AS d5w_direct_bolus_records,
       COUNT(c.orderid) FILTER(WHERE treatment='D5W' AND semantic_class='DIRECT_FLUID_RECORD' AND ordercategoryname<>'03-IV Fluid Bolus')::integer AS d5w_direct_continuous_records,
       COUNT(c.orderid) FILTER(WHERE treatment='D5W' AND semantic_class='MEDICATION_CARRIER_RECORD')::integer AS d5w_carrier_records,
       COUNT(c.orderid) FILTER(WHERE treatment='D5W' AND semantic_class='MEDICATION_CARRIER_RECORD' AND volume_allocation_rule='RULE_C_AMBIGUOUS')::integer AS d5w_carrier_rule_c_records,
       COALESCE(SUM(c.observable_window_ml) FILTER(WHERE treatment='D5W' AND semantic_class='MEDICATION_CARRIER_RECORD'),0) AS d5w_carrier_ab_ml,
       COUNT(c.orderid) FILTER(WHERE semantic_class IN ('MIXED_SOLUTION_RECORD','AMBIGUOUS_FLUID_RECORD','EXCLUDE_NONCLINICAL_OR_INVALID'))::integer AS any_semantic_ambiguous_or_invalid_records,
       COUNT(c.orderid) FILTER(WHERE volume_allocation_rule='RULE_C_AMBIGUOUS')::integer AS any_rule_c_records
FROM q05_windows w
LEFT JOIN q05_classified c USING(stay_id,window_hours)
GROUP BY w.stay_id,w.window_hours;

CREATE TEMP TABLE q05_window_source AS
SELECT w.subject_id,w.hadm_id,w.stay_id,w.anchor_year_group,w.temporal_split,
       w.first_careunit,w.window_hours,w.t0_time,w.t1_time,w.observation_end,
       w.first_event_time,w.in_t1_risk_set,
       p.post_t1_na_count,p.first_post_t1_na_time,p.first_post_t1_event_time,
       COALESCE(p.event_post_t1,false) AS event_post_t1,
       EXTRACT(EPOCH FROM (w.observation_end-w.t1_time))/3600.0 AS post_t1_followup_hours,
       i.all_inputevent_records,
       CASE WHEN i.all_inputevent_records>0 THEN 'ACTIVE' ELSE 'SILENT' END AS inputevent_interface_status,
       e.nacl_direct_records,e.nacl_direct_rule_a_records,e.nacl_direct_rule_b_records,e.nacl_direct_rule_c_records,
       e.nacl_direct_ab_ml,e.nacl_direct_a_ml,e.nacl_direct_bolus_records,e.nacl_direct_continuous_records,
       e.nacl_carrier_records,e.nacl_carrier_rule_c_records,e.nacl_carrier_ab_ml,
       e.lr_direct_records,e.lr_direct_rule_a_records,e.lr_direct_rule_b_records,e.lr_direct_rule_c_records,
       e.lr_direct_ab_ml,e.lr_direct_a_ml,e.lr_direct_bolus_records,e.lr_direct_continuous_records,
       e.lr_carrier_records,e.lr_carrier_rule_c_records,e.lr_carrier_ab_ml,
       e.d5w_direct_records,e.d5w_direct_rule_a_records,e.d5w_direct_rule_b_records,e.d5w_direct_rule_c_records,
       e.d5w_direct_ab_ml,e.d5w_direct_a_ml,e.d5w_direct_bolus_records,e.d5w_direct_continuous_records,
       e.d5w_carrier_records,e.d5w_carrier_rule_c_records,e.d5w_carrier_ab_ml,
       e.any_semantic_ambiguous_or_invalid_records,e.any_rule_c_records
FROM q05_windows w
JOIN q05_post_t1_process p USING(stay_id,window_hours)
JOIN q05_interface i USING(stay_id,window_hours)
JOIN q05_exposure_stay e USING(stay_id,window_hours);

\copy (SELECT * FROM q05_window_source ORDER BY stay_id,window_hours) TO :'window_source_csv' CSV HEADER

SELECT window_hours,
       COUNT(*) AS locked_stays,
       COUNT(*) FILTER(WHERE in_t1_risk_set) AS t1_risk,
       COUNT(*) FILTER(WHERE in_t1_risk_set AND post_t1_na_count>0) AS outcome_observed,
       COUNT(*) FILTER(WHERE in_t1_risk_set AND event_post_t1) AS events,
       COUNT(*) FILTER(WHERE in_t1_risk_set AND post_t1_na_count=0) AS outcome_unascertained,
       COUNT(*) FILTER(WHERE in_t1_risk_set AND post_t1_na_count>0 AND inputevent_interface_status='SILENT') AS interface_silent_observed_outcome
FROM q05_window_source
GROUP BY window_hours ORDER BY window_hours;

DROP TABLE q05_window_source;
DROP TABLE q05_exposure_stay;
DROP TABLE q05_classified;
DROP TABLE q05_linked_main;
DROP TABLE q05_order_keys;
DROP TABLE q05_candidate_raw;
DROP TABLE q05_interface;
DROP TABLE q05_post_t1_process;
DROP TABLE q05_windows;
DROP TABLE q05_first_event;
DROP TABLE q05_base;
DROP TABLE q05_locked_key;
