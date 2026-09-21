-- Public repository copy: paths are psql variables; no patient-level data are included.
\set ON_ERROR_STOP on
\pset pager off
SET enable_nestloop = off;
SET work_mem = '512MB';

-- Independent secondary cohort: MIMIC_SECONDARY_HARMONIZED_ICU_ONLY.
-- All objects are temporary and exist only in this psql session.
CREATE TEMP TABLE evc_ranked AS
SELECT i.subject_id,i.hadm_id,i.stay_id,i.intime,i.outtime,i.first_careunit,
       a.admittime,a.dischtime,a.deathtime,p.gender,p.anchor_year_group,
       p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year AS age,
       ROW_NUMBER() OVER(PARTITION BY i.subject_id ORDER BY i.intime,i.stay_id) AS icu_rank
FROM mimiciv_icu.icustays i
JOIN mimiciv_hosp.admissions a USING(subject_id,hadm_id)
JOIN mimiciv_hosp.patients p USING(subject_id);

CREATE TEMP TABLE evc_first AS SELECT * FROM evc_ranked WHERE icu_rank=1;
CREATE TEMP TABLE evc_adult AS SELECT * FROM evc_first WHERE age>=18;
CREATE TEMP TABLE evc_clock AS
SELECT *,intime+INTERVAL '24 hour' AS t0_time,intime+INTERVAL '36 hour' AS t1_time,
       LEAST(outtime,intime+INTERVAL '7 day',COALESCE(deathtime,outtime)) AS observation_end,
       (outtime>intime+INTERVAL '24 hour' AND dischtime>intime+INTERVAL '24 hour'
        AND (deathtime IS NULL OR deathtime>intime+INTERVAL '24 hour')) AS present_t0,
       (outtime>intime+INTERVAL '36 hour' AND dischtime>intime+INTERVAL '36 hour'
        AND (deathtime IS NULL OR deathtime>intime+INTERVAL '36 hour')) AS present_t1
FROM evc_adult;

CREATE TEMP TABLE evc_pre_na AS
SELECT c.stay_id,COUNT(ch.sodium)::integer AS pre_na_n,MIN(ch.sodium) AS pre_na_min,
       MAX(ch.sodium) AS pre_na_max,
       (ARRAY_AGG(ch.sodium ORDER BY ch.charttime DESC) FILTER(WHERE ch.sodium IS NOT NULL))[1] AS sodium_last_pre_t0
FROM evc_clock c
LEFT JOIN mimiciv_derived.chemistry ch ON ch.hadm_id=c.hadm_id
 AND ch.charttime>=c.intime AND ch.charttime<c.t0_time AND ch.sodium IS NOT NULL
GROUP BY c.stay_id;

CREATE TEMP TABLE evc_hyper AS
SELECT c.stay_id,COALESCE(BOOL_OR(i.itemid IN(225161,228341)),false) AS hypertonic_pre_t0
FROM evc_clock c
LEFT JOIN mimiciv_icu.inputevents i ON i.stay_id=c.stay_id
 AND i.starttime<c.t0_time AND COALESCE(i.endtime,i.starttime)>c.intime
 AND i.itemid IN(225161,228341)
GROUP BY c.stay_id;

CREATE TEMP TABLE evc_grace AS
SELECT c.stay_id,COALESCE(BOOL_OR(ch.sodium>=151),false) AS grace_high
FROM evc_clock c
LEFT JOIN mimiciv_derived.chemistry ch ON ch.hadm_id=c.hadm_id
 AND ch.charttime>=c.t0_time AND ch.charttime<c.t1_time AND ch.sodium IS NOT NULL
GROUP BY c.stay_id;

CREATE TEMP TABLE evc_flags AS
SELECT c.*,n.pre_na_n,n.pre_na_min,n.pre_na_max,n.sodium_last_pre_t0,
       h.hypertonic_pre_t0,g.grace_high,
       (n.pre_na_n>=1 AND n.pre_na_min>=135 AND n.pre_na_max<=145) AS pre_na_eligible
FROM evc_clock c JOIN evc_pre_na n USING(stay_id)
JOIN evc_hyper h USING(stay_id) JOIN evc_grace g USING(stay_id);

CREATE TEMP TABLE evc_risk AS
SELECT * FROM evc_flags
WHERE present_t0 AND pre_na_eligible AND NOT hypertonic_pre_t0 AND present_t1 AND NOT grace_high;
CREATE INDEX evc_risk_stay_idx ON evc_risk(stay_id);
CREATE INDEX evc_risk_hadm_idx ON evc_risk(hadm_id);

CREATE TEMP TABLE evc_post AS
SELECT r.stay_id,COUNT(ch.sodium)::integer AS post_t1_na_count,
       MIN(ch.charttime) AS first_post_t1_na_time,
       MIN(ch.charttime) FILTER(WHERE ch.sodium>=151) AS first_event_time,
       COALESCE(BOOL_OR(ch.sodium>=151),false) AS event
FROM evc_risk r
LEFT JOIN mimiciv_derived.chemistry ch ON ch.hadm_id=r.hadm_id
 AND ch.charttime>=r.t1_time AND ch.charttime<r.observation_end AND ch.sodium IS NOT NULL
GROUP BY r.stay_id;

CREATE TEMP TABLE evc_interface AS
SELECT r.stay_id,COUNT(i.itemid)::integer AS all_inputevent_records
FROM evc_risk r
LEFT JOIN mimiciv_icu.inputevents i ON i.stay_id=r.stay_id
 AND i.starttime<r.t1_time AND COALESCE(i.endtime,i.starttime)>r.t0_time
GROUP BY r.stay_id;

CREATE TEMP TABLE evc_raw AS
SELECT r.stay_id,i.orderid,i.linkorderid,i.itemid,d.label,d.category,d.unitname,
       i.starttime,i.endtime,i.amount,i.amountuom,i.rate,i.rateuom,
       i.ordercategoryname,i.ordercomponenttypedescription,
       CASE i.itemid WHEN 225158 THEN 'nacl' WHEN 225828 THEN 'lr' WHEN 220949 THEN 'd5w' END AS fluid,
       EXTRACT(EPOCH FROM(i.endtime-i.starttime))/3600.0 AS duration_hours,
       EXTRACT(EPOCH FROM(LEAST(i.endtime,r.t1_time)-GREATEST(i.starttime,r.t0_time)))/3600.0 AS overlap_hours
FROM evc_risk r JOIN mimiciv_icu.inputevents i ON i.stay_id=r.stay_id
JOIN mimiciv_icu.d_items d USING(itemid)
WHERE i.itemid IN(225158,225828,220949)
 AND i.starttime<r.t1_time AND i.endtime>r.t0_time;

CREATE TEMP TABLE evc_order_keys AS SELECT DISTINCT stay_id,orderid FROM evc_raw;
CREATE TEMP TABLE evc_linked AS
SELECT k.stay_id,k.orderid,COUNT(DISTINCT j.itemid)::integer AS linked_main_item_count
FROM evc_order_keys k JOIN mimiciv_icu.inputevents j USING(stay_id,orderid)
WHERE j.itemid NOT IN(225158,225828,220949)
 AND j.ordercomponenttypedescription='Main order parameter'
GROUP BY k.stay_id,k.orderid;

CREATE TEMP TABLE evc_classified AS
WITH x AS(
 SELECT r.*,COALESCE(l.linked_main_item_count,0) linked_main_item_count,
   CASE WHEN r.endtime>r.starttime AND r.duration_hours<=48 AND r.rate>0 AND r.rateuom IN('mL/hour','mL/min') THEN 'RULE_A_RATE_OVERLAP'
        WHEN r.endtime>r.starttime AND r.duration_hours<=24 AND r.amount>0 AND r.amountuom='mL' THEN 'RULE_B_TIME_PRORATED_AMOUNT'
        ELSE 'RULE_C_AMBIGUOUS' END AS allocation_rule
 FROM evc_raw r LEFT JOIN evc_linked l USING(stay_id,orderid)
)
SELECT x.*,
 CASE WHEN endtime<=starttime OR overlap_hours<=0 OR
        ((amount IS NULL OR amount<=0 OR amountuom<>'mL') AND (rate IS NULL OR rate<=0 OR rateuom NOT IN('mL/hour','mL/min')))
      THEN 'EXCLUDE_NONCLINICAL_OR_INVALID'
      WHEN ordercomponenttypedescription='Main order parameter' AND ordercategoryname IN('02-Fluids (Crystalloids)','03-IV Fluid Bolus')
      THEN 'DIRECT_FLUID_RECORD'
      WHEN ordercomponenttypedescription='Mixed solution' AND linked_main_item_count>0 THEN 'MEDICATION_CARRIER_RECORD'
      WHEN ordercomponenttypedescription='Mixed solution' THEN 'MIXED_SOLUTION_RECORD'
      ELSE 'AMBIGUOUS_FLUID_RECORD' END AS semantic_class,
 CASE WHEN allocation_rule='RULE_A_RATE_OVERLAP' AND rateuom='mL/hour' THEN rate*overlap_hours
      WHEN allocation_rule='RULE_A_RATE_OVERLAP' AND rateuom='mL/min' THEN rate*overlap_hours*60.0
      WHEN allocation_rule='RULE_B_TIME_PRORATED_AMOUNT' THEN amount*overlap_hours/duration_hours
      ELSE NULL END AS window_ml
FROM x;

CREATE TEMP TABLE evc_exposure AS
SELECT r.stay_id,
 COALESCE(SUM(c.window_ml) FILTER(WHERE fluid='nacl' AND semantic_class='DIRECT_FLUID_RECORD'),0) AS nacl_ml,
 COALESCE(SUM(c.window_ml) FILTER(WHERE fluid='lr' AND semantic_class='DIRECT_FLUID_RECORD'),0) AS lr_ml,
 COALESCE(SUM(c.window_ml) FILTER(WHERE fluid='d5w' AND semantic_class='DIRECT_FLUID_RECORD'),0) AS d5w_ml,
 COUNT(c.orderid) FILTER(WHERE fluid='nacl' AND semantic_class='DIRECT_FLUID_RECORD')::integer AS nacl_direct_records,
 COUNT(c.orderid) FILTER(WHERE fluid='lr' AND semantic_class='DIRECT_FLUID_RECORD')::integer AS lr_direct_records,
 COUNT(c.orderid) FILTER(WHERE fluid='d5w' AND semantic_class='DIRECT_FLUID_RECORD')::integer AS d5w_direct_records,
 COUNT(c.orderid) FILTER(WHERE allocation_rule='RULE_C_AMBIGUOUS')::integer AS any_rule_c_records,
 COUNT(c.orderid) FILTER(WHERE semantic_class IN('MIXED_SOLUTION_RECORD','AMBIGUOUS_FLUID_RECORD','EXCLUDE_NONCLINICAL_OR_INVALID'))::integer AS any_semantic_ambiguous_or_invalid_records
FROM evc_risk r LEFT JOIN evc_classified c USING(stay_id) GROUP BY r.stay_id;

\copy (SELECT r.subject_id,r.hadm_id,r.stay_id,r.anchor_year_group,r.first_careunit,r.age,r.gender,r.intime,r.outtime,r.t0_time,r.t1_time,r.observation_end,r.sodium_last_pre_t0,p.post_t1_na_count,p.first_post_t1_na_time,p.first_event_time,p.event,EXTRACT(EPOCH FROM(r.observation_end-r.t1_time))/3600.0 AS followup_hours,i.all_inputevent_records,CASE WHEN i.all_inputevent_records>0 THEN 'ACTIVE' ELSE 'SILENT' END AS inputevent_interface_status,e.nacl_ml,e.lr_ml,e.d5w_ml,e.nacl_direct_records,e.lr_direct_records,e.d5w_direct_records,e.any_rule_c_records,e.any_semantic_ambiguous_or_invalid_records FROM evc_risk r JOIN evc_post p USING(stay_id) JOIN evc_interface i USING(stay_id) JOIN evc_exposure e USING(stay_id) ORDER BY r.stay_id) TO :'mimic_source_csv' CSV HEADER

\copy (SELECT * FROM (VALUES (1,'all_icu_stays',(SELECT COUNT(*) FROM evc_ranked),'all mimiciv_icu.icustays rows'),(2,'first_icu_stay_per_patient',(SELECT COUNT(*) FROM evc_first),'icu_rank=1'),(3,'adult_first_icu',(SELECT COUNT(*) FROM evc_adult),'age>=18'),(4,'alive_and_in_icu_at_t0',(SELECT COUNT(*) FROM evc_flags WHERE present_t0),'in ICU/hospital and alive at T0'),(5,'pre_t0_na_eligible',(SELECT COUNT(*) FROM evc_flags WHERE present_t0 AND pre_na_eligible),'>=1 pre-T0 sodium and all 135-145'),(6,'no_pre_t0_hypertonic_saline',(SELECT COUNT(*) FROM evc_flags WHERE present_t0 AND pre_na_eligible AND NOT hypertonic_pre_t0),'exclude item 225161/228341 before T0'),(7,'alive_and_in_icu_at_t1',(SELECT COUNT(*) FROM evc_flags WHERE present_t0 AND pre_na_eligible AND NOT hypertonic_pre_t0 AND present_t1),'alive/in ICU at T1'),(8,'no_t0_t1_na_ge151',(SELECT COUNT(*) FROM evc_risk),'technical T1 risk set'),(9,'post_t1_na_observed_binary_cohort',(SELECT COUNT(*) FROM evc_post WHERE post_t1_na_count>0),'>=1 sodium in harmonized ICU-only outcome window'),(10,'post_t1_na_unascertained',(SELECT COUNT(*) FROM evc_post WHERE post_t1_na_count=0),'excluded; never coded non-event'),(11,'recorded_na_ge151_events',(SELECT COUNT(*) FROM evc_post WHERE event),'events among observed cohort'),(12,'formal_active_interface_cohort',(SELECT COUNT(*) FROM evc_post p JOIN evc_interface i USING(stay_id) WHERE p.post_t1_na_count>0 AND i.all_inputevent_records>0),'formal direct-fluid model cohort'),(13,'formal_active_interface_events',(SELECT COUNT(*) FROM evc_post p JOIN evc_interface i USING(stay_id) WHERE p.post_t1_na_count>0 AND i.all_inputevent_records>0 AND p.event),'events in formal model cohort')) AS x(step_order,step,n,definition) ORDER BY step_order) TO :'mimic_flow_csv' CSV HEADER

DROP TABLE evc_exposure; DROP TABLE evc_classified; DROP TABLE evc_linked; DROP TABLE evc_order_keys;
DROP TABLE evc_raw; DROP TABLE evc_interface; DROP TABLE evc_post; DROP TABLE evc_risk;
DROP TABLE evc_flags; DROP TABLE evc_grace; DROP TABLE evc_hyper; DROP TABLE evc_pre_na;
DROP TABLE evc_clock; DROP TABLE evc_adult; DROP TABLE evc_first; DROP TABLE evc_ranked;
