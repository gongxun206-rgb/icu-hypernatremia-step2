# Exposure semantic rules

## Direct versus carrier

Direct-fluid records represent documented direct fluid orders. Linked medication/therapy records consistent with diluent or carrier use remain medication-carrier records and are not merged into the primary direct-fluid exposure.

## Allocation

- Rule A: rate x overlap for records with an interpretable rate and interval.
- Rule B: documented amount x overlap/documented duration.
- Rule C: ambiguous allocation; no volume is assigned in the primary A/B metric.

## Interface states

- ACTIVE with no qualifying direct-fluid record: no qualifying record observed in the window.
- SILENT: documentation is unknown and must not be assigned zero.

## Manual semantic audit

The researcher-confirmed 60-case review yielded A=8, B=45, C=7, D=0 and semantic confirmation YES=53, PARTIAL=4, NO=3. Only these aggregate results and rules are public. Patient identifiers, raw medication lines, and row-level review files are excluded.
