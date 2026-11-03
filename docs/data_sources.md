# Data sources

## METAR (ground truth)

- **Source:** Iowa Environmental Mesonet (IEM), ASOS/METAR archive
- **Station:** LTAC, Ankara Esenboğa Airport
- **Period:** 2018-01-01 to 2024-12-31
- **File:** `data/raw/LTAC_2018_2024.csv` (124,483 reports, all 84 months present)
- **Time:** UTC (checked against the METAR report time)
- **Report frequency:** every 30 minutes, plus special reports
- **Fields used:** `valid` (time), `vsby` (visibility, miles), `wxcodes` (weather codes)

### Fog label

A report is labelled **fog** if both are true:

1. Visibility below 0.62 miles (1 km, the WMO fog definition)
2. `wxcodes` contains `FG` or `FZFG` as a whole word

Codes such as `BCFG` (patches), `MIFG` (shallow) and `PRFG` (partial) are not counted as fog. `BCFG` is kept as a separate flag.

### Fog events

Consecutive fog reports with gaps of at most 1 hour form one event.
Result: 656 fog reports in 77 events.

## SEVIRI (Meteosat Second Generation)

- **Source:** EUMETSAT Data Store, accessed with the `eumdac` Python API
- **Product:** SEVIRI Level 1.5 native image data, 0° service (collection `EO:EUM:DAT:MSG:HRSEVIRI`)
- **Satellites:** Meteosat-9 (MSG2), Meteosat-10 (MSG3) and Meteosat-11 (MSG4), depending on date
- **Selection:** for each time in `data/processed/sample_times.csv`, the nearest SEVIRI scan (scans every 15 minutes)
- **Extraction:** 5×5 pixel window around LTAC, 8 thermal infrared channels (IR3.9, WV6.2, WV7.3, IR8.7, IR9.7, IR10.8, IR12.0, IR13.4)
- **Reading and calibration:** `satpy`, brightness temperatures in kelvin
- **Raw files:** deleted after extraction; only features are kept in `data/processed/features.csv`

### Note on satellites

The BTD at the airport pixel (IR10.8 − IR3.9) is lower on MSG3 than on MSG4 in this sample: about 0.6 K for non-fog and 1.1 K for fog cases. MSG2 has only 21 samples, too few to compare. See Results.