Data sources
METAR (ground truth)
Source: Iowa Environmental Mesonet (IEM), ASOS/METAR archive
Stations used: LTAC Ankara Esenboğa (test airport), LTAN Konya, LTBY Eskişehir, LTCE Erzurum (set in config.yaml)
Period: 2018-01-01 to 2024-12-31
Files: data/raw/LTAC_2018_2024.csv (LTAC, 124,483 reports, all 84 months present) and data/raw/stations_2018_2024.csv (other airports; stations not listed in config.yaml are ignored)
Time: UTC
Report frequency: every 30 minutes at LTAC, plus special reports; LTAN and LTBY report hourly
Fields used: valid (time), vsby (visibility, miles), wxcodes (weather codes)
Download settings (IEM form): UTC, comma delimited, lat/lon and elevation included, missing data as M, trace as T, routine reports and specials
Fog label

A report is labelled fog if both are true:

Visibility below 0.62 miles (1 km, the WMO fog definition)
wxcodes contains FG or FZFG as a whole word

Codes such as BCFG (patches), MIFG (shallow) and PRFG (partial) are not counted as fog.

Fog events

Consecutive fog reports with a gap of at most 1 hour form one event (2 hours at LTAN and LTBY, which report hourly).

Sampling
Night only: solar zenith angle above 95° (computed with pyorbital)
At most 75 events per station and at most 6 reports per event
Non-fog cases: 3 per fog case, at least 3 hours away from any fog report
Random seed 42
SEVIRI (Meteosat Second Generation)
Source: EUMETSAT Data Store, accessed with the eumdac Python API (key and secret in ~/.eumdac/credentials)
Product: SEVIRI Level 1.5 native image data, 0° service (collection EO:EUM:DAT:MSG:HRSEVIRI)
Satellites: Meteosat-9 (MSG2), Meteosat-10 (MSG3) and Meteosat-11 (MSG4), depending on date
Selection: for each case, the nearest SEVIRI scan within ±10 minutes (scans every 15 minutes)
Extraction: 5×5 pixel window around the airport (a pixel is about 4–5 km over Turkey), 8 thermal infrared channels (IR3.9, WV6.2, WV7.3, IR8.7, IR9.7, IR10.8, IR12.0, IR13.4)
Reading and calibration: satpy, brightness temperatures in kelvin; windows with any value below 150 K are dropped
Raw files: deleted after extraction; only features are kept (data/processed/features_all.csv)
Note on satellites

The BTD at the airport pixel (IR10.8 − IR3.9) is lower on Meteosat-10 than on Meteosat-11: about 0.6 K for non-fog and 1.1 K for fog cases. See the README results.

ERA5-Land (surface weather)
Source: ERA5-Land hourly (ECMWF/ERA5_LAND/HOURLY), read through Google Earth Engine at 11,132 m scale
Access: your own Earth Engine project (era5.gee_project in config.yaml)
Variables: 2 m temperature, dew point, skin temperature and 10 m wind, giving four features: temperature_2m, dew point depression (dpd), t2m_minus_skin and wind10
Latency: the final ERA5-Land that Earth Engine serves is 2–3 months behind real time. The preliminary ERA5-Land-T is about 5 days behind but is not in Earth Engine. So ERA5 is not available for live use, and scripts/predict.py falls back to the satellite-only model when the hour is missing.
Raw files: monthly requests cached in data/raw/era5land
Cases not used

The 2023–2024 cases of LTAN, LTBY and LTCE are in neither the training set (2018–2022) nor the test set (LTAC 2023–2024): 941 cases (4,114 − 2,917 − 256).