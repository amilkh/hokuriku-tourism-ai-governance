# JMA Weather Data (Reproducible Pipeline)

This folder stores weather data used by `deep_analysis_tojinbo.py`.

## Canonical analysis input

The analysis script now uses this file by default:

- `jma_hourly_cleaned_merged_8fields.csv`

If this file is missing, it falls back to the legacy file:

- `jma_hourly_cleaned_merged_2024-01-01_2026-02-19.csv`

## 8-field schema (hourly)

- `timestamp`
- `snow_depth_cm`
- `snowfall_1h_cm`
- `temp_c`
- `precip_1h_mm`
- `sun_1h_h`
- `wind_speed_ms`
- `weather_type`
- `humidity_pct`

The analysis script internally maps these into legacy names for compatibility:

- `temp_c` -> `temp`
- `precip_1h_mm` -> `precip`
- `sun_1h_h` -> `sun`
- `wind_speed_ms` -> `wind`

## How to regenerate from JMA monthly downloads

1. Download hourly monthly CSVs from JMA ObsDL:
   - https://www.data.jma.go.jp/risk/obsdl/
2. Put files in:
   - `jma/orig-full-monthly/`
   - naming pattern: `data.csv`, `data (1).csv`, `data (2).csv`, ...
3. Run merger script:

```bash
cd jma/orig-full-monthly
python merge_clean_jma_8fields.py
```

4. Copy/rename output as canonical input:

```bash
cp jma_hourly_cleaned_merged_8fields_<start>_<end>.csv ../jma_hourly_cleaned_merged_8fields.csv
```

## Notes

- Raw encoding is CP932 (Shift-JIS family).
- Script extracts primary value columns and ignores quality/homogeneity side columns.
- See `jma/orig-full-monthly/README.md` for detailed field setup and JMA options.
