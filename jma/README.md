# JMA Weather Data (Reproducible Pipeline)

This folder stores the canonical JMA weather dataset used by `deep_analysis_tojinbo.py`.

## Canonical analysis input

```
jma/jma_mikuni_hourly_8.csv
```

8 hourly fields from Mikuni (三国) station, cleaned and merged.

## Schema

| Column | Description |
|---|---|
| `timestamp` | Hourly datetime (local JST) |
| `snow_depth_cm` | Snow depth (cm) |
| `snowfall_1h_cm` | 1-hour snowfall (cm) |
| `temp_c` | Temperature (°C) |
| `precip_1h_mm` | 1-hour precipitation (mm) |
| `sun_1h_h` | 1-hour sunshine duration (h) |
| `wind_speed_ms` | Wind speed (m/s) |
| `weather_type` | Weather symbol code |
| `humidity_pct` | Relative humidity (%) |

`deep_analysis_tojinbo.py` maps to legacy names internally:
`temp_c` → `temp`, `precip_1h_mm` → `precip`, `sun_1h_h` → `sun`, `wind_speed_ms` → `wind`

## Adding new months (extending the dataset)

1. Download new monthly hourly CSVs from JMA ObsDL:
   - https://www.data.jma.go.jp/risk/obsdl/
   - Station: Mikuni (三国), all 8 fields, hourly
2. Put the downloaded files in `jma/rawdata/`
   - Any `.csv` filename is accepted
   - Preferred naming for monthly files: `mikuni_hourly_8_YYYY_MM.csv` (example: `mikuni_hourly_8_2025_06.csv`)
   - Raw files are excluded from git (`.gitignore`) — they live only on your machine
3. Run the merger from the repo root:
   ```bash
   python jma/merge_clean_jma.py
   ```
4. The script extends `jma_mikuni_hourly_8.csv` in-place (new months merged, duplicates removed).
5. Commit the updated `jma_mikuni_hourly_8.csv`.

## Notes

- Raw encoding is CP932 (Shift-JIS family).
- The script auto-detects column positions from the JMA header rows.
- Upsert logic: newly parsed rows always win over existing rows for the same timestamp.
- See `jma/rawdata/README.md` for JMA ObsDL export settings used.
