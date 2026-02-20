# JMA Hourly 8-Field Dataset (Mikuni)

This folder contains raw monthly CSV downloads from JMA and a reproducible script to clean + merge them.

- Source: https://www.data.jma.go.jp/risk/obsdl/
- Station: 三国
- Time unit: 時別値 (hourly)

## Selected 8 fields

1. 積雪の深さ (snow depth)
2. 降雪の深さ（前1時間） (1h snowfall)
3. 気温 (temperature)
4. 降水量（前1時間） (1h precipitation)
5. 日照時間（前1時間） (1h sunshine)
6. 風向・風速 (wind direction/speed)
7. 天気 (weather type)
8. 相対湿度 (relative humidity)

> Primary analysis output uses 8 measurement fields with `wind_speed_ms` as wind metric.
> A second output also includes `wind_dir` (wind direction string).

## Download settings (JMA ObsDL)

Use these settings on the JMA site:

- 地点: 三国
- 期間: monthly chunks (e.g., 2024-12 to latest), hourly values
- 項目:
  - 積雪の深さ
  - 降雪の深さ（前1時間）
  - 気温
  - 降水量（前1時間）
  - 日照時間（前1時間）
  - 風向・風速
  - 天気
  - 相対湿度
- オプション:
  - 利用上注意が必要なデータを表示させる
  - 観測環境などの変化以前のデータを表示させる
  - ダウンロードデータはすべて数値で格納
  - ダウンロードデータに都道府県名を格納

Save files in this folder as `data.csv`, `data (1).csv`, `data (2).csv`, ...

## Reproducible merge/clean

Run:

```bash
cd jma/orig-full-monthly
python merge_clean_jma_8fields.py
```

Outputs:

- `jma_hourly_cleaned_merged_8fields_<start>_<end>.csv`
  - Columns: `timestamp`, `snow_depth_cm`, `snowfall_1h_cm`, `temp_c`, `precip_1h_mm`, `sun_1h_h`, `wind_speed_ms`, `weather_type`, `humidity_pct`
- `jma_hourly_cleaned_merged_8fields_with_wind_direction_<start>_<end>.csv`
  - Same as above plus `wind_dir`

## Notes

- Raw CSV encoding is CP932 (Shift-JIS family).
- JMA files include quality/homogeneity side columns; this script extracts only primary value columns.
- Duplicate timestamps across monthly files are deduplicated by timestamp (keeping the last occurrence).
