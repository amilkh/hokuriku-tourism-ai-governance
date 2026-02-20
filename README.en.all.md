# Fukui Prefecture Tourism DX AI Camera Open Data

This repository publishes aggregated open data from AI cameras installed at key tourist locations in Fukui Prefecture, Japan. Data is regularly collected, processed, and released for public use and visualization.

[Visualization App](https://code4fukui.github.io/fukui-kanko-people-flow-visualization/)
[Visualization App Source Code](https://github.com/code4fukui/fukui-kanko-people-flow-visualization)

## Camera Locations

As of January 2025, AI cameras are installed at:
- Fukui Station East Entrance (Tourist Information)
- Tojinbo Shopping Street
- Rainbow Line Summit Park (Parking Lot 1 & 2)

*Please do not contact the locations directly regarding the cameras.*

## Directory Structure

- `full/`: Aggregated CSV data per location and detection target
- `monthly/`: Data aggregated by day (daily aggregates; organized by location, target, year, month)
- `daily/`: Data aggregated by hour (hourly aggregates; organized by location, target, year, month, day)
- `hourly/`: Data at 5-minute intervals (not hourly aggregates; organized by location, target, year, month, day, hour)
- `tools/`: TypeScript scripts for data aggregation and processing (run with Deno)

## Data Processing Tools

Scripts in the `tools/` directory (run with Deno):
- `aggregate-day.deno.ts`, `aggregate-hour.deno.ts`, `aggregate5mins.deno.ts`: Aggregate raw CSV data into daily, hourly, and 5-minute intervals
- `csv2sqlite.deno.ts`: Convert CSV data to SQLite databases
- `check-csv.deno.ts`: Validate CSV files for errors or warnings
- `escape-age-data.deno.ts`, `escape-movement-data.deno.ts`: Fix formatting issues in age and movement data
- `license-plate-aggregation.deno.ts`: Aggregate license plate data from parking lots
- `thin-out-to-one-second-interval.deno.ts`: Reduce movement data to one-second intervals

## Data Format

CSV files contain aggregated counts and attributes for detected objects (Person, Face, LicensePlate), including age, gender, prefecture, and movement.

## License

MIT License © 2024 Code for FUKUI

---

# Fukui Prefecture Tourism DX AI Camera Open Data Visualization Web Application

This repository provides a monorepo for visualizing people flow and related data collected via AI cameras as part of the Fukui Prefecture Tourism DX initiative. The project consists of multiple Vite+React applications, each focused on a specific visualization or feature, and a shared component library for code reuse.

[Open the Application](https://code4fukui.github.io/fukui-kanko-people-flow-visualization/) (updated daily at 1:00 AM JST)

---

## Project Structure

```
fukui-kanko-people-flow-visualization/
├── packages/
│   ├── whole/           # Comprehensive data visualization app
│   ├── landing-page/    # Landing page for the project
│   ├── fukui-terminal/  # Fukui Station area visualization
│   ├── tojinbo/         # Tojinbo area visualization
│   ├── rainbow-line/    # Rainbow Line parking lot visualization
│   └── shared/          # Shared components, utilities, hooks, and types
├── data/                # Data submodule (people flow data)
└── tools/               # Utility scripts
```

## Monorepo Applications

### 1. `whole`

- **Purpose:** Main, comprehensive visualization of people flow and related data across all monitored areas.
- **Features:**
  - Interactive graphs and charts
  - Date range selection
  - Starred/favorite series
  - Data export and sharing
- **Tech:** React, Vite, Tailwind CSS, Radix UI, Recharts

### 2. `landing-page`

- **Purpose:** Entry point and navigation for the visualization suite.
- **Features:**
  - Links to each sub-application
  - Responsive design
- **Tech:** React, Vite, Tailwind CSS

### 3. `fukui-terminal`

- **Purpose:** Visualization focused on the Fukui Station East Entrance area.
- **Features:**
  - Period-based graphs
  - Data comparison mode
- **Tech:** React, Vite, Tailwind CSS, Recharts

### 4. `tojinbo`

- **Purpose:** Visualization for the Tojinbo Shotaro area.
- **Features:**
  - Similar to `fukui-terminal`, with area-specific data
- **Tech:** React, Vite, Tailwind CSS, Recharts

### 5. `rainbow-line`

- **Purpose:** Visualization for Rainbow Line parking lots.
- **Features:**
  - Parking lot-specific filters
  - Aggregated and daily data views
- **Tech:** React, Vite, Tailwind CSS, Recharts

### 6. `shared`

- **Purpose:** Shared library of UI components, hooks, utilities, types, and constants for all apps.
- **Exports:**
  - Components (UI, parts)
  - Hooks
  - Utilities
  - Types
  - Constants
  - Reducers

## Data

- Data is managed as a git submodule in the `data/people-flow-data` directory.
- Data is copied into the relevant app's `public` directory during development and build.

## Development

### Prerequisites

- [pnpm](https://pnpm.io/) (recommended)
- Node.js 18+

### Install dependencies

```bash
pnpm install
```

### Start development servers

- Start all apps and data server:
  ```bash
  pnpm dev
  ```
- Start only the main app (`whole`):
  ```bash
  pnpm dev:whole
  ```
- Start a specific app (e.g., `fukui-terminal`):
  ```bash
  pnpm dev:fukui-terminal
  ```
- Start landing page only:
  ```bash
  pnpm dev:landing
  ```
- Start all apps simultaneously:
  ```bash
  pnpm dev:all
  ```

### Accessing Apps

- Landing Page: [http://localhost:3004](http://localhost:3004)
- Whole (main): [http://localhost:3000](http://localhost:3000)
- Fukui Terminal: [http://localhost:3001](http://localhost:3001)
- Tojinbo: [http://localhost:3002](http://localhost:3002)
- Rainbow Line: [http://localhost:3003](http://localhost:3003)

### Build

```bash
pnpm build
```

### Lint

```bash
pnpm lint
```

### Data Submodule

To update the data submodule and copy the latest data:

```bash
pnpm submodule
```

## Deployment

- The project is deployed to GitHub Pages via the `upload.sh` script in the `tools/` directory.
- Data can be uploaded separately with `pnpm upload:data`.

## License

See [LICENSE](LICENSE).

---

## Acknowledgements

- Fukui Prefecture Tourism DX Project
- Code for Fukui

---

## Contact

For questions or contributions, please open an issue or pull request on GitHub.

---

# Japan Kanko Dashboard

A web-based dashboard for visualizing tourist visit statistics across Japan, by prefecture and municipality. This project provides interactive charts and data visualizations to help users explore trends in tourism data.

## Live Demo

- [Tourist Dashboard (観光者数ダッシュボード)](https://code4fukui.github.io/japan-kanko-dashboard/)
- [Year-on-Year Comparison (観光者数 前年比)](https://code4fukui.github.io/japan-kanko-dashboard/compprev.html)

## Features

- Interactive treemap and line charts of tourist visits by region, prefecture, and city
- Year-on-year comparison of visitor numbers
- Data selection by region, prefecture, city, and month
- Uses open data sources for up-to-date statistics
- Fully client-side, no backend required

## Data Sources

- [Japan Kanko Stat: Tourist Visits by Prefecture/City](https://github.com/code4fukui/japan-kanko-stat)
- [Digital Tourism Statistics Open Data (日本観光振興協会)](https://www.nihon-kankou.or.jp/home/jigyou/research/d-toukei/)
- [Local Government Codes](https://code4fukui.github.io/localgovjp/localgovjp-utf8.csv)
- [Region Classification (統計局ホームページ/地域区分)](https://www.stat.go.jp/data/shugyou/1997/3-1.html)

## Technology Stack

- HTML, JavaScript (ES Modules)
- [ApexCharts.js](https://apexcharts.com/) for data visualization
- [js.sabae.cc](https://js.sabae.cc/) utility modules (CSV, ArrayUtil, Num, etc.)
- No build tools or frameworks required

## Usage

1. Clone or download this repository.
2. Open `index.html` in your web browser to view the dashboard.
3. Open `compprev.html` for year-on-year comparison charts.

No server setup is required. All data is loaded from public sources via HTTP.

## File Structure

- `index.html` — Main dashboard UI and logic
- `compprev.html` — Year-on-year comparison tool
- `JAPAN_AREA.js` — Region and prefecture definitions
- `README.md` — Japanese project overview
- `README.en.md` — This English README
- `LICENSE` — MIT License

## Contributing

Contributions are welcome! Please open issues or pull requests on [GitHub](https://github.com/code4fukui/japan-kanko-dashboard).

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Taisuke Fukuno ([GitHub](https://github.com/code4fukui))

---

# merged_survey_csv_py

Python scripts to standardize and merge tourism survey CSV data from Toyama, Ishikawa, and Fukui prefectures.

## Overview

This project converts survey data collected by three prefectures into a unified CSV schema, then merges the results into a single dataset and splits outputs by year.

## Features

- Standardize different CSV formats into one schema
- JSON-based column mapping for each prefecture
- Preprocessing for dates, line endings, and anonymized member IDs
- Automatic information source flag generation
- Merge outputs across prefectures
- Auto-download the latest public data
- Year-based file splitting
- Cleanup of old outputs before processing

## Requirements

- Python 3.6+

## Quick Start

Run the full pipeline (download, convert, merge, split):

```bash
python merge_survey.py
```

Run a single prefecture conversion:

```bash
python convert_toyama.py
python convert_ishikawa.py
python convert_fukui.py
```

Download data only:

```bash
python download_data.py
python download_data.py --toyama
python download_data.py --ishikawa
python download_data.py --fukui
```

## Project Structure

```
merged_survey_csv_py/
├── download_data.py           # Data download script
├── convert_toyama.py          # Toyama conversion
├── convert_ishikawa.py        # Ishikawa conversion
├── convert_fukui.py           # Fukui conversion
├── merge_survey.py            # Main pipeline (download + convert + merge)
├── .github/
│   └── workflows/
│       └── run_python.yml     # GitHub Actions workflow
├── input/
│   ├── toyama/
│   │   ├── toyama.csv                    # Toyama survey data
│   │   └── column_mapping_toyama.json    # Toyama mapping
│   ├── ishikawa/
│   │   ├── ishikawa.csv                  # Ishikawa survey data
│   │   └── column_mapping_ishikawa.json  # Ishikawa mapping
│   └── fukui/
│       ├── fukui.csv                     # Fukui merged data
│       ├── fukui_2023.csv                # Fukui 2023 data
│       ├── fukui_2024.csv                # Fukui 2024 data
│       ├── fukui_2025.csv                # Fukui 2025 data
│       └── column_mapping_fukui.json     # Fukui mapping
├── output/
│   ├── toyama/
│   │   ├── toyama_converted.csv          # Full converted data (not pushed)
│   │   ├── toyama_converted_2023.csv     # 2023 split
│   │   └── ...
│   ├── ishikawa/
│   │   ├── ishikawa_converted.csv        # Full converted data (not pushed)
│   │   ├── ishikawa_converted_2023.csv   # 2023 split
│   │   └── ...
│   └── fukui/
│       ├── fukui_converted.csv           # Full converted data (not pushed)
│       ├── fukui_converted_2023.csv      # 2023 split
│       └── ...
└── output_merge/
    ├── merged_survey.csv      # Final merged data (not pushed)
    ├── merged_survey_2023.csv # 2023 split
    └── ...
```

For detailed developer notes, see [docs/development.md](docs/development.md).

## How It Works

1. Download the latest data for each prefecture.
2. Clean output directories.
3. Convert each dataset using JSON column mappings.
4. Generate flags for information sources.
5. Merge all converted rows into a single CSV.
6. Split merged and per-prefecture outputs by year.

## Data Details

### Standardized Schema

The unified output includes fields for:

- Demographics (residence, gender, age, occupation, income)
- Accommodation (area, nights, meals)
- Transportation modes and ratings
- Visit frequency
- Purpose and activity flags
- Information source flags
- Spending categories
- Satisfaction and feedback
- Additional metadata (consent, location, user agent)

### Information Source Flags

Flags are generated when the source text includes known keywords, for example:

- Facebook, RSI, online maps, Instagram, TikTok
- X (formerly Twitter), YouTube, social media ads
- Blog, roundup sites, internet or app
- Digital news, booking websites, accommodation
- TV or radio, newspaper or magazine
- Travel agencies, friends or acquaintances, local people
- Tourism pamphlets or posters
- Tourist information centers, fairs or exhibitions
- DMO or tourism association websites, other

## Output Files

- Per-prefecture converted files under output/<prefecture>/
- Merged files under output_merge/
- Each output is split by year (for example, 2023-2026)
- Full merged files can exceed 50MB and are not pushed to GitHub

## GitHub Actions

The workflow in .github/workflows/run_python.yml runs daily at 06:00 JST to:

1. Download the latest data
2. Convert and merge
3. Split files by year
4. Push year-split outputs

## Notes

- Input CSV files must be UTF-8 encoded.
- Column mapping JSON files must be valid JSON.
- Output directories are created automatically.
- Large files (full merged outputs) are ignored via .gitignore.

## License

- CC-BY 4.0 (Attribution) - Hokuriku Inbound Tourism DX and Data Consortium

## Attribution

This project is a modified aggregation of the following works:

- Toyama Prefecture data: https://ckan.tdcp.pref.toyama.jp/dataset/kanko_data (CC-BY)
- Ishikawa Prefecture data: https://sites.rsi.com/view/milli-ishikawa-pref/data (CC-BY 2.1 JP)
- Fukui Prefecture data: https://github.com/code4fukui/fukui-kanko-survey (CC-BY 4.0)

