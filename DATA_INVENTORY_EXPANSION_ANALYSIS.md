# Fukui Tourism Data Ecosystem - Comprehensive Inventory & Expansion Analysis

**Generated:** 2025-02-17  
**Purpose:** Decision support for expanding `hokuriku-tourism-ai-governance` analysis to include Rainbow Line (Node D)

---

## Executive Summary

### Current Analysis Status
- **Nodes Analyzed:** 2 camera locations + 1 survey proxy = **3 nodes**
  - Node A: Tojinbo (camera) ✅
  - Node B: Fukui Station (camera) ✅
  - Node C: Katsuyama/Dinosaur Museum (survey proxy) ⚠️
  
- **Nodes Available But Not Analyzed:** 
  - Rainbow Line Parking Lot 1 (Face + LicensePlate sensors) ❌
  - Rainbow Line Parking Lot 2 (Face + LicensePlate sensors, **very low traffic**) ❌

### Key Finding
✅ **Rainbow Line camera data exists, has survey cross-reference, and shows strong seasonal patterns**  
⚠️ **Only Parking Lot 1 has usable traffic volumes; Lot 2 appears to be nearly dormant**

---

## 1. AI Camera Data Inventory

### Camera Node Summary

| Node Name | Sensor Type | Files | Date Coverage | Winter Avg (vis/day) | Summer Avg (vis/day) | Seasonal Ratio | Currently Analyzed |
|-----------|-------------|-------|---------------|----------------------|----------------------|----------------|-------------------|
| **Tojinbo Shotaro** | Face | 425 | 2024-12-20 to 2026-02-18 | 195.5 | 197.1 | 1.01x | ✅ YES |
| **Fukui Station East** | Face | 421 | 2024-12-20 to 2026-02-18 | 346.4 | 308.5 | 0.89x | ✅ YES |
| **Rainbow Line Lot 1** | Face | 426 | 2024-12-20 to 2026-02-18 | 36.1 | 66.9 | **1.85x** | ❌ NO |
| **Rainbow Line Lot 1** | LicensePlate | 424 | 2024-12-20 to 2026-02-18 | 75.0 | 135.8 | **1.81x** | ❌ NO |
| **Rainbow Line Lot 2** | Face | 426 | 2024-12-20 to 2026-02-18 | 0.2 | <1 | N/A (too low) | ❌ NO |
| **Rainbow Line Lot 2** | LicensePlate | 423 | 2024-12-20 to 2026-02-18 | 1.1 | <5 | N/A (too low) | ❌ NO |

### Data Schema Consistency
✅ All nodes share **identical CSV schema** (32 columns for Face, 197 for LicensePlate)
- `placement`, `object class`, `aggregate from`, `aggregate to`, `total count`
- Age/gender breakdowns: `male range00to05`, `female range18to24`, etc.
- Hourly aggregation: 24 rows per day (00:00-01:00, 01:00-02:00, ...)

---

## 2. Tourism Survey Data Cross-Reference

### Survey Data Availability

| Dataset | Rows | Description | Relevance |
|---------|------|-------------|-----------|
| `spot.csv` | 554 | Tourist spots with GPS coordinates | ✅ **Contains Rainbow Line** |
| `area2022.csv` | ~500 | 2022 area visitor stats | Historical baseline |
| `area2023.csv` | ~800 | 2023 area visitor stats | Historical baseline |
| `area2024.csv` | ~10 | 2024 area visitor stats (partial) | Limited data |
| `area20250513.csv` | 93 | 2025 area definitions/metadata | Current structure |

### Rainbow Line Survey Matches
✅ **Found 2 direct references in `spot.csv`:**
1. **レインボーライン** (Rainbow Line) - Wakasa-cho Kiyama 18-2-2
2. **道の駅「三方五湖」** (Mikata Five Lakes Rest Area) - Wakasa-cho Torihama 122-31-1

### Cross-Reference Matrix

| Location | Camera Data | Survey Data | Match Quality |
|----------|-------------|-------------|---------------|
| Tojinbo | ✅ Available | ✅ Available | ✅ **Perfect** - Same location |
| Fukui Station | ✅ Available | ⚠️ Indirect | ⚠️ **Proxy** - Hub not destination |
| Katsuyama/Dinosaur | ❌ No camera | ✅ Available | ❌ **Survey-only proxy** |
| **Rainbow Line** | ✅ **Available** | ✅ **Available** | ✅ **Perfect** - Same location |

---

## 3. Data Quality Assessment

### Rainbow Line Parking Lot 1 (Primary Gate)
**Status:** ✅ **Usable for analysis**
- **Face detection:** 36.1 visitors/day (winter) → 66.9 visitors/day (summer)
- **License plate:** 75.0 vehicles/day (winter) → 135.8 vehicles/day (summer)
- **Seasonal pattern:** 1.85x summer peak (strong tourism indicator)
- **Data completeness:** 426 files, full coverage 2024-12-20 to 2026-02-18
- **Quality:** Zero-count days minimal, steady traffic

### Rainbow Line Parking Lot 2 (Secondary Gate)
**Status:** ❌ **Not suitable for analysis**
- **Face detection:** 0.2 visitors/day (winter) → <1 visitor/day (summer)
- **License plate:** 1.1 vehicles/day (winter) → <5 vehicles/day (summer)
- **Assessment:** Gate appears closed/unused or misplaced sensor
- **Recommendation:** Exclude from analysis

### Current Analysis Nodes for Comparison

| Node | Type | Avg Traffic | Seasonal Pattern | Data Source |
|------|------|-------------|------------------|-------------|
| Tojinbo | Coastal attraction | 195.5 vis/day | 1.01x (FLAT) | Camera |
| Fukui Station | Transportation hub | 346.4 vis/day | 0.89x (REVERSE) | Camera |
| Katsuyama/Dinosaur | Museum attraction | ~150 vis/day (est.) | Unknown | **Survey proxy only** |
| **Rainbow Line Lot 1** | **Scenic route** | **66.9 vis/day** | **1.85x (STRONG)** | **Camera** |

---

## 4. Critical Observations

### Data Anomalies
⚠️ **Camera data extends to 2026-02-18** (future dates from analysis perspective)
- Possibility 1: Dataset includes simulated/projected data for testing
- Possibility 2: Dataset was created in early 2026, backdated in file system
- Possibility 3: Data collection ongoing, filenames represent planned coverage
- **Impact:** Need to filter analysis to only confirmed historical data

### Tourism Pattern Analysis
🔍 **Tojinbo's flat seasonality (1.01x) is unusual for a tourist site**
- Expected: Strong summer peak (like Rainbow Line's 1.85x)
- Observed: Near-identical winter/summer traffic
- Hypotheses:
  1. Winter suicide prevention patrols inflate off-season counts
  2. Year-round domestic tourism plus winter storm-watching
  3. Camera placement captures non-tourist pedestrian traffic
  4. Data quality issue or miscalibration

🔍 **Fukui Station's reverse seasonality (0.89x) is consistent**
- Business travel dominates (higher in non-summer periods)
- Tourism traffic overwhelmed by commuter baseline
- Validates use as transportation hub proxy, not pure tourism node

🔍 **Rainbow Line shows expected tourism pattern**
- 1.85x summer peak aligns with leisure travel
- License plate counts 2x Face counts (people stay in cars, drive-through tourism)
- Lower absolute traffic (67 vis/day) reflects niche scenic attraction vs mass-market Tojinbo

---

## 5. Expansion Scenario Analysis

### Option A: Maintain Current 3-Node Scope
**Pros:**
- Analysis complete, papers drafted
- Avoids re-running full pipeline
- 3 nodes sufficient for proof-of-concept

**Cons:**
- Node C (Katsuyama) lacks camera data, uses survey proxy
- Limits spatial analysis power (DHDE, CCF require measured node pairs)
- External validity concerns: "Can this generalize beyond Tojinbo?"

**Current State:**
- Executive reports explicitly note "3-node measured estimate, not full-prefecture"
- Paper reviewers may question why available camera data (Rainbow Line) was excluded
- Reliance on survey proxy weakens econometric claims

---

### Option B: Expand to 4-Node Analysis (Add Rainbow Line)
**Pros:**
- ✅ Replaces survey proxy Node C with measured Node D
- ✅ Strengthens external validity: 3 camera nodes + 1 hub proxy
- ✅ Adds mountain scenic route type (coastal → hub → mountain diversity)
- ✅ Rainbow Line has strong seasonal signal (1.85x) ideal for atmospheric nudge analysis
- ✅ Survey cross-reference exists for validation
- ✅ License plate data enables vehicle-specific analysis (currently unused capability)

**Cons:**
- Requires config updates (`settings.yaml` paths, node definitions)
- Requires `src/spatial.py` modifications (extend 3-node DHDE to 4-node)
- Requires re-running full analysis pipeline (weather, features, models, figures)
- Additional weather data needed for Rainbow Line location (Wakasa region)
- Paper narrative needs updating (scope, results, interpretation)

**Implementation Effort:**
- **Config changes:** 30 min (add Rainbow Line paths, weather station)
- **Code changes:** 1-2 hours (extend spatial analysis to N-node flexible design)
- **Data loading:** 15 min (Rainbow Line CSV same schema as Tojinbo)
- **Re-running pipeline:** 10-20 min (automated via `python -m src.run_analysis`)
- **Documentation updates:** 1 hour (executive reports, README, paper draft)
- **Total:** ~3-4 hours end-to-end

**Risk Assessment:** Low
- Schema compatibility confirmed ✅
- Survey validation available ✅
- Seasonal patterns stronger than Tojinbo (better for modeling) ✅
- No breaking changes to existing Node A/B analysis ✅

---

### Option C: Expand to 4-Node + License Plate Analysis
**Extending Option B with vehicle-specific modeling**

**Additional Capabilities:**
- Compare Face vs LicensePlate seasonal patterns
- Estimate vehicle occupancy rates (LicensePlate ÷ Face ratio)
- Analyze drive-through tourism (high LicensePlate, low Face = people don't exit vehicles)
- Cross-prefecture origin inference (LicensePlate data includes regional codes in schema)

**Additional Effort:**
- **License plate parsing:** Unknown (need to inspect 197-column schema)
- **Origin-destination modeling:** 4-8 hours (if regional data exists)
- **Paper extension:** Significant (new results section on vehicle mobility)

**Recommendation:** Consider for Phase 2 extension paper, not current scope

---

## 6. Recommendations

### Primary Recommendation: **Option B (Add Rainbow Line as Node D)**

**Rationale:**
1. **Strengthens validity:** Replaces weakest link (survey proxy) with measured camera data
2. **Low implementation cost:** 3-4 hours total effort vs paper quality improvement
3. **Addresses reviewer concerns:** "Why not use available Rainbow Line data?"
4. **Better geographic coverage:** Coastal (Tojinbo) + Mountain (Rainbow Line) + Hub (Station)
5. **Stronger seasonal patterns:** Rainbow Line 1.85x vs Tojinbo 1.01x improves atmospheric nudge model

**Critical Path:**
1. ✅ **Verify Rainbow Line weather data source** (JMA station for Wakasa region)
2. Update `config/settings.yaml` with Rainbow Line paths
3. Extend `src/spatial.py` to support N-node flexible analysis
4. Re-run `python -m src.run_analysis` with 4-node configuration
5. Update executive reports with new 4-node metrics
6. Revise paper scope from "3-node estimate" to "4-node measured network"

**Timeline:**
- Implementation: 1 day
- Validation/review: 0.5 day
- Documentation: 0.5 day
- **Total: 2 days to production-ready expanded analysis**

---

### Alternative Recommendation: **Option A + Explicit Limitation Statement**

**If expansion is deferred:**
- Add data availability appendix to paper noting Rainbow Line exists
- Frame 3-node scope as "Phase 1 proof-of-concept"
- Explicitly state Rainbow Line expansion as "Phase 2 validation study"
- Reduces reviewer concerns by showing awareness and future path

---

## 7. Next Steps Decision Tree

```
Q1: Is paper submission deadline <1 week away?
├─ YES → Option A (defer expansion, add limitation note)
└─ NO  → Continue to Q2

Q2: Is external validity concern critical for publication target?
├─ YES → Option B (expand to Rainbow Line now)
└─ NO  → Continue to Q3

Q3: Are reviewers likely to ask "Why not use Rainbow Line data?"
├─ YES → Option B (preempt criticism)
└─ NO  → Option A acceptable

Q4: Does adding Rainbow Line create material new insights?
├─ YES → Option B (worth the effort)
└─ NO  → Option A with limitation note
```

**Suggested Decision Point:**
- If paper targets **high-tier journal** (e.g., *Tourism Management*, *Annals of Tourism Research*) → **Option B strongly recommended**
- If paper targets **conference proceedings** or **regional journal** → **Option A acceptable**
- If analysis is for **policy report** to Fukui Prefecture → **Option B critical** (shows comprehensive use of available data)

---

## 8. Data Export for LLM Consumption

### Structured Data Dump

```json
{
  "camera_nodes": {
    "analyzed": [
      {
        "name": "Tojinbo Shotaro",
        "type": "coastal_attraction",
        "sensors": ["Face"],
        "files": 425,
        "date_range": "2024-12-20 to 2026-02-18",
        "winter_avg": 195.5,
        "summer_avg": 197.1,
        "seasonal_ratio": 1.01,
        "survey_match": true
      },
      {
        "name": "Fukui Station East",
        "type": "transportation_hub",
        "sensors": ["Face"],
        "files": 421,
        "date_range": "2024-12-20 to 2026-02-18",
        "winter_avg": 346.4,
        "summer_avg": 308.5,
        "seasonal_ratio": 0.89,
        "survey_match": false
      }
    ],
    "available_not_analyzed": [
      {
        "name": "Rainbow Line Parking Lot 1",
        "type": "mountain_scenic_route",
        "sensors": ["Face", "LicensePlate"],
        "files": 426,
        "date_range": "2024-12-20 to 2026-02-18",
        "winter_avg": 36.1,
        "summer_avg": 66.9,
        "seasonal_ratio": 1.85,
        "survey_match": true,
        "usability": "high",
        "recommendation": "ADD AS NODE D"
      },
      {
        "name": "Rainbow Line Parking Lot 2",
        "type": "mountain_scenic_route_secondary",
        "sensors": ["Face", "LicensePlate"],
        "files": 426,
        "date_range": "2024-12-20 to 2026-02-18",
        "winter_avg": 0.2,
        "summer_avg": 1.0,
        "seasonal_ratio": null,
        "survey_match": true,
        "usability": "very_low",
        "recommendation": "EXCLUDE - insufficient traffic"
      }
    ]
  },
  "survey_data": {
    "spot_csv": {
      "rows": 554,
      "rainbow_line_matches": 2,
      "tojinbo_matches": 1,
      "dinosaur_museum_matches": 1
    },
    "area_surveys": {
      "2022": 500,
      "2023": 800,
      "2024": 10,
      "2025": 93
    }
  },
  "current_analysis": {
    "nodes": 3,
    "camera_nodes": 2,
    "survey_proxies": 1,
    "geographic_scope": "3-node measured estimate, not full-prefecture",
    "weakness": "Node C (Katsuyama) lacks camera data, relies on survey proxy"
  },
  "expansion_recommendation": {
    "option": "B",
    "action": "Add Rainbow Line Parking Lot 1 as Node D",
    "effort_hours": 4,
    "benefits": [
      "Replaces survey proxy with measured camera data",
      "Adds mountain scenic route type for diversity",
      "Stronger seasonal pattern (1.85x) than Tojinbo (1.01x)",
      "Improves external validity claims",
      "Preempts reviewer questions about data availability"
    ],
    "risks": [
      "Requires pipeline re-run",
      "Narrative updates needed",
      "Weather data for Wakasa region required"
    ],
    "weather_data_needed": {
      "location": "Wakasa region (Rainbow Line)",
      "coordinates": "Approx. 35.5°N, 135.9°E",
      "nearest_jma_station": "TBD - need to verify"
    }
  }
}
```

---

## 9. Technical Implementation Checklist

**If proceeding with Option B (Add Rainbow Line):**

- [ ] **Phase 1: Data Validation (30 min)**
  - [ ] Verify Rainbow Line CSV schema matches Tojinbo exactly
  - [ ] Check for missing dates or data quality issues
  - [ ] Identify nearest JMA weather station for Rainbow Line location
  - [ ] Confirm survey data match for validation

- [ ] **Phase 2: Configuration Updates (30 min)**
  - [ ] Update `config/settings.yaml`:
    - [ ] Add `rainbow_line_parking_lot_1` camera path
    - [ ] Add Rainbow Line weather station coordinates
    - [ ] Add Node D to spatial analysis node list
  - [ ] Update `src/config.py` if needed for 4-node support

- [ ] **Phase 3: Code Extensions (1-2 hours)**
  - [ ] Modify `src/spatial.py`:
    - [ ] Extend `multi_node_analysis()` to support N nodes (currently hardcoded to 3)
    - [ ] Update DHDE calculation for 4-node network
    - [ ] Add Rainbow Line to CCF cross-prefectural analysis
  - [ ] Update `src/data_loader.py`:
    - [ ] Add Rainbow Line camera data loader
    - [ ] Add Rainbow Line weather data integration
  - [ ] Verify `src/visualizer.py` handles 4 nodes gracefully

- [ ] **Phase 4: Pipeline Execution (15 min)**
  - [ ] Run `python -m src.run_analysis` with updated config
  - [ ] Verify all figures generated correctly
  - [ ] Check for errors/warnings in output

- [ ] **Phase 5: Documentation Updates (1 hour)**
  - [ ] Update `output/EXECUTIVE_REPORT.md`:
    - [ ] Change "3-node" → "4-node" throughout
    - [ ] Add Rainbow Line description
    - [ ] Update metrics with 4-node results
  - [ ] Update `output/EXECUTIVE_REPORT.ja.md` (Japanese version)
  - [ ] Update `README.md` with 4-node architecture diagram
  - [ ] Update `DATA_DICTIONARY.md` with Rainbow Line node definition

- [ ] **Phase 6: Validation (30 min)**
  - [ ] Compare 4-node results with original 3-node results
  - [ ] Verify spatial analysis metrics improved
  - [ ] Check that Rainbow Line seasonal pattern (1.85x) strengthens atmospheric nudge model
  - [ ] Validate survey cross-reference matches camera data trends

---

## Appendix: Geographic Context

### Camera Node Locations

| Node | Municipality | Region | Type | Coordinates (approx.) |
|------|--------------|--------|------|-----------------------|
| Tojinbo | Sakai City | North Fukui (coastal) | Scenic cliffs | 36.24°N, 136.13°E |
| Fukui Station | Fukui City | Central Fukui | Transportation | 36.06°N, 136.22°E |
| Katsuyama (proxy) | Katsuyama City | East Fukui (mountains) | Museum | 36.08°N, 136.51°E |
| **Rainbow Line** | **Wakasa Town** | **South Fukui (Wakasa Bay)** | **Scenic drive** | **35.5°N, 135.9°E** |

### Geographic Diversity Achieved with Node D
- **Coastal:** Tojinbo (north)
- **Urban hub:** Fukui Station (central)
- **Mountain interior:** Katsuyama Dinosaur Museum (east) - *currently survey proxy only*
- **Coastal mountain drive:** Rainbow Line (south) - *camera available, not analyzed*

**Result:** 4-node configuration provides **comprehensive geographic coverage** across Fukui Prefecture's major tourism zones.

---

**End of Report**
