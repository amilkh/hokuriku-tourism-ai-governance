# Data Provenance — MEAS-D-26-08690 (DHDE / Fukui Tourism)

Snapshot of the exact dataset states that reproduce the published statistics.
Locked and verified on **2026-06-10**. All repos carry a local annotated tag
**`meas-v2-data`** at the paper-correct commit.

## Repo → commit table

| Repo (local path under `~/active/`) | Paper-correct SHA | Commit date | Remote (origin) | Restore command |
| --- | --- | --- | --- | --- |
| `fukui-kanko-trend-report/public/data` (RSI submodule) | `bf2cfc4546229518b4e3d5ece6b19abdca2cf849` | 2026-02-17 ("Update open data for 2026-02-12") | `https://github.com/code4fukui/fukui-kanko-trend-data` | `cd ~/active/fukui-kanko-trend-report/public/data && git checkout meas-v2-data` |
| `fukui-kanko-trend-report` (superproject) | `8bbab3008117feb3fdaca5acbaaec3173548e162` | 2026-03-11 | `https://github.com/code4fukui/fukui-kanko-trend-report` | `cd ~/active/fukui-kanko-trend-report && git checkout meas-v2-data` |
| `fukui-kanko-people-flow-data` (camera counts) | `ca79a526ed500ca4550879fa8c5382c4b91f5d0d` | 2026-03-11 (data through 2026-03-10) | `https://github.com/code4fukui/fukui-kanko-people-flow-data` | `cd ~/active/fukui-kanko-people-flow-data && git checkout meas-v2-data` |
| `opendata` (merged surveys) | `c782c5185d5f6765a1682d44838d754f4c0bb66e` | 2026-02-18 | `git@github.com:hokuriku-inbound-kanko/opendata.git` | `cd ~/active/opendata && git checkout meas-v2-data` |
| `fukui-kanko-survey` | `30f8aa1c1a85c9819f56ea3061aa33637c22dfe4` | 2026-03-10 ("update data") | `https://github.com/code4fukui/fukui-kanko-survey` | `cd ~/active/fukui-kanko-survey && git checkout meas-v2-data` |

Notes:

- **Only the RSI submodule needs a checkout away from HEAD.** The other four
  repos' HEADs are already paper-correct (upstream stopped updating in
  March 2026); the tag pins them against future drift.
- The trend-report superproject's recorded gitlink (`456f2380`, Mar 11 data)
  is NOT the paper state — the paper uses the *older* `bf2cfc45` (Feb 12 data)
  in `public/data`. Always pin the submodule explicitly; do not trust
  `git submodule update`.
- Weather CSVs (`jma/*.csv`, through 2026-03-06) are committed inside this
  repo and are covered by its own history — no separate pin needed.
- **Upstream history rewrite (confirmed 2026-06-10):** code4fukui rewrote
  the history of `fukui-kanko-trend-data` — `bf2cfc45` is no longer an
  ancestor of their current `main`; 1,617 commits including the paper state
  were orphaned upstream. The paper-correct history now survives only in
  the `amilkh` fork and the bundles below. This is why every repo is both
  forked and bundled.

## Fork mirrors (tag `meas-v2-data` pushed 2026-06-10)

All five repos have forks under `github.com/amilkh/` carrying the
`meas-v2-data` annotated tag. These are independent of code4fukui deletions
or history rewrites:

| Fork | Tagged commit |
| --- | --- |
| `github.com/amilkh/fukui-kanko-trend-data` | `bf2cfc45` |
| `github.com/amilkh/fukui-kanko-trend-report` | `8bbab300` |
| `github.com/amilkh/fukui-kanko-people-flow-data` | `ca79a526` |
| `github.com/amilkh/opendata` | `c782c518` |
| `github.com/amilkh/fukui-kanko-survey` | `30f8aa1c` |

Researcher restore, per repo:

```bash
git clone https://github.com/amilkh/<repo>.git
cd <repo> && git checkout meas-v2-data
```

## Verification recipe

```bash
cd ~/active/fukui-kanko-trend-report/public/data
git checkout bf2cfc4546229518b4e3d5ece6b19abdca2cf849   # or: git checkout meas-v2-data
cd ~/active/hokuriku-tourism-ai-governance
python3 scripts/generate_fig2_ols_holdout.py
```

Expected output (verified 2026-06-10):

```text
rows: 397 (paper: 397 = 317 train + 80 holdout), end: 2026-02-12
in-sample R² = 0.8096 (paper: 0.810)
holdout   R² = 0.6834 (paper: 0.683)
holdout  MAE = 1,793 (paper: 1,793)
```

Failure signature with drifted (post-Feb) RSI data: **418 rows, in-sample
R² = 0.792, holdout R² = 0.527**. If you see those numbers, the submodule is
not at `bf2cfc45`.

After verification, restore the working state if you were on a newer commit
(as of 2026-06-10 the working state is `f723f053`, May 2 data):

```bash
cd ~/active/fukui-kanko-trend-report/public/data && git checkout f723f053
```

## Offline backups

Location: `~/backups/meas-v2-data/` (WSL disk, total ≈ 1.6 GB). Integrity:
`sha256sum -c SHA256SUMS` in that directory.

Each repo has two artifacts:

- `*.bundle` — full git history (`git bundle create … --all`), survives
  upstream history rewrites. Restore with
  `git clone <name>.bundle restored-repo && cd restored-repo && git checkout meas-v2-data`.
- `*-<sha>-tree.tar.gz` — plain `git archive` of the pinned tree, readable
  without git.

| Repo | Bundle | Pinned tree archive |
| --- | --- | --- |
| RSI data (public/data) | `rsi-trend-data.bundle` | `rsi-trend-data-bf2cfc45-tree.tar.gz` |
| trend-report (superproject) | `fukui-kanko-trend-report.bundle` | `fukui-kanko-trend-report-8bbab300-tree.tar.gz` |
| people-flow-data | `fukui-kanko-people-flow-data.bundle` | `fukui-kanko-people-flow-data-ca79a526-tree.tar.gz` |
| opendata | `opendata.bundle` | `opendata-c782c518-tree.tar.gz` |
| survey | `fukui-kanko-survey.bundle` | `fukui-kanko-survey-30f8aa1c-tree.tar.gz` |

The superproject archive contains an empty `public/data/` (git archive does
not descend into submodules) — use the RSI archive/bundle alongside it.

A true offsite copy (Takelab via `ssh tk`, RSI Drive, GCS, or a Zenodo
deposit with DOI) is still recommended; as of 2026-06-10 the copies are the
local WSL disk plus the GitHub fork mirrors.
