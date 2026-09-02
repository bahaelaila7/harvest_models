# Harvest-readiness models

Three neural nets that estimate the probability a stand — and an individual tree —
is cut, from the forest's state at the *prior* measurement:

* **Stand model:** `p(cut stand | stand status)` — full FIA stand attributes.
* **Tree model:** `p(cut tree | tree status, stand status)`.
* **Cohort model (LANDIS-deployable):** `p(cut stand | biomass cohorts)` — restricted
  to features a LANDIS-II Biomass Succession run emits (species / age / biomass),
  so it can be scored directly on simulation output.

All are two-layer MLPs (one hidden layer, sigmoid output), as requested.

## Why a prospective (transition) framing

FIA is a **remeasured panel**: every plot is revisited (~5–10 yr, `REMPER`).
"Readiness for harvest" is inherently a forward question — *given the stand as it
stands now, how likely is it to be cut before the next visit?* So every predictor
is taken from the **prior visit (T)** and the label is whether cutting happened in
the interval **[T, T+1]**. Using the state at the cutting visit would be circular:
a clearcut collapses basal area and AGB, which would then "predict" the cut
trivially. Everything here is pre-cut state → future cut.

## Labels (from raw FIA, `/workspace/FIADB.duckdb`)

The curated `data_eco_cohorts.duckdb` used elsewhere in this repo **cannot** train
these models — it aggregates trees into species cohorts and collapses all
disturbance into one `subp_has_dstrb` boolean, discarding harvest vs. fire/insect
and all tree status. The labels only exist in the raw FIA tables:

| Model | Positive label | Source |
|-------|----------------|--------|
| Stand | any forested condition at T+1 has `COND.TRTCD{1,2,3} = 10` (cutting) | `COND` |
| Tree  | a tree live at T (`STATUSCD=1`) is `STATUSCD=3` (cut/removed) at T+1 | `TREE` |

**Tree matching across visits:** `PREV_TRE_CN` is null for ~all cut trees in this
DB, so trees are linked by their physical tag key
`(STATECD, UNITCD, COUNTYCD, PLOT, SUBP, TREE)` ordered by `INVYR` — a window
`lag()` gives each tree its prior-visit state. This recovers 199k cut trees with a
real prior live measurement (vs. ~1k via `PREV_TRE_CN`).

Scope: CONUS (excludes AK/HI), plots forested at the prior visit.

## Features

All predictors describe the **prior visit**. `REMPER` (interval length) is included
as an exposure covariate — longer intervals mean more opportunity to be cut.

**Stand status** (dominant forested condition + live-tree aggregates):

| Feature | Meaning |
|---------|---------|
| `STDAGE` | stand age |
| `BALIVE` | live basal area |
| `ALSTKCD`, `GSSTKCD` | all-live / growing-stock stocking class |
| `fortyp_grp` | forest-type family (`FORTYPCD // 100`) — dominant-species group |
| `OWNGRPCD` | ownership group (federal / state / private) |
| `STDORGCD` | stand origin (natural / planted) |
| `SITECLCD`, `STDSZCD` | site productivity class, stand-size class |
| `agb_per_acre`, `tpa`, `qmd`, `max_dia`, `n_species`, `softwood_frac` | live-tree structure |
| `LAT`, `LON`, `ELEV`, `SLOPE` | location / terrain |

**Tree status** (adds to the full stand block above):

| Feature | Meaning |
|---------|---------|
| `dia`, `ht`, `totage` | maturity: DBH, height, tree age |
| `cr` | crown ratio |
| `cclcd` | crown class (dominant … overtopped) |
| `treeclcd` | tree class (growing stock / rough / rotten) |
| `any_damage`, `n_damage` | recorded damage agents (`DAMAGE_AGENT_CD1/2/3`) — a reason to thin |
| `drybio_ag`, `tpa_unadj`, `softwood` | biomass, expansion factor, softwood flag |

Numeric features are median-imputed with a missingness indicator for heavy-null
columns (e.g. `HT`, `TOTAGE`, `CR`); categoricals are one-hot encoded.

**Cohort model — LANDIS-deployable** (`build_model3.py`). Same stand cut label and
prospective framing, but features are restricted to what a LANDIS-II Biomass
Succession run emits: **stand cohorts** (species / age / biomass). No ownership,
stocking codes, DBH, or site class. Features are computed from the curated
`data_eco_cohorts` table (the LANDIS-aligned cohort representation used elsewhere
in this repo), joined to FIA cut labels on `plt_cn`:

| Feature | Meaning |
|---------|---------|
| `agb_density`, `ln_agb` | total live biomass (per-acre) |
| `mean_age`, `max_age`, `min_age`, `sd_age`, `dom_cohort_age` | biomass-weighted age structure (maturity) |
| `agb_f_0_10 … agb_f_80p` | biomass fraction in age classes 0–10, 10–20, 20–40, 40–80, 80+ |
| `n_cohorts`, `n_species` | stand structural / diversity counts |
| `softwood_frac` | softwood share of biomass |
| `dom_species_share`, `species_hhi` | species dominance & concentration |
| `remper` | interval length (LANDIS knows its own timestep) |

## Model & training

`X → Linear(H) → ReLU → Linear(1) → sigmoid`, Adam, mini-batch. Handling for the
class imbalance (stand ~14% cut, tree ~1.8% cut):

* **Weighted binary cross-entropy** (positives up-weighted to balance the mean
  gradient) so the net ranks rare positives instead of predicting all-negative.
* **Isotonic recalibration** fit on a held-out validation split, so reported
  probabilities are honest (the class weighting otherwise inflates them).
* **Plot-grouped** 70/15/15 train/val/test split (`prev_plt_cn`) — a plot's trees
  never straddle the split, so shared stand features can't leak.

Evaluation: ROC-AUC and average precision (threshold-free ranking) against the
base rate, plus Brier score and a decile calibration table (probability quality).

## Results

See `outputs/summary.json`, `outputs/{stand,tree}_metrics.json`, and
`outputs/train.log`. Headline on the plot-grouped test set:

| Model | Rows | Cut rate | Features | ROC-AUC | Avg precision (vs base) | Brier (calib. vs base) |
|-------|-----:|---------:|---------:|--------:|------------------------:|-----------------------:|
| **Stand** `p(cut stand \| status)` | 428,788 | 14.1% | 42 | **0.801** | **0.418** (base 0.140, 3.0×) | 0.099 vs 0.120 |
| **Tree** `p(cut tree \| status)` | 11,186,775 | 1.78% | 59 | **0.974** | **0.310** (base 0.018, 17.7×) | 0.0135 vs 0.0172 |
| **Cohort** `p(cut stand \| cohorts)` (LANDIS) | 327,214 | 11.1% | 19 | **0.730** | **0.302** (base 0.112, 2.7×) | 0.090 vs 0.100 |

All three beat their base-rate baselines on every metric and are well-calibrated
after isotonic recalibration (predicted ≈ observed across all deciles; e.g. the
stand top decile predicts 0.510 vs. 0.487 observed, the tree top decile 0.180 vs.
0.172, the cohort top decile 0.355 vs. 0.359). The tree model's very high ROC-AUC
reflects that the large majority of trees carry near-zero cut risk and the model
concentrates the ~1.8% of cuts into its top decile.

The **cohort model is deliberately weaker** (0.730 vs. the stand model's 0.801):
it cannot see ownership, stocking class, or site productivity — which are among the
strongest harvest predictors — because LANDIS does not emit them. Biomass, age
structure, and species composition alone still carry real signal (mature,
high-biomass stands are cut more often), which is what makes it usable on
simulation output.

## Reproduce

```bash
FIADB=/workspace/FIADB.duckdb ./run.sh      # venv + build datasets + train + validate
```

Requires the raw FIA DuckDB (`FIADB.duckdb`); point `FIADB` at its location.

The cohort model additionally needs the curated cohort DB
(`ECODB`, default `/workspace/data_eco_cohorts.duckdb`).

Outputs (all under `outputs/`, committed):
* `{stand,tree,cohort}_model.npz` — trained weights + feature names + isotonic calibrator.
* `{stand,tree,cohort}_metrics.json`, `summary.json` — validation metrics.
* `build.log`, `train*.log` — full run logs.

The parquet datasets under `data/` are large and reproducible from the builders,
so they are git-ignored.

## Files

| File | Purpose |
|------|---------|
| `build_dataset.py` | FIA → stand + tree labeled parquet datasets (SQL feature/label extraction) |
| `build_model3.py` | FIA + curated cohorts → LANDIS-deployable cohort dataset |
| `harvest_nn.py` | numpy MLP, weighted BCE, isotonic calibration, grouped split, metrics |
| `run.sh` | one-command reproduce (venv → build → train) |
| `requirements.txt` | pinned deps |
