#!/usr/bin/env python3
"""
Build the model-3 (LANDIS-deployable) dataset.

Model 3 answers the same question as the stand model -- p(cut stand | status) --
but is restricted to features that a LANDIS-II Biomass Succession run actually
emits: **stand cohorts** (species / age / biomass). No FIA-only fields (ownership,
stocking codes, DBH, site class) are used, so the trained model can be scored
directly on LANDIS community-output cohorts.

Label (unchanged from the stand model): a plot's *next* remeasurement records a
cutting treatment (COND.TRTCD{1,2,3}=10). Features are taken from the *prior*
visit's cohorts, so this stays a prospective transition model.

Feature source: the curated cohort table `data_eco_cohorts` (the LANDIS-aligned
representation used elsewhere in this repo -- it carries `age_calc` and `agb` per
species-cohort). Labels/transitions come from raw FIA (`PLOT.PREV_PLT_CN`, `COND`).
The two are joined on plt_cn.

Output: data/cohort_dataset.parquet
"""
import duckdb, os, time

FIADB = os.environ.get("FIADB", os.path.join(os.environ.get("WORKSPACE", "/workspace"), "FIA/truth/fia_truth_r4.duckdb"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)
CONUS = "STATECD<=56 AND STATECD NOT IN (2,15)"

con = duckdb.connect()
con.execute("PRAGMA threads=8")
con.execute(f"ATTACH '{FIADB}' AS t (READ_ONLY)")
# the plot-visit cohorts: Sim A of the truth DB, keyed by the natural visit id of schema raw
con.execute("""
CREATE OR REPLACE TEMP VIEW eco_cohorts AS
SELECT CAST(CAST(v.STATECD AS BIGINT) * 100000000000000 + CAST(v.UNITCD AS BIGINT) * 1000000000000 + CAST(v.COUNTYCD AS BIGINT) * 1000000000
            + CAST(v.PLOT AS BIGINT) * 10000 + CAST(v.INVYR AS BIGINT) AS VARCHAR) AS plt_cn,
       c.subp, c.species_symbol, c.agb, c.age_calc, c.sftwd_hrdwd
FROM t.main.curated_cohorts_landis_stdorg c
JOIN t.main.product_visits v ON v.STATECD = c.statecd AND v.UNITCD = c.unitcd AND v.COUNTYCD = c.countycd AND v.PLOT = c.plot AND v.MEASDATE = c.measdate
""")

t = time.time()
# --- cohort feature block, one row per plot-visit (plt_cn) --------------------
con.execute("""
CREATE OR REPLACE TEMP TABLE cohort_feat AS
WITH sp AS (   -- per-species biomass within a plot-visit
  SELECT plt_cn, species_symbol, SUM(agb) sagb
  FROM eco_cohorts GROUP BY plt_cn, species_symbol
),
sp2 AS (       -- species dominance + concentration (Herfindahl)
  SELECT plt_cn,
         MAX(sagb)/NULLIF(SUM(sagb),0)                    AS dom_species_share,
         SUM(sagb*sagb)/NULLIF(SUM(sagb)*SUM(sagb),0)     AS species_hhi
  FROM sp GROUP BY plt_cn
),
agg AS (
  SELECT plt_cn,
    COUNT(DISTINCT subp)                                  AS n_subp,
    COUNT(*)                                              AS n_cohorts,
    COUNT(DISTINCT species_symbol)                        AS n_species,
    SUM(agb)                                              AS agb_sum,
    SUM(agb*age_calc)/NULLIF(SUM(agb),0)                  AS mean_age,     -- biomass-weighted
    MAX(age_calc)                                         AS max_age,
    MIN(age_calc)                                         AS min_age,
    SQRT(GREATEST(SUM(agb*age_calc*age_calc)/NULLIF(SUM(agb),0)
         - POW(SUM(agb*age_calc)/NULLIF(SUM(agb),0),2), 0)) AS sd_age,
    arg_max(age_calc, agb)                                AS dom_cohort_age, -- age of largest cohort
    SUM(CASE WHEN age_calc <10            THEN agb ELSE 0 END)/NULLIF(SUM(agb),0) AS agb_f_0_10,
    SUM(CASE WHEN age_calc>=10 AND age_calc<20 THEN agb ELSE 0 END)/NULLIF(SUM(agb),0) AS agb_f_10_20,
    SUM(CASE WHEN age_calc>=20 AND age_calc<40 THEN agb ELSE 0 END)/NULLIF(SUM(agb),0) AS agb_f_20_40,
    SUM(CASE WHEN age_calc>=40 AND age_calc<80 THEN agb ELSE 0 END)/NULLIF(SUM(agb),0) AS agb_f_40_80,
    SUM(CASE WHEN age_calc>=80            THEN agb ELSE 0 END)/NULLIF(SUM(agb),0) AS agb_f_80p,
    SUM(CASE WHEN sftwd_hrdwd='S' THEN agb ELSE 0 END)/NULLIF(SUM(agb),0)          AS softwood_frac
  FROM eco_cohorts GROUP BY plt_cn
)
SELECT a.plt_cn,
       a.agb_sum / NULLIF(a.n_subp,0)          AS agb_density,   -- per-subplot (per-acre) biomass
       LN(1 + a.agb_sum/NULLIF(a.n_subp,0))    AS ln_agb,
       a.n_cohorts, a.n_species,
       a.mean_age, a.max_age, a.min_age, a.sd_age, a.dom_cohort_age,
       a.agb_f_0_10, a.agb_f_10_20, a.agb_f_20_40, a.agb_f_40_80, a.agb_f_80p,
       a.softwood_frac, s.dom_species_share, s.species_hhi
FROM agg a JOIN sp2 s ON s.plt_cn = a.plt_cn
""")
n = con.sql("SELECT count(*) FROM cohort_feat").fetchone()[0]
print(f"cohort_feat rows (plot-visits with cohorts): {n:,}  [{time.time()-t:.1f}s]")

# --- stand cut label per plot-visit (same as the stand model) ----------------
con.execute("""
CREATE OR REPLACE TEMP TABLE stand_cut AS
SELECT PLT_CN,
   MAX(CASE WHEN TRTCD1=10 OR TRTCD2=10 OR TRTCD3=10 THEN 1 ELSE 0 END) AS cut
FROM t.raw.COND GROUP BY PLT_CN
""")

# --- ownership per plot-visit (dominant forested condition) ------------------
# Supplied here from FIA; at LANDIS-deployment time ownership comes from an
# external land-ownership layer joined onto the simulated cells.
con.execute("""
CREATE OR REPLACE TEMP TABLE own AS
WITH rk AS (
  SELECT PLT_CN, OWNGRPCD,
         ROW_NUMBER() OVER (PARTITION BY PLT_CN
                            ORDER BY CONDPROP_UNADJ DESC NULLS LAST, CONDID) rn
  FROM t.raw.COND WHERE COND_STATUS_CD = 1
)
SELECT PLT_CN, OWNGRPCD FROM rk WHERE rn = 1
""")

# --- join: prior-visit cohorts (+ownership) -> cut during [T, T+1] -----------
t = time.time()
con.execute(f"""
COPY (
  SELECT sc.cut AS label, cur.REMPER AS remper, o.OWNGRPCD,
         cf.* EXCLUDE (plt_cn), cf.plt_cn AS prev_plt_cn
  FROM t.raw.PLOT cur
  JOIN cohort_feat cf ON cf.plt_cn = cur.PREV_PLT_CN     -- prior visit has cohorts
  JOIN stand_cut  sc ON sc.PLT_CN = cur.CN
  LEFT JOIN own   o  ON o.PLT_CN = cur.PREV_PLT_CN
  WHERE cur.PREV_PLT_CN IS NOT NULL
    AND {CONUS.replace('STATECD','cur.STATECD')}
    AND cur.PLOT_STATUS_CD < 3
) TO '{OUT}/cohort_dataset.parquet' (FORMAT parquet)
""")
r = con.sql(f"SELECT count(*) n, sum(label) pos, avg(label) rate FROM '{OUT}/cohort_dataset.parquet'").fetchone()
print(f"COHORT dataset: {r[0]:,} rows, {r[1]:,} positives ({r[2]*100:.2f}%)  [{time.time()-t:.1f}s]")
print("Done.")
