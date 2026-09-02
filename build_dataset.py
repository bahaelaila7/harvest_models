#!/usr/bin/env python3
"""
Build labeled datasets for the two harvest-readiness models from raw FIA data.

FIA is a remeasured panel. We model harvest PROSPECTIVELY: given a stand/tree's
state at a prior visit T, what is the probability it is cut during the interval
[T, T+1]? Using the state at the *cutting* visit would be circular (a clearcut
collapses AGB, trivially predicting the cut), so all predictive features are
taken from the prior visit and only the label comes from the current visit.

Labels
------
Stand cut (plot grain): any forested condition at the current visit has
    COND.TRTCD{1,2,3} = 10 (cutting/harvest).  ~14% of previously-forested,
    remeasured plots.
Tree cut: a tree that was live (STATUSCD=1) at its prior visit is recorded
    STATUSCD=3 (cut/removed) at the current visit.  Trees are matched across
    visits by physical tag key (STATECD,UNITCD,COUNTYCD,PLOT,SUBP,TREE) because
    PREV_TRE_CN is null for ~all cut trees in this DB.  ~1.8% positive.

Outputs two parquet files under data/:  stand_dataset.parquet, tree_dataset.parquet
"""
import duckdb, os, time, sys

DB = "/workspace/FIADB.duckdb"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)

# CONUS, exclude AK(2)/HI(15); sampled plots only.
CONUS = "STATECD<=56 AND STATECD NOT IN (2,15)"

con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=8")

def step(msg):
    print(f"\n=== {msg} ===", flush=True); return time.time()

# ---------------------------------------------------------------------------
# 1. Per-visit STAND feature table, keyed by PLT_CN.
#    Used as the prior-visit ("status at T") feature block for both models.
# ---------------------------------------------------------------------------
t = step("Building per-visit stand features (keyed by PLT_CN)")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE stand_feat AS
WITH
-- dominant forested condition per plot visit (largest forested CONDPROP)
cond_rk AS (
  SELECT PLT_CN, STDAGE, BALIVE, SITECLCD, STDORGCD, STDSZCD, OWNGRPCD, SLOPE,
         ALSTKCD, GSSTKCD, FORTYPCD, CONDPROP_UNADJ,
         ROW_NUMBER() OVER (PARTITION BY PLT_CN
                            ORDER BY CONDPROP_UNADJ DESC NULLS LAST, CONDID) rn
  FROM COND
  WHERE COND_STATUS_CD = 1                       -- forested conditions only
),
cond_feat AS (
  SELECT PLT_CN, STDAGE, BALIVE, SITECLCD, STDORGCD, STDSZCD, OWNGRPCD, SLOPE,
         ALSTKCD,                                  -- all-live stocking class (1 overstocked .. 5 nonstocked)
         GSSTKCD,                                  -- growing-stock stocking class
         (FORTYPCD // 100)::INT AS fortyp_grp      -- forest-type family (1 pine..9 other); dominant species
  FROM cond_rk WHERE rn = 1
),
-- forested proportion of the plot (0..1); marks whether the plot is forest at all
cond_prop AS (
  SELECT PLT_CN, SUM(CONDPROP_UNADJ) AS forest_prop
  FROM COND WHERE COND_STATUS_CD = 1 GROUP BY PLT_CN
),
-- live-tree stand aggregates
tree_feat AS (
  SELECT PLT_CN,
         SUM(DRYBIO_AG * TPA_UNADJ)                                  AS agb_per_acre,
         SUM(TPA_UNADJ)                                              AS tpa,
         SQRT(SUM(DIA*DIA*TPA_UNADJ) / NULLIF(SUM(TPA_UNADJ),0))     AS qmd,
         MAX(DIA)                                                    AS max_dia,
         COUNT(DISTINCT SPCD)                                        AS n_species,
         SUM(CASE WHEN SPGRPCD<=24 THEN TPA_UNADJ ELSE 0 END)
           / NULLIF(SUM(TPA_UNADJ),0)                                AS softwood_frac
  FROM TREE
  WHERE STATUSCD = 1 AND {CONUS} AND DRYBIO_AG IS NOT NULL AND TPA_UNADJ IS NOT NULL
  GROUP BY PLT_CN
)
SELECT p.CN AS PLT_CN, p.LAT, p.LON, p.ELEV, p.MEASYEAR,
       cf.STDAGE, cf.BALIVE, cf.SITECLCD, cf.STDORGCD, cf.STDSZCD, cf.OWNGRPCD, cf.SLOPE,
       cf.ALSTKCD, cf.GSSTKCD, cf.fortyp_grp,
       cp.forest_prop,
       tf.agb_per_acre, tf.tpa, tf.qmd, tf.max_dia, tf.n_species, tf.softwood_frac
FROM PLOT p
JOIN cond_prop cp ON cp.PLT_CN = p.CN            -- keep only plots forested at this visit
LEFT JOIN cond_feat cf ON cf.PLT_CN = p.CN
LEFT JOIN tree_feat tf ON tf.PLT_CN = p.CN
WHERE {CONUS.replace('STATECD','p.STATECD')}
""")
n = con.sql("SELECT count(*) FROM stand_feat").fetchone()[0]
print(f"stand_feat rows (forested plot-visits): {n:,}  [{time.time()-t:.1f}s]")

# ---------------------------------------------------------------------------
# 2. STAND dataset: prior-visit features -> cut in [T, T+1].
# ---------------------------------------------------------------------------
t = step("Building STAND dataset")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE stand_cut AS
SELECT PLT_CN,
   MAX(CASE WHEN TRTCD1=10 OR TRTCD2=10 OR TRTCD3=10 THEN 1 ELSE 0 END) AS cut
FROM COND GROUP BY PLT_CN
""")
con.execute(f"""
COPY (
  SELECT
      sc.cut                       AS label,
      cur.REMPER                   AS remper,        -- interval length T->T+1 (exposure)
      cur.CN                       AS cur_plt_cn,
      sf.PLT_CN                     AS prev_plt_cn,
      sf.LAT, sf.LON, sf.ELEV, sf.MEASYEAR AS prev_measyear,
      sf.STDAGE, sf.BALIVE, sf.SITECLCD, sf.STDORGCD, sf.STDSZCD, sf.OWNGRPCD, sf.SLOPE,
      sf.ALSTKCD, sf.GSSTKCD, sf.fortyp_grp,
      sf.forest_prop, sf.agb_per_acre, sf.tpa, sf.qmd, sf.max_dia, sf.n_species, sf.softwood_frac
  FROM PLOT cur
  JOIN stand_feat sf ON sf.PLT_CN = cur.PREV_PLT_CN   -- prior visit must be forested
  JOIN stand_cut  sc ON sc.PLT_CN = cur.CN
  WHERE cur.PREV_PLT_CN IS NOT NULL
    AND {CONUS.replace('STATECD','cur.STATECD')}
    AND cur.PLOT_STATUS_CD < 3
) TO '{OUT}/stand_dataset.parquet' (FORMAT parquet)
""")
r = con.sql(f"SELECT count(*) n, sum(label) pos, avg(label) rate FROM '{OUT}/stand_dataset.parquet'").fetchone()
print(f"STAND dataset: {r[0]:,} rows, {r[1]:,} positives ({r[2]*100:.2f}%)  [{time.time()-t:.1f}s]")

# ---------------------------------------------------------------------------
# 3. TREE dataset: prior-visit tree+stand state -> cut at current visit.
#    Match trees across visits by physical tag key; prior visit must be live.
# ---------------------------------------------------------------------------
t = step("Building TREE dataset")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE tree_trans AS
WITH w AS (
  SELECT STATECD,UNITCD,COUNTYCD,PLOT,SUBP,TREE, INVYR, PLT_CN, STATUSCD,
         lag(STATUSCD) OVER v prev_st,
         lag(PLT_CN)   OVER v prev_pltcn,
         lag(DIA)      OVER v prev_dia,
         lag(HT)       OVER v prev_ht,
         lag(CCLCD)    OVER v prev_cclcd,
         lag(CR)       OVER v prev_cr,
         lag(DRYBIO_AG)OVER v prev_bio,
         lag(TPA_UNADJ)OVER v prev_tpa,
         lag(SPGRPCD)  OVER v prev_spgrpcd,
         lag(TREECLCD) OVER v prev_treeclcd,
         lag(TOTAGE)   OVER v prev_totage,           -- tree age (maturity), sparse
         lag(DAMAGE_AGENT_CD1) OVER v prev_dmg1,
         lag(DAMAGE_AGENT_CD2) OVER v prev_dmg2,
         lag(DAMAGE_AGENT_CD3) OVER v prev_dmg3
  FROM TREE WHERE {CONUS}
  WINDOW v AS (PARTITION BY STATECD,UNITCD,COUNTYCD,PLOT,SUBP,TREE ORDER BY INVYR)
)
SELECT
    CASE WHEN STATUSCD=3 THEN 1 ELSE 0 END AS label,   -- cut vs (survived/died)
    PLT_CN AS cur_plt_cn, prev_pltcn AS prev_plt_cn,
    prev_dia, prev_ht, prev_cclcd, prev_cr, prev_bio, prev_tpa, prev_treeclcd, prev_totage,
    CASE WHEN prev_spgrpcd<=24 THEN 1 ELSE 0 END AS prev_softwood,
    -- damage: any/count of recorded damage agents at prior visit (a reason to thin)
    CASE WHEN coalesce(prev_dmg1,0)>0 OR coalesce(prev_dmg2,0)>0 OR coalesce(prev_dmg3,0)>0
         THEN 1 ELSE 0 END AS prev_any_damage,
    (CASE WHEN coalesce(prev_dmg1,0)>0 THEN 1 ELSE 0 END
   + CASE WHEN coalesce(prev_dmg2,0)>0 THEN 1 ELSE 0 END
   + CASE WHEN coalesce(prev_dmg3,0)>0 THEN 1 ELSE 0 END) AS prev_n_damage
FROM w
WHERE prev_st = 1 AND STATUSCD IN (1,2,3)            -- live at T, resolved at T+1
""")
con.execute(f"""
COPY (
  SELECT
      tt.label, cur.REMPER AS remper,
      -- tree state at prior visit (maturity, crown, damage)
      tt.prev_dia AS dia, tt.prev_ht AS ht, tt.prev_cclcd AS cclcd, tt.prev_cr AS cr,
      tt.prev_bio AS drybio_ag, tt.prev_tpa AS tpa_unadj, tt.prev_treeclcd AS treeclcd,
      tt.prev_totage AS totage, tt.prev_softwood AS softwood,
      tt.prev_any_damage AS any_damage, tt.prev_n_damage AS n_damage,
      -- stand state at prior visit
      sf.STDAGE, sf.BALIVE, sf.SITECLCD, sf.STDORGCD, sf.STDSZCD, sf.OWNGRPCD, sf.SLOPE,
      sf.ALSTKCD, sf.GSSTKCD, sf.fortyp_grp,
      sf.agb_per_acre, sf.tpa AS stand_tpa, sf.qmd, sf.max_dia, sf.n_species, sf.softwood_frac,
      sf.LAT, sf.LON, sf.ELEV,
      tt.prev_plt_cn
  FROM tree_trans tt
  JOIN stand_feat sf ON sf.PLT_CN = tt.prev_plt_cn
  JOIN PLOT cur ON cur.CN = tt.cur_plt_cn
) TO '{OUT}/tree_dataset.parquet' (FORMAT parquet)
""")
r = con.sql(f"SELECT count(*) n, sum(label) pos, avg(label) rate FROM '{OUT}/tree_dataset.parquet'").fetchone()
print(f"TREE dataset: {r[0]:,} rows, {r[1]:,} positives ({r[2]*100:.3f}%)  [{time.time()-t:.1f}s]")
print("\nDone.")
