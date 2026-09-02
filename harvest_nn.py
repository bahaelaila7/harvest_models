#!/usr/bin/env python3
"""
Two-layer MLP (one hidden layer, sigmoid output) for harvest readiness.

Trains and validates two models on the datasets from build_dataset.py:
  * STAND:  p(cut stand | stand status at prior visit)
  * TREE :  p(cut tree  | tree status + stand status at prior visit)

Design notes
------------
* Weighted binary cross-entropy handles the class imbalance (14% / 1.8% cut)
  so the network learns to rank rare positives instead of predicting all-negative.
* Because class weighting distorts the output scale, predicted probabilities are
  recalibrated with isotonic regression fit on a held-out validation split, so
  reported probabilities are honest (low Brier, diagonal calibration curve).
* Train/val/test are split by PLOT (prev_plt_cn) so a plot's trees never straddle
  the split — no stand-feature leakage inflating the tree model.
* Metrics: ROC-AUC & average precision (ranking, threshold-free), Brier score &
  a decile calibration table (probability quality), against the base-rate baseline.
"""
import numpy as np, duckdb, json, os, sys
from sklearn.model_selection import GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)

def sigmoid(z):
    """Numerically stable logistic."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

# ----------------------------------------------------------------------------
# Feature specification per model
# ----------------------------------------------------------------------------
# Stand status at prior visit: age, stocking, live basal area, forest type
# (dominant-species family), ownership, productivity, structure, location.
STAND_NUM = ["remper","LAT","LON","ELEV","prev_measyear","STDAGE","BALIVE","SLOPE",
             "ALSTKCD","GSSTKCD","forest_prop","agb_per_acre","tpa","qmd","max_dia",
             "n_species","softwood_frac","SITECLCD","STDSZCD"]
STAND_MISS = ["STDAGE","BALIVE","agb_per_acre","qmd","SITECLCD","ALSTKCD","GSSTKCD","SLOPE"]
STAND_ONEHOT = {"OWNGRPCD":[10,20,30,40], "STDORGCD":[0,1],
                "fortyp_grp":[1,2,3,4,5,6,7,8,9]}

# Tree status at prior visit: maturity (age, DBH, height, crown ratio), crown class,
# tree class, damage, softwood; PLUS the full stand-status block above.
TREE_NUM = ["remper","dia","ht","cr","drybio_ag","tpa_unadj","totage","n_damage",
            "STDAGE","BALIVE","SLOPE","ALSTKCD","GSSTKCD","agb_per_acre","stand_tpa",
            "qmd","max_dia","n_species","softwood_frac","LAT","LON","ELEV","SITECLCD","STDSZCD"]
TREE_MISS = ["ht","cr","totage","drybio_ag","STDAGE","BALIVE","SITECLCD","ALSTKCD"]
TREE_ONEHOT = {"CCLCD":[1,2,3,4,5], "TREECLCD":[2,3,4], "OWNGRPCD":[10,20,30,40],
               "STDORGCD":[0,1], "softwood":[0,1], "any_damage":[0,1],
               "fortyp_grp":[1,2,3,4,5,6,7,8,9]}

# Model 3 (LANDIS-deployable): features derivable ONLY from stand biomass cohorts
# (species / age / biomass) -- no FIA-only fields. Deployable on LANDIS outputs.
COHORT_NUM = ["remper","agb_density","ln_agb","n_cohorts","n_species",
              "mean_age","max_age","min_age","sd_age","dom_cohort_age",
              "agb_f_0_10","agb_f_10_20","agb_f_20_40","agb_f_40_80","agb_f_80p",
              "softwood_frac","dom_species_share","species_hhi"]
COHORT_MISS = ["remper"]
COHORT_ONEHOT = {}

def load(name, num, miss, onehot, group_col):
    """Load parquet, build feature matrix X, label y, group ids, and feature names."""
    cols = list(dict.fromkeys(num + list(onehot) + ["label", group_col]))
    # duckdb resolves column names case-insensitively; alias TREECLCD/CCLCD which
    # live under different casings in the two files.
    df = duckdb.sql(f"SELECT * FROM '{DATA}/{name}'").df()
    df.columns = [c.lower() for c in df.columns]
    y = df["label"].to_numpy(np.float32)
    groups = df[group_col.lower()].to_numpy()
    feats, names = [], []
    # numeric: median impute (+ missingness indicator for heavy-null cols)
    for c in num:
        v = df[c.lower()].to_numpy(np.float64)
        m = np.isnan(v)
        med = np.nanmedian(v) if not np.all(m) else 0.0
        v = np.where(m, med, v)
        feats.append(v); names.append(c)
        if c in miss:
            feats.append(m.astype(np.float64)); names.append(c + "_isna")
    # one-hot categoricals (explicit level set; unknown/na -> all zeros)
    for c, levels in onehot.items():
        v = df[c.lower()].to_numpy(np.float64)
        for lv in levels:
            feats.append((v == lv).astype(np.float64)); names.append(f"{c}={lv}")
    X = np.column_stack(feats).astype(np.float64)
    return X, y, groups, names

# ----------------------------------------------------------------------------
# Two-layer MLP: X -> Linear(H) -> ReLU -> Linear(1) -> sigmoid, Adam + weighted BCE
# ----------------------------------------------------------------------------
class MLP:
    def __init__(self, d_in, hidden=64, lr=1e-3, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.standard_normal((d_in, hidden)) * np.sqrt(2.0/d_in)
        self.b1 = np.zeros(hidden)
        self.W2 = r.standard_normal((hidden, 1)) * np.sqrt(2.0/hidden)
        self.b2 = np.zeros(1)
        self.lr = lr
        self._m = {k: np.zeros_like(v) for k,v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k,v in self._params().items()}
        self._t = 0

    def _params(self):
        return {"W1":self.W1,"b1":self.b1,"W2":self.W2,"b2":self.b2}

    def logits(self, X):
        self._h_pre = X @ self.W1 + self.b1
        self._h = np.maximum(self._h_pre, 0.0)
        return (self._h @ self.W2 + self.b2).ravel()

    def predict_proba(self, X, bs=100000):
        out = np.empty(len(X))
        for i in range(0, len(X), bs):
            z = self.logits(X[i:i+bs])
            out[i:i+bs] = sigmoid(z)
        return out

    def _adam(self, grads):
        self._t += 1; b1,b2,eps = 0.9,0.999,1e-8
        for k,g in grads.items():
            self._m[k] = b1*self._m[k] + (1-b1)*g
            self._v[k] = b2*self._v[k] + (1-b2)*(g*g)
            mh = self._m[k]/(1-b1**self._t); vh = self._v[k]/(1-b2**self._t)
            self._params()[k] -= self.lr * mh/(np.sqrt(vh)+eps)

    def fit(self, X, y, w, epochs=12, bs=4096, seed=0):
        r = np.random.default_rng(seed); n = len(X)
        for ep in range(epochs):
            idx = r.permutation(n)
            for s in range(0, n, bs):
                b = idx[s:s+bs]; Xb,yb,wb = X[b],y[b],w[b]
                z = self.logits(Xb)
                p = sigmoid(z)
                # d/dz of weighted BCE = w*(p - y)
                dz = (wb*(p - yb))/len(b)
                dW2 = self._h.T @ dz[:,None]
                db2 = dz.sum(keepdims=True)
                dh = dz[:,None] @ self.W2.T
                dh[self._h_pre <= 0] = 0.0
                dW1 = Xb.T @ dh
                db1 = dh.sum(0)
                self._adam({"W1":dW1,"b1":db1,"W2":dW2,"b2":db2})

def standardize(Xtr, *others):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0] = 1.0
    return [( (X-mu)/sd ) for X in (Xtr,)+others]

def decile_calibration(y, p, bins=10):
    order = np.argsort(p); y, p = y[order], p[order]
    out = []
    for chunk_y, chunk_p in zip(np.array_split(y,bins), np.array_split(p,bins)):
        out.append((float(chunk_p.mean()), float(chunk_y.mean()), len(chunk_y)))
    return out

def run(name, parquet, num, miss, onehot, group_col, hidden, epochs, tag):
    print(f"\n{'='*70}\n{tag}\n{'='*70}", flush=True)
    X, y, groups, names = load(parquet, num, miss, onehot, group_col)
    print(f"rows={len(y):,}  features={X.shape[1]}  positives={int(y.sum()):,} ({y.mean()*100:.3f}%)")

    # plot-grouped 70/15/15 split
    tr, tmp = next(GroupShuffleSplit(1, test_size=0.30, random_state=0).split(X,y,groups))
    g2 = groups[tmp]
    va_rel, te_rel = next(GroupShuffleSplit(1, test_size=0.50, random_state=1).split(X[tmp],y[tmp],g2))
    va, te = tmp[va_rel], tmp[te_rel]
    Xtr,Xva,Xte = standardize(X[tr], X[va], X[te])
    ytr,yva,yte = y[tr],y[va],y[te]

    # balanced sample weights (mean ~1): w_pos = 0.5 N/Npos, w_neg = 0.5 N/Nneg
    npos, nneg = ytr.sum(), len(ytr)-ytr.sum()
    w = np.where(ytr==1, 0.5*len(ytr)/npos, 0.5*len(ytr)/nneg)

    net = MLP(Xtr.shape[1], hidden=hidden, lr=1e-3, seed=0)
    net.fit(Xtr, ytr, w, epochs=epochs, bs=4096, seed=0)

    # isotonic recalibration on validation, then evaluate on test
    pva_raw = net.predict_proba(Xva)
    iso = IsotonicRegression(out_of_bounds="clip").fit(pva_raw, yva)
    pte_raw = net.predict_proba(Xte)
    pte = iso.predict(pte_raw)

    res = {
        "model": tag, "n_rows": int(len(y)), "n_features": X.shape[1],
        "base_rate": float(y.mean()),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "roc_auc": float(roc_auc_score(yte, pte_raw)),          # ranking unaffected by calibration
        "avg_precision": float(average_precision_score(yte, pte_raw)),
        "brier_raw": float(brier_score_loss(yte, pte_raw)),
        "brier_calibrated": float(brier_score_loss(yte, pte)),
        "brier_baseline": float(brier_score_loss(yte, np.full_like(yte, yte.mean()))),
        "calibration_deciles": decile_calibration(yte, pte),
        "features": names,
    }
    print(f"test ROC-AUC        : {res['roc_auc']:.4f}")
    print(f"test avg-precision  : {res['avg_precision']:.4f}   (baseline={yte.mean():.4f})")
    print(f"test Brier (calib)  : {res['brier_calibrated']:.5f}  (baseline={res['brier_baseline']:.5f})")
    print("calibration (pred -> obs, n):")
    for pp,oo,nn in res["calibration_deciles"]:
        print(f"   {pp:6.3f} -> {oo:6.3f}  (n={nn:,})")

    np.savez(os.path.join(OUT, f"{name}_model.npz"),
             W1=net.W1,b1=net.b1,W2=net.W2,b2=net.b2,
             feat_names=np.array(names), iso_x=iso.X_thresholds_, iso_y=iso.y_thresholds_)
    with open(os.path.join(OUT, f"{name}_metrics.json"),"w") as f:
        json.dump(res, f, indent=2)
    return res

if __name__ == "__main__":
    results = {}
    results["stand"] = run("stand","stand_dataset.parquet", STAND_NUM, STAND_MISS,
                           STAND_ONEHOT, "prev_plt_cn", hidden=64, epochs=15,
                           tag="STAND  p(cut stand | stand status)")
    results["tree"] = run("tree","tree_dataset.parquet", TREE_NUM, TREE_MISS,
                          TREE_ONEHOT, "prev_plt_cn", hidden=96, epochs=8,
                          tag="TREE   p(cut tree | tree status, stand status)")
    results["cohort"] = run("cohort","cohort_dataset.parquet", COHORT_NUM, COHORT_MISS,
                          COHORT_ONEHOT, "prev_plt_cn", hidden=48, epochs=15,
                          tag="COHORT p(cut stand | biomass cohorts)  [LANDIS-deployable]")
    with open(os.path.join(OUT,"summary.json"),"w") as f:
        json.dump({k:{m:v[m] for m in ["roc_auc","avg_precision","base_rate",
                   "brier_calibrated","brier_baseline"]} for k,v in results.items()}, f, indent=2)
    print("\nSaved models + metrics to", OUT)
