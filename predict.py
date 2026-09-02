#!/usr/bin/env python3
"""
Score a cohort (or stand / tree) table with a trained harvest model.

Loads a self-contained `*_model.npz` (weights + standardization + preprocessing
spec + isotonic calibrator) and emits a calibrated cut probability per input row.
The feature encoding is reproduced via `harvest_nn.build_features`, the exact
function used at training time, so scoring matches the validated model.

Typical LANDIS use — score simulated stand cohorts with model 3:

    python predict.py --model outputs/cohort_model.npz \
        --input landis_cohort_summary.parquet \
        --output preds.parquet --id-cols cell_id,timestep

The input table must contain the model's feature columns (see the spec inside the
npz / the README feature tables); a `label` column is not required. Extra columns
are ignored except those named in --id-cols, which are copied to the output.
"""
import argparse, json, os, sys
import numpy as np, duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvest_nn import build_features, sigmoid   # single source of truth for encoding


def load_model(path):
    d = np.load(path, allow_pickle=False)
    spec = json.loads(str(d["spec"]))
    # json turns int one-hot levels into ints already; keys stay column names
    onehot = {k: list(v) for k, v in spec["onehot"].items()}
    return {
        "W1": d["W1"], "b1": d["b1"], "W2": d["W2"], "b2": d["b2"],
        "mu": d["mu"], "sd": d["sd"], "iso_x": d["iso_x"], "iso_y": d["iso_y"],
        "num": spec["num"], "miss": spec["miss"], "onehot": onehot,
        "medians": spec["medians"], "feat_names": spec["feat_names"],
    }


def predict(m, df):
    """df: DataFrame with lower-cased columns. Returns calibrated probabilities."""
    need = set(m["num"]) | set(m["onehot"])
    have = set(df.columns)
    missing = sorted({c.lower() for c in need} - have)
    if missing:
        raise SystemExit(f"ERROR: input is missing required feature columns: {missing}")
    X, names = build_features(df, m["num"], m["miss"], m["onehot"], m["medians"])
    if names != m["feat_names"]:
        raise SystemExit("ERROR: feature layout does not match the model spec.")
    Xs = (X - m["mu"]) / m["sd"]
    h = np.maximum(Xs @ m["W1"] + m["b1"], 0.0)          # ReLU hidden
    z = (h @ m["W2"] + m["b2"]).ravel()
    p_raw = sigmoid(z)
    return np.interp(p_raw, m["iso_x"], m["iso_y"])       # isotonic recalibration (clipped)


def main():
    ap = argparse.ArgumentParser(description="Score a cohort table with a harvest model.")
    ap.add_argument("--model", required=True, help="path to a *_model.npz")
    ap.add_argument("--input", required=True, help="parquet or csv table to score")
    ap.add_argument("--output", required=True, help="output path (.parquet or .csv)")
    ap.add_argument("--id-cols", default="", help="comma-separated id columns to carry through")
    args = ap.parse_args()

    m = load_model(args.model)
    df = duckdb.sql(f"SELECT * FROM '{args.input}'").df()
    df.columns = [c.lower() for c in df.columns]

    p = predict(m, df)

    id_cols = [c.strip().lower() for c in args.id_cols.split(",") if c.strip()]
    out = df[id_cols].copy() if id_cols else df.iloc[:, :0].copy()
    out["p_cut"] = p

    duckdb.register("out_tbl", out)
    fmt = "csv" if args.output.lower().endswith(".csv") else "parquet"
    duckdb.sql(f"COPY out_tbl TO '{args.output}' (FORMAT {fmt})")
    print(f"scored {len(p):,} rows -> {args.output}  (mean p_cut={p.mean():.4f}, "
          f"p90={np.quantile(p,0.9):.4f}, max={p.max():.4f})")


if __name__ == "__main__":
    main()
