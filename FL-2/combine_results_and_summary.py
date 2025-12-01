#!/usr/bin/env python3
"""
overall_round_summary.py

Combine FL CSV results and produce a single per-country aggregated CSV
containing means of accuracy, f1, loss, auroc.

Usage:
    python src/Fedprox/overall_round_summary.py --results-folder results/results_fedprox
"""

import argparse
import glob
import os
import tempfile
import pandas as pd
import numpy as np
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        logger.info("Loaded %s (%d rows, %d cols)", path, df.shape[0], df.shape[1])
        return df
    except Exception:
        logger.exception("Failed to read %s", path)
        return None


def _atomic_write(df: pd.DataFrame, out_path: str):
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=out_dir)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_path)
        logger.info("Wrote %s (%d rows)", out_path, len(df))
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def find_csv_files(results_folder):
    # common result file patterns used by your pipeline
    patterns = [
        os.path.join(results_folder, "all_clients_rounds.csv"),
        os.path.join(results_folder, "round_*_clients*.csv"),
        os.path.join(results_folder, "*_rounds.csv"),
        os.path.join(results_folder, "round_*_clients_eval.csv"),
        os.path.join(results_folder, "*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    # unique, existing
    files = sorted(set([f for f in files if os.path.isfile(f)]))
    return files


def canonical_country_col(df, filename):
    """
    Determine a country/client column. Preference order:
      client_file, client_id, client, country
    If none present, infer from filename.
    Returns dataframe with new column 'country' (string).
    """
    col_candidates = ["client_file", "client_id", "client", "country"]
    for c in col_candidates:
        if c in df.columns:
            out = df[c].astype(str).str.strip()
            return out.fillna("").replace("", None)
    # fallback: infer from filename like "Denmark_overall" or base filename
    base = os.path.splitext(os.path.basename(filename))[0]
    # try to strip "round_\d+_" prefix
    m = re.sub(r"^round[_\-]?\d+[_\-]?", "", base)
    return pd.Series([m] * len(df))


def normalize_metrics(df):
    """
    Ensure numeric columns accuracy, f1, loss, auroc exist if possible.
    Try common alternate names.
    """
    # lowercase column keys
    cols_lower = {c.lower(): c for c in df.columns}
    # mapping of desired -> potential alternatives (in order)
    alt_map = {
        "accuracy": ["accuracy", "acc", "accuracy_score", "acc_score"],
        "f1": ["f1", "f1_score", "f1_macro"],
        "loss": ["loss", "train_loss", "val_loss"],
        "auroc": ["auroc", "auc", "roc_auc"],
    }
    out = pd.DataFrame(index=df.index)
    for target, alts in alt_map.items():
        found = None
        for alt in alts:
            if alt in cols_lower:
                found = cols_lower[alt]
                break
        if found is not None:
            out[target] = pd.to_numeric(df[found], errors="coerce").astype(float)
        else:
            out[target] = np.nan
    return out


def combine_and_aggregate(results_folder, out_path):
    files = find_csv_files(results_folder)
    if not files:
        logger.error("No CSV files found in %s", results_folder)
        return False

    rows = []
    for f in files:
        df = _safe_read_csv(f)
        if df is None:
            continue

        # create country series
        country_series = canonical_country_col(df, f)
        # normalize metrics
        metrics_df = normalize_metrics(df)

        # combine into a minimal DF
        minimal = pd.DataFrame({
            "country": country_series,
            "accuracy": metrics_df["accuracy"],
            "f1": metrics_df["f1"],
            "loss": metrics_df["loss"],
            "auroc": metrics_df["auroc"],
        })

        # drop rows where country is missing
        minimal = minimal[~minimal["country"].isnull()]
        if minimal.empty:
            continue

        # drop rows where all metrics are NaN
        minimal = minimal.dropna(how="all", subset=["accuracy", "f1", "loss", "auroc"])
        if minimal.empty:
            continue

        rows.append(minimal)

    if not rows:
        logger.error("No usable metric rows found in any CSVs.")
        return False

    combined = pd.concat(rows, ignore_index=True, sort=False)

    # normalize country strings
    combined["country"] = combined["country"].astype(str).str.strip()

    # group by country and compute means (skip NaNs automatically)
    agg = combined.groupby("country", dropna=True).agg({
        "accuracy": "mean",
        "f1": "mean",
        "loss": "mean",
        "auroc": "mean",
    }).reset_index()

    # rename columns to explicit names and round to reasonable precision
    agg = agg.rename(columns={
        "country": "country",
        "accuracy": "accuracy_mean",
        "f1": "f1_mean",
        "loss": "loss_mean",
        "auroc": "auroc_mean",
    })
    # round numeric columns to 6 decimals for neatness
    for c in ["accuracy_mean", "f1_mean", "loss_mean", "auroc_mean"]:
        if c in agg.columns:
            agg[c] = agg[c].round(6)

    # write atomically
    _atomic_write(agg, out_path)
    logger.info("Combined summary written to %s", out_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Produce overall_round_summary.csv by country")
    parser.add_argument("--results-folder", "-r", required=True, help="Path to results folder")
    parser.add_argument("--out-file", "-o", default=None, help="Output CSV path (default: results_folder/overall_round_summary.csv)")
    args = parser.parse_args()

    results_folder = args.results_folder
    if not os.path.isdir(results_folder):
        logger.error("results-folder does not exist: %s", results_folder)
        return

    out_file = args.out_file or os.path.join(results_folder, "overall_round_summary.csv")
    ok = combine_and_aggregate(results_folder, out_file)
    if not ok:
        logger.error("Failed to produce overall summary.")
    else:
        logger.info("Success. Output: %s", out_file)


if __name__ == "__main__":
    main()
