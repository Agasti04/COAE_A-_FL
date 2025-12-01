# src/Fedprox/utils.py
import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

META_COLS = [
    'userid', 'day_period_evening', 'day_period_morning', 'day_period_night',
    'day_period_noon', 'start_interval', 'end_interval', 'timestamp', 'time_diff_hours'
]

# -------------------------
# Basic file / data helpers
# -------------------------
def _ensure_folder(path):
    """Create folder if not exists (idempotent)."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # best-effort; ignore failures here, caller will handle write errors
        pass

def list_client_files(data_folder):
    return sorted(glob.glob(os.path.join(data_folder, "*.csv")))

def read_client_df(path):
    return pd.read_csv(path)

def compute_availability_vector(df, exclude_cols=META_COLS + ['label']):
    cols = [c for c in df.columns if c not in exclude_cols]
    avail = {c: int(df[c].notna().mean() > 0.5) for c in cols}
    return avail

def aggregate_availability(avail_list):
    keys = sorted({k for d in avail_list for k in d.keys()})
    agg = {}
    for k in keys:
        vals = [d.get(k, 0) for d in avail_list]
        agg[k] = int(all(v == 1 for v in vals))
    shared = [k for k, v in agg.items() if v == 1]
    return shared, agg

def split_features(all_cols, shared_features):
    shared = list(shared_features)
    local = [c for c in all_cols if c not in shared and c not in META_COLS + ['label']]
    return shared, local

def group_kfold_splits(df, userid_col='userid', k=5, seed=42):
    gkf = GroupKFold(n_splits=k)
    X = df.index.values
    groups = df[userid_col].values
    splits = []
    for train_idx, test_idx in gkf.split(X, groups=groups):
        splits.append((train_idx, test_idx))
    return splits

# -------------------------
# Metrics
# -------------------------
def metrics_from_preds_multiclass(y_true, y_pred, y_proba=None, average="macro", auroc_multiclass="ovr"):
    # y_true: array-like of labels 0..C-1
    # y_pred: array-like predicted labels 0..C-1
    # y_proba: ndarray (n_samples, n_classes) or None
    try:
        acc = float(accuracy_score(y_true, y_pred))
    except Exception:
        acc = 0.0
    try:
        f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    except Exception:
        f1 = 0.0
    auroc = None
    if y_proba is not None:
        try:
            # roc_auc_score requires y_true contains at least two classes
            auroc = float(roc_auc_score(y_true, y_proba, multi_class=auroc_multiclass))
        except Exception:
            auroc = None
    return {"accuracy": acc, "f1": f1, "auroc": auroc}

# -------------------------
# Round parsing (robust)
# -------------------------
def parse_round_from_config(config):
    """
    Robustly extract an integer round index from various config shapes.
    Returns an int (>=0) when found, else returns -1 as sentinel.

    Tries keys: "round", "rnd", "round_idx", "roundNumber", "server_round",
    then searches any digits inside the string representation.
    """
    if config is None:
        return -1

    # direct keys to try in order
    for k in ("round", "rnd", "round_idx", "roundNumber", "server_round"):
        if isinstance(config, dict) and (k in config):
            v = config.get(k)
            try:
                if isinstance(v, (int, float)):
                    return int(v)
                if isinstance(v, str) and v.strip() != "":
                    return int(float(v.strip()))
            except Exception:
                pass

    # fallback: search any string-like values in config for digits
    try:
        s = str(config)
        m = re.search(r"(\d+)", s)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    return -1

# -------------------------
# Per-country CSV writer (UPSERT)
# -------------------------
def save_round_client_csv(results_folder, round_idx, country_name, row):
    """
    Robust upsert that ensures:
      - No -1 rounds are written. If round_idx is unknown/non-numeric, assign next available
        non-negative integer for that country's CSV.
      - Replace any previous rows that used negative rounds for the same client.
      - Keep one row per (round, client): latest wins.
      - Best-effort writes; won't raise to caller.
    """
    import os
    import pandas as pd
    try:
        _ensure_folder(results_folder)
    except Exception:
        try:
            os.makedirs(results_folder, exist_ok=True)
        except Exception:
            pass

    country_path = os.path.join(results_folder, f"{country_name}.csv")

    # coerce incoming round to int if possible; negative or non-coercible -> sentinel -1
    try:
        round_clean = int(float(round_idx))
    except Exception:
        round_clean = -1

    # prepare row dict
    row_with_round = dict(row)
    # temporarily store original requested round for diagnostics (optional)
    row_with_round["_requested_round"] = row_with_round.get("round", row_with_round.get("req_round", None))
    # ensure a client field
    if "client" not in row_with_round or row_with_round.get("client") in (None, ""):
        row_with_round["client"] = country_name

    # Load existing CSV if present (best-effort)
    try:
        if os.path.exists(country_path):
            existing = pd.read_csv(country_path)
        else:
            existing = pd.DataFrame()
    except Exception:
        existing = pd.DataFrame()

    # If round_clean is negative, determine next round index to use
    if round_clean < 0:
        try:
            if ("round" in existing.columns) and (not existing.empty):
                # consider only non-negative numeric rounds
                rounds_numeric = pd.to_numeric(existing["round"], errors="coerce")
                nonneg = rounds_numeric[rounds_numeric.notna() & (rounds_numeric >= 0)]
                if len(nonneg) > 0:
                    next_round = int(nonneg.max()) + 1
                else:
                    next_round = 0
            else:
                next_round = 0
        except Exception:
            next_round = 0
        round_clean = next_round

    # set the resolved round in the row to store
    row_with_round["round"] = int(round_clean)

    # Remove any prior rows for same client that used negative rounds (cleanup)
    try:
        if not existing.empty and "client" in existing.columns:
            # rows with same client and round < 0
            try:
                rounds_existing = pd.to_numeric(existing["round"], errors="coerce")
                neg_mask = rounds_existing.isna() | (rounds_existing < 0)
            except Exception:
                # fallback: treat non-numeric as negative-like
                neg_mask = pd.Series([False]*len(existing))
                for i, v in enumerate(existing.get("round", [])):
                    try:
                        if int(float(v)) < 0:
                            neg_mask.iloc[i] = True
                    except Exception:
                        neg_mask.iloc[i] = True
            same_client_mask = existing["client"] == row_with_round["client"]
            drop_mask = neg_mask & same_client_mask
            if drop_mask.any():
                existing = existing.loc[~drop_mask].reset_index(drop=True)
    except Exception:
        # ignore cleanup failures
        pass

    # Build ordered row
    ordered_keys = ["round", "userid", "split_mode", "accuracy", "loss", "f1", "auroc", "client"]
    ordered_row = {}
    for k in ordered_keys:
        ordered_row[k] = row_with_round.get(k, None)
    for k in sorted(row_with_round.keys()):
        if k not in ordered_row:
            ordered_row[k] = row_with_round[k]

    # Upsert: drop any existing row with same (round, client) then append
    try:
        if existing.empty:
            final_df = pd.DataFrame([ordered_row])
        else:
            # ensure columns present
            for c in ordered_row.keys():
                if c not in existing.columns:
                    existing[c] = pd.NA

            # create mask for same (round, client)
            try:
                mask_round = pd.to_numeric(existing["round"], errors="coerce") == float(ordered_row["round"])
            except Exception:
                mask_round = existing["round"] == ordered_row["round"]
            mask_client = existing["client"] == ordered_row["client"]
            mask_same = mask_round & mask_client
            if mask_same.any():
                existing = existing.loc[~mask_same]
            # append
            new_df = pd.DataFrame([ordered_row])
            final_df = pd.concat([existing, new_df], ignore_index=True, sort=False)
            # reorder columns: ordered_keys first, then others
            remaining = [c for c in final_df.columns if c not in ordered_row.keys()]
            final_df = final_df.loc[:, list(ordered_row.keys()) + remaining]
    except Exception:
        final_df = pd.DataFrame([ordered_row])

    # Save final CSV (atomic write attempt)
    try:
        tmp_path = country_path + ".tmp"
        final_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, country_path)
    except Exception:
        try:
            final_df.to_csv(country_path, index=False)
        except Exception:
            try:
                final_df.to_csv(country_path.replace(".csv", "_err.csv"), index=False)
            except Exception:
                pass


# -------------------------
# Summary builders
# -------------------------
def build_summary_all_rounds(results_folder, out_name="summary_all_rounds.csv"):
    """
    Read all per-country CSVs and produce a deduplicated 'summary_all_rounds.csv'
    containing every row (one per round+client) across countries.
    Returns path or None on failure.
    """
    pattern = os.path.join(results_folder, "*.csv")
    files = [f for f in glob.glob(pattern) if not os.path.basename(f).startswith("summary")]
    if not files:
        return None

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            if "client" not in df.columns:
                df["client"] = os.path.splitext(os.path.basename(f))[0]
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True, sort=False)

    # deduplicate by (round, client) keeping last (should be the latest upsert result)
    if "round" in all_df.columns and "client" in all_df.columns:
        try:
            all_df = all_df.drop_duplicates(subset=["round", "client"], keep="last")
        except Exception:
            # fallback: keep as-is
            pass

    out_path = os.path.join(results_folder, out_name)
    try:
        all_df.to_csv(out_path, index=False)
        return out_path
    except Exception:
        try:
            all_df.to_csv(out_path.replace(".csv", "_err.csv"), index=False)
            return out_path.replace(".csv", "_err.csv")
        except Exception:
            return None

# -------------------------
# Torch helpers and misc
# -------------------------
def torch_state_dict_to_ndarrays(state_dict):
    arrays = []
    for k, v in state_dict.items():
        arrays.append(v.cpu().numpy())
    return arrays

def ndarrays_to_torch_state_dict(param_list, reference_state_dict):
    import torch
    state = {}
    keys = list(reference_state_dict.keys())
    assert len(keys) == len(param_list)
    for k, arr in zip(keys, param_list):
        state[k] = torch.tensor(arr)
    return state

def free_gpu():
    import torch, gc
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

from collections import defaultdict

def stratified_group_kfold(df, group_col='userid', label_col='label', n_splits=5, seed=42):
    """
    Stratified + Group K-Fold:
    - Keeps ALL rows of the same userid in the same split.
    - Ensures each fold has similar class distribution.
    """
    rng = np.random.RandomState(seed)

    # Unique groups (unique userids)
    groups = df[group_col].unique()

    # Compute per-group label distribution
    group_to_labels = {}
    for g in groups:
        labels = df[df[group_col] == g][label_col].values
        counts = np.bincount(labels, minlength=df[label_col].nunique())
        group_to_labels[g] = counts

    # Sort groups by total size (largest groups first)
    groups_sorted = sorted(groups, key=lambda g: group_to_labels[g].sum(), reverse=True)

    # Prepare folds
    fold_counts = [np.zeros(df[label_col].nunique(), dtype=int) for _ in range(n_splits)]
    fold_groups = [[] for _ in range(n_splits)]

    # Greedy assignment of groups to best fold (keeps stratification)
    for g in groups_sorted:
        best_fold = None
        best_score = None
        counts = group_to_labels[g]
        for f in range(n_splits):
            temp = fold_counts[f] + counts
            score = temp.std()
            if (best_score is None) or (score < best_score):
                best_score = score
                best_fold = f
        fold_groups[best_fold].append(g)
        fold_counts[best_fold] += counts

    # Convert group folds into row index folds
    splits = []
    for f in range(n_splits):
        test_groups = fold_groups[f]
        test_idx = df[df[group_col].isin(test_groups)].index.values
        train_idx = df[~df[group_col].isin(test_groups)].index.values
        splits.append((train_idx, test_idx))

    return splits

def apply_smote(X_shared, X_local, y, k_neighbors=5):
    """
    Apply SMOTE separately on the shared and local feature spaces.
    Works only on training data. Returns new (X_shared, X_local, y).
    """
    from imblearn.over_sampling import SMOTE
    import numpy as _np

    X_combined = _np.hstack([X_shared, X_local])
    sm = SMOTE(k_neighbors=k_neighbors)
    X_resampled, y_resampled = sm.fit_resample(X_combined, y)
    shared_dim = X_shared.shape[1]
    Xs_new = X_resampled[:, :shared_dim]
    Xl_new = X_resampled[:, shared_dim:]
    return Xs_new, Xl_new, y_resampled

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights (tensor or None)
        self.gamma = gamma

    def forward(self, logits, targets):
        device = logits.device
        if self.alpha is not None:
            alpha = self.alpha.to(device)
        else:
            alpha = None
        ce_loss = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()

def compute_class_weights(labels, num_classes=3):
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum()
    return weights.astype(np.float32)

def save_summary_all_rounds(results_folder, rows=None, out_name="summary_all_rounds.csv"):
    import os
    path = os.path.join(results_folder, out_name)
    try:
        if rows is not None:
            # Write passed-in rows (same behavior as old function)
            pd.DataFrame(rows).to_csv(path, index=False)
            return path
        else:
            # No rows provided: attempt to build from per-country CSVs
            return build_summary_all_rounds(results_folder, out_name=out_name)
    except Exception:
        try:
            # fallback: attempt writing to error file
            if rows is not None:
                pd.DataFrame(rows).to_csv(path.replace(".csv", "_err.csv"), index=False)
                return path.replace(".csv", "_err.csv")
        except Exception:
            pass
    return None
