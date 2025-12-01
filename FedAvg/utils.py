"""
utils.py

Utility functions for:
- Defining constants (meta columns, label column, file paths)
- Computing intersection of feature columns across countries
- Loading the common features from JSON
- Creating GroupKFold splits based on userid (leave-n-participant-out)
- Preparing tensors and label indices
- Computing metrics (accuracy, F1, AUROC)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# --------- CONSTANTS (EDIT THESE TO MATCH YOUR DATA) ---------

# Meta columns to KEEP in the DataFrame but EXCLUDE from training features
META_COLS = [
    "userid",
    "day_period_evening",
    "day_period_morning",
    "day_period_night",
    "day_period_noon",
    "start_interval",
    "end_interval",
    "timestamp",
    "time_diff_hours",
]

# Label column name (change if your label column is named differently)
# Here we assume: 1 = positive, 2 = neutral, 3 = negative
LABEL_COL = "label"  # mood label

# Map "country name" to the path of its cleaned CSV file
COUNTRY_FILES: Dict[str, str] = {
    "Denmark": "Data/Denmark_overall.csv",
    "UK": "Data/UK_overall.csv",
    "Mexico": "Data/Mexico_overall.csv",
    "China": "Data/China_overall.csv",
    "Paraguay": "Data/Paraguay_overall.csv",
    "India": "Data/India_overall.csv",
    # Add/remove as needed
}

COMMON_FEATURES_PATH = "common_features.json"


# --------- COMMON FEATURE COMPUTATION ---------

def compute_and_save_common_features() -> List[str]:
    """
    Load all country CSVs, compute the intersection of feature columns
    (excluding meta columns and label column), and save them to JSON.

    Returns the list of common feature column names.
    """
    feature_sets = []

    for country, path in COUNTRY_FILES.items():
        df = pd.read_csv(path)

        # All columns in this country's dataframe
        all_cols = set(df.columns)

        # Remove meta columns (if present) and the label column
        cols_to_exclude = set(META_COLS + [LABEL_COL])
        feature_cols = list(all_cols - cols_to_exclude)

        feature_sets.append(set(feature_cols))
        print(f"{country}: {len(feature_cols)} candidate feature columns.")

    # Intersection of all feature sets
    common_features = sorted(list(set.intersection(*feature_sets)))
    print(f"Found {len(common_features)} common feature columns across all countries.")

    # Save to JSON so server and clients can load the same list
    with open(COMMON_FEATURES_PATH, "w") as f:
        json.dump(common_features, f, indent=2)

    print(f"Saved common features to {COMMON_FEATURES_PATH}")
    return common_features


def load_common_features() -> List[str]:
    """
    Load the list of common feature columns from JSON file.
    """
    path = Path(COMMON_FEATURES_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{COMMON_FEATURES_PATH} not found. "
            "Run compute_and_save_common_features() once before training."
        )
    with open(path, "r") as f:
        common_features = json.load(f)
    return common_features


# --------- GROUPED SPLITS (LEAVE-N-PARTICIPANT-OUT) ---------

def grouped_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "userid",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Single GroupShuffleSplit on the dataframe using 'userid' as groups.

    Approximates "leave-n-participant-out" once:
    - ~20% of unique userids go to test
    - Remaining to train.

    Returns:
        train_df, test_df

    NOTE: Kept for reference / debugging, not used in the k-fold pipeline.
    """
    groups = df[group_col].values
    indices = np.arange(len(df))

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(gss.split(indices, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(
        f"Group split done: "
        f"{len(train_df)} train rows, {len(test_df)} test rows, "
        f"{len(np.unique(train_df[group_col]))} train users, "
        f"{len(np.unique(test_df[group_col]))} test users."
    )
    return train_df, test_df


def make_group_kfold_indices(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "userid",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create GroupKFold splits using 'userid' as the group.

    This implements a k-fold "leave-n-participant-out" strategy:

    - Each fold uses a different subset of userids as test
    - A user appears in exactly one test fold
    - For each fold: remaining users are in train

    Returns:
        List of (train_idx, test_idx) index arrays for each fold.
    """
    groups = df[group_col].values
    indices = np.arange(len(df))
    gkf = GroupKFold(n_splits=n_splits)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(indices, groups=groups)):
        print(
            f"Fold {fold_idx+1}/{n_splits}: "
            f"{len(train_idx)} train rows, {len(test_idx)} test rows, "
            f"{len(np.unique(df.iloc[train_idx][group_col]))} train users, "
            f"{len(np.unique(df.iloc[test_idx][group_col]))} test users."
        )
        folds.append((train_idx, test_idx))

    return folds


# --------- FEATURE / LABEL PREPARATION ---------

def prepare_features_and_labels(
    df: pd.DataFrame,
    common_features: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a dataframe and a list of common feature names, return:
    - X: feature matrix (n_samples, n_features)
    - y_idx: label indices (0..C-1)
    - classes: sorted unique label values as they appear in LABEL_COL

    Assumes LABEL_COL is in df and labels are integers (e.g. 1,2,3).
    """
    # Filter the dataframe to only the common feature columns
    X = df[common_features].values.astype(np.float32)

    # Raw labels (e.g. 1,2,3 for positive, neutral, negative)
    y_raw = df[LABEL_COL].values

    # Map raw labels to indices 0..C-1 for PyTorch CrossEntropy
    classes = np.sort(np.unique(y_raw))
    label_to_index = {lab: i for i, lab in enumerate(classes)}
    y_idx = np.array([label_to_index[lab] for lab in y_raw], dtype=np.int64)

    return X, y_idx, classes


# --------- METRICS ---------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Accuracy, macro-F1, and macro AUROC.

    Args:
        y_true: ground-truth label indices (shape: [N])
        y_pred: predicted label indices (shape: [N])
        y_proba: predicted probabilities (shape: [N, C])

    Returns:
        Dict with 'accuracy', 'f1', 'auroc'
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    # AUROC for multi-class using one-vs-rest
    num_classes = y_proba.shape[1]
    if num_classes > 1:
        try:
            auroc = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            # Fallback if AUROC cannot be computed (e.g. one class present)
            auroc = float("nan")
    else:
        auroc = float("nan")

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "auroc": float(auroc),
    }


# --------- MAIN (OPTIONAL SCRIPT) ---------

if __name__ == "__main__":
    # If you run: python utils.py
    # This will compute and save the common feature intersection once.
    compute_and_save_common_features()
