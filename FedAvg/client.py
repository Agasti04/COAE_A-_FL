"""
client.py

Defines a Flower NumPyClient that:
- Loads the country-specific dataset
- Uses only the intersection (common) features
- Performs k=5 GroupKFold splits by userid (leave-n-participant-out)
- For each FL round r, uses fold (r-1) as the current train/test split
- Trains a local PyTorch model for several epochs
- Evaluates on local test fold and returns Accuracy, F1, AUROC
"""

import argparse
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    COUNTRY_FILES,
    META_COLS,
    LABEL_COL,
    load_common_features,
    make_group_kfold_indices,
    prepare_features_and_labels,
    compute_metrics,
)
from model import MoodNet


# --------- HELPER FUNCTIONS FOR MODEL PARAMS ---------

def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """
    Convert PyTorch model parameters to a list of NumPy arrays
    for Flower to send to the server.
    """
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load Flower parameters (NumPy arrays) into the PyTorch model.
    """
    state_dict = model.state_dict()
    new_state_dict = {}
    for (key, _), param in zip(state_dict.items(), parameters):
        new_state_dict[key] = torch.tensor(param)
    model.load_state_dict(new_state_dict, strict=True)


# --------- FLOWER CLIENT ---------

class CountryClient(fl.client.NumPyClient):
    """
    Flower client representing a single country.

    Responsibilities:
    - Load and hold the country's dataframe
    - Precompute GroupKFold (k=5) splits by userid
    - For each round, pick the fold indicated by the server (using config['round'])
    - Train a local model on the current train fold
    - Evaluate on the current test fold
    """

    def __init__(
        self,
        country: str,
        batch_size: int = 64,
        local_epochs: int = 5,
        lr: float = 1e-3,
        device: str = "cpu",
        n_splits: int = 5,
    ):
        super().__init__()
        self.country = country
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = torch.device(device)
        self.n_splits = n_splits

        # Load common features that all clients must use
        self.common_features = load_common_features()

        # Load this country's CSV
        csv_path = COUNTRY_FILES[country]
        df = pd.read_csv(csv_path)

        # Save full dataframe for reference
        self.df = df

        # Optionally keep meta columns + label for later analysis
        # (Not used in training directly)
        meta_cols_present = [c for c in META_COLS if c in df.columns]
        self.meta_df = df[meta_cols_present + [LABEL_COL]].copy()

        # Prepare full feature matrix and labels once
        X_all, y_all, classes_all = prepare_features_and_labels(
            df, self.common_features
        )
        self.X_all = X_all  # shape: [N, D]
        self.y_all = y_all  # shape: [N]
        self.classes = classes_all  # e.g. [1, 2, 3] for pos/neu/neg

        # In your case, this should be 3 (positive/neutral/negative)
        self.num_classes = len(self.classes)
        self.input_dim = self.X_all.shape[1]

        # Build GroupKFold index splits (leave-n-participant-out, k=5)
        self.folds = make_group_kfold_indices(
            df, n_splits=self.n_splits, group_col="userid"
        )
        self.num_folds = len(self.folds)

        # Initialize train/test loaders for fold 0 (will be updated each round)
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.set_fold(0)

        # Initialize local model
        self.model = MoodNet(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dim=64,
        ).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print(
            f"[{self.country}] Client initialised: "
            f"{self.X_all.shape[0]} total samples, "
            f"{self.input_dim} features, "
            f"{self.num_classes} classes."
        )

    # --------- FOLD HANDLING ---------

    def set_fold(self, fold_idx: int) -> None:
        """
        Set the current train/test datasets and dataloaders
        according to the given fold index (0..k-1).

        The fold uses:
        - Specific subset of userids as test
        - Remaining userids as train
        """
        fold_idx = int(fold_idx) % self.num_folds
        train_idx, test_idx = self.folds[fold_idx]

        X_train = self.X_all[train_idx]
        y_train = self.y_all[train_idx]
        X_test = self.X_all[test_idx]
        y_test = self.y_all[test_idx]

        self.train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        self.test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        print(
            f"[{self.country}] Using fold {fold_idx+1}/{self.num_folds}: "
            f"{len(self.train_dataset)} train samples, "
            f"{len(self.test_dataset)} test samples."
        )

    # --------- Flower NumPyClient Methods ---------

    def get_parameters(self, config: Dict[str, str]):
        """
        Called by Flower to get the current local model parameters.
        """
        return get_model_parameters(self.model)

    def fit(self, parameters, config: Dict[str, str]):
        """
        Called by Flower to:
        - Receive the current global model parameters from the server
        - Select the current fold using 'round' from config
        - Perform local training for a few epochs
        - Return the updated model parameters and the number of training examples
        """
        # Load global parameters into local model
        set_model_parameters(self.model, parameters)

        # Determine which fold to use for this round
        round_number = int(config.get("round", 1))
        fold_idx = (round_number - 1) % self.num_folds
        self.set_fold(fold_idx)

        self.model.train()
        for epoch in range(self.local_epochs):
            total_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * X_batch.size(0)

            avg_loss = total_loss / len(self.train_dataset)
            print(
                f"[{self.country}] Round {round_number}, "
                f"Fold {fold_idx+1}/{self.num_folds}, "
                f"Epoch {epoch+1}/{self.local_epochs}, "
                f"Train loss: {avg_loss:.4f}"
            )

        # Return updated parameters + number of examples
        return get_model_parameters(self.model), len(self.train_dataset), {}

    def evaluate(self, parameters, config: Dict[str, str]):
        """
        Called by Flower at the end of each round to:
        - Load the latest global parameters
        - Use the same fold for evaluation as in fit (based on 'round')
        - Evaluate on this client's local test fold
        - Return loss, number of examples, and metrics (accuracy, F1, AUROC)
        """
        set_model_parameters(self.model, parameters)

        # Ensure we use the same fold index as in fit for this round
        round_number = int(config.get("round", 1))
        fold_idx = (round_number - 1) % self.num_folds
        self.set_fold(fold_idx)

        self.model.eval()

        all_logits = []
        all_labels = []

        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item() * X_batch.size(0)

                all_logits.append(logits.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        avg_loss = total_loss / len(self.test_dataset)

        # Concatenate all predictions and labels
        logits = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_labels, axis=0)

        # Compute probabilities and predicted labels
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        y_pred = probs.argmax(axis=1)

        metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_proba=probs)

        print(
            f"[{self.country}] Round {round_number}, Fold {fold_idx+1}/{self.num_folds} Eval -> "
            f"loss: {avg_loss:.4f}, "
            f"acc: {metrics['accuracy']:.4f}, "
            f"f1: {metrics['f1']:.4f}, "
            f"auroc: {metrics['auroc']:.4f}"
        )

        # Flower expects (loss, num_examples, metrics_dict)
        metrics["loss"] = float(avg_loss)

        # ----------------------------------------------------------
# SAVE ALL CLIENT PREDICTIONS INTO ONE GLOBAL FILE (FEDAVG)
# ----------------------------------------------------------
        # ----------------------------------------------------------
# SAFE: Append predictions without duplicating header
# ----------------------------------------------------------
        import csv, os

        out_file = "fedavg_predictions.csv"

# Write header once
        write_header = not os.path.exists(out_file)

        with open(out_file, "a", newline="") as f:
           writer = csv.writer(f)
           if write_header:
               writer.writerow(["country", "y_true", "y_pred"])

    # Append predictions
           for yt, yp in zip(y_true, y_pred):
               writer.writerow([self.country, int(yt), int(yp)])

        print(f"[{self.country}] Appended predictions to {out_file}")
# ----------------------------------------------------------



        return float(avg_loss), len(self.test_dataset), metrics


# --------- MAIN: START A SINGLE CLIENT PROCESS ---------

def main():
    """
    Parse arguments, construct a CountryClient, and start it with Flower.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country name, e.g. Denmark, UK, Mexico (must match keys in COUNTRY_FILES).",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="0.0.0.0:8080",
        help="Flower server address (host:port).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use, e.g. 'cpu' or 'cuda'.",
    )
    args = parser.parse_args()

    client = CountryClient(
        country=args.country,
        batch_size=64,
        local_epochs=5,   # <- 5 epochs
        lr=1e-3,
        device=args.device,
        n_splits=5,       # <- k=5 folds
    )

    # Start the client and connect to the server
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
