# src/client.py
import os
import traceback
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
from .model import SharedTower, LocalTower, Combiner
from .utils import (
    read_client_df,
    group_kfold_splits,
    metrics_from_preds_multiclass,
    torch_state_dict_to_ndarrays,
    ndarrays_to_torch_state_dict,
    save_round_client_csv,
    META_COLS,
    free_gpu,
    stratified_group_kfold, FocalLoss, compute_class_weights,save_round_client_csv, parse_round_from_config, _ensure_folder,
)
from torch.utils.data import WeightedRandomSampler

# -----------------------
# Logging config (minimal output by default)
# -----------------------
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FedProx proximal hyperparameter (tune this; set to 0.0 to disable)
MU = 0.01
SHARED_EMBED_DIM = 64
LOCAL_OUT_DIM = 64
NUM_CLASSES = 3  # classes 0,1,2 internally

class TowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, data_path: str, shared_feature_list=None,
                 results_folder="results", local_epochs=5, batch_size=64, lr=1e-3):
        self.cid = cid  # country name
        self.data_path = data_path
        self.df = read_client_df(data_path)
        self.results_folder = results_folder
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.shared_features = list(shared_feature_list) if shared_feature_list is not None else []
        self.all_feature_cols = [c for c in self.df.columns if c not in META_COLS + ["label"]]
        self.local_features = [c for c in self.all_feature_cols if c not in self.shared_features]

        # labels: keep three classes 1,2,3 -> convert to 0,1,2 internally
        self.df["label"] = pd.to_numeric(self.df["label"], errors="coerce").fillna(1).astype(int)
        # map 1->0, 2->1, 3->2
        self.df["label"] = self.df["label"].map({1: 0, 2: 1, 3: 2}).fillna(0).astype(int)

        # standardization stats for features
        self.shared_means = {}
        self.shared_stds = {}
        for c in self.shared_features:
            col = pd.to_numeric(self.df[c], errors="coerce").fillna(0.0).astype(float)
            m = float(col.mean())
            s = float(col.std()) if float(col.std()) > 0 else 1.0
            self.shared_means[c] = m
            self.shared_stds[c] = s

        self.local_means = {}
        self.local_stds = {}
        for c in self.local_features:
            col = pd.to_numeric(self.df[c], errors="coerce").fillna(0.0).astype(float)
            m = float(col.mean())
            s = float(col.std()) if float(col.std()) > 0 else 1.0
            self.local_means[c] = m
            self.local_stds[c] = s

        # models
        if len(self.shared_features) == 0:
            class DummyShared(nn.Module):
                def __init__(self, out_dim):
                    super().__init__()
                    self.out_dim = out_dim
                def forward(self, x):
                    batch = x.shape[0]
                    return torch.zeros(batch, self.out_dim, device=x.device)
            self.shared_model = DummyShared(SHARED_EMBED_DIM).to(DEVICE)
        else:
            self.shared_model = SharedTower(len(self.shared_features), embed_dim=SHARED_EMBED_DIM).to(DEVICE)

        if len(self.local_features) == 0:
            class DummyLocal(nn.Module):
                def __init__(self, shared_dim, out_dim):
                    super().__init__()
                    self.net = nn.Sequential(nn.Linear(shared_dim, 64), nn.ReLU(), nn.Linear(64, out_dim), nn.ReLU())
                def forward(self, local_x, shared_rep_detached):
                    return self.net(shared_rep_detached)
            self.local_model = DummyLocal(shared_dim=SHARED_EMBED_DIM, out_dim=LOCAL_OUT_DIM).to(DEVICE)
        else:
            self.local_model = LocalTower(len(self.local_features), shared_dim=SHARED_EMBED_DIM, out_dim=LOCAL_OUT_DIM).to(DEVICE)

        self.combiner = Combiner(shared_dim=SHARED_EMBED_DIM, local_dim=LOCAL_OUT_DIM, num_classes=NUM_CLASSES).to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()
        labels_np = self.df["label"].values
        cw = compute_class_weights(labels_np, num_classes=NUM_CLASSES)
        self.class_weights = torch.tensor(cw, dtype=torch.float32, device=DEVICE)

        # Optimizers
        self.opt_shared = optim.Adam(self.shared_model.parameters(), lr=self.lr)
        self.opt_local = optim.Adam(list(self.local_model.parameters()) + list(self.combiner.parameters()), lr=self.lr)

        # Splits used for client-side evaluation
        self.splits = stratified_group_kfold(self.df, group_col='userid', label_col='label', n_splits=5)
        self._round_counter = 0
        
    def get_parameters(self):
        state = self.shared_model.state_dict()
        return torch_state_dict_to_ndarrays(state)

    def set_parameters(self, parameters):
        ref = self.shared_model.state_dict()
        state = ndarrays_to_torch_state_dict(parameters, ref)
        self.shared_model.load_state_dict(state)

    def fit(self, parameters, config):
        """
        Must return tuple: (ndarrays, num_examples, metrics_dict)
        """
        try:
            # Set incoming global/shared parameters
            self.set_parameters(parameters)

            # Save a detached copy of the global/shared parameters (used for proximal term)
            global_params = {name: param.detach().clone() for name, param in self.shared_model.named_parameters()}
            mu = MU  # FedProx mu (can be 0.0)

            n = len(self.df)
            # Build standardized full arrays
            if len(self.shared_features) == 0:
                X_shared = torch.zeros((n, 1), dtype=torch.float32)
            else:
                arr = []
                for c in self.shared_features:
                    col = pd.to_numeric(self.df[c], errors="coerce").fillna(0.0).astype(float)
                    norm = (col - self.shared_means[c]) / (self.shared_stds[c] + 1e-9)
                    arr.append(norm.values.reshape(-1, 1))
                X_shared = np.hstack(arr).astype(np.float32)
                X_shared = torch.tensor(X_shared)

            if len(self.local_features) == 0:
                X_local = torch.zeros((n, 1), dtype=torch.float32)
            else:
                arr = []
                for c in self.local_features:
                    col = pd.to_numeric(self.df[c], errors="coerce").fillna(0.0).astype(float)
                    norm = (col - self.local_means[c]) / (self.local_stds[c] + 1e-9)
                    arr.append(norm.values.reshape(-1, 1))
                X_local = np.hstack(arr).astype(np.float32)
                X_local = torch.tensor(X_local)

            y = torch.tensor(self.df["label"].values.astype(np.int64))

            X_shared = X_shared.to(DEVICE)
            X_local = X_local.to(DEVICE)
            y = y.to(DEVICE)

            # build dataset / loader
            dataset = TensorDataset(X_shared, X_local, y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

            # Set models to train mode
            self.shared_model.train()
            self.local_model.train()
            self.combiner.train()

            total_loss = 0.0
            total_batches = 0

            # Local training
            for epoch in range(self.local_epochs):
                epoch_loss = 0.0
                epoch_batches = 0
                for xb_s, xb_l, yb in loader:
                    xb_s = xb_s.to(DEVICE)
                    xb_l = xb_l.to(DEVICE)
                    yb = yb.to(DEVICE)

                    # forward
                    shared_rep = self.shared_model(xb_s)
                    shared_rep_det = shared_rep.detach()
                    local_rep = self.local_model(xb_l, shared_rep_det)
                    logits = self.combiner(shared_rep, local_rep)
                    loss = self.criterion(logits, yb.long())

                    # FEDPROX proximal term
                    if mu and mu > 0.0:
                        prox = 0.0
                        for name, param in self.shared_model.named_parameters():
                            prox += ((param - global_params[name]) ** 2).sum()
                        loss = loss + (mu / 2.0) * prox

                    # backward + step
                    self.opt_shared.zero_grad()
                    self.opt_local.zero_grad()
                    loss.backward()
                    self.opt_shared.step()
                    self.opt_local.step()

                    batch_loss = float(loss.detach().cpu().numpy())
                    epoch_loss += batch_loss
                    epoch_batches += 1

                # accumulate
                total_loss += epoch_loss
                total_batches += max(1, epoch_batches)

            # compute average loss across all batches seen
            if total_batches > 0:
                avg_loss = total_loss / total_batches
            else:
                avg_loss = float("nan")

            # free GPU memory
            free_gpu()

            # Return updated shared params (ndarrays), number examples, and metrics dict
            updated_params = self.get_parameters()
            metrics = {"loss": float(avg_loss)}
            if total_batches > 0:
                avg_loss = total_loss / total_batches
            else:
                avg_loss = float("nan")

            # --- NEW: compute metrics on entire local dataset so server can save them ---
            try:
                self.shared_model.eval()
                self.local_model.eval()
                self.combiner.eval()
                with torch.no_grad():
                    # run full dataset through model to get probs/preds
                    logits_all = self.combiner(
                        self.shared_model(X_shared.to(DEVICE)),
                        self.local_model(X_local.to(DEVICE), self.shared_model(X_shared.to(DEVICE)).detach())
                    )
                    probs_all = torch.softmax(logits_all.detach().cpu(), dim=1).numpy()
                    preds_all = np.argmax(probs_all, axis=1)
                    y_all = y.detach().cpu().numpy()
                    m = metrics_from_preds_multiclass(y_all, preds_all, probs_all)
                    # ensure python floats (not numpy types)
                    accuracy_val = float(m.get("accuracy", np.nan)) if m is not None else float("nan")
                    f1_val = float(m.get("f1", np.nan)) if m is not None else float("nan")
                    auroc_val = float(m.get("auroc", np.nan)) if m is not None and m.get("auroc") is not None else None
            except Exception as e:
                # if metric computation fails, set to NaN/None but continue
                logger.exception("Metric computation in fit() failed for client %s", self.cid)
                accuracy_val = float("nan")
                f1_val = float("nan")
                auroc_val = None

            # free GPU memory
            free_gpu()

            # Return updated shared params (ndarrays), number examples, and metrics dict
            updated_params = self.get_parameters()
            # send client_file so server can identify the data file
            client_file = os.path.basename(self.data_path) if self.data_path else str(self.cid)
            metrics = {
                "loss": float(avg_loss) if not np.isnan(avg_loss) else None,
                "accuracy": accuracy_val,
                "f1": f1_val,
                "auroc": auroc_val,
                "client_file": client_file,
            }
            return updated_params, int(n), metrics

        except Exception as e:
            # log exception with stack trace and re-raise so Flower sees the failure
            logger.exception("Client %s fit() failed", self.cid)
            raise

    def evaluate(self, parameters, config):
        try:
            self.set_parameters(parameters)
            self.shared_model.eval()
            self.local_model.eval()
            self.combiner.eval()

            all_metrics = []
            all_losses = []
            all_counts = []

            for split_idx, (train_idx, test_idx) in enumerate(self.splits):
                df_test = self.df.iloc[test_idx]
                if len(df_test) == 0:
                    continue

                n = len(df_test)

                # ----- Build standardized shared features -----
                if len(self.shared_features) == 0:
                    Xs = torch.zeros((n, 1), dtype=torch.float32)
                else:
                    arr_s = []
                    for c in self.shared_features:
                        col = pd.to_numeric(df_test[c], errors="coerce").fillna(0.0).astype(float)
                        norm = (col - self.shared_means[c]) / (self.shared_stds[c] + 1e-9)
                        arr_s.append(norm.values.reshape(-1, 1))
                    Xs = torch.tensor(np.hstack(arr_s).astype(np.float32))

                # ----- Build standardized local features -----
                if len(self.local_features) == 0:
                    Xl = torch.zeros((n, 1), dtype=torch.float32)
                else:
                    arr_l = []
                    for c in self.local_features:
                        col = pd.to_numeric(df_test[c], errors="coerce").fillna(0.0).astype(float)
                        norm = (col - self.local_means[c]) / (self.local_stds[c] + 1e-9)
                        arr_l.append(norm.values.reshape(-1, 1))
                    Xl = torch.tensor(np.hstack(arr_l).astype(np.float32))

                Xs = Xs.to(DEVICE)
                Xl = Xl.to(DEVICE)
                y_true = df_test["label"].astype(int).values

                # ----- Forward pass -----
                with torch.no_grad():
                    shared_rep = self.shared_model(Xs)
                    local_rep = self.local_model(Xl, shared_rep.detach())
                    logits = self.combiner(shared_rep, local_rep)
                    logits_cpu = logits.detach().cpu()

                    # clamp extreme values
                    max_abs = float(torch.max(torch.abs(logits_cpu)).item())
                    if max_abs > 1e4:
                        logger.warning("large logits client %s fold %d max_abs=%f", self.cid, split_idx, max_abs)
                        logits_cpu = torch.clamp(logits_cpu, -1e4, 1e4)

                    probs = torch.softmax(logits_cpu, dim=1).numpy()
                    preds = np.argmax(probs, axis=1)

                # ----- Metrics -----
                m = metrics_from_preds_multiclass(y_true, preds, probs)

                y_true_tensor = torch.tensor(y_true, dtype=torch.long)
                loss_val = float(self.criterion(logits_cpu, y_true_tensor).item())

                all_metrics.append(m)
                all_losses.append(loss_val)
                all_counts.append(n)

            # No test data
            if len(all_metrics) == 0:
                client_file = os.path.basename(self.data_path) if getattr(self, "data_path", None) else str(self.cid)
                return float("nan"), 0, {"accuracy": 0.0, "f1": 0.0, "client_file": client_file}

            # ----- Weighted averages -----
            total_n = sum(all_counts)
            avg_acc = sum(m["accuracy"] * c for m, c in zip(all_metrics, all_counts)) / total_n
            avg_f1 = sum(m["f1"] * c for m, c in zip(all_metrics, all_counts)) / total_n

            auroc_vals = [(m["auroc"], c) for m, c in zip(all_metrics, all_counts) if m.get("auroc") is not None]
            avg_auroc = (sum(a * c for a, c in auroc_vals) / sum(c for _, c in auroc_vals)) if auroc_vals else None

            avg_loss = sum(all_losses) / len(all_losses)

            # ----- Determine round index -----
            _round_candidate = parse_round_from_config(config)
            if isinstance(_round_candidate, int) and _round_candidate >= 0:
                round_idx = _round_candidate
            else:
                round_idx = self._round_counter
                self._round_counter += 1

            country = str(self.cid).strip()

            logger.warning("DEBUG_SAVE round=%s client=%s saving to %s",
                           round_idx, country, self.results_folder)

            # ----- Save per-client CSV -----
            out_row = {
                "userid": "all",
                "client": country,
                "split_mode": "leave_n_out",
                "accuracy": avg_acc,
                "f1": avg_f1,
                "auroc": avg_auroc,
                "loss": avg_loss,
            }
            save_round_client_csv(self.results_folder, round_idx, country, out_row)

            free_gpu()

            # ----- Build metrics dict to send to server -----
            client_file = os.path.basename(self.data_path) if getattr(self, "data_path", None) else str(self.cid)

            metrics_dict = {
                "accuracy": float(avg_acc),
                "f1": float(avg_f1),
                "loss": float(avg_loss),
                "client_file": client_file,
            }
            if avg_auroc is not None:
                metrics_dict["auroc"] = float(avg_auroc)

            # ----- Return -----
            return float(avg_loss), int(total_n), metrics_dict

        except Exception:
            logger.exception("Client %s evaluate() failed", self.cid)
            raise

