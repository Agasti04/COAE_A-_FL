import os
import shutil
import tempfile
import gc
import json
import logging
import pandas as pd
import numpy as np
import torch
import flwr as fl
from flwr.server.strategy.fedavg import FedAvg
from .utils import ndarrays_to_torch_state_dict, torch_state_dict_to_ndarrays

logger = logging.getLogger(__name__)

class SaveEveryRoundFedAvg(FedAvg):
    """
    Replacement FedAvg strategy:
      - Saves aggregated weights (shared_weights_round_<rnd>.pt)
      - Logs raw results returned by clients (for debugging)
      - Extracts accuracy, f1, loss, auroc, client_file robustly
      - Writes per-round CSV and master all_clients_rounds.csv
    """

    def __init__(self, results_folder, reference_state_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_folder = results_folder
        self.ref_state = reference_state_dict
        os.makedirs(self.results_folder, exist_ok=True)
        self.round = 0

    def _safe_mkdir(self, path):
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def _atomic_write_csv(self, df: pd.DataFrame, out_path: str):
        out_dir = os.path.dirname(out_path) or "."
        self._safe_mkdir(out_dir)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=out_dir)
        os.close(fd)
        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, out_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _to_number(self, v):
        """Convert numpy/json types to python numbers/None safely."""
        try:
            if v is None:
                return None
            if isinstance(v, (float, int)):
                return float(v)
            if isinstance(v, (np.floating, np.integer)):
                return float(np.asscalar(np.array(v))) if hasattr(np, "asscalar") else float(v)
            # some metrics might come as numpy arrays of length 1
            if isinstance(v, (np.ndarray, list)) and len(v) == 1:
                return float(v[0])
            # string numbers
            if isinstance(v, str):
                try:
                    return float(v)
                except Exception:
                    return None
            return None
        except Exception:
            return None

    def _safe_json(self, obj):
        try:
            return json.dumps(obj)
        except Exception:
            return str(obj)

    def _extract_from_result_obj(self, res_obj):
        """
        Try to pull (num_examples, metrics_dict) from res_obj which could be:
         - an object with attributes .num_examples and .metrics
         - a tuple like (num_examples, metrics)
         - a dict
        """
        num_examples = None
        metrics = {}

        # try attributes
        try:
            if hasattr(res_obj, "num_examples"):
                num_examples = getattr(res_obj, "num_examples")
        except Exception:
            pass

        try:
            if hasattr(res_obj, "metrics"):
                metrics = getattr(res_obj, "metrics") or {}
        except Exception:
            pass

        # try tuple unpacking (common shapes)
        if (num_examples is None or metrics == {}) and isinstance(res_obj, (list, tuple)):
            # common shapes:
            # - FitRes: (params, num_examples, metrics) or (num_examples, metrics)
            # - EvaluateRes: (num_examples, metrics)
            tup = list(res_obj)
            # look for first integer-like
            for item in tup:
                if isinstance(item, (int, np.integer)):
                    if num_examples is None:
                        try:
                            num_examples = int(item)
                        except Exception:
                            pass
            # look for dict-like metrics
            for item in tup[::-1]:
                if isinstance(item, dict):
                    metrics = item
                    break

        # try dict
        if isinstance(res_obj, dict):
            if "num_examples" in res_obj and num_examples is None:
                num_examples = res_obj.get("num_examples")
            if "metrics" in res_obj and (metrics == {}):
                metrics = res_obj.get("metrics") or {}
            # maybe metrics at root
            for k in ("accuracy", "f1", "loss", "auroc", "client_file"):
                if k in res_obj and metrics.get(k) is None:
                    metrics[k] = res_obj.get(k)

        # ensure metrics is a dict
        if not isinstance(metrics, dict):
            try:
                metrics = dict(metrics)
            except Exception:
                metrics = {}

        return num_examples, metrics

    def _resolve_client_info(self, client_proxy, metrics):
        """
        Return (client_file, client_id)
        Try client_proxy.get_properties(), .properties, then metrics keys.
        """
        client_id = None
        client_file = None
        try:
            client_id = getattr(client_proxy, "cid", None) or getattr(client_proxy, "client_id", None)
        except Exception:
            client_id = str(client_proxy)

        # try get_properties()
        try:
            if client_proxy is not None and hasattr(client_proxy, "get_properties"):
                props = client_proxy.get_properties()
                if isinstance(props, dict):
                    for key in ("file_name", "filename", "data_file", "client_file"):
                        if props.get(key):
                            client_file = props.get(key)
                            break
        except Exception:
            pass

        # try attribute
        if client_file is None:
            try:
                props = getattr(client_proxy, "properties", None)
                if isinstance(props, dict):
                    for key in ("file_name", "filename", "data_file", "client_file"):
                        if props.get(key):
                            client_file = props.get(key)
                            break
            except Exception:
                pass

        # fallback to metrics
        if client_file is None and isinstance(metrics, dict):
            for key in ("client_file", "file_name", "filename", "data_file", "client"):
                if key in metrics and metrics[key]:
                    client_file = metrics[key]
                    break

        # final fallbacks
        if client_file is None:
            client_file = str(client_id) if client_id is not None else str(client_proxy)

        if client_id is None:
            try:
                client_id = str(client_proxy)
            except Exception:
                client_id = "unknown"

        return str(client_file), str(client_id)

    # ---------------- Overrides -----------------
    def aggregate_fit(self, rnd, results, failures):
        # cleanup
        try:
            shutil.rmtree("ray_tmp", ignore_errors=True)
            os.makedirs("ray_tmp", exist_ok=True)
        except Exception:
            pass

        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is None:
            return None

        # Save aggregated weights (defensive)
        try:
            params = aggregated[0] if isinstance(aggregated, tuple) else aggregated
            params_nd = fl.common.parameters_to_ndarrays(params) if not isinstance(params, list) else params
            state = ndarrays_to_torch_state_dict(params_nd, self.ref_state)
            fname = os.path.join(self.results_folder, f"shared_weights_round_{rnd}.pt")
            torch.save(state, fname)
        except Exception:
            logger.exception("Failed to save shared weights for round %s", rnd)

        self.round = rnd
        return aggregated

    def aggregate_evaluate(self, rnd, results, failures):
        # Let parent do aggregation (keeps Flower behavior)
        aggregated = super().aggregate_evaluate(rnd, results, failures)

        # Log raw results for debugging (very helpful)
        try:
            logger.info("aggregate_evaluate: round=%s got %d results, %d failures", rnd, len(results or []), len(failures or []))
            # Log compact representations of the results - don't flood with huge dumps
            for i, entry in enumerate(results or []):
                try:
                    # entry commonly is (client_proxy, EvaluateRes)
                    client_proxy = entry[0] if isinstance(entry, (list, tuple)) and len(entry) >= 1 else None
                    res_obj = entry[1] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else entry
                    logger.info("raw_result[%d] client=%s res_repr=%s", i, getattr(client_proxy, "cid", str(client_proxy)), repr(res_obj))
                except Exception:
                    logger.exception("Failed to log raw result entry %d", i)
        except Exception:
            logger.exception("Failed to log results in aggregate_evaluate")

        # Build rows for CSV
        rows = []
        for entry in results or []:
            try:
                client_proxy = entry[0] if isinstance(entry, (list, tuple)) and len(entry) >= 1 else None
                res_obj = entry[1] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else entry

                num_examples, metrics = self._extract_from_result_obj(res_obj)

                # Resolve client file/id
                client_file, client_id = self._resolve_client_info(client_proxy, metrics)

                # pick explicit metrics
                accuracy = self._to_number(metrics.get("accuracy") if isinstance(metrics, dict) else None)
                f1 = self._to_number(metrics.get("f1") if isinstance(metrics, dict) else None)
                loss = self._to_number(metrics.get("loss") if isinstance(metrics, dict) else None)
                auroc = self._to_number(metrics.get("auroc") if isinstance(metrics, dict) else None)

                # if our explicit fields are missing, try alternative keys
                if accuracy is None:
                    for alt in ("acc", "accuracy_score"):
                        if alt in metrics:
                            accuracy = self._to_number(metrics.get(alt))
                            if accuracy is not None:
                                break
                if f1 is None:
                    for alt in ("f1_score", "f1_macro"):
                        if alt in metrics:
                            f1 = self._to_number(metrics.get(alt))
                            if f1 is not None:
                                break
                if loss is None:
                    for alt in ("train_loss", "val_loss"):
                        if alt in metrics:
                            loss = self._to_number(metrics.get(alt))
                            if loss is not None:
                                break
                if auroc is None:
                    for alt in ("auc", "roc_auc"):
                        if alt in metrics:
                            auroc = self._to_number(metrics.get(alt))
                            if auroc is not None:
                                break

                other_metrics = {k: v for k, v in (metrics.items() if isinstance(metrics, dict) else []) if k not in ("accuracy", "f1", "loss", "auroc", "client_file")}
                row = {
                    "round": int(rnd),
                    "client_file": client_file,
                    "client_id": client_id,
                    "result_type": "evaluate",
                    "num_examples": int(num_examples) if num_examples is not None else None,
                    "accuracy": accuracy,
                    "f1": f1,
                    "loss": loss,
                    "auroc": auroc,
                    "other_metrics": self._safe_json(other_metrics) if other_metrics else None,
                }
                rows.append(row)
            except Exception:
                logger.exception("Failed to extract row for one evaluate result")

        # write per-round and master CSVs
        try:
            if rows:
                df_round = pd.DataFrame(rows)
                out_round = os.path.join(self.results_folder, f"round_{rnd}_clients_eval.csv")
                self._atomic_write_csv(df_round, out_round)

                master_path = os.path.join(self.results_folder, "all_clients_rounds.csv")
                if os.path.exists(master_path):
                    existing = pd.read_csv(master_path)
                    combined = pd.concat([existing, df_round], ignore_index=True)
                else:
                    combined = df_round
                # optional: drop duplicates if any
                combined.drop_duplicates(subset=["round", "client_id", "result_type"], keep="last", inplace=True)
                self._atomic_write_csv(combined, master_path)
            else:
                logger.info("No rows extracted for round %s evaluate", rnd)
        except Exception:
            logger.exception("Failed to write evaluate CSVs for round %s", rnd)

        # resource cleanup
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

        return aggregated
