# src/Fedprox/results_manager.py
import os
import threading
from typing import List, Optional, Dict
import pandas as pd
import numpy as _np

class ResultsManager:
    """
    Thread-safe per-process results manager. Use ResultsManager.init(...) once per process.
    """

    _instance = None
    _lock_inst = threading.Lock()

    @classmethod
    def init(cls, results_folder: str, countries: List[str]):
        with cls._lock_inst:
            if cls._instance is None:
                cls._instance = cls(results_folder, countries)
            return cls._instance

    @classmethod
    def get(cls):
        if cls._instance is None:
            raise RuntimeError("ResultsManager not initialized. Call ResultsManager.init(...) first.")
        return cls._instance

    def __init__(self, results_folder: str, countries: List[str]):
        self.results_folder = os.path.abspath(results_folder)
        os.makedirs(self.results_folder, exist_ok=True)
        self.lock = threading.Lock()

        self.countries = [str(c).strip() for c in countries]

        cols = ["round", "loss", "accuracy", "f1", "auroc"]
        self.country_dfs: Dict[str, pd.DataFrame] = {c: pd.DataFrame(columns=cols) for c in self.countries}
        self.overall_df = pd.DataFrame(columns=["round", "loss_mean", "accuracy_mean", "f1_mean", "auroc_mean", "n_countries"])

        # try to load existing CSVs
        for c in list(self.countries):
            p = os.path.join(self.results_folder, f"{c}.csv")
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    df = df.loc[:, [col for col in ["round", "loss", "accuracy", "f1", "auroc"] if col in df.columns]]
                    self.country_dfs[c] = df.reindex(columns=cols)
                except Exception:
                    pass

        overall_p = os.path.join(self.results_folder, "overall_rounds.csv")
        if os.path.exists(overall_p):
            try:
                self.overall_df = pd.read_csv(overall_p)
            except Exception:
                pass

    def _atomic_write_csv(self, df: pd.DataFrame, path: str):
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)

    def save_round(self, country: str, round_idx: int, loss: Optional[float],
                   accuracy: Optional[float], f1: Optional[float], auroc: Optional[float]):
        country = str(country).strip()
        if country not in self.country_dfs:
            with self.lock:
                if country not in self.country_dfs:
                    cols = ["round", "loss", "accuracy", "f1", "auroc"]
                    self.country_dfs[country] = pd.DataFrame(columns=cols)
                    self.countries.append(country)

        row = {
            "round": int(round_idx),
            "loss": float(loss) if loss is not None else _np.nan,
            "accuracy": float(accuracy) if accuracy is not None else _np.nan,
            "f1": float(f1) if f1 is not None else _np.nan,
            "auroc": float(auroc) if auroc is not None else _np.nan,
        }

        with self.lock:
            df = self.country_dfs[country]
            if not df.empty:
                try:
                    df = df.loc[~(df["round"].astype(int) == int(round_idx))]
                except Exception:
                    pass
            new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True, sort=False)
            new_df = new_df.sort_values("round").reset_index(drop=True)
            self.country_dfs[country] = new_df

            country_path = os.path.join(self.results_folder, f"{country}.csv")
            try:
                self._atomic_write_csv(new_df, country_path)
            except Exception:
                try:
                    new_df.to_csv(country_path, index=False)
                except Exception:
                    pass

            # build overall for this round
            rows_for_round = []
            for c, cdf in self.country_dfs.items():
                try:
                    if not cdf.empty:
                        mask = (cdf["round"].astype(int) == int(round_idx))
                        if mask.any():
                            rows_for_round.append(cdf.loc[mask].iloc[-1].to_dict())
                except Exception:
                    continue

            if rows_for_round:
                def mean_safe(key):
                    vals = [r.get(key) for r in rows_for_round if r.get(key) is not None and (not (_np.isnan(r.get(key)) if isinstance(r.get(key), float) else False))]
                    return float(_np.mean(vals)) if vals else _np.nan

                agg = {
                    "round": int(round_idx),
                    "loss_mean": mean_safe("loss"),
                    "accuracy_mean": mean_safe("accuracy"),
                    "f1_mean": mean_safe("f1"),
                    "auroc_mean": mean_safe("auroc"),
                    "n_countries": len(rows_for_round),
                }

                odf = self.overall_df
                if not odf.empty:
                    try:
                        odf = odf.loc[~(odf["round"].astype(int) == int(round_idx))]
                    except Exception:
                        pass
                odf = pd.concat([odf, pd.DataFrame([agg])], ignore_index=True, sort=False)
                odf = odf.sort_values("round").reset_index(drop=True)
                self.overall_df = odf

                overall_path = os.path.join(self.results_folder, "overall_rounds.csv")
                try:
                    self._atomic_write_csv(self.overall_df, overall_path)
                except Exception:
                    try:
                        self.overall_df.to_csv(overall_path, index=False)
                    except Exception:
                        pass

    def get_country_df(self, country: str) -> pd.DataFrame:
        country = str(country).strip()
        if country in self.country_dfs:
            return self.country_dfs[country].copy()
        else:
            return pd.DataFrame(columns=["round", "loss", "accuracy", "f1", "auroc"])

    def get_overall_df(self) -> pd.DataFrame:
        return self.overall_df.copy()
