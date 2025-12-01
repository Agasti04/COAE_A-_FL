from logging import config
import os

# Dedicated Ray temp directory
RAY_TMP = os.path.abspath("ray_tmp")
os.makedirs(RAY_TMP, exist_ok=True)

# Tell Ray & system to use this temp directory
os.environ["RAY_TMPDIR"] = RAY_TMP
os.environ["TMPDIR"] = RAY_TMP

# Ray object spilling config (prevents /tmp filling)
os.environ["RAY_object_spilling_config"] = (
    '{"type":"filesystem","params":{"directory_path": "ray_tmp"}}'
)

import argparse
import flwr as fl
from flwr.common.parameter import ndarrays_to_parameters
from .utils import (
    list_client_files, read_client_df, compute_availability_vector, aggregate_availability,
    free_gpu
)
from src.Fedprox.utils import build_summary_all_rounds, save_summary_all_rounds
from .client import TowerClient
from .model import SharedTower
# replace whatever import you have with this exact import
from src.Fedprox.strategy import SaveEveryRoundFedAvg



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--results_folder", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clients_fraction", type=float, default=1.0)
    parser.add_argument("--gpu_fraction", type=float, default=0.15)
    args = parser.parse_args()

    data_folder = args.data_folder
    results_folder = args.results_folder
    os.makedirs(results_folder, exist_ok=True)

    client_files = list_client_files(data_folder)
    print(f"Found {len(client_files)} client files")

    avail_list = []
    for path in client_files:
        df = read_client_df(path)
        avail = compute_availability_vector(df)
        avail_list.append(avail)

    shared_features, _ = aggregate_availability(avail_list)
    print("Shared features (S):", shared_features)

    
    # strict mapping: "0".."N-1" -> files
    client_files = list_client_files(data_folder)
    client_objs = {}
    for idx, path in enumerate(client_files):
        cid = str(idx)
        country = os.path.basename(path).replace(".csv", "")
        client_objs[cid] = TowerClient(
            cid=country,
            data_path=path,
            shared_feature_list=shared_features,
            results_folder=results_folder,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    reference_shared = SharedTower(len(shared_features)).state_dict()

    strategy = SaveEveryRoundFedAvg(
        results_folder=results_folder,
        reference_state_dict=reference_shared,
        fraction_fit=args.clients_fraction,
        min_fit_clients=len(client_objs),
        min_available_clients=len(client_objs),
        fraction_evaluate=1.0,
        min_evaluate_clients=len(client_objs),
        initial_parameters=ndarrays_to_parameters([v.cpu().numpy() for v in reference_shared.values()]),
    )

    # Ray tmp short path
    os.environ["RAY_TMPDIR"] = "/tmp/ray_tower"
    os.makedirs("/tmp/ray_tower", exist_ok=True)

    def client_fn(cid: str):
        cid = str(cid)
        if cid not in client_objs:
            raise KeyError(f"Flower passed unknown client id {cid}. Keys: {list(client_objs.keys())}")
        obj = client_objs[cid]
        return obj.to_client()

    client_resources = {"num_cpus": 1, "num_gpus": args.gpu_fraction}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_objs),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    from src.Fedprox.utils import build_summary_all_rounds, build_summary_rounds_from_countries
    print("summary_all:", build_summary_all_rounds(results_folder))
    print("summary_rounds:", build_summary_rounds_from_countries(results_folder))

    # build final summary after simulation
    import pandas as pd, glob, re
    summary_rows = []
    for entry in sorted(os.listdir(results_folder)):
        if not entry.startswith("round_"):
            continue
        round_folder = os.path.join(results_folder, entry)
        if not os.path.isdir(round_folder):
            continue
        m = re.match(r"round_(\d+)", entry)
        round_idx = int(m.group(1)) if m else -1
        for f in glob.glob(os.path.join(round_folder, "*.csv")):
            try:
                df = pd.read_csv(f)
                if df.empty:
                    continue
                row = df.iloc[0].to_dict()
                row["round"] = round_idx if round_idx != -1 else "unknown"
                summary_rows.append(row)
            except Exception:
                continue

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(results_folder, "summary_all_rounds.csv"),
            index=False
        )
        print("Summary written:", os.path.join(results_folder, "summary_all_rounds.csv"))
    else:
        print("Warning: summary_all_rounds.csv is empty (no rows found).")

    free_gpu()
    print("Training finished. Results saved at:", results_folder)
    # after training finishes (replace any old final-summary code)
    from .utils import build_summary_all_rounds, build_summary_rounds_from_countries

    summary_all = build_summary_all_rounds(results_folder)
    if summary_all:
        print("Summary (all rows) written:", summary_all)
    else:
        print("Warning: summary_all_rounds.csv could not be written (no per-country rows found).")

    summary_agg = build_summary_rounds_from_countries(results_folder)
    if summary_agg:
        print("Aggregated per-round summary written:", summary_agg)
    else:
        print("Warning: summary_rounds.csv could not be built (no data).")

if __name__ == "__main__":
    main()
    
