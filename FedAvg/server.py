"""
server.py

Starts the Flower server, using FedAvg strategy, and
aggregates Accuracy, F1, and AUROC across clients each round.

Also saves global metrics to a CSV file and generates plots.
"""

from typing import Dict, List, Optional, Tuple

import flwr as fl
import matplotlib.pyplot as plt
import pandas as pd

from utils import compute_and_save_common_features, load_common_features


class MoodStrategy(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy that:
    - Aggregates evaluation metrics from all clients each round
    - Computes weighted global Accuracy, F1, and AUROC
    """

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """
        Called by Flower after receiving evaluation results from clients.
        We override this to compute weighted global metrics.

        Returns:
            aggregated_loss, aggregated_metrics
        """
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            print(f"[Server] Round {server_round}: no evaluation results.")
            return aggregated_loss, {}

        # Weighted sums of metrics
        total_examples = 0
        sum_acc = 0.0
        sum_f1 = 0.0
        sum_auroc = 0.0

        for client_proxy, eval_res in results:
            num_examples = eval_res.num_examples
            metrics = eval_res.metrics

            acc = float(metrics.get("accuracy", 0.0))
            f1 = float(metrics.get("f1", 0.0))
            auroc = float(metrics.get("auroc", 0.0))

            total_examples += num_examples
            sum_acc += acc * num_examples
            sum_f1 += f1 * num_examples
            sum_auroc += auroc * num_examples

        # Avoid division by zero
        if total_examples > 0:
            global_acc = sum_acc / total_examples
            global_f1 = sum_f1 / total_examples
            global_auroc = sum_auroc / total_examples
        else:
            global_acc = 0.0
            global_f1 = 0.0
            global_auroc = 0.0

        metrics_agg = {
            "global_accuracy": float(global_acc),
            "global_f1": float(global_f1),
            "global_auroc": float(global_auroc),
        }

        # Nicely formatted loss string (fixing earlier bug)
        if aggregated_loss is not None:
            loss_str = f"{aggregated_loss:.4f}"
        else:
            loss_str = "NA"

        print(
            f"[Server] Round {server_round} aggregated metrics -> "
            f"loss: {loss_str}, "
            f"acc: {metrics_agg['global_accuracy']:.4f}, "
            f"f1: {metrics_agg['global_f1']:.4f}, "
            f"auroc: {metrics_agg['global_auroc']:.4f}"
        )

        return aggregated_loss, metrics_agg


def save_history_to_csv_and_plots(history: fl.server.history.History) -> None:
    """
    Take the History returned by Flower and:
    - Save global_accuracy, global_f1, global_auroc per round into a CSV file
    - Generate line plots of each metric vs round and save as PNG
    """
    # history.metrics_distributed is a dict:
    #   {metric_name: [(round, value), ...], ...}
    metrics_dist = history.metrics_distributed

    # Collect rounds and metrics into a DataFrame
    rows = {}
    for metric_name in ["global_accuracy", "global_f1", "global_auroc"]:
        if metric_name not in metrics_dist:
            continue
        for rnd, val in metrics_dist[metric_name]:
            if rnd not in rows:
                rows[rnd] = {"round": rnd}
            rows[rnd][metric_name] = val

    # Convert dict to DataFrame sorted by round
    rounds_sorted = sorted(rows.keys())
    data = [rows[rnd] for rnd in rounds_sorted]
    df = pd.DataFrame(data)
    df = df.sort_values("round").reset_index(drop=True)

    # Save to CSV
    csv_path = "fl_metrics_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Server] Saved metrics history to {csv_path}")

    # Plot each metric vs round
    for metric_name in ["global_accuracy", "global_f1", "global_auroc"]:
        if metric_name not in df.columns:
            continue
        plt.figure()
        plt.plot(df["round"], df[metric_name], marker="o")
        plt.xlabel("Round")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs Round")
        plt.grid(True)
        png_path = f"{metric_name}.png"
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()
        print(f"[Server] Saved plot to {png_path}")


def main():
    """
    Main entry point to:
    - Compute and save common features (intersection) if not already done
    - Start the Flower server with our custom strategy
    - After training, save metrics history to CSV and plots
    """
    # Compute the common features once (creates common_features.json)
    compute_and_save_common_features()

    # Just to show we can load them (server doesn't strictly need them)
    common_features = load_common_features()
    print(f"[Server] Loaded {len(common_features)} common feature columns.")

    # Functions to pass configuration (round number) to clients
    def fit_config_fn(server_round: int) -> Dict[str, int]:
        """
        Configuration dict for client.fit().
        Includes current round (1-based) so clients can pick the proper fold.
        """
        return {"round": server_round}

    def eval_config_fn(server_round: int) -> Dict[str, int]:
        """
        Configuration dict for client.evaluate().
        Uses the same 'round' so that evaluation uses the same fold
        as training in that round.
        """
        return {"round": server_round}

    # Create strategy
    strategy = MoodStrategy(
        fraction_fit=1.0,       # use all available clients each round
        fraction_evaluate=1.0,  # evaluate on all clients each round
        min_fit_clients=6,      # adjust depending on number of clients
        min_evaluate_clients=6,
        min_available_clients=6,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=eval_config_fn,
    )

    # Configure number of federated rounds (k=5 folds => 5 rounds)
    config = fl.server.ServerConfig(num_rounds=5)

    # Start the Flower server
    # This returns a History object, which we can then save to CSV and plots
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )

    # After training: save metrics history
    save_history_to_csv_and_plots(history)


if __name__ == "__main__":
    main()
