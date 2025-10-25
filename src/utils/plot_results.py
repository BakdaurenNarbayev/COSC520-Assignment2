import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import numpy as np
import os

RESULTS_FILE = "..\\results\\benchmark_results.csv"
OUTPUT_DIR = "..\\results\\plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_results():
    results = defaultdict(lambda: defaultdict(dict))
    with open(RESULTS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row["Metric"]
            n = int(row["N"])
            algo = row["Algorithm"]
            mean = float(row["Mean"])
            std = float(row["StdDev"])
            mem = row.get("Memory_MB")
            mem_val = float(mem) if mem else None
            results[metric][n][algo] = {"mean": mean, "std": std, "mem": mem_val}
    return results

def plot_metric(results, metric_name, ylabel, filename):
    """
    Creates a log-log plot of the specified metric vs dataset size.
    """
    sizes = sorted(results.keys())
    all_algorithms = set()
    for n in sizes:
        all_algorithms.update(results[n].keys())

    plt.figure(figsize=(8, 6))
    for algo in sorted(all_algorithms):
        Ns, means, stds = [], [], []
        for n in sizes:
            entry = results[n].get(algo)
            if entry is None:
                continue
            Ns.append(n)
            means.append(entry["mean"])
            stds.append(entry["std"])
        if not Ns:
            continue
        Ns = np.array(Ns)
        means = np.array(means)
        stds = np.array(stds)
        plt.plot(Ns, means, marker='o', label=algo)
        plt.fill_between(Ns, means - stds, means + stds, alpha=0.2)

    plt.title(f"{ylabel} vs Array Size (log-log scale)")
    plt.xlabel("Array Size (N)")
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(visible=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.pdf"))
    plt.close()

def plot_memory_usage(results, filename="memory_usage"):
    """
    Creates a log-log plot of memory usage vs array size.
    """
    sizes = sorted(results.keys())
    all_algorithms = set()
    for n in sizes:
        all_algorithms.update(results[n].keys())

    plt.figure(figsize=(8, 6))
    for algo in sorted(all_algorithms):
        Ns, mems = [], []
        for n in sizes:
            entry = results[n].get(algo)
            if entry is None or entry["mem"] is None:
                continue
            Ns.append(n)
            mems.append(entry["mem"])
        if not Ns:
            continue
        Ns = np.array(Ns)
        mems = np.array(mems)
        plt.plot(Ns, mems, marker='o', label=algo)

    plt.title("Memory Usage vs Array Size (log-log scale)")
    plt.xlabel("Array Size (N)")
    plt.ylabel("Memory Usage (MB)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(visible=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.pdf"))
    plt.close()

if __name__ == "__main__":
    results = load_results()
    metrics_to_plot = [
        ("build", "Preprocessing Time (s)"),
        ("query", "Query Time (s)"),
        ("update", "Update Time (s)")
    ]
    for metric, ylabel in metrics_to_plot:
        plot_metric(results[metric], metric.capitalize(), ylabel, metric)
    plot_memory_usage(results["build"], filename="memory_usage")
    print(f"All plots saved in {OUTPUT_DIR}")
