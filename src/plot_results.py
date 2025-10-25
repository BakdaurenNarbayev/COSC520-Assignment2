import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import numpy as np

def load_results():
    results = defaultdict(lambda: defaultdict(dict))

    with open("benchmark_results.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row["Metric"]
            n = int(row["N"])
            algo = row["Algorithm"]
            mean = float(row["Mean"])
            std = float(row["StdDev"])
            results[metric][n][algo] = (mean, std)
    
    return results


# --- PLOTTING ---
def plot_with_ci(results, metric_name, ylabel, loglog=False):
    """Plot with mean ± std as confidence intervals."""
    sizes = sorted(results.keys())

    # Collect all algorithms that appear for any N
    all_algorithms = set()
    for n in sizes:
        all_algorithms.update(results[n].keys())

    plt.figure(figsize=(8, 6))

    for algo in sorted(all_algorithms):
        Ns, means, stds = [], [], []
        for n in sizes:
            entry = results[n].get(algo)
            if entry is None:
                continue  # skip missing combinations
            mean, std = entry
            Ns.append(n)
            means.append(mean)
            stds.append(std)

        if not Ns:
            continue  # skip algorithms that had no data at all

        Ns = np.array(Ns)
        means = np.array(means)
        stds = np.array(stds)
        plt.plot(Ns, means, marker='o', label=algo)
        plt.fill_between(Ns, means - stds, means + stds, alpha=0.2)

    plt.title(f"{metric_name} vs Dataset Size")
    plt.xlabel("Array Size (N)")
    plt.ylabel(ylabel)
    if loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()


# --- MAIN ENTRY ---
if __name__ == "__main__":
    results = load_results()

    for metric, label in [
        ("build", "Build Time (s)"),
        ("query", "Average Query Time (s)"),
        ("update", "Average Update Time (s)")
    ]:
        plot_with_ci(results[metric], metric.capitalize(), label, loglog=False)
        plot_with_ci(results[metric], metric.capitalize() + " (Log-Log)", label, loglog=True)