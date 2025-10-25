import json
import os
import random
import timeit
from tqdm import tqdm
import numpy as np
import csv
import concurrent.futures
import tracemalloc

from approaches.naive import Naive
from approaches.srd import SRD
from approaches.segment_tree import SegmentTree
from approaches.sparse_table import SparseTable

# --- CONFIG ---
NUM_RUNS = 5
NUM_QUERIES = 500
NUM_UPDATES = 500
DATASET_DIR = "datasets"
EXPORT_CSV = True
TIMEOUT_SECONDS = 120

def try_run_with_timeout(func, *args, timeout=TIMEOUT_SECONDS, default=None, **kwargs):
    """Run a function with timeout; return default if it exceeds the limit."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return default

# --- DATA LOADING ---
def load_datasets(dataset_dir=DATASET_DIR):
    """Load all JSON datasets from the specified folder."""
    datasets = {}
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".json"):
            size = int("".join(filter(str.isdigit, filename)))
            with open(os.path.join(dataset_dir, filename), "r") as f:
                datasets[size] = json.load(f)
    return dict(sorted(datasets.items()))

# --- BENCHMARK HELPERS ---
def benchmark_build(structure_class, data):
    """Measure time and memory to build the data structure."""
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()

    timer = timeit.Timer(lambda: structure_class(data.copy()))
    build_time = timer.timeit(number=1)

    end_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    total_mem = sum(stat.size_diff for stat in stats)
    total_mem_mb = max(total_mem / (1024 * 1024), 0)

    return build_time, total_mem_mb

def benchmark_query(structure_class, data, num_queries=NUM_QUERIES, pbar=None):
    """Measure average query time using timeit."""
    rmq = structure_class(data.copy())
    n = len(data)
    queries = [(min(a,b), max(a,b)) for _ in range(num_queries)
           for a, b in [(random.randint(0, n-1), random.randint(0, n-1))]]

    def run_queries():
        for l, r in queries:
            rmq.query(l, r)
            if pbar: pbar.update(1)

    timer = timeit.Timer(run_queries)
    total_time = timer.timeit(number=1)
    return total_time / num_queries

def benchmark_update(structure_class, data, num_updates=NUM_UPDATES, pbar=None):
    """Measure average update time using timeit."""
    rmq = structure_class(data.copy())
    n = len(data)
    updates = [(random.randint(0, n - 1), random.uniform(-1000, 1000)) for _ in range(num_updates)]

    def run_updates():
        for i, val in updates:
            rmq.update(i, val)
            if pbar: pbar.update(1)

    timer = timeit.Timer(run_updates)
    total_time = timer.timeit(number=1)
    return total_time / num_updates

# --- MAIN BENCHMARK FUNCTION ---
def run_benchmarks():
    datasets = load_datasets()
    algorithms = [Naive, SRD, SegmentTree, SparseTable]

    results = {"build": {}, "query": {}, "update": {}}
    skip_algos = {"build": set(), "query": set(), "update": set()}  # Track skipped algorithms

    print(f"\nRunning RMQ Benchmarks ({len(algorithms)} algorithms × {len(datasets)} datasets × {NUM_RUNS} runs)\n")

    for n, data in datasets.items():
        print(f"\nDataset size: {n}")
        results["build"][n] = {}
        results["query"][n] = {}
        results["update"][n] = {}

        for algo in algorithms:
            algo_name = algo.__name__

            # --- BUILD ---
            if algo_name not in skip_algos["build"]:
                build_times = []
                with tqdm(total=NUM_RUNS, desc=f"{algo_name} Build (N={n})", ncols=100, leave=False) as pbar_build:
                    for _ in range(NUM_RUNS):
                        result = try_run_with_timeout(benchmark_build, algo, data)
                        if result is None:
                            print(f"{algo_name} (N={n}) — build exceeded {TIMEOUT_SECONDS}s, skipping this and future builds")
                            skip_algos["build"].add(algo_name)
                            build_times = None
                            break
                        build_time, build_mem = result
                        build_times.append(build_time)
                        pbar_build.update(1)
                if build_times is not None:
                    results["build"][n][algo_name] = (np.mean(build_times), np.std(build_times), build_mem)
            else:
                results["build"][n][algo_name] = None

            # --- QUERY ---
            if algo_name not in skip_algos["query"]:
                query_times = []
                with tqdm(total=NUM_RUNS*NUM_QUERIES, desc=f"{algo_name} Query (N={n})", ncols=100, leave=False) as pbar_query:
                    for _ in range(NUM_RUNS):
                        result = try_run_with_timeout(benchmark_query, algo, data, pbar=pbar_query)
                        if result is None:
                            print(f"{algo_name} (N={n}) — query exceeded {TIMEOUT_SECONDS}s, skipping this and future queries")
                            skip_algos["query"].add(algo_name)
                            query_times = None
                            break
                        query_times.append(result)
                if query_times is not None:
                    results["query"][n][algo_name] = (np.mean(query_times), np.std(query_times))
            else:
                results["query"][n][algo_name] = None

            # --- UPDATE ---
            if algo_name not in skip_algos["update"]:
                update_times = []
                with tqdm(total=NUM_RUNS*NUM_UPDATES, desc=f"{algo_name} Update (N={n})", ncols=100, leave=False) as pbar_update:
                    for _ in range(NUM_RUNS):
                        result = try_run_with_timeout(benchmark_update, algo, data, pbar=pbar_update)
                        if result is None:
                            print(f"{algo_name} (N={n}) — update exceeded {TIMEOUT_SECONDS}s, skipping this and future updates")
                            skip_algos["update"].add(algo_name)
                            update_times = None
                            break
                        update_times.append(result)
                if update_times is not None:
                    results["update"][n][algo_name] = (np.mean(update_times), np.std(update_times))
            else:
                results["update"][n][algo_name] = None

            # --- PRINT SUMMARY ---
            update_str = f"{np.mean(update_times)*1e6:9.2f}" if update_times else "N/A"
            query_str  = f"{np.mean(query_times)*1e6:9.2f}" if query_times else "N/A"
            build_str  = f"{np.mean(build_times):7.6f}" if build_times else "N/A"
            mem_str    = f"{build_mem:6.2f}" if build_times else "N/A"

            print(f"{algo_name:<15} | Build: {build_str} s | Mem: {mem_str} MB | "
                  f"Query: {query_str} µs | Update: {update_str} µs")

    print("\nAll benchmarks completed!\n")
    return results

# --- CSV EXPORT ---
def export_to_csv(results, filename="results/benchmark_results.csv"):
    """Save results to CSV for external plotting or analysis."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "N", "Algorithm", "Mean", "StdDev", "Memory_MB"])
        for metric in results:
            for n in results[metric]:
                for algo, values in results[metric][n].items():
                    if values is None:
                        continue
                    if metric == "build":
                        mean, std, mem = values
                    else:
                        mean, std = values
                        mem = ""
                    writer.writerow([metric, n, algo, mean, std, mem])
    print(f"Results saved to {filename}")

# --- MAIN ---
if __name__ == "__main__":
    results = run_benchmarks()
    if EXPORT_CSV:
        export_to_csv(results)
