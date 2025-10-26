# RMQ Benchmarking

A Python-based benchmarking for **Range Minimum Query (RMQ)** data structures.  
This project evaluates the performance of multiple RMQ implementations, including `Naive`, `SRD`, `SegmentTree`, and `SparseTable`, across different dataset sizes, operations (`preprocessing` or `build`, `query`, `update`), and `memory usage`.

---

## Project Structure

```
./src/
├── benchmark.py                # Main benchmarking script
├── test_approach.py            # Pytest unit tests for RMQ implementations
├── approaches/                 # Implementations of the approaches
│   ├── naive.py
│   ├── srd.py
│   ├── segment_tree.py
│   └── sparse_table.py
├── datasets/
│   ├── random_int_1000.json
│   ├── random_int_10000.json
│   └── ...
└── results/
    └── benchmark_results.csv   # Output from benchmark runs
    └── plots/                  # Plots comparing the approaches
        └── build.pdf
        └── memory_usage
        └── query.pdf
        └── update.pdf
└── utils/
    └── generate_datasets.py    # Datasets generation script
    └── plot_results.py         # Plots generation script
```

---

## Installation

### Requirements
- Python 3.8+

### Install dependencies
Please install the main dependencies manually:
```bash
pip install pytest timeit tqdm numpy matplotlib
```

---

## Usage

### 1. Unit Tests

Unit tests are written using **pytest** to validate correctness and error handling of all RMQ approaches.

Run all tests:
```bash
pytest -vv
```
or
```bash
python -m pytest -vv
```

### 2. Prepare Datasets
If you are using the link from the report, place your JSON datasets in the `./src/datasets/` directory.  
To reduce benchmarking time, consider removing larger datasets.

Alternatively, you can run `generate_datasets.py` script in `./src/utils/` directory via
```bash
python generate_datasets.py
```

### 3. Run Benchmarks
Execute the main benchmarking script:
```bash
python benchmark.py
```

This will:
- Load all datasets from `datasets/`
- Run all approaches (`Naive`, `SRD`, `SegmentTree`, `SparseTable`)
- Perform multiple runs per dataset and operation
- Export results to `./src/results/benchmark_results.csv`

#### Configuration

You can modify these constants in `benchmark.py`:

| Variable | Description | Default |
|-----------|--------------|----------|
| `NUM_RUNS` | Number of runs per test | 5 |
| `NUM_QUERIES` | Queries per run | 500 |
| `NUM_UPDATES` | Updates per run | 500 |
| `DATASET_DIR` | Dataset folder path | `"datasets"` |
| `EXPORT_CSV` | Save results to CSV | `True` |
| `TIMEOUT_SECONDS` | Timeout per benchmark run | 120 |

#### Output Example
Results are saved in CSV format (Memory_MB is available only for `build` operation):
```
Metric,N,Algorithm,Mean,StdDev,Memory_MB
build,1000,Naive,9.54E-06,4.38E-06,0.001583099
query,1000,Naive,1.15E-05,1.20E-06
...
```

### 4. Visualization

To plot graphs based on the benchmarking results, you can run `plot_results.py` script in `./src/utils/` directory via
```bash
python plot_results.py
```
Plots are generated in `./src/results/plots/` directory by default.

---