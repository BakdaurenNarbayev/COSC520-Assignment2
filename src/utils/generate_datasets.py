import numpy as np
import random
from pathlib import Path
import json

def generate_datasets(
    sizes=[10**3, 10**4, 10**5, 10**6, 10**7],
    seed=42,
    output_dir="..\\datasets"
):
    """
    Generate multiple numeric datasets with different statistical distributions
    for use in Range Minimum Query (RMQ) benchmarking.

    Each dataset is stored as a `.json` file containing a list of numeric values.

    Args:
        sizes (list[int]): Dataset sizes (number of elements per file).
        seed (int): Random seed to ensure reproducibility across runs.
        output_dir (str): Target directory where all JSON files will be saved.

    Notes:
        - This script supports several built-in distributions:
            • random_uniform    — Float values sampled uniformly from [-1000, 1000]
            • random_int        — Integer values sampled uniformly from [-1000, 1000]
            • sorted_ascending  — Linearly increasing values from -1000 to 1000
            • sorted_descending — Linearly decreasing values from 1000 to -1000
            • repeated_values   — Repeated small set of values (1.0-5.0)
    """
    np.random.seed(seed)
    random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    distributions = {
        "random_uniform": lambda n: np.random.uniform(-1000, 1000, n),
        "random_int": lambda n: np.random.randint(-1000, 1000, n),
        "sorted_ascending": lambda n: np.linspace(-1000, 1000, n),
        "sorted_descending": lambda n: np.linspace(1000, -1000, n),
        "repeated_values": lambda n: np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n),
    }

    for dist_name, generator in distributions.items():
        for size in sizes:
            data = generator(size)
            filename = f"{dist_name}_{size}.json"
            file_path = output_path / filename

            # Save as JSON
            with open(file_path, "w") as f:
                json.dump(data.tolist(), f)

            print(f"Generated {filename}")

    print("\nDataset generation complete!")
    print(f"Saved in: {output_path.resolve()}")

if __name__ == "__main__":
    generate_datasets()