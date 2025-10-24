import numpy as np
import random
from pathlib import Path
import json

def generate_datasets(
    sizes=[10**3, 10**4, 10**5, 10**6, 10**7],
    seed=42,
    output_dir="..\datasets"
):
    """
    Generate numeric datasets of various distributions for RMQ benchmarking.

    Args:
        sizes (list[int]): List of dataset sizes (number of elements).
        seed (int): Random seed for reproducibility.
        output_dir (str): Folder where datasets will be saved.
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