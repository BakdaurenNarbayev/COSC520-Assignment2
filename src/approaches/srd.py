from math import ceil, sqrt

class SRD:
    """
    A Range Minimum Query (RMQ) structure using Square Root Decomposition (SRD).
    Source: https://iq.opengenus.org/range-minimum-query-square-root-decomposition

    This approach divides the array into √n blocks, precomputes the minimum for
    each block, and answers queries by combining block and element-level checks.

    Time Complexity:
        - Update: O(1)
        - Query: O(√n)

    Attributes:
        array (list[float]): The list of numeric values for queries and updates.
        feed (list[float]): Precomputed minimums for each block.
        n (int): Amount of numbers in the array.
        block_size (int): Amount of numbers in each block.
    """

    def __init__(self, array: list[float]) -> None:
        """
        Initialize the SRD structure with an input array.

        Args:
            array (list[float]): List of numbers to be queried.

        Raises:
            TypeError: If the input is not a list.
            ValueError: If the list is empty.
        """
        if not isinstance(array, list):
            raise TypeError("Input array must be of type list.")
        if not array:
            raise ValueError("Input array cannot be empty.")
        
        self.array: list[float] = array
        self.n: int = len(array)
        self.block_size: int = ceil(sqrt(self.n))
        self.feed: list[float] = [float("inf")] * ceil(self.n / self.block_size)

        # Precompute the minimum value for each block
        for index in range(self.n):
            block_idx = index // self.block_size
            self.feed[block_idx] = min(self.feed[block_idx], self.array[index])

    def update(self, index: int, new_value: float) -> None:
        """
        Update the value at a specific index in the array and refresh its block minimum.

        Args:
            index (int): Index to update.
            new_value (float): The new value to assign.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer or new_value is not a float.
        """
        if not isinstance(index, int):
            raise TypeError("Index must be an integer.")
        if not isinstance(new_value, float):
            raise TypeError("New value must be a float.")
        if not (0 <= index < len(self.array)):
            raise IndexError(f"Index {index} is out of bounds.")
        
        # Update value
        self.array[index] = new_value

        # Recompute minimum for the affected block
        block_idx = index // self.block_size
        self.feed[block_idx] = min(self.feed[block_idx], new_value)

    def query(self, left: int, right: int) -> float:
        """
        Find the minimum value in the array between two indices, inclusive.

        Args:
            left (int): Starting index of the range.
            right (int): Ending index of the range.

        Returns:
            float: The minimum value within the specified range.

        Raises:
            IndexError: If indices are out of range.
            ValueError: If left > right.
        """
        if not (isinstance(left, int) and isinstance(right, int)):
            raise TypeError("Both left and right indices must be integers.")
        if not (0 <= left < len(self.array)) or not (0 <= right < len(self.array)):
            raise IndexError("Range indices are out of bounds.")
        if left > right:
            raise ValueError("Left index cannot be greater than right index.")

        current_min = float("inf")
        
        # Compare minimum values in three sub-ranges
        # First range (left-most block, might not be fully present)
        while left < right and left % self.block_size != 0:
            current_min = min(current_min, self.array[left])
            left += 1

        # Second range (full blocks in-between)
        while left + self.block_size <= right:
            current_min = min(current_min, self.feed[left // self.block_size])
            left += self.block_size

        # Third range (right-most block, might not be fully present)
        while left <= right:
            current_min = min(current_min, self.array[left])
            left += 1

        return current_min