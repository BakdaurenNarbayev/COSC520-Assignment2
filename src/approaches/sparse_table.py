from math import log2, ceil

class SparseTable:
    """
    A Range Minimum Query (RMQ) structure using a Sparce Table.
    Source: https://iq.opengenus.org/sparse-table

    This approach converts the array to n x log n table with precomputed minimums for
    ranges of size 2^j, and answers queries by combining ranges.
    It is a static structure - updates are not supported efficiently (rebuilding required).

    Time Complexity:
        - Build: O(N log N)
        - Update: O(N log N) (by rebuilding)
        - Query: O(1)

    Attributes:
        array (list[float]): The list of numeric values for queries and updates.
        n (int): Amount of numbers in the array.
        lt (list[int]): Precomputed floor(log2) values for quick access.
        st (list[list[float]]): Sparse table storing range minimums.
    """

    def __init__(self, array: list[float]) -> None:
        """
        Initialize the Sparse Table with an input array.

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
        self.lt: list[int] = [0] * (self.n + 1)

        # Precompute logarithms
        for i in range(2, self.n + 1):
            self.lt[i] = self.lt[i // 2] + 1

        # Precompute Sparse Table
        k = ceil(log2(self.n))
        self.st: list[list[float]] = [[float("inf")] * k for _ in range(self.n)]

        self._build_sparse_table()

    def _build_sparse_table(self) -> None:
        """Build the sparse table for range minimum queries."""
        # Initialize level 0 (intervals of size 1)
        for i in range(self.n):
            self.st[i][0] = self.array[i]

        # Compute values for intervals with length 2^j
        j = 1
        while (1 << j) <= self.n:
            i = 0
            while (i + (1 << j)) <= self.n:
                self.st[i][j] = min(
                    self.st[i][j - 1],
                    self.st[i + (1 << (j - 1))][j - 1],
                )
                i += 1
            j += 1

    def update(self, index: int, new_value: float) -> None:
        """
        Update the value at a specific index. This triggers a full rebuild.

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
        if not (0 <= index < self.n):
            raise IndexError(f"Index {index} is out of bounds.")
        
        self.array[index] = new_value
        self._build_sparse_table()

    def query(self, left: int, right: int) -> float:
        """
        Find the minimum value in the array between two indices, inclusive.

        Args:
            left (int): Starting index of the range.
            right (int): Ending index of the range.

        Returns:
            float: The minimum value within the specified range.

        Raises:
            TypeError: If indices are not integers.
            IndexError: If indices are out of range.
            ValueError: If left > right.
        """
        if not (isinstance(left, int) and isinstance(right, int)):
            raise TypeError("Both left and right indices must be integers.")
        if not (0 <= left < len(self.array)) or not (0 <= right < len(self.array)):
            raise IndexError("Range indices are out of bounds.")
        if left > right:
            raise ValueError("Left index cannot be greater than right index.")

        j = self.lt[right - left + 1]
        return min(self.st[left][j], self.st[right - (1 << j) + 1][j])