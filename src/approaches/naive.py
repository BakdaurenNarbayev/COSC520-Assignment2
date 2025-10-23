class Naive:
    """
    A simple Range Minimum Query (RMQ) structure using a naive approach.
    Source: https://iq.opengenus.org/range-minimum-query-naive

    Time complexity:
        - Update: O(1)
        - Query: O(N) per range query

    Attributes:
        array (list[float]): The list of numeric values for queries and updates.
    """

    def __init__(self, array: list[float]) -> None:
        """
        Initialize the RMQ structure with an input array.

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
        self.array = array

    def update(self, index: int, new_value: float) -> None:
        """
        Update the value at a specific index in the array.

        Args:
            index (int): Index to update.
            new_value (float): The new value to assign.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the index is not an integer.
            TypeError: If the new value is not a float.
        """
        if not isinstance(index, int):
            raise TypeError("Index must be an integer.")
        if not isinstance(new_value, float):
            raise TypeError("New value must be a float.")
        if not (0 <= index < len(self.array)):
            raise IndexError(f"Index {index} is out of bounds.")
        self.array[index] = new_value

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
        for i in range(left, right + 1):
            if self.array[i] < current_min:
                current_min = self.array[i]
        return current_min