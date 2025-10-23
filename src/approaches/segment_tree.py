class SegmentTree:
    """
    A Range Minimum Query (RMQ) structure using a Segment Tree.
    Source: https://iq.opengenus.org/range-minimum-query-segment-tree

    This approach constructs a binary tree where each node stores
    the minimum value for a segment of the array.

    Time complexity:
        - Build: O(N)
        - Update: O(log N)
        - Query: O(log N)

    Attributes:
        array (list[float]): The list of numeric values for queries and updates.
        tree (list[float]): Segment tree storing range minimums.
        n (int): Amount of numbers in the array.
    """

    def __init__(self, array: list[float]) -> None:
        """
        Initialize the Segment Tree structure with an input array.

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
        # Safely allocate tree nodes for recursion
        self.tree: list[float] = [float("inf")] * (4 * self.n)

        self._construct_tree_recursive(0, 0, self.n - 1)

    def _construct_tree_recursive(self, node, start, end) -> None:
        """
        Recursively build the segment tree.

        node represents the root
        start represents the starting index
        end represents the last index
        """
        # Leaf node will have a single element
        if start == end:
            self.tree[node] = self.array[start]
        else:
            mid = (start + end) // 2
            # Recurse on the left child
            self._construct_tree_recursive(2 * node + 1, start, mid)
            # Recurse on the right child
            self._construct_tree_recursive(2 * node + 2, mid + 1, end)
            # Internal node will hold the minimum of two children
            self.tree[node] = min(self.tree[2 * node + 1], self.tree[2 * node + 2])

    def update(self, index: int, new_value: float) -> None:
        """
        Update the value at a specific index in the array and refresh its segment minimum.

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
        
        self._update_recursive(0, 0, self.n - 1, index, new_value)

    def _update_recursive(self, node, start, end, index, new_value) -> None:
        """Recursively update a single element in the segment tree."""
        if start == end:
            self.array[index] = new_value
            self.tree[node] = new_value
        else:
            mid = (start + end) // 2
            # If index is in the left child, recurse on the left child
            if index <= mid:
                self._update_recursive(2 * node + 1, start, mid, index, new_value)
            # if index is in the right child, recurse on the right child
            else:
                self._update_recursive(2 * node + 2, mid + 1, end, index, new_value)
            # Internal node will hold the minimum of two children
            self.tree[node] = min(self.tree[2 * node + 1], self.tree[2 * node + 2])

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
        
        return self._query_recursive(0, 0, self.n - 1, left, right)

    def _query_recursive(self, node, start, end, left, right) -> float:
        """Recursively find the minimum value in the given range."""
        # Case 1
        # range represented by a node is completely outside the range
        # return the maximum value 
        if right < start or end < left:
            return float("inf")
        
        # Case 2
        # range represented by a node is completely inside the given range
        # return the node value itself
        if left <= start and end <= right:
            return self.tree[node]
        
        # Case 3     
        # range represented by a node is partially inside and partially outside the given range
        # recurse through the left and right subtree
        mid = (start + end) // 2
        q1 = self._query_recursive(2 * node + 1, start, mid, left, right)
        q2 = self._query_recursive(2 * node + 2, mid + 1, end, left, right)
        return min(q1, q2)