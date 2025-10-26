"""
This module contains unit tests for validating Range Minimum Query (RMQ)
implementations, including:
    • Naive
    • SRD (Square Root Decomposition)
    • SegmentTree
    • SparseTable

The tests ensure:
    - Correct initialization behavior
    - Proper exception handling for invalid inputs
    - Correctness of update and query operations
"""

import pytest
from approaches.naive import Naive
from approaches.srd import SRD
from approaches.segment_tree import SegmentTree
from approaches.sparse_table import SparseTable


# --- FIXTURES ---

@pytest.fixture(params=[Naive, SRD, SegmentTree, SparseTable])
def rmq_class(request):
    """
    Parametrized fixture that provides each RMQ implementation one by one.

    This allows all test cases to automatically run across all implementations.
    """
    return request.param

@pytest.fixture
def rmq(rmq_class):
    """
    Fixture that initializes a reusable RMQ instance with a fixed dataset.

    Returns:
        An instance of the RMQ class initialized with [5.0, 3.0, 8.0, 2.0, 7.0]
    """
    return rmq_class([5.0, 3.0, 8.0, 2.0, 7.0])


# --- TESTS FOR INITIALIZATION ---
def test_init_valid(rmq_class):
    """Verify that a valid list initializes correctly."""
    rmq = rmq_class([1.0, 2.0, 3.0])
    assert rmq.array == [1.0, 2.0, 3.0]

def test_init_empty_list(rmq_class):
    """Ensure initializing with an empty list raises ValueError."""
    with pytest.raises(ValueError):
        rmq_class([])

def test_init_non_list(rmq_class):
    """Ensure non-list inputs raise TypeError."""
    with pytest.raises(TypeError):
        rmq_class("not a list")


# --- TESTS FOR UPDATE METHOD ---
def test_update_valid(rmq):
    """Check that a valid update modifies the array correctly."""
    rmq.update(2, 1.5)
    assert rmq.array[2] == 1.5

def test_update_index_out_of_bounds(rmq):
    """Ensure updates outside valid indices raise IndexError."""
    with pytest.raises(IndexError):
        rmq.update(-1, 4.5)
    with pytest.raises(IndexError):
        rmq.update(10, 4.5)
    
def test_update_invalid_index_type(rmq):
    """Ensure non-integer indices raise TypeError."""
    with pytest.raises(TypeError):
        rmq.update("2", 4.5)

def test_update_invalid_value_type(rmq):
    """Ensure non-float update values raise TypeError."""
    with pytest.raises(TypeError):
        rmq.update(1, "not a float")


# --- TESTS FOR QUERY METHOD ---
def test_query_valid_range(rmq):
    """Check that query returns the correct minimum for a valid range."""
    assert rmq.query(1, 3) == 2.0

def test_query_single_element(rmq):
    """Ensure querying a single element returns that element."""
    assert rmq.query(0, 0) == 5.0

def test_query_entire_range(rmq):
    """Ensure querying the full range returns the global minimum."""
    assert rmq.query(0, 4) == 2.0

def test_query_left_greater_than_right(rmq):
    """Ensure left > right raises ValueError."""
    with pytest.raises(ValueError):
        rmq.query(3, 1)

def test_query_indices_out_of_bounds(rmq):
    """Ensure out-of-bounds indices raise IndexError."""
    with pytest.raises(IndexError):
        rmq.query(-5, -1)
    with pytest.raises(IndexError):
        rmq.query(-1, 3)
    with pytest.raises(IndexError):
        rmq.query(1, 10)
    with pytest.raises(IndexError):
        rmq.query(10, 12)

def test_query_invalid_index_types(rmq):
    """Ensure non-integer query indices raise TypeError."""
    with pytest.raises(TypeError):
        rmq.query("0", 3)
    with pytest.raises(TypeError):
        rmq.query(0, 3.5)


# --- TESTS FOR QUERY METHOD AFTER UPDATES ---
def test_query_after_update_single(rmq):
    """
    Verify that queries reflect updated values correctly after a single update.
    """
    # Original minimum in [0, 4] is 2.0
    assert rmq.query(0, 4) == 2.0

    # Update index 3 (value 2.0) to 10.0, now minimum should be 3.0
    rmq.update(3, 10.0)
    assert rmq.query(0, 4) == 3.0

    # Update index 1 (value 3.0) to -5.0, new global minimum should be -5.0
    rmq.update(1, -5.0)
    assert rmq.query(0, 4) == -5.0


def test_query_after_multiple_updates(rmq):
    """
    Ensure multiple sequential updates produce correct query results.
    """
    # Perform several updates
    updates = [(0, 9.0), (2, 1.0), (4, -2.0)]
    for i, val in updates:
        rmq.update(i, val)

    # Now array should be [9.0, 3.0, 1.0, 2.0, -2.0]
    # Minimum in [0, 4] should be -2.0
    assert rmq.query(0, 4) == -2.0

    # Minimum in [1, 3] should be 1.0
    assert rmq.query(1, 3) == 1.0

    # Minimum in [0, 2] should be 1.0
    assert rmq.query(0, 2) == 1.0


def test_query_after_update_outside_range(rmq):
    """
    Ensure updates outside the queried range do not affect results.
    """
    original_min = rmq.query(1, 3)  # 2.0
    rmq.update(0, -10.0)            # Update index outside range
    assert rmq.query(1, 3) == original_min  # Should remain 2.0