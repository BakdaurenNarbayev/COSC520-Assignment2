import pytest
from approaches.naive import Naive
from approaches.srd import SRD


# --- FIXTURES ---

@pytest.fixture(params=[Naive, SRD])
def rmq_class(request):
    """
    Fixture that supplies each RMQ class (Naive and SRD (Square Root Decomposition))
    one by one to all tests that depend on it.
    """
    return request.param

@pytest.fixture
def rmq(rmq_class):
    """Fixture that initializes an RMQ instance with a default test array."""
    return rmq_class([5.0, 3.0, 8.0, 2.0, 7.0])


# --- TESTS FOR INITIALIZATION ---
def test_init_valid(rmq_class):
    rmq = rmq_class([1.0, 2.0, 3.0])
    assert rmq.array == [1.0, 2.0, 3.0]

def test_init_empty_list(rmq_class):
    with pytest.raises(ValueError):
        rmq_class([])

def test_init_non_list(rmq_class):
    with pytest.raises(TypeError):
        rmq_class("not a list")


# --- TESTS FOR UPDATE METHOD ---
def test_update_valid(rmq):
    rmq.update(2, 1.5)
    assert rmq.array[2] == 1.5

def test_update_index_out_of_bounds(rmq):
    with pytest.raises(IndexError):
        rmq.update(-1, 4.5)
    with pytest.raises(IndexError):
        rmq.update(10, 4.5)
    
def test_update_invalid_index_type(rmq):
    with pytest.raises(TypeError):
        rmq.update("2", 4.5)

def test_update_invalid_value_type(rmq):
    with pytest.raises(TypeError):
        rmq.update(1, "not a float")


# --- TESTS FOR QUERY METHOD ---
def test_query_valid_range(rmq):
    assert rmq.query(1, 3) == 2.0

def test_query_single_element(rmq):
    assert rmq.query(0, 0) == 5.0

def test_query_entire_range(rmq):
    assert rmq.query(0, 4) == 2.0

def test_query_left_greater_than_right(rmq):
    with pytest.raises(ValueError):
        rmq.query(3, 1)

def test_query_indices_out_of_bounds(rmq):
    with pytest.raises(IndexError):
        rmq.query(-5, -1)
    with pytest.raises(IndexError):
        rmq.query(-1, 3)
    with pytest.raises(IndexError):
        rmq.query(1, 10)
    with pytest.raises(IndexError):
        rmq.query(10, 12)

def test_query_invalid_index_types(rmq):
    with pytest.raises(TypeError):
        rmq.query("0", 3)
    with pytest.raises(TypeError):
        rmq.query(0, 3.5)