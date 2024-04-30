import pytest
from numpy import int16, unique
from ridge.utils import neighbour_vectors


@pytest.mark.parametrize("dims", [2, 3, 4, 5])
def test_neighbour_vectors(dims: int):
    v = neighbour_vectors(n=dims, cutoff=1, dtype=int16)
    # verify shape of output
    assert v.shape == (2 * dims, dims)
    # check total displacement doesn't exceed the cutoff
    assert (abs(v).sum(axis=1) == 1).all()
    # verify that all returned vectors are unique
    assert v.shape == unique(v, axis=0).shape

    # repeat test including all neighbours
    v = neighbour_vectors(n=dims, cutoff=dims, dtype=int16)
    assert v.shape == (3**dims - 1, dims)
    assert (abs(v).sum(axis=1) <= dims).all()
    assert v.shape == unique(v, axis=0).shape
