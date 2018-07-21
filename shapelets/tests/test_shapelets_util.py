import pytest, random
from shapelets.shapelet_utils import shapelet_utils


@pytest.mark.parametrize('execution_number', range(5))
def test_merge(execution_number):
    fst = [int(random.random()*100) for x in range(100)]
    fst.sort()
    snd = [x for x in range(0,75,3)]

    from copy import deepcopy
    expected = deepcopy(fst) + snd
    expected.sort()

    out = shapelet_utils.merge(fst, snd)
    assert expected == out


@pytest.mark.parametrize('a,b,expected',[ 
    ([1,2,3], [1,2,3], 0),
    ([1,2,3], [3,2,1], 1.490711)
])
def test_subsequence_distance_offset(a, b, expected):
    dist = shapelet_utils.subsequence_distance(a,b)
    assert abs(dist - expected) < 0.0001