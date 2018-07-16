import pytest, random
from shapelets.shapelet_utils import shapelet_utils

def test_merge():
    fst = [int(random.random()*100) for x in range(100)]
    fst.sort()
    snd = [x for x in range(0,20,2)]

    from copy import deepcopy
    expected = deepcopy(fst) + snd
    expected.sort()

    out = shapelet_utils.merge(fst, snd)

    assert expected == out