"""The fast multiset must retain the scalar instrument's exact byte semantics."""

from collections import Counter

import numpy as np
import pytest

from forge_mind.workspace.mesh_evidence import triangle_inventory


def scalar_inventory(triangles):
    values = np.array(triangles, dtype=np.float32, copy=True)
    values[values == 0] = 0
    result = Counter()
    for triangle in values:
        if any(np.array_equal(triangle[a], triangle[b]) for a, b in ((0, 1), (1, 2), (2, 0))):
            continue
        start = min(range(3), key=lambda i: tuple(triangle[i]))
        result[np.roll(triangle, -start, axis=0).tobytes()] += 1
    return result


def test_batched_inventory_matches_scalar_across_boundary_and_lexicographic_ties():
    rng = np.random.default_rng(452)
    triangles = rng.integers(-3, 4, size=(32777, 3, 3)).astype(np.float32)
    triangles[::13, 1] = triangles[::13, 0]  # collapsed triangles
    triangles[::17, 2] = triangles[::17, 0]
    triangles[::19, 2] = triangles[::19, 1]
    triangles[32768:] = triangles[:9]  # multiplicities span the batch boundary
    assert triangle_inventory(triangles) == scalar_inventory(triangles)
    assert triangle_inventory(triangles[:, ::-1]) == scalar_inventory(triangles[:, ::-1])


def test_inventory_preserves_readonly_input_and_signed_zero_bytes():
    triangles = np.array([[[-0.0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    before = triangles.tobytes()
    triangles.flags.writeable = False
    assert triangle_inventory(triangles) == scalar_inventory(triangles)
    assert triangles.tobytes() == before


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_even_in_collapsed_triangles_is_rejected(value):
    triangles = np.zeros((2, 3, 3))
    triangles[1, 0, 0] = value
    with pytest.raises(ValueError, match="Nonfinite"):
        triangle_inventory(triangles)


def test_empty_and_all_collapsed_inventory():
    assert triangle_inventory(np.empty((0, 3, 3))) == Counter()
    assert triangle_inventory(np.zeros((4, 3, 3))) == Counter()


@pytest.mark.parametrize("shape", [(3, 3), (4, 3, 2), (1, 4, 3)])
def test_malformed_triangle_array_is_rejected(shape):
    with pytest.raises(ValueError, match="shape"):
        triangle_inventory(np.zeros(shape))
