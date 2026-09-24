import math

import pytest

from aegisrover.mapping.inflation import distance_field, inflate


def test_inflation_uses_geometric_radius():
    field = distance_field(12, 12, [(5, 5)], 0.5)
    cost = inflate(field, 1.0)
    assert cost[5][5] == pytest.approx(1.0)
    assert cost[5][7] == pytest.approx(0.0)
    assert cost[5][8] == pytest.approx(0.0)


def test_distance_field_is_euclidean_on_diagonals():
    field = distance_field(12, 12, [(5, 5)], 0.5)
    # One diagonal step is sqrt(2) cells, not 1 cell.
    assert field[6][6] == pytest.approx(math.sqrt(2) * 0.5)
    # A (2, 1) offset is hypot(2, 1) cells, not 2 grid steps.
    assert field[6][7] == pytest.approx(math.hypot(2, 1) * 0.5)
    assert field[4][3] == pytest.approx(math.hypot(2, 1) * 0.5)


def test_inflation_boundary_is_circular_not_square():
    field = distance_field(12, 12, [(5, 5)], 0.5)
    cost = inflate(field, 1.0)
    # Equal Euclidean distances must give equal cost on axis and diagonal.
    assert cost[5][6] == pytest.approx(0.5)                     # 0.5 m along +x
    assert cost[6][6] == pytest.approx(1 - math.sqrt(2) * 0.5)  # 0.5*sqrt(2) m diagonal
    # (7,7) is sqrt(2) m out: outside a 1 m circle.
    assert cost[7][7] == pytest.approx(0.0)
    # With a 1.2 m radius the old square zone (2 grid steps counted as "1.0 m")
    # would still cover (7,7); the circular one must not.
    assert inflate(field, 1.2)[7][7] == pytest.approx(0.0)


def test_distance_field_takes_nearest_of_multiple_obstacles():
    field = distance_field(12, 12, [(0, 0), (11, 11)], 1.0)
    assert field[2][2] == pytest.approx(math.sqrt(8))
    assert field[5][8] == pytest.approx(math.hypot(3, 6))


def test_distance_field_without_obstacles_is_infinite():
    field = distance_field(4, 3, [], 0.5)
    assert all(v == math.inf for row in field for v in row)
