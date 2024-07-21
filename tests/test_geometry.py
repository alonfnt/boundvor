import numpy as np
import pytest

from boundvor.geometry import point_in_polygon


def test_point_inside_polygon():
    polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]
    point = (2, 2)
    assert point_in_polygon(point, polygon) == True


def test_point_outside_polygon():
    polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]
    point = (5, 5)
    assert point_in_polygon(point, polygon) == False


def test_point_on_vertex():
    polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]
    point = (0, 0)
    assert point_in_polygon(point, polygon, check_boundary=True) == True


def test_point_on_horizontal_edge():
    polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]
    point = (2, 0)
    assert point_in_polygon(point, polygon, check_boundary=True) == True


def test_point_on_vertical_edge():
    polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]
    point = (4, 2)
    assert point_in_polygon(point, polygon, check_boundary=True) == True


def test_point_on_diagonal_edge():
    polygon = [(0, 0), (2, 2), (0, 4), (4, 4), (4, 0)]
    point = (1, 1)
    assert point_in_polygon(point, polygon, check_boundary=True) == True


def test_point_outside_near_edge():
    polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]
    point = (4.1, 2)
    assert point_in_polygon(point, polygon) == False


def test_point_outside_near_vertex():
    polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]
    point = (0, 4.1)
    assert point_in_polygon(point, polygon) == False


def test_point_on_concave_edge():
    polygon = [(0, 0), (2, 2), (0, 4), (4, 4), (2, 2), (4, 0)]
    point = (2, 2)
    assert point_in_polygon(point, polygon, check_boundary=True) == True


def test_point_inside_concave():
    polygon = [(0, 0), (2, 2), (0, 4), (4, 4), (2, 2), (4, 0)]
    point = (1, 1)
    assert point_in_polygon(point, polygon, check_boundary=True) == True


def test_point_outside_concave():
    polygon = [(0, 0), (2, 2), (0, 4), (4, 4), (2, 2), (4, 0)]
    point = (4, 5)
    assert point_in_polygon(point, polygon) == False


def test_invalid_polygon_exception():
    with pytest.raises(ValueError, match="Polygon must have at least three points"):
        point_in_polygon((1, 1), [(0, 0), (1, 0)])
