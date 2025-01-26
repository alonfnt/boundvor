import numpy as np
import pytest

from boundvor.geometry import is_left_of_edge, compute_intersection, clip, sort_polygon, point_in_polygon, line_intersect, polygon_intersection



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

def test_is_left_of_edge():
    p = np.array([1.0, 1.0])
    edge = np.array([[0.0, 0.0], [2.0, 2.0]])
    assert is_left_of_edge(p, edge) == True

    p = np.array([3.0, 1.0])
    assert is_left_of_edge(p, edge) == False

def test_compute_intersection():
    s = np.array([0.0, 0.0])
    e = np.array([2.0, 2.0])
    edge = np.array([[1.0, 0.0], [1.0, 2.0]])
    intersection = compute_intersection(s, e, edge)
    expected_intersection = np.array([1.0, 1.0])
    assert np.allclose(intersection, expected_intersection)

def test_clip():
    subject_polygon = np.array([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
    clip_polygon = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]])
    clipped_polygon = clip(subject_polygon, clip_polygon)
    expected_clipped_polygon = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]])
    expected_clipped_polygon = sort_polygon(expected_clipped_polygon)
    clipped_polygon = sort_polygon(clipped_polygon)
    assert np.array_equal(clipped_polygon, expected_clipped_polygon)

def test_sort_polygon():
    poly = np.array([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
    sorted_poly = sort_polygon(poly)
    expected_sorted_poly = np.array([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
    assert np.allclose(sorted_poly, expected_sorted_poly)

def test_point_in_polygon():
    polygon = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    point = np.array([2.0, 2.0])
    assert point_in_polygon(point, polygon) == True

    point = np.array([5.0, 5.0])
    assert point_in_polygon(point, polygon) == False

    point = np.array([4.0, 4.0])
    assert point_in_polygon(point, polygon, check_boundary=True) == True

def test_line_intersect():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([2.0, 2.0])
    q1 = np.array([0.0, 2.0])
    q2 = np.array([2.0, 0.0])
    assert line_intersect(p1, p2, q1, q2) == True

    q1 = np.array([2.0, 2.0])
    q2 = np.array([4.0, 4.0])
    assert line_intersect(p1, p2, q1, q2) == False

def test_polygon_intersection():
    poly1 = np.array([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
    poly2 = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]])
    intersection = polygon_intersection(poly1, poly2)
    expected_intersection = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]])
    intersection = sort_polygon(intersection)
    expected_intersection = sort_polygon(expected_intersection)
    assert np.allclose(intersection, expected_intersection)