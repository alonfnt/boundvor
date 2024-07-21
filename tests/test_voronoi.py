import numpy as np
import pytest

from boundvor import BoundedVoronoi
from boundvor.geometry import point_in_polygon

np.random.seed(0)

def test_basic_voronoi_within_bounds():
    points = np.random.rand(10, 2)
    bounding_box = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    voronoi = BoundedVoronoi(points, bounds=bounding_box)

    assert len(voronoi.regions) == len(points)
    for region in voronoi.regions:
        for vertex in region:
            assert point_in_polygon(voronoi.vertices[vertex], bounding_box, check_boundary=True)


def test_voronoi_with_points_outside():
    points = np.array([[0.0, 3.0], [3.0, 0.0], [8.0, 8.0]])
    bounding_polygon = np.array(
        [[0.0, 0.0], [0.5, 0.2], [1.0, 0.0], [0.8, 0.5], [1.0, 1.0], [0.0, 1.0], [0.2, 0.5]]
    )
    with pytest.raises(ValueError, match="All points must be within the bounds"):
        BoundedVoronoi(points, bounds=bounding_polygon)


def test_default_bounds():
    points = np.random.rand(10, 2)
    voronoi = BoundedVoronoi(points)

    min_x, min_y = voronoi.min_bound - 0.1
    max_x, max_y = voronoi.max_bound + 0.1
    default_bounds = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]

    assert np.allclose(voronoi.bounds, default_bounds)


def test_furthest_site():
    points = np.random.rand(10, 2)
    bounding_box = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    voronoi = BoundedVoronoi(points, bounds=bounding_box, furthest_site=True)

    assert len(voronoi.regions) == len(points)
    for region in voronoi.regions:
        for vertex in region:
            assert point_in_polygon(voronoi.vertices[vertex], bounding_box, check_boundary=True)


def test_incremental():
    points = np.array([[0.3, 0.3], [0.7, 0.7], [0.5, 0.5], [0.5, 0.3], [0.5, 0.7]])
    bounding_box = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    voronoi = BoundedVoronoi(points, bounds=bounding_box, incremental=True)

    assert len(voronoi.regions) == len(points)
    for region in voronoi.regions:
        for vertex in region:
            assert point_in_polygon(voronoi.vertices[vertex], bounding_box, check_boundary=True)


def test_qhull_options():
    points = np.array([[0.3, 0.3], [0.7, 0.7], [0.5, 0.5], [0.5, 0.3], [0.5, 0.7]])
    bounding_box = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    voronoi = BoundedVoronoi(points, bounds=bounding_box, qhull_options="Qbb Qc")

    assert len(voronoi.regions) == len(points)
    for region in voronoi.regions:
        for vertex in region:
            assert point_in_polygon(voronoi.vertices[vertex], bounding_box, check_boundary=True)
