from collections import defaultdict

import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi

from boundvor.geometry import point_in_polygon, polygon_intersection

class BoundedVoronoi(ScipyVoronoi):
    def __init__(self, points, bounds=None, furthest_site=False, incremental=False, qhull_options=None):
        super().__init__(points, furthest_site=furthest_site, incremental=incremental, qhull_options=qhull_options)

        self.bounds = bounds if bounds is not None else self._default_bounds()
        self.bounds = np.asarray(self.bounds)
        print(self.bounds)

        if len(self.bounds) < 3 or self.bounds.shape[1] != 2:
            raise ValueError("Bounds must have at least three points")

        if any(not point_in_polygon(point, self.bounds) for point in points):
            raise ValueError("All points must be within the bounds")

        self._make_regions_finite()

    def _default_bounds(self):
        min_x, min_y = self.min_bound - 0.1
        max_x, max_y = self.max_bound + 0.1
        return [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]

    def _make_regions_finite(self):
        if self.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        bounded_regions = []
        new_vertices = self.vertices.tolist()
        new_vertices.extend(self.bounds)
        vertex_map = {tuple(vertex): i for i, vertex in enumerate(new_vertices)}

        center = self.points.mean(axis=0)
        _points = np.vstack((self.points, self.bounds))
        radius = np.ptp(_points).max() * 10

        all_ridges = defaultdict(list)
        for (p1, p2), (v1, v2) in zip(self.ridge_points, self.ridge_vertices):
            all_ridges[p1].append((p2, v2, v1))
            all_ridges[p2].append((p1, v1, v2))

        for p1, region in enumerate(self.point_region):
            vertices_id = self.regions[region]

            if all(v >= 0 for v in vertices_id):
                if all(point_in_polygon(self.vertices[v], self.bounds) for v in vertices_id):
                    bounded_regions.append(vertices_id)
                    continue

                clipped_vertices = polygon_intersection(self.bounds, [self.vertices[v] for v in vertices_id])
                for vertex in clipped_vertices:
                    if tuple(vertex) not in vertex_map:
                        vertex_map[tuple(vertex)] = len(new_vertices)
                        new_vertices.append(vertex)

                region_points = [vertex_map[tuple(vertex)] for vertex in clipped_vertices]
                bounded_regions.append(region_points)
                continue

            ridges = all_ridges[p1]
            region_points = [v for v in vertices_id if v >= 0]
            vertices_in_region = [new_vertices[v] for v in region_points]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                t = self.points[p2] - self.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])

                midpoint = self.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = self.vertices[v2] + direction * radius
                vertices_in_region.append(far_point)

            clipped_vertices = polygon_intersection(self.bounds, vertices_in_region)

            for vertex in clipped_vertices:
                if tuple(vertex) not in vertex_map:
                    vertex_map[tuple(vertex)] = len(new_vertices)
                    new_vertices.append(vertex)

            region_points = [vertex_map[tuple(vertex)] for vertex in clipped_vertices]
            bounded_regions.append(region_points)

        self.regions = bounded_regions
        self.vertices = np.asarray(new_vertices)

