from collections import defaultdict
import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi


class Voronoi(ScipyVoronoi):

    def __init__(self, points, bounds, furthest_site=False, incremental=False, qhull_options=None):
        super().__init__(points, furthest_site, incremental, qhull_options)
        self.bounds = bounds
        self.extend_regions()

    def extend_regions(self):
        new_regions = defaultdict(list)
        ptp = np.ptp(np.vstack((self.points, self.bounds)), axis=0)
        center = np.mean(self.points, axis=0)
        for pointidx, simplex in zip(self.ridge_points, self.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                # Evaluate of the simplex is within the bounds if not find the interesction with the polygon
                simplex_1_inside = self.point_is_inside(self.vertices[simplex[0]], self.bounds)
                simplex_2_inside = self.point_is_inside(self.vertices[simplex[1]], self.bounds)
                if simplex_1_inside and simplex_2_inside:
                    new_regions[pointidx[0]].extend(list(map(int, simplex)))
                    new_regions[pointidx[1]].extend(list(map(int, simplex)))
                else:
                    # Find the point that is inside and the one that is outside, and replace the outside one with the interesction with the polygon
                    intersection_point = self.intersection_point(self.vertices[simplex], self.bounds)
                    self.vertices = np.vstack([self.vertices, intersection_point])
                    new_vertex_index = int(len(self.vertices) - 1)
                    new_simplex = [simplex[0], new_vertex_index] if not simplex_1_inside else [simplex[1], new_vertex_index]
                    new_regions[pointidx[0]].extend(list(map(int, new_simplex)))
                    new_regions[pointidx[1]].extend(list(map(int, new_simplex)))

            else:
                i = int(simplex[simplex >= 0][0])
                t = self.points[pointidx[1]] - self.points[pointidx[0]]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = self.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                if self.furthest_site:
                    direction = -direction
                new_point = self.vertices[i] + direction * ptp.max()
                line = [self.vertices[i], new_point]
                intersection = self.intersection_point(line, self.bounds)
                self.vertices = np.vstack([self.vertices, intersection])
                new_vertex_index = len(self.vertices) - 1
                new_regions[pointidx[0]].extend([i, new_vertex_index])
                new_regions[pointidx[1]].extend([i, new_vertex_index])

        #self.regions = [new_regions[i] for i in range(len(self.points))]
        sizes = [len(r) for r in self.regions]
        infinity_point = sizes.index(0)
        #self.regions = [list(set(new_regions[p])) for p in self.point_region]
        print(self.regions)
        print(new_regions)
        self.regions = [list(set(new_regions[p])) for p in range(len(self.points))]

        # Sort the points on the region to be clockwise
        for i, region in enumerate(self.regions):
            if len(region) > 2:
                region = np.array(region)
                region = region[np.argsort(np.arctan2(self.vertices[region, 1] - self.points[i, 1], self.vertices[region, 0] - self.points[i, 0]))]
                self.regions[i] = region.tolist()

        self.regions.insert(infinity_point, [])


    @staticmethod
    def point_is_inside(point, polygon):
        # Winding number algorithm
        vertx, verty = polygon.T
        rev_vertx, rev_verty = np.roll(polygon, -1, axis=0).T
        y_crosses = (verty > point[1]) != (rev_verty > point[1])
        x_crosses = point[0] < (rev_vertx - vertx) * (point[1] - verty) / (rev_verty - verty) + vertx
        return np.sum(y_crosses & x_crosses) % 2 == 1



    @staticmethod
    def intersection_point(line, polygon):
        # Get all the edges of polygon
        edges = [[polygon[i], polygon[(i + 1) % len(polygon)]] for i in range(len(polygon))]
        for edge in edges:
            try:
                return Voronoi.intersection(line, edge)
            except:
                pass
        raise ValueError('No intersection found')

    @staticmethod
    def intersection(AB, CD):
        """Finds the intersection point of two lines AB and CD"""
        (A, B), (C, D) = AB, CD
        mAB = (B[1] - A[1]) / (B[0] - A[0] + 1e-10)
        mCD = (D[1] - C[1]) / (D[0] - C[0] + 1e-10)

        if mAB == mCD:
            raise ValueError('Lines are parallel')

        if (B[0] - A[0]) == 0:
            # Line AB is vertical
            x = A[0]
            y = mCD * (x - C[0]) + C[1]
        elif (D[0] - C[0]) == 0:
            # Line CD is vertical
            x = C[0]
            y = mAB * (x - A[0]) + A[1]
        else:
            x = (mAB * A[0] - A[1] - mCD * C[0] + C[1]) / (mAB - mCD)
            y = mAB * (x - A[0]) + A[1]
        return np.array([x, y])


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    np.random.seed(1234)
    points = np.random.rand(15, 2) * 0.9 + 0.05
    polygon = np.array([[0., 0.],[0., 1.], [1., 1.], [1., 0]])

    assert Voronoi.point_is_inside([0.5, 0.5], polygon) == True
    assert Voronoi.point_is_inside([0.5, 1.5], polygon) == False
    assert Voronoi.point_is_inside([0.5, -0.5], polygon) == False
    assert Voronoi.point_is_inside([0.5, 0.31], polygon) == True



    plt.figure(figsize=(8,4), dpi=200)
    ax = plt.subplot(121, aspect='equal')
    ax.set_title('Scipy Voronoi')
    vor = ScipyVoronoi(points)
    ax.add_patch(Polygon(polygon, edgecolor='red', fill=None, zorder=20))
    ax.plot(points[:, 0], points[:, 1], 'ko')
    for p in vor.point_region:
        region = vor.regions[p]
        print(region)
        if not -1 in region  and len(region) > 0:
            cell = Polygon([vor.vertices[i] for i in region], edgecolor='k')
            ax.add_patch(cell)
    for i, point in enumerate(points):
        ax.text(point[0], point[1], str(i), color='red', fontsize=6)
    for i, vertex in enumerate(vor.vertices):
        ax.text(vertex[0], vertex[1], str(i), color='orange', fontsize=6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)

    ax = plt.subplot(122, aspect='equal')
    ax.set_title('Extended Voronoi')
    ax.add_patch(Polygon(polygon, edgecolor='red', fill=None, zorder=20))
    vor = Voronoi(points, polygon)
    print('--'*20)
    for p in vor.point_region:
        region = vor.regions[p]
        print(region)
        if not -1 in region and len(region) > 0:
            cell = Polygon([vor.vertices[i] for i in region], edgecolor='k')
            ax.add_patch(cell)
    ax.plot(points[:, 0], points[:, 1], 'ko')
    for i, point in enumerate(points):
        ax.text(point[0], point[1], str(i), color='red', fontsize=6)
    for i, vertex in enumerate(vor.vertices):
        ax.text(vertex[0], vertex[1], str(i), color='orange', fontsize=6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)


    plt.tight_layout()
    plt.show()

