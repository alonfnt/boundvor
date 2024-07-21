import numpy as np

def point_in_polygon(point, polygon, check_boundary=False):

    if len(polygon) < 3:
        raise ValueError("Polygon must have at least three points")

    x, y = point
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    if check_boundary:
        for i in range(n):
            p1x, p1y = polygon[i]
            p2x, p2y = polygon[(i + 1) % n]
            if (min(p1x, p2x) <= x <= max(p1x, p2x)) and (min(p1y, p2y) <= y <= max(p1y, p2y)):
                if (p2x - p1x) * (y - p1y) == (p2y - p1y) * (x - p1x):
                    return True

    return inside

def line_intersect(p1, p2, q1, q2):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

def polygon_intersection(poly1, poly2):
    def clip(subject_polygon, clip_polygon):
        def inside(p, edge):
            return (edge[1][0] - edge[0][0]) * (p[1] - edge[0][1]) >= (edge[1][1] - edge[0][1]) * (p[0] - edge[0][0])

        def compute_intersection(s, e, edge):
            dc = [edge[0][0] - edge[1][0], edge[0][1] - edge[1][1]]
            dp = [s[0] - e[0], s[1] - e[1]]
            n1 = edge[0][0] * edge[1][1] - edge[0][1] * edge[1][0]
            n2 = s[0] * e[1] - s[1] * e[0]
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
            return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

        output_list = subject_polygon
        clip_polygon.append(clip_polygon[0])  # Ensure the clip polygon is closed
        for clip_edge in zip(clip_polygon[:-1], clip_polygon[1:]):
            input_list = output_list
            output_list = []
            if len(input_list) == 0:
                break
            s = input_list[-1]
            for e in input_list:
                if inside(e, clip_edge):
                    if not inside(s, clip_edge):
                        output_list.append(compute_intersection(s, e, clip_edge))
                    output_list.append(e)
                elif inside(s, clip_edge):
                    output_list.append(compute_intersection(s, e, clip_edge))
                s = e
        return output_list

    def sort_polygon(poly):
        center = np.mean(poly, axis=0)
        return sorted(poly, key=lambda x: np.arctan2(x[1] - center[1], x[0] - center[0]))

    poly1 = sort_polygon(poly1)
    poly2 = sort_polygon(poly2)

    clipped = clip(poly1, poly2)
    return clipped

