import numpy as np
cimport numpy as np


def point_in_polygon(point, polygon, bint check_boundary=False):

    # Ensure inputs are numpy arrays
    point = np.asarray(point, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)

    cdef int n, i
    cdef double x, y, p1x, p1y, p2x, p2y, xinters
    cdef bint inside = False

    if polygon.shape[0] < 3:
        raise ValueError("Polygon must have at least three points")

    x, y = point
    n = polygon.shape[0]
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
                if abs((p2x - p1x) * (y - p1y) - (p2y - p1y) * (x - p1x)) < 1e-9:
                    return True

    return inside

def line_intersect(np.ndarray[np.float64_t, ndim=1] p1, 
                   np.ndarray[np.float64_t, ndim=1] p2, 
                   np.ndarray[np.float64_t, ndim=1] q1, 
                   np.ndarray[np.float64_t, ndim=1] q2):
    def ccw(np.ndarray[np.float64_t, ndim=1] A, 
            np.ndarray[np.float64_t, ndim=1] B, 
            np.ndarray[np.float64_t, ndim=1] C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


def is_left_of_edge(np.ndarray[np.float64_t, ndim=1] p, 
           np.ndarray[np.float64_t, ndim=2] edge):
    return (edge[1, 0] - edge[0, 0]) * (p[1] - edge[0, 1]) >= (edge[1, 1] - edge[0, 1]) * (p[0] - edge[0, 0])


def compute_intersection(np.ndarray[np.float64_t, ndim=1] s, 
                         np.ndarray[np.float64_t, ndim=1] e, 
                         np.ndarray[np.float64_t, ndim=2] edge):
    cdef double dc0 = edge[0, 0] - edge[1, 0]
    cdef double dc1 = edge[0, 1] - edge[1, 1]
    cdef double dp0 = s[0] - e[0]
    cdef double dp1 = s[1] - e[1]
    cdef double n1 = edge[0, 0] * edge[1, 1] - edge[0, 1] * edge[1, 0]
    cdef double n2 = s[0] * e[1] - s[1] * e[0]
    cdef double n3 = 1.0 / (dc0 * dp1 - dc1 * dp0)
    return np.array([(n1 * dp0 - n2 * dc0) * n3, (n1 * dp1 - n2 * dc1) * n3])


def clip(np.ndarray[np.float64_t, ndim=2] subject_polygon, 
         np.ndarray[np.float64_t, ndim=2] clip_polygon):
    
    cdef list output_list = subject_polygon.tolist()
    clip_polygon = np.vstack([clip_polygon, clip_polygon[0]])  # Ensure the clip polygon is closed

    for clip_edge in zip(clip_polygon[:-1], clip_polygon[1:]):
        clip_edge = np.asarray(clip_edge)
        input_list = output_list[:]
        output_list = []
        if len(input_list) == 0:
            break
        s = input_list[-1]
        s = np.asarray(s, dtype=np.float64)
        for e in input_list: 
            e = np.asarray(e, dtype=np.float64)
            if is_left_of_edge(e, clip_edge):
                if not is_left_of_edge(s, clip_edge):
                    output_list.append(compute_intersection(s, e, clip_edge))
                output_list.append(e)
            elif is_left_of_edge(s, clip_edge):
                output_list.append(compute_intersection(s, e, clip_edge))
            s = e

    if len(output_list) == 0:
        return np.empty((0, 2))
    return np.array(output_list)


def sort_polygon(np.ndarray[np.float64_t, ndim=2] poly):
    center = np.mean(poly, axis=0)
    return np.asarray(sorted(poly, key=lambda x: np.arctan2(x[1] - center[1], x[0] - center[0])))


def polygon_intersection(poly1, poly2):
    # Ensure inputs are numpy arrays
    poly1 = np.array(poly1, dtype=np.float64)
    poly2 = np.array(poly2, dtype=np.float64)

    poly1 = sort_polygon(poly1)
    poly2 = sort_polygon(poly2)

    clipped = clip(poly1, poly2)
    return clipped