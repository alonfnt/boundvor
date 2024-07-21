import numpy as np
import boundvor.geometry_c as c_api

def point_in_polygon(point, polygon, check_boundary=False):

    if len(polygon) < 3:
         raise ValueError("Polygon must have at least three points")

    point = np.asarray(point, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)

    return c_api.point_in_polygon(point, polygon, check_boundary)


def _sort_counter_clockwise(polygon):
    (center_x, center_y) = np.mean(polygon, axis=0)
    key_fun = lambda x: np.arctan2(x[1] - center_y, x[0] - center_x)
    return sorted(polygon, key=key_fun)


def polygon_intersection(subject_polygon, clipping_polygon):

    # Sort counter-clockwise
    subject_polygon = _sort_counter_clockwise(list(subject_polygon))
    clipping_polygon = _sort_counter_clockwise(list(clipping_polygon))

    clipped_polygon = c_api.polygon_intersection(subject_polygon, clipping_polygon)
    clipped_polygon = np.asarray(clipped_polygon, dtype=np.float64)
    return clipped_polygon
