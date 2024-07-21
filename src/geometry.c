#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

static int point_in_polygon(double x, double y, double *polygon, int n, int check_boundary) {
    int i, j;
    int inside = 0;

    // Check if the point is inside the polygon
    for (i = 0, j = n - 1; i < n; j = i++) {
        double xi = polygon[2 * i];
        double yi = polygon[2 * i + 1];
        double xj = polygon[2 * j];
        double yj = polygon[2 * j + 1];

        int intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) {
            inside = !inside;
        }
    }

    // Check if the point is on the boundary of the polygon
    if (check_boundary) {
        for (i = 0, j = n - 1; i < n; j = i++) {
            double xi = polygon[2 * i];
            double yi = polygon[2 * i + 1];
            double xj = polygon[2 * j];
            double yj = polygon[2 * j + 1];
            if ((x >= fmin(xi, xj)) && (x <= fmax(xi, xj)) && (y >= fmin(yi, yj)) && (y <= fmax(yi, yj))) {
                if ((xj - xi) * (y - yi) == (yj - yi) * (x - xi)) {
                    return 1;  // Point is on the boundary
                }
            }
        }
    }

    return inside;
}

// Helper function to determine if a point is inside a clipping edge
static int inside(double *p, double *edge_start, double *edge_end) {
    return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) >= (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0]);
}

static void compute_intersection(double *s, double *e, double *edge_start, double *edge_end, double *result) {
    double dc[2] = {edge_start[0] - edge_end[0], edge_start[1] - edge_end[1]};
    double dp[2] = {s[0] - e[0], s[1] - e[1]};
    double n1 = edge_start[0] * edge_end[1] - edge_start[1] * edge_end[0];
    double n2 = s[0] * e[1] - s[1] * e[0];
    double det = dc[0] * dp[1] - dc[1] * dp[0];

    if (det == 0) {
        result[0] = result[1] = NAN;
        return;
    }

    result[0] = (n1 * dp[0] - n2 * dc[0]) / det;
    result[1] = (n1 * dp[1] - n2 * dc[1]) / det;
}


// Computes the interesection between two polygons using the Sutherland-Hodgman algorithm
static PyObject* polygon_intersection(double *subject_polygon, int subject_size, double *clip_polygon, int clip_size) {
    PyObject *result = PyList_New(0);
    if (!result) return NULL;

    // Allocate memory for closed clip polygon (add the first point at the end)
    double *clip_polygon_closed = malloc((clip_size + 1) * 2 * sizeof(double));
    if (!clip_polygon_closed) {
        Py_DECREF(result);
        return NULL;
    }
    memcpy(clip_polygon_closed, clip_polygon, clip_size * 2 * sizeof(double));
    clip_polygon_closed[2 * clip_size] = clip_polygon[0];
    clip_polygon_closed[2 * clip_size + 1] = clip_polygon[1];

    // Allocate memory for output list
    double *output_list = malloc(subject_size * 2 * sizeof(double));
    if (!output_list) {
        free(clip_polygon_closed);
        Py_DECREF(result);
        return NULL;
    }
    memcpy(output_list, subject_polygon, subject_size * 2 * sizeof(double));
    int output_size = subject_size * 2;

    for (int i = 0; i < clip_size; ++i) {
        double *clip_edge_start = clip_polygon_closed + 2 * i;
        double *clip_edge_end = clip_polygon_closed + 2 * (i + 1);

        double *input_list = malloc(output_size * sizeof(double));
        if (!input_list) {
            free(clip_polygon_closed);
            free(output_list);
            Py_DECREF(result);
            return NULL;
        }
        memcpy(input_list, output_list, output_size * sizeof(double));
        int input_size = output_size / 2;
        output_size = 0;

        double *s = input_list + 2 * (input_size - 1);
        for (int j = 0; j < input_size; ++j) {
            double *e = input_list + 2 * j;

            double intersection[2];
            compute_intersection(s, e, clip_edge_start, clip_edge_end, intersection);

            if (inside(e, clip_edge_start, clip_edge_end)) {
                if (!inside(s, clip_edge_start, clip_edge_end)) {
                    output_list[output_size++] = intersection[0];
                    output_list[output_size++] = intersection[1];
                }
                output_list[output_size++] = e[0];
                output_list[output_size++] = e[1];
            } else if (inside(s, clip_edge_start, clip_edge_end)) {
                output_list[output_size++] = intersection[0];
                output_list[output_size++] = intersection[1];
            }
            s = e;
        }
        free(input_list);
    }

    free(clip_polygon_closed);

    for (int i = 0; i < output_size; i += 2) {
        PyObject *point = Py_BuildValue("(dd)", output_list[i], output_list[i + 1]);
        if (!point) {
            // If Py_BuildValue fails, clean up and return NULL
            free(output_list);
            Py_DECREF(result);
            return NULL;
        }
        PyList_Append(result, point);
        Py_DECREF(point);
    }

    free(output_list);
    return result;
}

static PyObject* py_point_in_polygon(PyObject* self, PyObject* args) {
    PyObject *point_obj, *polygon_obj;
    int check_boundary;

    if (!PyArg_ParseTuple(args, "OOi", &point_obj, &polygon_obj, &check_boundary)) {
        return NULL;
    }

    PyArrayObject *point_array = (PyArrayObject*) PyArray_FROM_OTF(point_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *polygon_array = (PyArrayObject*) PyArray_FROM_OTF(polygon_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!point_array || !polygon_array) {
        Py_XDECREF(point_array);
        Py_XDECREF(polygon_array);
        return NULL;
    }

    double x = *(double*) PyArray_GETPTR1(point_array, 0);
    double y = *(double*) PyArray_GETPTR1(point_array, 1);
    double *polygon = (double*) PyArray_DATA(polygon_array);
    int n = PyArray_DIM(polygon_array, 0);

    int result = point_in_polygon(x, y, polygon, n, check_boundary);

    Py_DECREF(point_array);
    Py_DECREF(polygon_array);

    return PyBool_FromLong(result);
}

static PyObject* py_polygon_intersection(PyObject* self, PyObject* args) {
    PyObject *poly1_obj, *poly2_obj;
    if (!PyArg_ParseTuple(args, "OO", &poly1_obj, &poly2_obj)) {
        return NULL;
    }

    PyArrayObject *poly1_array = (PyArrayObject*) PyArray_FROM_OTF(poly1_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *poly2_array = (PyArrayObject*) PyArray_FROM_OTF(poly2_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!poly1_array || !poly2_array) {
        Py_XDECREF(poly1_array);
        Py_XDECREF(poly2_array);
        return NULL;
    }

    double *poly1 = (double*) PyArray_DATA(poly1_array);
    int poly1_size = PyArray_DIM(poly1_array, 0);
    double *poly2 = (double*) PyArray_DATA(poly2_array);
    int poly2_size = PyArray_DIM(poly2_array, 0);

    PyObject *result = polygon_intersection(poly1, poly1_size, poly2, poly2_size);

    Py_DECREF(poly1_array);
    Py_DECREF(poly2_array);

    return result;
}

// Method definitions
static PyMethodDef GeometryMethods[] = {
    {"point_in_polygon", py_point_in_polygon, METH_VARARGS, "Check if a point is inside a polygon"},
    {"polygon_intersection", py_polygon_intersection, METH_VARARGS, "Compute polygon intersection"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef geometrymodule = {
    PyModuleDef_HEAD_INIT,
    "geometry_c",
    NULL,
    -1,
    GeometryMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_geometry_c(void) {
    PyObject *module = PyModule_Create(&geometrymodule);
    import_array();
    return module;
}
