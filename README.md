# boundvor
A minimal Python library that provides a wrapper for [`scipy.spatial.Voronoi`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html), clipping the resulting cells to a bounding polygon.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6c585fbf-d893-4cb1-8f7d-d59c56f3b0e2" height="320" width="320"/>
</p>

## Installation
Install the package using pip:

```bash
pip install boundvor
```

## Usage

```python
import numpy as np
from boundvor import BoundedVoronoi

# Generate random points
points = np.random.rand(10, 2)

# Define a bounding box
bounding_box = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])

# Create a bounded Voronoi diagram
voronoi = BoundedVoronoi(points, bounds=bounding_box)
```

## License
MIT License
