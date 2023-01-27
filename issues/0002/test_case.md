## Test case reported by Ronan Danno
- 3 vertices: (0, 0, 0), (1, 0, 0), (0, 1, 0)
- 1 single triangle, linking vertex indices (0, 1, 2)
- 10 vertical rays with x varying from 0.0 to 0.9 by steps of 0.1, y = 0.5, z = 1 (for rayFrom) or -1 (for rayTo)

### Status
- 26/01/2023 Pending further investigation.
- 27/01/2023 Solution implemented in gpu_ray_surface_intersect.cu main().
  Add extra triangle when h_triangles contains only one triplet.

```python
import numpy as np
import os, platform, shutil
```


```python
# Specify my source and target
USER = 'raymondleung8'
GPU_CODE_DIR = '/home/{}/data/source'.format(USER)
WORK_DIR = os.path.join(GPU_CODE_DIR, 'workspace')
DATA_DIR = os.path.join(WORK_DIR, 'input')
GPU_BIN_TARGET = 'gpu_ray_surface_intersect'
if 'Windows' in platform.system():
    GPU_BIN_TARGET += '.exe'
```


```python
# Set up work directory
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

shutil.copy2(os.path.join(GPU_CODE_DIR, GPU_BIN_TARGET), WORK_DIR)
for module in ['gpu_ray_surface_intersect', '__init__']:
    shutil.copy2(os.path.join(GPU_CODE_DIR, 'scripts/{}.py'.format(module)), WORK_DIR)

os.chdir(WORK_DIR)
```


```python
from gpu_ray_surface_intersect import PyGpuRSI
```


```python
# Synthesize the input data
vertices = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32)
triangles = np.array([[0,1,2]], dtype=np.int32)
raysFrom = np.c_[np.arange(0,1,0.1), [[0.5,1]]*10]
raysTo = np.c_[np.arange(0,1,0.1), [[0.5,-1]]*10]

# Generate binary files
rasterscan = lambda x, fmt: np.array(x.flatten(), fmt)
fw = lambda f: os.path.join(DATA_DIR, f)
object_format_filename = [(vertices, 'float32', 'test_vertices_f32'),
                          (triangles, 'int32', 'test_triangles_i32'),
                          (raysFrom, 'float32', 'test_rayFrom_f32'),
                          (raysTo, 'float32', 'test_rayTo_f32')]

for item in object_format_filename:
    obj, fmt, name = item
    with open(fw(name), 'wb') as f:
        rasterscan(obj, fmt).tofile(f)
```


```python
def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

# The directory structure should look like this
list_files(os.getcwd())
```

    workspace/
        gpu_ray_surface_intersect
        gpu_ray_surface_intersect.py
        __init__.py
        input/
            test_vertices_f32
            test_triangles_i32
            test_rayFrom_f32
            test_rayTo_f32


#### GPU ray-segment surface intersection tests are invoked below


```python
# Configuration
parms = {'mode':'barycentric', 'keep_cuda_binary': True}
```


```python
# GPU ray-segment surface intersection tests are invoked below

rsi = PyGpuRSI(GPU_CODE_DIR, WORK_DIR, cfg=parms)

triangles_ = np.tile(triangles[0],(2,1))

'''
COMMENTS

Currently, using the code from commit SHA 77cf088a4525e117,
when the number of given triangles is 1, it throws an exception
"illegal memory access was encountered in gpu_ray_surface_intersect.cu at line 270"

For testing, when the 2nd argument is changed from `triangles` to `triangles_`,
i.e. if the mesh contains multiple triangles, it produces the correct results.

This has to do with how the binary radix tree is constructed. The bvhTraverse
function expects there to be at least one split node where the left and right
child nodes are defined. For this corner case, they are undefined in the root node.
'''
intersecting_rays, distances, hit_triangles, hit_points = \
      rsi.test(vertices, triangles, raysFrom, raysTo)
```

    input/vertices_f32 contains 36 bytes, 9 <f>, 3 elements
    input/triangles_i32 contains 12 bytes, 3 <i>, 1 elements
    input/rayFrom_f32 contains 120 bytes, 30 <f>, 10 elements
    input/rayTo_f32 contains 120 bytes, 30 <f>, 10 elements
    1024 threads/block, grids: {triangles: 1, rays: 16}
    [0, 1] delta: 4.76837e-07
    [0, 1] delta: 4.76837e-07
    [0, 0] delta: 0
    checking sortMortonCode
    0: (1) 0
    1: (0) 439208192231179800
    Results for last few elements:
    0: 0
    1: 0
    2: 0
    3: 0
    4: 0
    5: 0
    6: -1
    7: -1
    8: -1
    9: -1
    Processing time: 0.27904 ms



```python
print(f'intersecting_rays: {intersecting_rays}')
print(f'distances: {distances}')
print(f'hit_triangles: {hit_triangles}')
print(f'hit_points: {hit_points}')
```

    intersecting_rays: [0 1 2 3 4 5]
    distances: [1. 1. 1. 1. 1. 1.]
    hit_triangles: [0 0 0 0 0 0]
    hit_points: [[0. 0.5  0.]
     [0.1        0.5 0.]
     [0.2        0.5 0.]
     [0.30000001 0.5 0.]
     [0.40000001 0.5 0.]
     [0.5        0.5 0.]]

```python
'''
COMMENTS
With the changes introduced in this commit, when a root node with
undefined left/right child nodes is prevented from appearing in
the binary radix tree, running rsi.test with `triangles` as the
second argument will produce the same results.
''';
```

```python
list_files(os.getcwd())
```

    workspace/
        gpu_ray_surface_intersect
        gpu_ray_surface_intersect.py
        __init__.py
        intersectTriangle_i32
        barycentricT_f32
        barycentricU_f32
        barycentricV_f32
        input/
            test_vertices_f32
            test_triangles_i32
            test_rayFrom_f32
            test_rayTo_f32
            vertices_f32
            triangles_i32
            rayFrom_f32
            rayTo_f32
        __pycache__/
            gpu_ray_surface_intersect.cpython-39.pyc



```python
rsi.cleanup()

#### Running in boolean and intersect_count mode
```


```python
for mode in ['boolean', 'intersect_count']:
    parms = {'mode':'boolean', 'keep_cuda_binary': True}
    rsi = PyGpuRSI(GPU_CODE_DIR, WORK_DIR, quiet=True, cfg=parms)
    results = rsi.test(vertices, triangles, raysFrom, raysTo)
    print(f'{mode} mode: {results}')
```

    boolean mode: [1 1 1 1 1 1 0 0 0 0]
    intersect_count mode: [1 1 1 1 1 1 0 0 0 0]

