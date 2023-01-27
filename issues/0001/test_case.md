#### Start jupyter notebook in docker

Require: Nvidia CUDA-capable GPU and Nvidia tools (e.g. nvcc) already installed.'
In my test environment, this .ipynb is located in "GPU_CODE_DIR/scripts"


```python
'''
$ /ccfsxx/r.leung/scripts/jupyter-notebook-nvidia-docker ${EXPERIMENT_DIR}
# where EXPERIMENT_DIR=/ccfsxx/r.leung/experiments/gpu-ray-triangle
''';
```

### Part 1: Prepare test data
- Objective: Convert the existing input (rtcma surface and rays) to UTM coordinates



```python
import numpy as np
import platform, os, shutil
```


```python
#Paths are relative to the notebook directory specified in the 1st argument
#when the jupyter-notebook-nvidia-docker-gpursi bash script was run.
#e.g. /home/USER/data mounts the volume at /ccfsxx/r.leung/experiments/gpu-ray-triangle/

USER = 'raymondleung8'
GPU_CODE_DIR = 'c:/{}/1'.format(USER)
GPU_BIN_TARGET = 'gpu_ray_surface_intersect'
if 'Windows' in platform.system():
    GPU_BIN_TARGET += '.exe'

WORK_DIR = os.path.join(GPU_CODE_DIR, 'workspace')
DEFAULT_DATA_DIR = os.path.join(WORK_DIR, 'input')

#Input (existing) data have been translated to occupy the positive octant
#TRANSLATED_DATA_DIR = f'/home/{USER}/data/source/input/rtcma'
TRANSLATED_DATA_DIR = os.path.join(GPU_CODE_DIR, 'input/rtcma')

#Output data will undo this translation, converting them back to UTM coordinates
UTM_DATA_DIR = TRANSLATED_DATA_DIR.replace('rtcma','rtcma_utm')

DATA_DIRS = [TRANSLATED_DATA_DIR, UTM_DATA_DIR]
```


```python
for d in [DEFAULT_DATA_DIR, UTM_DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

shutil.copy2(os.path.join(GPU_CODE_DIR, GPU_BIN_TARGET), WORK_DIR)
for module in ['gpu_ray_surface_intersect', '__init__']:
    shutil.copy2(os.path.join(GPU_CODE_DIR, 'scripts/{}.py'.format(module)), WORK_DIR)

for basename in os.listdir(GPU_CODE_DIR):
    if basename.endswith('.cu') or basename.endswith('.h') or \
       basename == GPU_BIN_TARGET:
          shutil.copy2(os.path.join(GPU_CODE_DIR, basename), WORK_DIR)

for module in ['gpu_ray_surface_intersect', '__init__']:
    shutil.copy2(os.path.join(GPU_CODE_DIR, 'scripts/{}.py'.format(module)), WORK_DIR)

os.chdir(WORK_DIR)
```


```python
fr = lambda x: os.path.join(TRANSLATED_DATA_DIR, x)
fw = lambda x: os.path.join(UTM_DATA_DIR, x)
as_triplets = lambda x: x.reshape((int(len(x)/3), 3))

#Read vertices, ray start and end points as triplets
offset = np.genfromtxt(fr('vertices_f32_offset.csv'))
vertices = as_triplets(np.fromfile(fr('vertices_f32'), dtype=np.float32))
triangles = as_triplets(np.fromfile(fr('triangles_i32'), dtype=np.int32))
rayfrom = as_triplets(np.fromfile(fr('rayFrom_f32'), dtype=np.float32))
rayto = as_triplets(np.fromfile(fr('rayTo_f32'), dtype=np.float32))
```

#### Inspect data loaded from TRANSLATED_DATA_DIR
offset = array([514574.616, 7497890.690, 305.646])
vertices[:5] = array([[  0.   , 540.028, 801.62 ],
                      [ 44.211, 576.443, 575.299],
                      [  4.971, 512.627, 751.494],
                      [ 44.427, 295.096, 805.18 ],
                      [ 22.068, 418.369, 577.724]], dtype=float32)

```python
#add the minimum vertex (offset) back into these arrays

#notes: these objects will subsequently be read in
#       gpu_ray_surface_intersect_linux_docker_demo_utm.ipynb
# - saving with float32 precision allows numerical rounding
#   effects to be emulated.
# - saving with float64 precision allows the mesh vertices
#   and ray end points to be "faithfully" preserved.

for prec, fmt in [('f32', 'float32'), ('f64', 'float64')]:
    with open(fw(f'vertices_{prec}'), 'wb') as f:
        np.array((vertices + offset).flatten(), fmt).tofile(f)
    with open(fw(f'rayFrom_{prec}'), 'wb') as f:
        np.array((rayfrom + offset).flatten(), fmt).tofile(f)
    with open(fw(f'rayTo_{prec}'), 'wb') as f:
        np.array((rayto + offset).flatten(), fmt).tofile(f)

with open(fw('triangles_i32'), 'wb') as f:
        np.array(triangles.flatten(), 'int32').tofile(f)

shutil.copy2(fr('ground_truth.csv'), UTM_DATA_DIR)
shutil.copy2(fr('vertices_f32_offset.csv'), UTM_DATA_DIR)
```



```python
print('Input data is bounded by {} and {}'.format(
    np.min(vertices, axis=0), np.max(vertices, axis=0)))

new_vertices = as_triplets(np.fromfile(fw('vertices_f32'), dtype=np.float32))
vmin = np.min(new_vertices, axis=0)
vmax = np.max(new_vertices, axis=0)
print('Output data is bounded by {} and {}'.format(
    '[%.3f %.3f %.3f]' % (vmin[0], vmin[1], vmin[2]),
    '[%.3f %.3f %.3f]' % (vmax[0], vmax[1], vmax[2])))
```

    Input data is bounded by [0. 0. 0.] and [5060.16 3565.47  805.18]
    Output data is bounded by [514574.625 7497890.500 305.646] and [519634.781 7501456.000 1110.826]
    

### Part 2: Illustrate numerical rounding effects

- Objective: To show the changes introduced in this commit handle numerical issues without altering the existing behaviour of the code.
- Note: In ```gpu_ray_surface_intersect.py```, ```translate_data``` automatically shifts the data when large spatial coordinates (UTM coordinates) are encountered. 

#### First, emulate rounding effects

- Read in distorted data from *_f32 files (large UTM coordinates being stored as float32)

#### Notes
- Normally, the distortion is introduced when double-precision floating numbers are casted to single-precision floats. This occurs when the original mesh vertices and rays (from high-precision numpy arrays) are written to float32 binary files. If gpu_ray_surface_intersect.py is used, this is done by the existing ```PyGpuRSI.acquire_data_``` method.
- For testing purpose, these numpy arrays are generated from files. Distortion is introduced in the ```_f32``` binary files instead while the ```_f64``` version remains distortion-free.
- This approach is due to the new data handling policy in the committed code, because it always anchors the mesh vertices at the origin (and likewise translates the rays by the same offset) when it sees large data points that resemble UTM coordinates, it no longer introduces rounding artefacts unlike the previous code.
- So, to emulate the rounding effects encountered in the previous code, we deliberately sacrifice precision (pre-distort the data ever so slightly) by reading large UTM coordinates directly from the curated ```_f32``` files.


```python
from gpu_ray_surface_intersect import PyGpuRSI

options = {'keep_cuda_binary': True, 'debug': True}
results = {}
for n, data_dir in enumerate(DATA_DIRS):
    f = lambda x: os.path.join(data_dir, x)
    # We still need to have access to the vertices, triangles and rays.
    # Normally, these would exist at the point where the code is run.
    # Here, we read them from files.
    print('processing {}'.format(data_dir))
    as_triplets = lambda x: x.reshape((int(len(x)/3), 3))
    vertices = as_triplets(np.fromfile(f('vertices_f32'), dtype=np.float32))
    triangles = as_triplets(np.fromfile(f('triangles_i32'), dtype=np.int32))
    rayFrom = as_triplets(np.fromfile(f('rayFrom_f32'), dtype=np.float32))
    rayTo = as_triplets(np.fromfile(f('rayTo_f32'), dtype=np.float32))

    with PyGpuRSI(GPU_CODE_DIR, WORK_DIR, quiet=True, cfg=options) as rsi:
        results[n] = rsi.test(vertices, triangles, rayFrom, rayTo)
```

    processing c:/raymondleung8/1\input/rtcma
    shift_required: False
    vertices min: [0. 0. 0.]
    vertices max: [5060.16 3565.47  805.18]
    rayfrom min: [ 1.231553 14.643824 -4.999214]
    rayfrom max: [5056.8647 3095.6218  785.946 ]
    rayto min: [ 1.231553 14.643824 -4.969767]
    rayto max: [5056.8647 3095.6218  792.6286]
    cleaning up...
    processing c:/raymondleung8/1\input/rtcma_utm
    shift_required: True
    attempted data translation...
    vertices min: [0. 0. 0.]
    vertices max: [5060.1562  3565.5      805.18005]
    rayfrom min: [ 1.21875   15.        -4.9992065]
    rayfrom max: [5056.8438  3096.       785.94604]
    rayto min: [ 1.21875  15.       -4.969757]
    rayto max: [5056.8438  3096.       792.62866]
    cleaning up...
    


```python
print('Agreement: {}/{}'.format(np.sum(results[0]==results[1]), len(results[0])))
```

    Agreement: 9845637/10000000
    

#### Next, show the new implementation minimises these errors
- Provided we read in _f64 files (large UTM coordinates stored as float64)
- Show that ```gpu_ray_surface_intersect``` correctly handles large UTM coordinates.

#### Now, the real deal...
- The test code is almost identical as before. The difference here is that we are switching between ```rtcma/*_f32``` (pre-translated vertices/rays which have a narrow dynamic range) and ```rtcma_utm/*_f64``` (high precision vertices/rays in raw UTM coordinates).
- The former is not affected by rounding since the magnitude of the numbers are small. The latter triggers coordinates translation internally which reduces the dynamic range and eliminates any significant rounding errors.


```python
from gpu_ray_surface_intersect import PyGpuRSI

options = {'keep_cuda_binary': True, 'debug': True}
results = {}
for n, data_dir in enumerate(DATA_DIRS):
    f = lambda x: os.path.join(data_dir, x)
    print('processing {}'.format(data_dir))
    as_triplets = lambda x: x.reshape((int(len(x)/3), 3))
    if 'utm' in data_dir:
        prec, fmt = 'f64', np.float64
    else:
        prec, fmt = 'f32', np.float32
    vertices = as_triplets(np.fromfile(f(f'vertices_{prec}'), dtype=fmt))
    triangles = as_triplets(np.fromfile(f('triangles_i32'), dtype=np.int32))
    rayFrom = as_triplets(np.fromfile(f(f'rayFrom_{prec}'), dtype=fmt))
    rayTo = as_triplets(np.fromfile(f(f'rayTo_{prec}'), dtype=fmt))

    with PyGpuRSI(GPU_CODE_DIR, WORK_DIR, quiet=True, cfg=options) as rsi:
        results[n] = rsi.test(vertices, triangles, rayFrom, rayTo)
```

    processing c:/raymondleung8/1\input/rtcma
    shift_required: False
    vertices min: [0. 0. 0.]
    vertices max: [5060.16 3565.47  805.18]
    rayfrom min: [ 1.231553 14.643824 -4.999214]
    rayfrom max: [5056.8647 3095.6218  785.946 ]
    rayto min: [ 1.231553 14.643824 -4.969767]
    rayto max: [5056.8647 3095.6218  792.6286]
    cleaning up...
    processing c:/raymondleung8/1\input/rtcma_utm
    shift_required: True
    attempted data translation...
    vertices min: [0. 0. 0.]
    vertices max: [5060.16015625 3565.4699707   805.17999268]
    rayfrom min: [ 1.23155296 14.64382362 -4.99921417]
    rayfrom max: [5056.86474609 3095.62182617  785.94598389]
    rayto min: [ 1.23155296 14.64382362 -4.96976709]
    rayto max: [5056.86474609 3095.62182617  792.62860107]
    cleaning up...
    


```python
print('Agreement: {}/{}'.format(np.sum(results[0]==results[1]), len(results[0])))
```

    Agreement: 10000000/10000000
    
