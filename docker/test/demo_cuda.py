#!/usr/bin/env python
# coding: utf-8

# ## gpu-rsi demonstration
# This notebook uses the PyGpuRSI wrapper class defined in the
# PyGpuRaySurfaceIntersect module to execute the CUDA code that
# checks for line-segment and surface-triangle intersection.

__copyright__ = "Copyright (c) 2022, Raymond Leung"
__license__   = "BSD-3-clause"

import numpy as np
import os, shutil, sys
from input_synthesis import synthesize_data
from gpu_ray_surface_intersect import PyGpuRSI


GPU_CODE_DIR = '/gpu-ray-surface-intersection-in-cuda-main/'
WORK_DIR = os.getcwd().replace('scripts', 'scratch')
DATA_DIR = os.path.join(WORK_DIR, 'input')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
os.chdir(WORK_DIR)

# Synthesize the input data
synthesize_data(DATA_DIR, n_triangles_approx=5000, n_rays=10000,
                show_graphics=False, save_results_in_binary=True)

# This step is not needed in standard workflow since (vertices, triangles, rayFrom,
# rayTo) would normally exist as numpy arrays. Here, we read these from files instead.
def bin2array(filename, precision, dims=2):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=precision)
    return data.reshape([int(len(data)/3), 3]) if dims==2 else data

vertices = bin2array('input/vertices_f32', np.float32)
triangles = bin2array('input/triangles_i32', np.int32)
rayFrom = bin2array('input/rayFrom_f32', np.float32)
rayTo = bin2array('input/rayTo_f32', np.float32)

# #### (Part A) Run CUDA program in "boolean" mode to return 0/1 intersection results
# #### GPU ray-segment surface intersection tests are applied in two ways

# Approach 1:
rsi = PyGpuRSI(GPU_CODE_DIR, WORK_DIR)
# - compile CUDA code the first time
# - subsequently, doesn't compile if GPU_BIN_TARGET already exists

results = rsi.test(vertices, triangles, rayFrom, rayTo)
# - The WORK_DIR contains source code and binary at this point
# - You may run .test again using another surface by specifying
#   new vertices and triangles, with both rayFrom and rayTo
#   omitted, which means you will reuse the same line segments.

'''
The test method basically invokes
!{rsi.gpu_bin_target} {rsi.vertices_file} {rsi.triangles_file} {rsi.rayfrom_file} {rsi.rayto_file}

but it also returns boolean results in a numpy array (dtype=int).
    0 => ray does NOT intersect with surface
    1 => ray intersects with surface
'''
#- user is responsible for clean-up when work is finished
rsi.cleanup()

# Approach 2: using with statement (auto clean-up)
# - The `quiet` flag (optional) suppresses console output
# - The setting dictionary is optional, configuring 'keep_cuda_binary' to
# True ensures the binary targets are not removed upon exit, so if the
# PyGpuRSI object is initialised again in the future, it will not
# compile the source code if "gpu_ray_surface_intersect" is found.
# This is okay assuming the CUDA implementation remains unchanged.
setting = {'keep_cuda_binary': True}
with PyGpuRSI(GPU_CODE_DIR, WORK_DIR, quiet=True, cfg=setting) as rsi:
    results2 = rsi.test(vertices, triangles, rayFrom, rayTo)

all(results == results2)

# #### (Part B) Run CUDA program in "barycentric" mode to return the distance to surface, intersecting triangle and intersecting point for each intersecting ray. 

from input_synthesis import synthesize_data

fw = lambda x: os.path.join('ínput', x)
geom_info = {}

synthesize_data(outdir=DATA_DIR, n_triangles_approx=5000, n_rays=10000,
                show_graphics=False, save_results_in_binary=True,
                perturb_centroid=True, feedback=geom_info)
print('Created {}'.format(geom_info))
vertices = bin2array('input/vertices_f32', np.float32)
triangles = bin2array('input/triangles_i32', np.int32)
rayFrom = bin2array('input/rayFrom_f32', np.float32)
rayTo = bin2array('input/rayTo_f32', np.float32)

# Configuration
# Set operating mode to 'barycentric' to return
# - intersecting rays
# - distance from the starting point of the ray to the surface
# - intersecting triangle
# - intersecting point
parms = {'mode':'barycentric', 'keep_cuda_binary': True}

# GPU ray-segment surface intersection tests are invoked below
rsi = PyGpuRSI(GPU_CODE_DIR, WORK_DIR, cfg=parms)

intersecting_rays, distances, hit_triangles, hit_points = rsi.test(vertices, triangles, rayFrom, rayTo)

# Compare results with ground truth
gt_hit_points = bin2array('input/intercepts', np.float32)
gt_hit_triangles = bin2array('input/intersect_triangle', np.int32, dims=1)
gt_intersecting_rays = gtidx = np.where(gt_hit_triangles >= 0)[0]
gt_hit_points = gt_hit_points[gtidx]
gt_hit_triangles = gt_hit_triangles[gtidx]
gt_distances = np.sqrt(np.sum((gt_hit_points - rayFrom[gtidx])**2, axis=1))

import pandas as pd
print('Running GPU code in barycentric output mode\n')
print('Ray-surface intersections: detected:{} (actual:{})'.format(len(intersecting_rays), len(gt_intersecting_rays)))
print('Undetected rays (FN): {}'.format(np.setdiff1d(gt_intersecting_rays, intersecting_rays)))
print('False detections (FP): {}'.format(np.setdiff1d(intersecting_rays, gt_intersecting_rays)))

df1 = pd.DataFrame(np.c_[hit_triangles, hit_points, distances],
                   columns=['triangle','px','py','pz','dist'],
                   index=intersecting_rays)
df2 = pd.DataFrame(np.c_[gt_hit_triangles, gt_hit_points, gt_distances],
                   columns=['gt_triangle','gt_px','gt_py','gt_pz','gt_dist'],
                   index=intersecting_rays)
dfc = pd.merge(df1, df2, left_index=True, right_index=True)

n_same = sum(dfc['triangle']==dfc['gt_triangle'])
print('Intersecting triangles: {}/{} ({}%) identical'.format(
       n_same, len(dfc), (100.*n_same)/len(dfc)))

p1 = dfc[['px','py','pz']].values
p2 = dfc[['gt_px','gt_py','gt_pz']].values
pc = np.abs(np.sqrt(np.sum((p2 - p1)**2, axis=1))) < 0.001
print('Intersecting points:    {}/{} ({}%) equivalent'.format(
       sum(pc), len(pc), (100.*sum(pc))/len(pc)))

dc = np.isclose(dfc['dist'], dfc['gt_dist'])
print('Computed intersecting distances: {}/{} ({}%) equivalent'.format(
       sum(dc), len(dc), (100.*sum(dc))/len(dc)))

#- check out some values
dfc.iloc[:8]

rsi.cleanup()
