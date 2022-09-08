#Copyright (c) 2022, Raymond Leung
#All rights reserved.
#
#This source code is licensed under the BSD-3-clause license found
#in the LICENSE.md file in the root directory of this source tree.
#
import numpy as np
import os, shutil, sys
import platform
import subprocess
from pdb import set_trace as bp


class PyGpuRSI(object):
    '''
    Thin wrapper to handle gpu_ray_surface_intersect.cu compilation,
    input/output data manipulation, and clean up.
    '''
    def __init__(self, src_path, wrk_dir=None, quiet=False, cfg={}):
        #establish paths
        self.src_dir = src_path
        self.wrk_dir = src_path if wrk_dir is None else wrk_dir
        self.data_dir = os.path.join(self.wrk_dir, 'input')
        self.gpu_cuda_source = cfg.get('gpu_cuda_source', 'gpu_ray_surface_intersect.cu')
        self.gpu_bin_target = cfg.get('gpu_bin_target', 'gpu_ray_surface_intersect')
        self.keep_cuda_binary = cfg.get('keep_cuda_binary', False)
        self.nvcc_compile = 'nvcc'
        #operating mode
        #  'boolean' (default) returns {0,1} outcome for ray-surface intersection
        #  'barycentric' returns intersecting rays, intersecting triangles f (-1
        #                for None), barycentric coords (t,u,v), and intersecting
        #                point P such that P=(1-u-v)*V[0]+u*V[1]+v*V[2] when
        #                there is an intersection and t measures the distance
        #                from the ray starting point to the surface.
        #  'intercept_count' returns number of ray-surface intersections
        self.mode = cfg.get('mode', 'boolean')
        if self.mode not in ['boolean', 'barycentric', 'intercept_count']:
            print('Unrecognised operating mode "{}",'.format(cfg['mode']))
            print('switched back to default "boolean" output.')
            self.mode = 'boolean'
        #name target
        if 'Windows' in platform.system():
            self.gpu_bin_target += '.exe'
        if 'Linux' in platform.system():
            self.nvcc_compile = cfg.get('nvcc_compile', '/usr/local/cuda/bin/nvcc')
        #expected input/output
        f = lambda x: os.path.join(self.data_dir, x)
        self.vertices_file = f('vertices_f32')
        self.triangles_file = f('triangles_i32')
        self.rayfrom_file = f('rayFrom_f32')
        self.rayto_file = f('rayTo_f32')
        self.results_file = f('results_i32')
        #set up file structure
        self.setup()
        self.quiet_flag = "silent" if quiet else ""
        self.large_positive_value = 2.5e+8

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.cleanup()

    def setup(self):
        #ensure data directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        os.chdir(self.wrk_dir)
        #nothing to do if target is already built
        if os.path.isfile(os.path.join(self.wrk_dir, self.gpu_bin_target)):
            return
        #else, replicate relevant source files
        for basename in os.listdir(self.src_dir):
            if basename.endswith('.cu') or basename.endswith('.h'):
                shutil.copy2(os.path.join(self.src_dir, basename), self.wrk_dir)
        for module in ['input_synthesis', '__init__']:
            shutil.copy2(os.path.join(self.src_dir, 'scripts/{}.py'.format(module)), self.wrk_dir)
        #build target
        try:
            print('compiling CUDA code...')
            if sys.version_info.major > 2:
                out = subprocess.run([self.nvcc_compile, self.gpu_cuda_source,
                                      "-o", self.gpu_bin_target])
                retval = out.returncode
            else:
                retval = subprocess.call([self.nvcc_compile, self.gpu_cuda_source,
                                          "-o", self.gpu_bin_target])
            if retval == 0:
                print("nvcc compilation succeeded!")
            else:
                print("nvcc compilation failed (exit code: %d)" % out.returncode)
        except:
            pass

    def acquire_data_(self, vertices, triangles, rayfrom=None, rayto=None):
        #convert user-supplied numpy arrays into binaries for CUDA program
        with open('input/vertices_f32', 'wb') as f:
            np.array(vertices.flatten(),'float32').tofile(f)
        with open('input/triangles_i32', 'wb') as f:
            np.array(triangles.flatten(),'int32').tofile(f)
        if rayfrom is not None:
            with open('input/rayFrom_f32', 'wb') as f:
                np.array(rayfrom.flatten(),'float32').tofile(f)
        if rayto is not None:
            with open('input/rayTo_f32', 'wb') as f:
                np.array(rayto.flatten(),'float32').tofile(f)
        
    def test(self, vertices, triangles, rayfrom=None, rayto=None):
        #create corresponding binary files
        self.acquire_data_(vertices, triangles, rayfrom, rayto)
        #perform ray-surface intersection test
        if sys.version_info.major > 2:
            subprocess.run([os.path.join(self.wrk_dir, self.gpu_bin_target),
                      self.vertices_file,
                      self.triangles_file,
                      self.rayfrom_file,
                      self.rayto_file,
                      self.quiet_flag,
                      self.mode])
        else:
            subprocess.call([os.path.join(self.wrk_dir, self.gpu_bin_target),
                      self.vertices_file,
                      self.triangles_file,
                      self.rayfrom_file,
                      self.rayto_file,
                      self.quiet_flag,
                      self.mode])
        #retrieve results
        if self.mode == 'boolean':
            return np.fromfile('results_i32', dtype=np.int32)
        elif self.mode == 'barycentric':
            result = np.fromfile('intersectTriangle_i32', dtype=np.int32)
            intersecting_rays = idx = np.where(result >= 0)[0]
            n = len(idx)
            hit_triangles = f = result[idx]
            t = np.fromfile('barycentricT_f32', dtype=np.float32)[idx]
            u = np.fromfile('barycentricU_f32', dtype=np.float32)[idx]
            v = np.fromfile('barycentricV_f32', dtype=np.float32)[idx]
            distances = t * np.sqrt(np.sum((rayto[idx] - rayfrom[idx])**2, axis=1))
            hit_points = np.empty((n,3))
            for ax in [0,1,2]:
                hit_points[:,ax] = (1-u-v) * vertices[triangles[f,0],ax] + \
                                    u * vertices[triangles[f,1],ax] + \
                                    v * vertices[triangles[f,2],ax]
            return intersecting_rays, distances, hit_triangles, hit_points
        else: #'intercept_count'
            return np.fromfile('results_i32', dtype=np.int32)

    def cleanup(self):
        def match_ext(f, extensions):
            for e in extensions:
                if f.endswith(e):
                    return True
            return False

        print('cleaning up...')
        for basename in os.listdir(self.wrk_dir):
            if match_ext(basename, ['.cu','.h','.exp','.lib','.pyc','_f32','_i32']):
                os.remove(os.path.join(self.wrk_dir, basename))
            elif basename in ['input_synthesis.py', '__init__.py']:
                os.remove(os.path.join(self.wrk_dir, basename))
        try:
            shutil.rmtree(self.data_dir)
            if not self.keep_cuda_binary:
                os.remove(os.path.join(self.wrk_dir, self.gpu_bin_target))
        except:
            pass
