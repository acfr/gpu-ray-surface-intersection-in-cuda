#Copyright (c) 2023, Raymond Leung
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
        self.shift_required = None
        self.debug = cfg.get('debug', False)
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
        #translate spatial coordinates if required
        self.translate_data(vertices, rayfrom, rayto)

        #convert user-supplied numpy arrays into binaries for CUDA program
        with open('input/vertices_f32', 'wb') as f:
            np.array(self.vertices.flatten(),'float32').tofile(f)
        with open('input/triangles_i32', 'wb') as f:
            np.array(triangles.flatten(),'int32').tofile(f)
        if rayfrom is not None:
            with open('input/rayFrom_f32', 'wb') as f:
                np.array(self.rayfrom.flatten(),'float32').tofile(f)
        if rayto is not None:
            with open('input/rayTo_f32', 'wb') as f:
                np.array(self.rayto.flatten(),'float32').tofile(f)

    def translate_data(self, vertices, rayfrom, rayto):
        '''
        When necessary, reduce the effective range to maximise float precision.

        Background:
        - IEEE754 represents float32 using 8 exponent and 23 fraction bits:
           X = (-1)^{b_{31}} * 2^{(b_{30}b_{29}...b_{23})_2 -127} * (1.b_{22}b_{21}...b_0)_2
             = (-1)^sign * 2^{E-127} * (1 + sum_{i=1:23} b_{23-i} 2^{-i})
        - This gives 6 to 9 significant decimal digits precision which may not be enough.
        - For decimals between 2^n and 2^{n+1}, precision is limited to 2^{n-23}.
          For floats between 2^23=8,388,608 and 2^24=16,777,216, precision is down to 2^0=1.
        Consideration:
        - Using Universal Transverse Mercator (UTM) projections, we do not reach these
          limits in longitude, as the UTM zones each cover at most 668km over an arc
          of 6 degrees. However, the latitude bands each cover 8 degrees, with northing
          measured from the equator, the y-coordinate is upper-bounded by 9,300,000 in
          the northern hemisphere (at 84 deg N) and 10,000,000 in the southern hemisphere.
        - For UTM coordinates, translating the data and expressing everything relative
          to the minimum coordinates before the geometry tests are applied can eliminate
          rounding errors provided the effective/intrinsic range of the data is less
          than approx. 8km (or 64km) depending on the required level of precision.
        - For a desired accuracy between (a) 0.001 and (b) 0.01, we can calculate
          when this is needed by solving n-23 = floor(log2(delta)) for n:
         (case a): n-23 = floor(-9.9658), n = 13 => 2^n = 8192
         (case b): n-23 = floor(-6.6439), n = 16 => 2^n = 65536
        - When np.max(np.abs(vertices)) > 2^n, translation should be applied.
        - To be clear, for a precision of 0.001 to 0.01, float32 can faithfully
          represent a local area with a span of 8.192 to 65.536 km if we choose
          to keep all coordinates positive. If the data is centered instead, the
          effective distance is doubled using signed representation [-R,R].
        '''
        if (rayfrom is None and rayto is not None or
            rayfrom is not None and rayto is None):
            raise Exception('Inconsistent ray[from|to] specification, only one is None.')

        rays_unchanged = rayfrom is None

        #perform test once, compute the shift only when this is unknown
        if self.shift_required is None:
            self.shift_required = np.max(np.abs(vertices)) > 16384
            if self.shift_required:
                self.min_coords = np.min(vertices, axis=0)
            else:
                self.min_coords = np.zeros(3)

        if self.shift_required:
            self.vertices = np.array(vertices) - self.min_coords
            if rays_unchanged is False:
                self.rayfrom = np.array(rayfrom) - self.min_coords
                self.rayto = np.array(rayto) - self.min_coords
        else:
            self.vertices = vertices
            self.rayfrom = rayfrom
            self.rayto = rayto
        if self.debug:
            print('shift_required: {}'.format(self.shift_required))
            if self.shift_required:
                print('attempted data translation...')
            print('vertices min: {}'.format(np.min(self.vertices,axis=0)))
            print('vertices max: {}'.format(np.max(self.vertices,axis=0)))
            print('rayfrom min: {}'.format(np.min(self.rayfrom,axis=0)))
            print('rayfrom max: {}'.format(np.max(self.rayfrom,axis=0)))
            print('rayto min: {}'.format(np.min(self.rayto,axis=0)))
            print('rayto max: {}'.format(np.max(self.rayto,axis=0)))

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
                hit_points[:,ax] = rayfrom[idx,ax] + t * (
                                   rayto[idx,ax] - rayfrom[idx,ax])
                '''
                hit_points[:,ax] = (1-u-v) * vertices[triangles[f,0],ax] + \
                                    u * vertices[triangles[f,1],ax] + \
                                    v * vertices[triangles[f,2],ax]
                '''
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
