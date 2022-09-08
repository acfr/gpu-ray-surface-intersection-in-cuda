#Copyright (c) 2022, Raymond Leung
#All rights reserved.
#
#This source code is licensed under the BSD-3-clause license found
#in the LICENSE.md file in the root directory of this source tree.
#
#Purpose: Create synthetic surface (triangle mesh) and rays for testing
#Output:  Binary files "vertices", "triangles", "rayFrom" and "rayTo"
#Require: Python 3
#
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import array

#Configuration parameters
#------------------------------
#- surface intervals
x_min, x_max = 100, 3100
y_min, y_max = 200, 2700
#- number of triangles and rays
default_num_triangles = 30000
default_num_rays = 10000000
#- enable/disable visualisation
visualise_surface = True
#------------------------------

#define spatial frequencies (w = 2*pi*f) for surface undulation
w = 2 * np.pi * np.array(
    [0.0487, 0.0912,
     0.0125, 0.0318,
     0.00672, 0.00543,
     0.00196, 0.00282,
     0.000571, 0.000386,
     0.000345, 0.000571,
     0.000251, 0.000145,
     0.000208, 0.000139])

#method to compute surface elevation using an analytic expression
def f_xy(x, y):
    return  1 * (np.cos(w[0]*x) + np.cos(w[1]*y)) \
          + 2 * (np.cos(w[2]*x - 0.3*np.pi) + np.cos(w[3]*y - 0.7*np.pi)) + \
          + 2.5 * (np.cos(w[4]*x + 1.2*np.pi) + np.cos(w[5]*y - 1.83*np.pi)) + \
          - 1.8 * (np.cos(w[6]*x + 2.93*np.pi) + np.cos(w[7]*y - 0.67*np.pi)) + \
          + 5 * (np.cos(w[8]*x + 0.04*np.pi) + np.cos(w[9]*y - 1.05*np.pi)) + \
          + 25 * (np.cos(w[10]*x - 0.61*np.pi) + np.cos(w[11]*y + 0.51*np.pi)) + \
          + 4 * (np.cos(w[12]*x - 0.61*np.pi) * np.cos(w[13]*y + 0.51*np.pi)) + \
          + 1.3 * (np.cos(w[14]*x * w[15]*y + 0.12*np.pi)) + \
          + 3 * np.sin(0.642*w[15]*y - 0.573*w[14]*x)

#method to plot mesh surface
def draw_mesh_surface(vertices, triangles, heading=None,
                      rgb_base=np.r_[0,0,1], colornoise=0.35, randseed=4562):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    mesh_polys = [[vertices[t] for t in tri] for tri in triangles]
    np.random.seed(randseed)
    fc = (1 - colornoise) * rgb_base + colornoise * np.random.rand(len(triangles),3)
    ax.add_collection3d(Poly3DCollection(mesh_polys, facecolors=fc, linewidths=1))
    xyz_min = np.min(vertices, axis=0)
    xyz_max = np.max(vertices, axis=0)
    ax.set_xlim(xyz_min[0], xyz_max[0])
    ax.set_ylim(xyz_min[1], xyz_max[1])
    ax.set_zlim(xyz_min[2], xyz_max[2])
    if heading is not None:
        plt.title(heading)

#method to locate triangles well inside the surface
def well_within_boundary(centroids, x_min, x_range, y_min, y_range, percent):
    return (centroids[:,0] > x_min + percent * x_range) & \
           (centroids[:,0] < x_min + (1 - percent) *x_range) & \
           (centroids[:,1] > y_min + percent * y_range) & \
           (centroids[:,1] < y_min + (1 - percent) * y_range)

#API for creating a surface and saving the data in binary format
def synthesize_data(outdir,
                    n_triangles_approx=default_num_triangles,
                    n_rays=default_num_rays,
                    show_graphics=True,
                    save_results_in_binary=True,
                    skip_ground_truth=False,
                    perturb_centroid=False,
                    feedback=dict()):
    x_range = x_max - x_min
    y_range = y_max - y_min
    aspect = y_range / x_range
    n_vertices_approx = int(n_triangles_approx / 2)

    #discretisation
    nX = int(np.sqrt(n_vertices_approx / aspect))
    nY = int(aspect * nX)
    xi = np.linspace(x_min, x_max, nX)
    yi = np.linspace(y_min, y_max, nY)
    delta = min(xi[1] - xi[0], yi[1] - yi[0])
    #add some noise to perturb xy coordinates
    np.random.seed(7065)
    noise_x = 0.25 * delta * (np.random.rand(nX) - 0.5)
    noise_y = 0.25 * delta * (np.random.rand(nY) - 0.5)
    xi += noise_x
    yi += noise_y

    #create mesh surface
    vertices = []
    triangles = []
    for y in yi:
        for x in xi:
            vertices.append([x, y, f_xy(x,y)])

    for y in range(nY-1):
        for x in range(nX-1):
            #vertices are ordered consistently in clockwise direction
            triangles.append([y*nX+x, y*nX+x+1, (y+1)*nX+x])
            triangles.append([y*nX+x+1, (y+1)*nX+x+1, (y+1)*nX+x])

    vertices = np.array(vertices, dtype=float)
    vertices[:,-1] -= min(vertices[:,-1])
    triangles = np.array(triangles, dtype=int)

    feedback['nVertices'] = len(vertices)
    feedback['nTriangles'] = len(triangles)
    feedback['nRays'] = n_rays

    if show_graphics:
        draw_mesh_surface(vertices, triangles, 'Simulated surface')
        plt.show()

    #compute centroids and normal vectors for surface patches
    centroids = []
    normals = []
    for t in triangles:
        n = np.cross(vertices[t[1]] - vertices[t[0]],
                     vertices[t[2]] - vertices[t[0]])
        normals.append(n / np.linalg.norm(n))
        centroids.append(np.mean(vertices[t], axis=0))

    normals = np.array(normals)
    centroids = np.array(centroids)
    if perturb_centroid:
        np.random.seed(9571)
        a1 = 0.2 * (np.random.rand(n_rays) - 0.5)
        a2 = 0.2 * (np.random.rand(n_rays) - 0.5)
        for i, t in enumerate(triangles):
            centroids[i] += a1[i] * (vertices[t[1]] - vertices[t[0]]) \
                         +  a2[i] * (vertices[t[2]] - vertices[t[0]])

    #create rays
    #idea: Line segment starts from "centroid - (k/2) * normal"
    #      and extends for distance k*rand() in the normal direction.
    #      In the end, about half will intersect the surface.
    #- rand() generates random variates in union{(0,0.498],[0.502,1]}
    #  introduce deadzone (0.498,0.502) to make the result unambiguous.
    def rand(n):
        r = np.random.rand(n)
        r[r < 0.5] *= 0.996
        r[r >= 0.5] = 0.502 + (r[r >= 0.5] - 0.5) * 0.996
        return r

    np.random.seed(8215)
    r = rand(n_rays)
    s = 0.6 + 0.4 * np.random.rand(n_rays) #stochastic segment length scaling factor
    t = np.random.randint(len(triangles), size=n_rays) #random triangle selections
    rayFrom = []
    rayTo = []
    lower = []
    upper = []
    crossing = []
    max_segment_length = 4 * delta
    magnitude = max_segment_length * s
    lower = -0.5 * magnitude
    upper = r * magnitude
    rayFrom = centroids[t] + lower[:,np.newaxis] * normals[t]
    rayTo = rayFrom + upper[:,np.newaxis] * normals[t]
    crossing = np.array(r > 0.5, dtype=np.int32)

    if show_graphics:
        M = min(200, n_rays)
        #- visualise first 200 rays relative to surface
        plt.plot([range(M), range(M)], [lower[:M], lower[:M] + upper[:M]])
        plt.plot([0,M], [0,0], 'k')
        plt.title('Illustration: Rays that cross the surface rise above y=0')
        plt.show()
        #- cdf
        plt.plot(np.sort(r), np.arange(n_rays)/n_rays)
        plt.ylabel('cdf')
        plt.xlabel('r')
        plt.title(r"y(r=0.5) $\rightarrow$ proportion of rays that don't intersect the surface")
        plt.axis('tight')
        plt.grid(True)
        plt.show()
        #- show some rays piercing through the surface
        '''
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        draw_mesh_surface(vertices, triangles, 'Some rays (red) intersecting, (green) not intersecting the surface')
        for c,v in zip(['g', 'r'], [0, 1]):
            margin = 0.2 if v == 1 else 0
            idx = np.where((crossing == v) & well_within_boundary((rayFrom + rayTo)/2.,
                            x_min, x_range, y_min, y_range, margin))[0][:25]
            ls = np.hstack([rayFrom[idx], rayTo[idx]]).copy()
            ls = ls.reshape((-1,2,3))
            lc = Line3DCollection(ls, linewidths=2, colors=c)
            plt.gca().add_collection(lc)
            plt.gca().scatter(rayTo[idx,0], rayTo[idx,1], rayTo[idx,2], c=c)
        plt.show()
        '''

    #shift the coordinates to anonymise data and preserve precision as float32
    xyz_min = np.min(vertices, axis=0)
    vertices -= xyz_min
    rayFrom -= xyz_min
    rayTo -= xyz_min

    #write data to bin files
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fw = lambda f: os.path.join(outdir, f)
    t0 = time.time()
    verts = np.array(vertices.flatten(),'float32')
    tris = np.array(triangles.flatten(),'int32')
    pFrom = np.array(rayFrom.flatten(),'float32')
    pTo = np.array(rayTo.flatten(),'float32')
    with open(fw('vertices_f32'), 'wb') as f:
        verts.tofile(f)
    with open(fw('triangles_i32'), 'wb') as f:
        tris.tofile(f)
    with open(fw('rayFrom_f32'), 'wb') as f:
        pFrom.tofile(f)
    with open(fw('rayTo_f32'), 'wb') as f:
        pTo.tofile(f)
    t1 = time.time()
    print('Essential files written in {}s'.format(t1 - t0))
    '''
    np.savetxt(fw('vertices.csv'), verts, delimiter=',', fmt='%.6f')
    np.savetxt(fw('triangles.csv'), tris, delimiter=',', fmt='%d')
    np.savetxt(fw('rayFrom.csv'), pFrom, delimiter=',', fmt='%.6f')
    np.savetxt(fw('rayTo.csv'), pTo, delimiter=',', fmt='%.6f')
    '''
    if skip_ground_truth:
        return
    print('Saving ground-truth...')
    if save_results_in_binary:
        with open(fw('ground_truth'), 'wb') as f:
            crossing.tofile(f)
        if perturb_centroid:
            intercepts = centroids[t] - xyz_min
            intercepts[crossing==0] = 0
            intersect_triangle = t
            intersect_triangle[crossing==0] = -1
            with open(fw('intercepts'), 'wb') as f:
                np.array(intercepts.flatten(),'float32').tofile(f)
            with open(fw('intersect_triangle'), 'wb') as f:
                np.array(intersect_triangle.flatten(),'int32').tofile(f)
    else:
        np.savetxt(fw('ground_truth.csv'), crossing, fmt='%d', delimiter=',')


if __name__ == "__main__":
    outdir = os.path.join(os.getcwd().replace('scripts', 'input'))
    synthesize_data(outdir, visualise_surface)
