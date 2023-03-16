//Copyright (c) 2022, Raymond Leung
//All rights reserved.
//
//This source code is licensed under the BSD-3-clause license found
//in the LICENSE.md file in the root directory of this source tree.
//
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <vector>

#include <stdint.h>
#include "bvh_structure.h"
#include "rsi_geometry.h"

using namespace std;
using namespace lib_bvh;
using namespace lib_rsi;

//-------------------------------------------------
// This implementation corresponds to version v3
// with support for barycentric mode and the
// intercept_count experimental feature
//-------------------------------------------------

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

template <class T>
int readData(string fname, vector<T> &v, int dim=1, bool silent=false)
{
    ifstream infile(fname.c_str(), ios::binary | ios::ate);
    if (! infile) {
        cerr << "File " << fname << " not found" << endl;
        exit(1);
    }
    ifstream::pos_type nbytes = infile.tellg();
    infile.seekg(0, infile.beg);
    const int elements = nbytes / sizeof(T);
    v.resize(elements);
    infile.read(reinterpret_cast<char*>(v.data()), nbytes);
    if (! silent) {
        cout << fname << " contains " << nbytes << " bytes, "
             << v.size() << " <" << typeid(v.front()).name() << ">, "
             << v.size() / dim << " elements" << endl;
    }
    return elements / dim;
}

template <class T>
void writeData(string fname, vector<T> &v)
{
    ofstream outfile(fname.c_str(), ios::out | ios::binary);
    if (! outfile) {
        cerr << "Cannot create " << fname << " for writing" << endl;
        exit(1);
    }
    outfile.write(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
    outfile.close();
}


int main(int argc, char *argv[])
{
    const bool checkEnabled(true);
    const float largePosVal(2.5e+8);
    vector<float> h_vertices;
    vector<int>   h_triangles;
    vector<float> h_rayFrom;
    vector<float> h_rayTo;
    vector<int>   h_crossingDetected;
    vector<int>   h_intersectTriangle;
    vector<float> h_baryT, h_baryU, h_baryV, h_debug;
    int nVertices, nTriangles, nRays;

    if (argc == 2 && strcmp(argv[1], "--help")==0)
    {
        cout << "CUDA GPU implementation of Moller-Trumbore ray-triangle intersection test\n"
             << "Optional args:\n"
             << "[1] vertices file, (nVertices,3) as binary float32[]\n"
             << "[2] triangles file, (nTriangles,3) as binary int32[]\n"
             << "[3] segment start points, (nRays,3) as binary float32[]\n"
             << "[4] segment end points, (nRays,3) as binary float32[]\n"
             << "[5] suppress cout with string \"silent\"\n"
             << "[6] output format \"boolean\" or \"barycentric\" or \"intercept_count\"\n"
             << "[7] ray index for which Moller-Trumbore diagnostic params are extracted, as int32\n";
        return 0;
    }
    //optional arguments
    std::string fileVertices(argc > 1? argv[1]: "input/vertices_f32"),
                fileTriangles(argc > 2? argv[2]: "input/triangles_i32"),
                fileFrom(argc > 3? argv[3]: "input/rayFrom_f32"),
                fileTo(argc > 4? argv[4]: "input/rayTo_f32");
    bool quietMode(argc > 5? strcmp(argv[5], "silent") == 0 : false);
    /* 
    Ray-surface intersection results are reported as follows:
      barycentric = false
      |  if interceptsCount is false (by default)
      |     return boolean array, h_crossingDetected[r] is set to 0 or 1
      |  else report the number of surface intersections for each ray
      |     return integer array, h_crossingDetected[r] >= 0
      barycentric = true 
      |  return index of intersecting triangle (f) via h_intersectTriangle[r]
      |  (-1 if none) and the intersecting point P via barycentric coordinates
      |  (t[r], u[r], v[r]) where t = distance(rayFrom, surface), P =
      |  (1-u-v)*V[0] + u*V[1] + v*V[2], V[i] = vertices[triangles[f][i]].
    */
    bool barycentric(argc > 6? strcmp(argv[6], "barycentric") == 0 : false);
    bool interceptsCount(argc > 6? strcmp(argv[6], "intercept_count") == 0 : false);
    int  queryRayIdx(argc > 7? atoi(argv[7]) : 0);

    //read input data into host memory
    nVertices = readData(fileVertices, h_vertices, 3, quietMode);
    nTriangles = readData(fileTriangles, h_triangles, 3, quietMode);

    if (h_triangles.size() == 3) {
        //Add an extra triangle so that BVH traversal works in an
        //uncomplicated way without throwing an exception. It
        //expects at least one split node at the top of the binary
        //radix tree where the left and right child nodes are defined.
        for (int i = 0; i < 3; i++)
            h_triangles.push_back(0);
        nTriangles += 1;
    }

    nRays = readData(fileFrom, h_rayFrom, 3, quietMode);
    assert(readData(fileTo, h_rayTo, 3, quietMode) == nRays);
    h_crossingDetected.resize(nRays);
 
    cudaEvent_t start, end;
    float time = 0;
    float *d_vertices, *d_rayFrom, *d_rayTo;
    int   *d_triangles, *d_crossingDetected, *d_intersectTriangle;
    float *d_baryT, *d_baryU, *d_baryV, *d_debug;
    AABB  *d_rayBox;
    int sz_vertices(3 * nVertices * sizeof(float)),
        sz_triangles(3 * nTriangles * sizeof(int)),
        sz_rays(3 * nRays * sizeof(float)),
        sz_rbox(nRays * sizeof(AABB)),
        sz_id(nRays * sizeof(int)),
        sz_debug(1536 * sizeof(float)),
        sz_bary(nRays * sizeof(float));
    cudaMalloc(&d_vertices, sz_vertices);
    cudaMalloc(&d_triangles, sz_triangles);
    cudaMalloc(&d_rayFrom, sz_rays);
    cudaMalloc(&d_rayTo, sz_rays);
    cudaMalloc(&d_rayBox, sz_rbox);

    if (! barycentric) {
        cudaMalloc(&d_crossingDetected, sz_id);
        cudaMemset(d_crossingDetected, 0, sz_id);
    }
    else {
        h_intersectTriangle.resize(nRays);
        h_baryT.resize(nRays);
        h_baryU.resize(nRays);
        h_baryV.resize(nRays);
        h_debug.resize(1536);
        cudaMalloc(&d_intersectTriangle, sz_id);
        cudaMalloc(&d_baryT, sz_bary);
        cudaMalloc(&d_baryU, sz_bary);
        cudaMalloc(&d_baryV, sz_bary);
        cudaMalloc(&d_debug, sz_debug);
    }
    cudaMemcpy(d_vertices, h_vertices.data(), sz_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, h_triangles.data(), sz_triangles, cudaMemcpyHostToDevice);

    //grid partitions
    int blockX = 1024,
        gridXr = (int)ceil((float)nRays / blockX),
        gridXt = (int)ceil((float)nTriangles / blockX),
        gridXLambda = 16; //N_{grids}
    if (! quietMode) {
        cout << blockX << " threads/block, grids: {triangles: "
             << gridXt << ", rays: " << gridXLambda << "}" << endl;
    }
    float minval[3], maxval[3], half_delta[3], inv_delta[3];
    vector<uint64_t> h_morton;
    vector<int> h_sortedTriangleIDs;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    cudaMemcpy(d_rayFrom, h_rayFrom.data(), sz_rays, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rayTo, h_rayTo.data(), sz_rays, cudaMemcpyHostToDevice);

    //initialise arrays
    if (barycentric) {
        initArrayKernel<<<gridXr, blockX>>>(d_intersectTriangle, -1, nRays);
        initArrayKernel<<<gridXr, blockX>>>(d_baryT, largePosVal, nRays);
        initArrayKernel<<<gridXr, blockX>>>(d_debug, 0.0f, 1536);
    }
    cudaDeviceSynchronize();

    //compute ray-segment bounding boxes
    rbxKernel<<<gridXr, blockX>>>(d_rayFrom, d_rayTo, d_rayBox, nRays);
    cudaDeviceSynchronize();

    //order triangles using Morton code
    //- normalise surface vertices to canvas coords
    getMinMaxExtentOfSurface<float>(h_vertices, minval, maxval, half_delta,
                                    inv_delta, nVertices, quietMode);
    //- convert centroid of triangles to morton code
    createMortonCode<float, uint64_t>(h_vertices, h_triangles,
                                      minval, half_delta, inv_delta,
                                      h_morton, nTriangles);
    //- sort before constructing binary radix tree
    sortMortonCode<uint64_t>(h_morton, h_sortedTriangleIDs);
    if (!quietMode && checkEnabled) {
        cout << "checking sortMortonCode" << endl;
        for (int j = 0; j < min(12, nTriangles); j++) {
            cout << j << ": (" << h_sortedTriangleIDs[j] << ") "
                 << h_morton[j] << endl;
        }
    }
    //data structures used in agglomerative LBVH construction
    BVHNode *d_leafNodes, *d_internalNodes;
    uint64_t *d_morton;
    int *d_sortedTriangleIDs;
    CollisionList *d_hitIDs;
    int sz_morton(nTriangles * sizeof(uint64_t)),
        sz_sortedIDs(nTriangles * sizeof(int)),
        sz_hitIDs(gridXLambda * blockX * sizeof(CollisionList));
    InterceptDistances *d_interceptDists;
    int sz_interceptDists(gridXLambda * blockX * sizeof(InterceptDistances));
    cudaMalloc(&d_leafNodes, nTriangles * sizeof(BVHNode));
    cudaMalloc(&d_internalNodes, nTriangles * sizeof(BVHNode));
    cudaMalloc(&d_morton, sz_morton);
    cudaMalloc(&d_sortedTriangleIDs, sz_sortedIDs);
    cudaMalloc(&d_hitIDs, sz_hitIDs);
    if (interceptsCount) {
        cudaMalloc(&d_interceptDists, sz_interceptDists);
    }
    cudaMemcpy(d_morton, h_morton.data(), sz_morton, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sortedTriangleIDs, h_sortedTriangleIDs.data(), sz_sortedIDs, cudaMemcpyHostToDevice);
    std::vector<uint64_t>().swap(h_morton);
    std::vector<int>().swap(h_sortedTriangleIDs);

    bvhResetKernel<<<gridXt, blockX>>>(d_vertices, d_triangles,
                                       d_internalNodes, d_leafNodes,
                                       d_sortedTriangleIDs, nTriangles);
    cudaDeviceSynchronize();

    bvhConstruct<uint64_t><<<gridXt, blockX>>>(d_internalNodes, d_leafNodes,
                                               d_morton, nTriangles);
    cudaDeviceSynchronize();

    if (barycentric) {
        bvhIntersectionKernel<<<gridXLambda, blockX>>>(
                    d_vertices, d_triangles, d_rayFrom, d_rayTo,
                    d_internalNodes, d_rayBox, d_hitIDs,
                    d_intersectTriangle, d_baryT, d_baryU, d_baryV,
                    d_debug, nTriangles, nRays, queryRayIdx);
    }
    else if (interceptsCount) {
        bvhIntersectionKernel<<<gridXLambda, blockX>>>(
                    d_vertices, d_triangles, d_rayFrom, d_rayTo,
                    d_internalNodes, d_rayBox, d_hitIDs,
                    d_interceptDists, d_crossingDetected,
                    nTriangles, nRays);
    }
    else {
        bvhIntersectionKernel<<<gridXLambda, blockX>>>(
                            d_vertices, d_triangles, d_rayFrom, d_rayTo,
                            d_internalNodes, d_rayBox, d_hitIDs,
                            d_crossingDetected, nTriangles, nRays);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);

    if (! barycentric) {
        HANDLE_ERROR(cudaMemcpy(h_crossingDetected.data(), d_crossingDetected,
                                sz_id, cudaMemcpyDeviceToHost));
        writeData("results_i32", h_crossingDetected);
    }
    else {
        HANDLE_ERROR(cudaMemcpy(h_intersectTriangle.data(), d_intersectTriangle,
                                sz_id, cudaMemcpyDeviceToHost));
        cudaMemcpy(h_baryT.data(), d_baryT, sz_bary, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_baryU.data(), d_baryU, sz_bary, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_baryV.data(), d_baryV, sz_bary, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_debug.data(), d_debug, sz_debug, cudaMemcpyDeviceToHost);
        writeData("intersectTriangle_i32", h_intersectTriangle);
        writeData("barycentricT_f32", h_baryT);
        writeData("barycentricU_f32", h_baryU);
        writeData("barycentricV_f32", h_baryV);
        writeData("debug_f32", h_debug);
    }

    //sanity check
    vector<int> &outcome = !barycentric ? h_crossingDetected : h_intersectTriangle;
    if (! quietMode) {
        cout << "Results for last few elements:" << endl;
        for (int i = nRays - min(20, nRays); i < nRays; i++) {
            cout << i << ": " << outcome[i] << endl;
        }
        cout << "Processing time: ";
        cout << time << " ms" << endl;
    }
} 
