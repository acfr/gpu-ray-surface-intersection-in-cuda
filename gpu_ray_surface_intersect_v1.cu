//Copyright (c) 2022, Raymond Leung
//All rights reserved.
//
//This source code is licensed under the BSD-3-clause license found
//in the LICENSE.md file in the root directory of this source tree.
//
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <vector>

using namespace std;

#define EPSILON 0.000001

//-----------------------------------------------
// This implementation corresponds to version v1
//-----------------------------------------------

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ void triangleBbox(const float *v0, const float *v1,
                             const float *v2, float *vMin, float *vMax)
{
    for (int i = 0; i < 3; i++) {
        if (v0[i] > v1[i]) {
            if (v0[i] > v2[i]) {
                vMax[i] = v0[i];
                vMin[i] = min(v1[i], v2[i]);
            }
            else {
                vMax[i] = v2[i];
                vMin[i] = v1[i];
            }
        }
        else {
            if (v1[i] > v2[i]) {
                vMax[i] = v1[i];
                vMin[i] = min(v0[i], v2[i]);
            }
            else {
                vMax[i] = v2[i];
                vMin[i] = v0[i];
            }
        }
    }
}

__device__ void lineSegmentBbox(const float *p0, const float *p1,
                                float *vMin, float *vMax)
{
    for (int i = 0; i < 3; i++) {
        if (p0[i] > p1[i]) {
            vMax[i] = p0[i];
            vMin[i] = p1[i];
        }
        else {
            vMax[i] = p1[i];
            vMin[i] = p0[i];
        }
    }
}

__device__ bool notOverlap(const float *tMin, const float *tMax,
                           const float *rayMin, const float *rayMax)
{   //this version uses precomputed rayMin and rayMax
    if (rayMin[0] > tMax[0] || rayMax[0] < tMin[0])
        return true;
    if (rayMin[1] > tMax[1] || rayMax[1] < tMin[1])
        return true;
    if (rayMin[2] > tMax[2] || rayMax[2] < tMin[2])
        return true;
    return false;
}

__device__ void subtract(const float *a, const float *b, float *out)
{
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

__device__ void dot(const float *a, const float *b, float &out)
{
    out = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ void cross(const float *a, const float *b, float *out)
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ int intersectMoller(
                const float *v0, const float *v1, const float *v2,
                const float *edge1, const float *edge2,
                const float *q0, const float *q1)
{
    float direction[3], avec[3], bvec[3], tvec[3], t, u, v, det, inv_det;
    subtract(q1, q0, direction);
    cross(direction, edge2, avec);
    dot(avec, edge1, det);
    if (det > EPSILON) {
        subtract(q0, v0, tvec);
        dot(avec, tvec, u);
        if (u < 0 || u > det)
            return 0;
        cross(tvec, edge1, bvec);
        dot(bvec, direction, v);
        if (v < 0 || u + v > det)
            return 0;
    }
    else if (det < -EPSILON) {
        subtract(q0, v0, tvec);
        dot(avec, tvec, u);
        if (u > 0 || u < det)
            return 0;
        cross(tvec, edge1, bvec);
        dot(bvec, direction, v);
        if (v > 0 || u + v < det)
            return 0;
    }
    else
        return 0;
    inv_det = 1.0 / det;
    dot(bvec, edge2, t);
    t *= inv_det;
    if (t < 0 || t > 1) {
        return 0;
    }
    else
        return 1;
}

__global__ void rbxKernel(const float* __restrict__ rayFrom,
                          const float* __restrict__ rayTo,
                          float* __restrict__ vMin,
                          float* __restrict__ vMax, int numRays)
{   //Pre-compute ray bounding box for all line segments,
    //instead of repeating the same in each thread-block.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRays) {
        const float *start = &rayFrom[3*i], *finish = &rayTo[3*i];
        lineSegmentBbox(start, finish, &vMin[3*i], &vMax[3*i]);
    }
}

__global__ void rsiKernel(const float* __restrict__ vertices,
                          const int* __restrict__ triangles,
                          const float* __restrict__ rayFrom,
                          const float* __restrict__ rayTo,
                          const float* __restrict__ rayMin,
                          const float* __restrict__ rayMax,
                          int* __restrict__ resultsR,
                          int numTriangles, int numRays)
{
    //Implement the ray-segment surface intersection test strategy
    //of Jimenez et al (GRAPP 2014) with intersection prescreening
    const int t = blockIdx.y * gridDim.x + blockIdx.x;
    const int i = threadIdx.x;

    //load triangle attributes into shared memory
    __shared__ float triangleVerts[9], tMin[3], tMax[3], edge1[3], edge2[3];
    const float *v0 = &triangleVerts[0],
                *v1 = &triangleVerts[3],
                *v2 = &triangleVerts[6];
    if ((i == 0) && (t < numTriangles)) {
        for(int j = 0; j < 3; j++) {
            int v = triangles[3*t+j];
            for (int k = 0; k < 3; k++) {
                triangleVerts[3*j+k] = vertices[3*v+k];
            }
        }
        triangleBbox(v0, v1, v2, tMin, tMax);
        subtract(v1, v0, edge1);
        subtract(v2, v0, edge2);
    }
    __syncthreads();
    //apply the test if triangle and ray bounding boxes overlap
    if (t < numTriangles) {
        for (int idx = i; idx < numRays; idx += blockDim.x) {
            const float *sMin = &rayMin[3*idx], *sMax = &rayMax[3*idx];
            if (!notOverlap(tMin, tMax, sMin, sMax)) {
                const float *start = &rayFrom[3*idx], *finish = &rayTo[3*idx];
                if (intersectMoller(v0, v1, v2, edge1, edge2, start, finish)) {
                    resultsR[idx] = 1;
                }
            }
        }
    }
}

template <class T>
int readData(string fname, vector<T> &v, int dim=1)
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
 
    cout << fname << " contains " << nbytes << " bytes, "
         << v.size() << " <" << typeid(v.front()).name() << ">, "
         << v.size() / dim << " elements" << endl;
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
    vector<float> h_vertices;
    vector<int>   h_triangles;
    vector<float> h_rayFrom;
    vector<float> h_rayTo;
    vector<int>   h_crossingDetected;
    int nVertices, nTriangles, nRays;

    if (argc == 2 && strcmp(argv[1], "--help")==0)
    {
        cout << "CUDA GPU implementation of Moller-Trumbore ray-triangle intersection test\n"
             << "Optional args:\n"
             << "[1] vertices file, (nVertices,3) as binary float32[]\n"
             << "[2] triangles file, (nTriangles,3) as binary int32[]\n"
             << "[3] segment start points, (nRays,3) as binary float32[]\n"
             << "[4] segment end points, (nRays,3) as binary float32[]\n\n";
        return 0;
    }
    //optional arguments
    std::string fileVertices(argc > 1? argv[1]: "input/vertices_f32"),
                fileTriangles(argc > 2? argv[2]: "input/triangles_i32"),
                fileFrom(argc > 3? argv[3]: "input/rayFrom_f32"),
                fileTo(argc > 4? argv[4]: "input/rayTo_f32");

    //read input data into host memory
    nVertices = readData(fileVertices, h_vertices, 3);
    nTriangles = readData(fileTriangles, h_triangles, 3);
    nRays = readData(fileFrom, h_rayFrom, 3);
    assert(readData(fileTo, h_rayTo, 3) == nRays);
    h_crossingDetected.resize(nRays);
 
    cudaEvent_t start, end;
    float *d_vertices, *d_rayFrom, *d_rayTo, *d_rayMin, *d_rayMax;
    int   *d_triangles, *d_crossingDetected;
    int sz_vertices(3 * nVertices * sizeof(float)),
        sz_triangles(3 * nTriangles * sizeof(int)),
        sz_rays(3 * nRays * sizeof(float));
    cudaMalloc(&d_vertices, sz_vertices);
    cudaMalloc(&d_triangles, sz_triangles);
    cudaMalloc(&d_rayFrom, sz_rays);
    cudaMalloc(&d_rayTo, sz_rays);
    cudaMalloc(&d_rayMin, sz_rays);
    cudaMalloc(&d_rayMax, sz_rays);
    cudaMalloc(&d_crossingDetected, nRays * sizeof(int));
    cudaMemcpy(d_vertices, h_vertices.data(), sz_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, h_triangles.data(), sz_triangles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rayFrom, h_rayFrom.data(), sz_rays, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rayTo, h_rayTo.data(), sz_rays, cudaMemcpyHostToDevice);
    cudaMemset(d_crossingDetected, 0, nRays * sizeof(int));

    //grid partitions: logGridX = ceil(log2(sqrt(n)));
    //                 logGridY = ceil(log2(n)-logGridX);
    //                 2^(logGridX+logGridY) >= n;
    int logGridX = ceil(log2f(sqrt(static_cast<float>(nTriangles))));
    int logGridY = ceil(log2f(nTriangles) - logGridX);
    int gridsX = static_cast<int>(pow(2, logGridX)),
        gridsY = static_cast<int>(pow(2, logGridY)),
        blockX = 1024;
    dim3 dimGrid(gridsX, gridsY, 1);
    dim3 dimBlock(blockX, 1, 1);
    cout << blockX << " threads/block, grids: (" << gridsX
         << "," << gridsY << ")" << endl;
 
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    int gridX = (int)ceil((float)nRays / blockX);
    rbxKernel<<<gridX, blockX>>>(d_rayFrom, d_rayTo, d_rayMin, d_rayMax, nRays);
    cudaDeviceSynchronize();

    rsiKernel<<<dimGrid, dimBlock>>>(d_vertices, d_triangles,
                                     d_rayFrom, d_rayTo, d_rayMin, d_rayMax,
                                     d_crossingDetected, nTriangles, nRays);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
 
    float time = 0;
    cudaEventElapsedTime(&time, start, end);
 
    HANDLE_ERROR(cudaMemcpy(h_crossingDetected.data(), d_crossingDetected,
                            nRays * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Device Variable Copying:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    ofstream fw("results_v1.csv", std::ofstream::out);
    if (fw.is_open()) {
        for (int i = 0; i < nRays; i++) {
            fw << h_crossingDetected[i] << "\n";
        }
        fw.close();
    }
    writeData("results_i32_v1", h_crossingDetected);

    cout << "Processing time: ";
    cout << time << " ms" << endl;
} 
