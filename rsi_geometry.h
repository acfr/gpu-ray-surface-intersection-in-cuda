//Copyright (c) 2022, Raymond Leung
//All rights reserved.
//
//This source code is licensed under the BSD-3-clause license found
//in the LICENSE.md file in the root directory of this source tree.
//
#pragma once

//Implement ray-surface (line-segment and triangle) intersection tests

#include <vector>
#include <stdint.h>

namespace lib_rsi {

#define EPSILON 0.000001
#define MAX_INTERSECTIONS 32

using namespace std;

//axes-aligned bounding box
struct AABB
{
    float xMin, xMax, yMin, yMax, zMin, zMax;
};

struct InterceptDistances
{
    float t[MAX_INTERSECTIONS];
    int count;
};

//declaration
__device__ void subtract(const float *aVec3, const float *bVec3, float *outVec3);

__device__ void dot(const float *aVec3, const float *bVec3, float &out);

__device__ void cross(const float *aVec3, const float *bVec3, float *outVec3);

__device__ void lineSegmentBbox(const float *p0, const float *p1, AABB &box);

__device__ bool notOverlap(const float *tMin, const float *tMax,
                           const float *rayMin, const float *rayMax);

__device__ int intersectMoller(
                const float *v0, const float *v1, const float *v2,
                const float *edge1, const float *edge2,
                const float *q0, const float *q1);

__device__ int intersectMoller(
                const float *v0, const float *v1, const float *v2,
                const float *edge1, const float *edge2,
                const float *q0, const float *q1,
                float &t, float &u, float &v, float &det, float &epsi, 
                float *avec, float *tvec, float *bvec);

__device__ void checkRayTriangleIntersection(const float* __restrict__ vertices,
                                             const int* __restrict__ triangles,
                                             const float* __restrict__ rayFrom,
                                             const float* __restrict__ rayTo,
                                             int* __restrict__ results,
                                             int rayIdx, int triangleID);

__device__ void checkRayTriangleIntersection(const float* __restrict__ vertices,
                                             const int* __restrict__ triangles,
                                             const float* __restrict__ rayFrom,
                                             const float* __restrict__ rayTo,
                                             int* __restrict__ intersectTriangle,
                                             float* baryT, float* baryU, float* baryV,
                                             float* debug, int rayIdx, int triangleID,
                                             int candidate, int queryRayIdx);

__device__ void checkRayTriangleIntersection(const float* __restrict__ vertices,
                                             const int* __restrict__ triangles,
                                             const float* __restrict__ rayFrom,
                                             const float* __restrict__ rayTo,
                                             InterceptDistances* __restrict__ interceptDists,
                                             int* __restrict__ results,
                                             int rayIdx, int triangleID);

__global__ void rbxKernel(const float* __restrict__ rayFrom,
                          const float* __restrict__ rayTo,
                          AABB* __restrict__ rayBox, int numRays);

template <typename T>
__global__ void initArrayKernel(T* array, T value, int numElements);

//implementation
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

__device__ void lineSegmentBbox(const float *p0, const float *p1, AABB &box)
{
    if (p0[0] > p1[0]) { box.xMin = p1[0]; box.xMax = p0[0]; }
    else               { box.xMin = p0[0]; box.xMax = p1[0]; }
    if (p0[1] > p1[1]) { box.yMin = p1[1]; box.yMax = p0[1]; }
    else               { box.yMin = p0[1]; box.yMax = p1[1]; }
    if (p0[2] > p1[2]) { box.zMin = p1[2]; box.zMax = p0[2]; }
    else               { box.zMin = p0[2]; box.zMax = p1[2]; }
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

// Implement the Moller-Trumbore ray-triangle intersection algorithm
// Only care about distance t, NOT the remaining barycentric coordinates (u,v)
// - Ray model: R(t) = Q0 + t *(Q1 - Q0), where Q0, Q1 denote segment end points
// - Point on triangle: T(u,v) = (1-u-v)*V0 + u*V1 + v*V2
//
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
    else {
        return 1;
    }
}

__device__ int intersectMoller(
                const float *v0, const float *v1, const float *v2,
                const float *edge1, const float *edge2,
                const float *q0, const float *q1,
                float &t, float &u, float &v, float &det, float &epsi,
                float *avec, float *tvec, float *bvec)
{
    float direction[3], inv_det;
    subtract(q1, q0, direction);
    cross(direction, edge2, avec);
    dot(avec, edge1, det);
    epsi = EPSILON;
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
    else {
        u *= inv_det;
        v *= inv_det;
        return 1;
    }
}

/* Report intersection as boolean in `results` */
__device__ void checkRayTriangleIntersection(const float* __restrict__ vertices,
                                             const int* __restrict__ triangles,
                                             const float* __restrict__ rayFrom,
                                             const float* __restrict__ rayTo,
                                             int* __restrict__ results,
                                             int rayIdx, int triangleID)
{
    float triangleVerts[9], edge1[3], edge2[3];
    const float *v0 = &triangleVerts[0],
                *v1 = &triangleVerts[3],
                *v2 = &triangleVerts[6];
    for(int j = 0; j < 3; j++) {
        int v = triangles[3*triangleID+j];
        for (int k = 0; k < 3; k++) {
            triangleVerts[3*j+k] = vertices[3*v+k];
        }
    }
    subtract(v1, v0, edge1);
    subtract(v2, v0, edge2);

    //apply Moller-Trumbore ray-triangle intersection test
    const float *start = &rayFrom[3*rayIdx], *finish = &rayTo[3*rayIdx];
    if (intersectMoller(v0, v1, v2, edge1, edge2, start, finish)) {
        results[rayIdx] = 1;
    }
}

/* @overload Report barycentric coordinates (t,u,v) where t=distance(rayFrom,surface) */
__device__ void checkRayTriangleIntersection(const float* __restrict__ vertices,
                                             const int* __restrict__ triangles,
                                             const float* __restrict__ rayFrom,
                                             const float* __restrict__ rayTo,
                                             int* __restrict__ intersectTriangle,
                                             float* baryT, float* baryU, float* baryV,
                                             float* debug, int rayIdx, int triangleID,
                                             int candidate,
                                             int queryRayIdx)
{
    float triangleVerts[9], edge1[3], edge2[3];
    const float *v0 = &triangleVerts[0],
                *v1 = &triangleVerts[3],
                *v2 = &triangleVerts[6];
    for(int j = 0; j < 3; j++) {
        int v = triangles[3*triangleID+j];
        for (int k = 0; k < 3; k++) {
            triangleVerts[3*j+k] = vertices[3*v+k];
        }
    }
    subtract(v1, v0, edge1);
    subtract(v2, v0, edge2);

    const float *start = &rayFrom[3*rayIdx], *finish = &rayTo[3*rayIdx];
    float t, u, v, det, epsilon, avec[3], tvec[3], bvec[3];
    if (intersectMoller(v0, v1, v2, edge1, edge2, start, finish,
                        t, u, v, det, epsilon, avec, tvec, bvec)) {
        if (t < baryT[rayIdx]) {
            intersectTriangle[rayIdx] = triangleID;
            baryT[rayIdx] = t;
            baryU[rayIdx] = u;
            baryV[rayIdx] = v;
        }
    }
    //hardcoded: only 1 thread in one particular grid-block should write out this data
    if (rayIdx == queryRayIdx) {
        int os = candidate * 48; //offset
        debug[os] = candidate;
        debug[os+1] = rayIdx;
        debug[os+2] = triangleID;
        debug[os+3] = det;
        debug[os+4] = t;
        debug[os+5] = u;
        debug[os+6] = v;
        debug[os+7] = triangles[3*triangleID];
        debug[os+8] = triangles[3*triangleID+1];
        debug[os+9] = triangles[3*triangleID+2];
        debug[os+10] = triangleVerts[0];
        debug[os+11] = triangleVerts[1];
        debug[os+12] = triangleVerts[2];
        debug[os+13] = triangleVerts[3];
        debug[os+14] = triangleVerts[4];
        debug[os+15] = triangleVerts[5];
        debug[os+16] = triangleVerts[6];
        debug[os+17] = triangleVerts[7];
        debug[os+18] = triangleVerts[8];
        debug[os+19] = rayFrom[3*rayIdx];
        debug[os+20] = rayFrom[3*rayIdx+1];
        debug[os+21] = rayFrom[3*rayIdx+2];
        debug[os+22] = rayTo[3*rayIdx];
        debug[os+23] = rayTo[3*rayIdx+1];
        debug[os+24] = rayTo[3*rayIdx+2];
        debug[os+25] = sqrtf((rayFrom[3*rayIdx]-rayTo[3*rayIdx])*(rayFrom[3*rayIdx]-rayTo[3*rayIdx])
                           + (rayFrom[3*rayIdx+1]-rayTo[3*rayIdx+1])*(rayFrom[3*rayIdx+1]-rayTo[3*rayIdx+1])
                           + (rayFrom[3*rayIdx+2]-rayTo[3*rayIdx+2])*(rayFrom[3*rayIdx+2]-rayTo[3*rayIdx+2]));
        debug[os+26] = rayFrom[3*rayIdx] + t * (rayTo[3*rayIdx] - rayFrom[3*rayIdx]);
        debug[os+27] = rayFrom[3*rayIdx+1] + t * (rayTo[3*rayIdx+1] - rayFrom[3*rayIdx+1]);
        debug[os+28] = rayFrom[3*rayIdx+2] + t * (rayTo[3*rayIdx+2] - rayFrom[3*rayIdx+2]);
        debug[os+29] = (1-u-v) * triangleVerts[0] + u * triangleVerts[3] + v * triangleVerts[6]; //(1-u-v)*V0 + u*V1 + v*V2
        debug[os+30] = (1-u-v) * triangleVerts[1] + u * triangleVerts[4] + v * triangleVerts[7];
        debug[os+31] = (1-u-v) * triangleVerts[2] + u * triangleVerts[5] + v * triangleVerts[8];
        debug[os+32] = edge1[0];
        debug[os+33] = edge1[1];
        debug[os+34] = edge1[2];
        debug[os+35] = edge2[0];
        debug[os+36] = edge2[1];
        debug[os+37] = edge2[2];
        debug[os+38] = avec[0];
        debug[os+39] = avec[1];
        debug[os+40] = avec[2];
        debug[os+41] = tvec[0];
        debug[os+42] = tvec[1];
        debug[os+43] = tvec[2];
        debug[os+44] = bvec[0];
        debug[os+45] = bvec[1];
        debug[os+46] = bvec[2];
        debug[os+47] = epsilon;
    }
}

/* @overload Report number of unique ray-surface intersections (limited to < 32) */
__device__ void checkRayTriangleIntersection(const float* __restrict__ vertices,
                                             const int* __restrict__ triangles,
                                             const float* __restrict__ rayFrom,
                                             const float* __restrict__ rayTo,
                                             InterceptDistances &interceptDists,
                                             int* __restrict__ results,
                                             int rayIdx, int triangleID)
{
    float triangleVerts[9], edge1[3], edge2[3];
    const float tol(EPSILON);
    const float *v0 = &triangleVerts[0],
                *v1 = &triangleVerts[3],
                *v2 = &triangleVerts[6];
    for(int j = 0; j < 3; j++) {
        int v = triangles[3*triangleID+j];
        for (int k = 0; k < 3; k++) {
            triangleVerts[3*j+k] = vertices[3*v+k];
        }
    }
    subtract(v1, v0, edge1);
    subtract(v2, v0, edge2);

    const float *start = &rayFrom[3*rayIdx], *finish = &rayTo[3*rayIdx];
    float *tp = interceptDists.t; //circular buffer
    float t, u, v, det, epsilon, avec[3], tvec[3], bvec[3];
    if (intersectMoller(v0, v1, v2, edge1, edge2, start, finish,
                        t, u, v, det, epsilon, avec, tvec, bvec)) {
        bool newIntercept(true);
        for (int i = 0; i < MAX_INTERSECTIONS; i++) {
            if ((t > tp[i] - tol) && (t < tp[i] + tol)) {
                newIntercept = false;
                break;
            }
        }
        if (newIntercept) {
            tp[interceptDists.count & (MAX_INTERSECTIONS - 1)] = t;
            interceptDists.count++;
            results[rayIdx] += 1;
        }
    }
}

__global__ void rbxKernel(const float* __restrict__ rayFrom,
                          const float* __restrict__ rayTo,
                          AABB* __restrict__ rayBox, int numRays)
{   //Pre-compute min/max coordinates for all line segments,
    //instead of repeating the same in each thread-block.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numRays) {
        const float *start = &rayFrom[3*i], *finish = &rayTo[3*i];
        lineSegmentBbox(start, finish, rayBox[i]);
    }
}

template <typename T>
__global__ void initArrayKernel(T* array, T value, int numElements)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements) {
        array[i] = value;
    }
}

}