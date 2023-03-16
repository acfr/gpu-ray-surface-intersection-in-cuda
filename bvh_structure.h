//Copyright (c) 2022, Raymond Leung
//All rights reserved.
//
//This source code is licensed under the BSD-3-clause license found
//in the LICENSE.md file in the root directory of this source tree.
//
#pragma once

#include <algorithm>//std::stable_sort
#include <numeric>  //std::iota
#include <queue>
#include <vector>
#include <sstream>
#include <assert.h>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include "morton3D.h"
#include "rsi_geometry.h"

namespace lib_bvh {
//Tools for building bounding volume hierarchy

#define LEFT 0
#define RIGHT 1
#define ROOT -2
#define QUANT_LEVELS (1 << 21) - 1
#define MAX_COLLISIONS 32
#define MAX_STACK_PTRS 64
#define COMPILE_NON_ESSENTIAL 1

using namespace std;
using namespace lib_morton;
using namespace lib_rsi;

struct BVHNode
{
    AABB bounds;
    BVHNode *childLeft, *childRight;
#if COMPILE_NON_ESSENTIAL
    //not strictly necessary
    BVHNode *parent;
    BVHNode *self;
    int idxSelf, idxChildL, idxChildR, isLeafChildL, isLeafChildR;
#endif
    int triangleID;
    int atomic;
    int rangeLeft, rangeRight;
};

struct CollisionList
{
    uint32_t hits[MAX_COLLISIONS];
    int count;
};

typedef BVHNode* NodePtr;

//declaration

/* transforming object coordinates to morton codes */
template <typename T, typename M>
void inline createMortonCode(const vector<T> &vertices, const vector<int> &triangles,
                             const T *minval, const T *half_delta,  const T *inv_delta,
                             vector<M> &morton, int nTriangles); //for triangle mesh

template <typename T, typename M>
void inline createMortonCode(const vector<T> &rayFrom, const vector<T> &rayTo,
                             const T *minval, const T *half_delta,  const T *inv_delta,
                             vector<M> &morton, int nRays); //for line segments

template <typename M>
void inline sortMortonCode(vector<M> &morton, vector<int> &idx);

template <typename M>
void inline sortMortonCodeThrust(thrust::host_vector<M> &morton,
                                 thrust::host_vector<int> &idx);

/* auxiliary functions */
template <typename T>
void inline getMinMaxExtentOfSurface(const vector<T> &vertices, T *minval, T *maxval,
                                     T *half_delta, T *inv_delta, int nVertices);

template <typename T>
void inline getMinMaxExtentOfRays
           (const vector<T> &rayFrom, const vector<T> &rayTo,
            T *minval, T *maxval, T *half_delta, T *inv_delta, int nRays);

template <typename T>
void inline reorderRays(vector<T> &rayFrom, vector<T> &rayTo, const vector<int> &idx);

void inline unscrambleIntersectionResults(vector<int> &results, const vector<int> &idx);

__device__ bool inline isLeaf(const BVHNode *node); 

__device__ bool inline overlap(const AABB &queryBox, const BVHNode *candidate);

template <typename M>
__device__ M inline highestBit(int i, M *morton);

__device__ void computeTriangleBounds(const float *triangleVerts, AABB &box);

/* core functions */
template <typename M>
__device__ void bvhUpdateParent(BVHNode *node, BVHNode* internalNodes,
                                BVHNode *leafNodes, M *morton, int nNodes);

__device__ bool inline bvhInsert(CollisionList &collisions, int value);

__device__ void bvhTraverse(const AABB& queryBox, NodePtr &bvhNode,
                               NodePtr* &stackPtr, CollisionList &hits);

__device__ void bvhFindCollisions(const float* vertices,
                                  const int* triangles,
                                  const float* rayFrom,
                                  const float* rayTo,
                                  const AABB* rayBox,
                                  const NodePtr bvhRoot,
                                  CollisionList &collisions,
                                  int* detected,
                                  int rayIdx);

__device__ void bvhFindCollisions(const float* vertices,
                                  const int* triangles,
                                  const float* rayFrom,
                                  const float* rayTo,
                                  const AABB* rayBox,
                                  const NodePtr bvhRoot,
                                  CollisionList &collisions,
                                  int* intersectTriangle,
                                  float* t,
                                  float* u,
                                  float* v,
                                  float* debug,
                                  int rayIdx,
                                  int queryRayIdx);

__device__ void bvhFindCollisions(const float* vertices,
                                  const int* triangles,
                                  const float* rayFrom,
                                  const float* rayTo,
                                  const AABB* rayBox,
                                  const NodePtr bvhRoot,
                                  CollisionList &collisions,
                                  InterceptDistances &interceptDists,
                                  int* detected,
                                  int rayIdx);

__global__ void bvhResetKernel(const float* __restrict__ vertices,
                               const int* __restrict__ triangles,
                               BVHNode* __restrict__ internalNodes,
                               BVHNode* __restrict__ leafNodes,
                               int* __restrict__ sortedObjectIDs, int nNodes);

template <typename M>
__global__ void bvhConstruct(BVHNode *internalNodes, BVHNode *leafNodes, M *morton, int nNodes);

__global__ void bvhIntersectionKernel(const float* __restrict__ vertices,
                                      const int* __restrict__ triangles,
                                      const float* __restrict__ rayFrom,
                                      const float* __restrict__ rayTo,
                                      const BVHNode* __restrict__ internalNodes,
                                      const AABB* __restrict__ rayBox,
                                      CollisionList* __restrict__ raytriBoxHitIDs,
                                      int* __restrict__ detected,
                                      int numTriangles, int numRays);

__global__ void bvhIntersectionKernel(const float* __restrict__ vertices,
                                      const int* __restrict__ triangles,
                                      const float* __restrict__ rayFrom,
                                      const float* __restrict__ rayTo,
                                      const BVHNode* __restrict__ internalNodes,
                                      const AABB* __restrict__ rayBox,
                                      CollisionList* __restrict__ raytriBoxHitIDs,
                                      int* __restrict__ intersectTriangle,
                                      float* __restrict__ baryT,
                                      float* __restrict__ baryU,
                                      float* __restrict__ baryV,
                                      float* __restrict__ debug,
                                      int numTriangles, int numRays, int queryRayIdx);

__global__ void bvhIntersectionKernel(const float* __restrict__ vertices,
                                      const int* __restrict__ triangles,
                                      const float* __restrict__ rayFrom,
                                      const float* __restrict__ rayTo,
                                      const BVHNode* __restrict__ internalNodes,
                                      const AABB* __restrict__ rayBox,
                                      CollisionList* __restrict__ raytriBoxHitIDs,
                                      InterceptDistances* __restrict__ rayInterceptDists,
                                      int* __restrict__ detected,
                                      int numTriangles, int numRays);

#if COMPILE_NON_ESSENTIAL
/* diagnostic functions */
BVHNode* testDecipherDescendent(const BVHNode &node, int which,
                                vector<BVHNode> &internalNodes, vector<BVHNode> &leafNodes);

void inline testSimulateTreeExpansion(const BVHNode &node, int levels,
                                      vector<BVHNode> &internalNodes, vector<BVHNode> &leafNodes);

void inline testPrintNode(const BVHNode &node, string desc="");
#endif

//implementation
template <typename T, typename M>
void inline createMortonCode(const vector<T> &vertices,
                             const vector<int> &triangles,
                             const T *minval,
                             const T *half_delta,
                             const T *inv_delta,
                             vector<M>& morton,
                             int nTriangles)
{   //normalise centroid vertices (convert from real to integer)
    //- scale each dimension to use up to 21 bits
    vector<unsigned int> vC[3];
    morton.resize(nTriangles);
    vC[0].resize(nTriangles);
    vC[1].resize(nTriangles);
    vC[2].resize(nTriangles);
    for (int i = 0; i < nTriangles; i++) {
        const float *v0 = &vertices[3*triangles[3*i]],
                    *v1 = &vertices[3*triangles[3*i+1]],
                    *v2 = &vertices[3*triangles[3*i+2]];
        for (int c = 0; c < 3; c++) {
            float centroid = ((v0[c] + v1[c] + v2[c]) / 3.0 - minval[c]);
            vC[c][i] = static_cast<unsigned int>((centroid + half_delta[c]) * inv_delta[c]);
        }
        //- compute morton code
        morton[i] = m3D_e_magicbits<M, unsigned int>(vC[0][i], vC[1][i], vC[2][i]);
    }
}

template <typename T, typename M>
void inline createMortonCode(const vector<T> &rayFrom, const vector<T> &rayTo,
                             const T *minval, const T *half_delta,  const T *inv_delta,
                             vector<M> &morton, int nRays)
{   //overloaded function for line segments
    //minval refers to the rays not the surface
    vector<unsigned int> vC[3];
    morton.resize(nRays);
    vC[0].resize(nRays);
    vC[1].resize(nRays);
    vC[2].resize(nRays);
    for (int i = 0; i < nRays; i++) {
        const float *v0 = &rayFrom[3*i],
                    *v1 = &rayTo[3*i];
        for (int c = 0; c < 3; c++) {
            float centroid = ((v0[c] + v1[c]) / 2.0 - minval[c]);
            vC[c][i] = static_cast<unsigned int>((centroid + half_delta[c]) * inv_delta[c]);
        }
        morton[i] = m3D_e_magicbits<M, unsigned int>(vC[0][i], vC[1][i], vC[2][i]);
    }
}

template <typename M>
void inline sortMortonCode(vector<M> &morton, vector<int> &idx)
{   //initialise original index locations
    idx.resize(morton.size());
    iota(idx.begin(), idx.end(), 0);

    //sort indexes based on comparing values in morton
    stable_sort(idx.begin(), idx.end(),
         [&morton](size_t i1, size_t i2) {return morton[i1] < morton[i2];});
    stable_sort(morton.begin(), morton.end());
}

template <typename M>
void inline sortMortonCodeThrust(thrust::host_vector<M> &h_morton,
                                 thrust::host_vector<int> &h_idx)
{
    //enumerate ray indices which will subsequently be rearranged
    thrust::sequence(thrust::host, h_idx.begin(), h_idx.end(), 0);
    thrust::device_vector<M> d_morton = h_morton;
    thrust::device_vector<int> d_idx = h_idx;
    thrust::sort_by_key(thrust::device, d_morton.begin(), d_morton.end(), d_idx.begin());
    thrust::copy(d_morton.begin(), d_morton.end(), h_morton.begin());
    thrust::copy(d_idx.begin(), d_idx.end(), h_idx.begin());
}

template <typename T>
void inline getMinMaxExtentOfSurface
           (const vector<T> &vertices, T *minval, T *maxval,
            T *half_delta, T *inv_delta, int nVertices, bool silent=false)
{
    vector<T> component;
    component.resize(nVertices);
    const T *sp;
    T *dp;
    for (int c = 0; c < 3; c++) {
        sp = vertices.data() + c;
        dp = component.data();
        for (int i = 0; i < nVertices; i++, sp+=3) {
            *dp++ = *sp;
        }
        minval[c] = *min_element(component.begin(), component.end());
        maxval[c] = *max_element(component.begin(), component.end());
        inv_delta[c] = float(QUANT_LEVELS) / (maxval[c] - minval[c]);
        half_delta[c] = 0.5 / inv_delta[c];
        if (! silent) {
            cout << "[" << minval[c] << ", " << maxval[c] << "] "
                 << "delta: " << 1.0 / inv_delta[c] << endl;
        }
    }
}

template <typename T>
void inline getMinMaxExtentOfRays
           (const vector<T> &rayFrom, const vector<T> &rayTo,
            T *minval, T *maxval, T *half_delta, T *inv_delta, int nRays)
{
    vector<T> component;
    component.resize(2*nRays);
    const T *fp, *tp;
    T *dp;
    for (int c = 0; c < 3; c++) {
        fp = rayFrom.data() + c;
        tp = rayTo.data() + c;
        dp = component.data();
        for (int i = 0; i < nRays; i++, fp+=3, tp+=3) {
            *dp++ = *fp;
            *dp++ = *tp;
        }
        minval[c] = *min_element(component.begin(), component.end());
        maxval[c] = *max_element(component.begin(), component.end());
        inv_delta[c] = float(QUANT_LEVELS) / (maxval[c] - minval[c]);
        half_delta[c] = 0.5 / inv_delta[c];
    }
}

template <typename T>
void inline reorderRays(vector<T> &rayFrom, vector<T> &rayTo, const vector<int> &idx)
{
    vector<float> h_origRayFrom, h_origRayTo;
    h_origRayFrom.swap(rayFrom);
    h_origRayTo.swap(rayTo);
    int n = h_origRayFrom.size() / 3;
    for (int i = 0; i < n; i++) {
        int j = 3 * idx[i];
        rayFrom.push_back(h_origRayFrom[j]);
        rayFrom.push_back(h_origRayFrom[j+1]);
        rayFrom.push_back(h_origRayFrom[j+2]);
        rayTo.push_back(h_origRayTo[j]);
        rayTo.push_back(h_origRayTo[j+1]);
        rayTo.push_back(h_origRayTo[j+2]);
    }
}

void inline unscrambleIntersectionResults(vector<int> &results, const vector<int> &idx)
{
    int n = results.size();
    vector<int> h_permuted(n);
    h_permuted.swap(results);
    for (int i = 0; i < n; i++) {
        results[idx[i]] = h_permuted[i];
    }
}

__device__ bool inline isLeaf(const BVHNode *node) {
    return node->triangleID >= 0;
}

__device__ bool inline overlap(const AABB &queryBox, const BVHNode *candidate) {
    const AABB &tBox = candidate->bounds;
    if (queryBox.xMin > tBox.xMax || queryBox.xMax < tBox.xMin)
        return false;
    if (queryBox.yMin > tBox.yMax || queryBox.yMax < tBox.yMin)
        return false;
    if (queryBox.zMin > tBox.zMax || queryBox.zMax < tBox.zMin)
        return false;
    return true;
}

template <typename M>
__device__ M inline highestBit(int i, M *morton)
{   //find the highest differing bit between two keys: morton[i]
    //and morton[i+1]. In practice, an XOR operation suffices.
    return morton[i] ^ morton[i+1];
}

__device__ void computeTriangleBounds(const float *triangleVerts, AABB &box)
{
    const float *v0 = &triangleVerts[0],
                *v1 = &triangleVerts[3],
                *v2 = &triangleVerts[6];
    if (v0[0] > v1[0]) {
        if (v0[0] > v2[0]) { box.xMin = min(v1[0], v2[0]); box.xMax = v0[0]; }
        else { box.xMin = v1[0]; box.xMax = v2[0]; }
    }
    else { // v1 >= v0
        if (v1[0] > v2[0]) { box.xMax = v1[0]; box.xMin = min(v0[0], v2[0]); }
        else { box.xMax = v2[0]; box.xMin = v0[0]; }
    }
    if (v0[1] > v1[1]) {
        if (v0[1] > v2[1]) { box.yMin = min(v1[1], v2[1]); box.yMax = v0[1]; }
        else { box.yMin = v1[1]; box.yMax = v2[1]; }
    }
    else {
        if (v1[1] > v2[1]) { box.yMin = min(v0[1], v2[1]); box.yMax = v1[1]; }
        else { box.yMin = v0[1]; box.yMax = v2[1]; }
    }
    if (v0[2] > v1[2]) {
        if (v0[2] > v2[2]) { box.zMin = min(v1[2], v2[2]); box.zMax = v0[2]; }
        else { box.zMin = v1[2]; box.zMax = v2[2]; }
    }
    else {
        if (v1[2] > v2[2]) { box.zMax = v1[2]; box.zMin = min(v0[2], v2[2]); }
        else { box.zMax = v2[2]; box.zMin = v0[2]; }
    }
}

template <typename M>
__device__ void bvhUpdateParent(BVHNode* node, BVHNode* internalNodes,
                                BVHNode *leafNodes, M *morton, int nNodes)
{
    /* This is a recursive function. It sets parent node bounding box and
       traverse to the root node, see approach in
       Robbin Marcus, "Real-time Raytracing part 2.1", Accessed: June 2022, URL:
       https://robbinmarcus.blogspot.com/2015/12/real-time-raytracing-part-21.html
    */
    //allow only one thread to process a node
    //  => for leaf nodes: always go through
    //  => for internal nodes: only when both children have been discovered
    if (atomicAdd(&node->atomic, 1) != 1)
        return;
#ifdef COMPILE_NON_ESSENTIAL
    node->self = node;
#endif
    if (! isLeaf(node))
    {   //expand bounds using children's axis-aligned bounding boxes
        const BVHNode *dL = node->childLeft, //descendants
                      *dR = node->childRight;
        node->bounds.xMin = min(dL->bounds.xMin, dR->bounds.xMin);
        node->bounds.xMax = max(dL->bounds.xMax, dR->bounds.xMax);
        node->bounds.yMin = min(dL->bounds.yMin, dR->bounds.yMin);
        node->bounds.yMax = max(dL->bounds.yMax, dR->bounds.yMax);
        node->bounds.zMin = min(dL->bounds.zMin, dR->bounds.zMin);
        node->bounds.zMax = max(dL->bounds.zMax, dR->bounds.zMax);
    }
    /* Deduce parent node index based on split properties described in
       Ciprian Apetrei, "Fast and Simple Agglomerative LBVH Construction",
       EG UK Computer Graphics & Visual Computing, 2014
    */
    int left = node->rangeLeft, right = node->rangeRight;
    BVHNode *parent;
    if (left == 0 || (right != nNodes - 1 &&
        highestBit(right, morton) < highestBit(left - 1, morton)))
    {
        parent = &internalNodes[right];
        parent->childLeft = node;
        parent->rangeLeft = left;
#ifdef COMPILE_NON_ESSENTIAL
        parent->idxChildL = node->idxSelf;
        parent->isLeafChildL = isLeaf(node);
        node->parent = parent;
#endif
    }
    else
    {
        parent = &internalNodes[left - 1];
        parent->childRight = node;
        parent->rangeRight = right;
#ifdef COMPILE_NON_ESSENTIAL
        parent->idxChildR = node->idxSelf;
        parent->isLeafChildR = isLeaf(node);
        node->parent = parent;
#endif
    }
    if (left == 0 && right == nNodes - 1)
    {   //current node represents the root,
        //set left child in last internal node to root
        internalNodes[nNodes - 1].childLeft = node;
        node->triangleID = ROOT;
        return;
    }
    bvhUpdateParent<M>(parent, internalNodes, leafNodes, morton, nNodes);
}

__device__ bool inline bvhInsert(CollisionList &collisions, int value)
{
    //insert value into the hits[] array. Returned value indicates
    //if buffer is full (true => not enough room for two elements).
    collisions.hits[collisions.count++] = static_cast<uint32_t>(value);
    return (collisions.count < MAX_COLLISIONS - 1)? false : true;
}

__device__ void bvhTraverse(
           const AABB& queryBox, NodePtr &bvhNode,
           NodePtr* &stackPtr, CollisionList &hits)
{
    //Note: both reference variables `bvhNode` and `stackPtr` are mutable.
    //When `bvhNode` corresponds to the root node, `stackPtr` points to
    //the next element after the NULL ptr and starts from a clean
    //slate. However, when this function is called a second time,
    //the previous state of the stack is preserved by the calling
    //function bvhFindCollisions within thread-scope, `bvhNode` will
    //point somewhere below the root where BVH tree expansion (path
    //traversal) will resume to prevent overrun of the hits array.

    //traverse nodes starting from the root iteratively
    //recipe is based on Tero Karras's post at
    //https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu
    NodePtr node(bvhNode);
    bool bufferFull(false);
    do
    {
        //check each child node for overlap.
        NodePtr childL = node->childLeft;
        NodePtr childR = node->childRight;
        bool overlapL = overlap(queryBox, childL);
        bool overlapR = overlap(queryBox, childR);

        //query overlaps a leaf node => report collision
        if (overlapL && isLeaf(childL))
            bufferFull = bvhInsert(hits, childL->triangleID);

        if (overlapR && isLeaf(childR))
            bufferFull |= bvhInsert(hits, childR->triangleID);

        //query overlaps an internal node => traverse
        bool traverseL = (overlapL && !isLeaf(childL));
        bool traverseR = (overlapR && !isLeaf(childR));

        if (!traverseL && !traverseR)
            node = *--stackPtr; //pop
        else
        {
            node = (traverseL) ? childL : childR;
            if (traverseL && traverseR)
                *stackPtr++ = childR; //push
        }
    }
    while (node != NULL && !bufferFull);
    //when buffer is full, the input/output argument `bvhNode` is
    //assigned a non-NULL NodePtr to permit resumption after the
    //triangle candidates in the hits buffer have been tested.
    bvhNode = node;
}

/* bvh_structure.h */
__device__ void bvhFindCollisions(const float* vertices,
                                  const int* triangles,
                                  const float* rayFrom,
                                  const float* rayTo,
                                  const AABB* rayBox,
                                  const NodePtr bvhRoot,
                                  CollisionList &collisions,
                                  int* detected,
                                  int rayIdx)
{
    NodePtr stack[MAX_STACK_PTRS];
    NodePtr* stackPtr = stack;
    *stackPtr++ = NULL;
    NodePtr nextNode(bvhRoot);
    do {
        //find potential collisions (subset of triangles to test for)
        //- if collisions buffer is full and there are more nodes
        //  remaining to check, the returned nextNode won't be NULL
        //- importantly, the stack content will persist in memory
        collisions.count = 0;
        bvhTraverse(rayBox[rayIdx], nextNode, stackPtr, collisions);

        //check for actual intersections with the triangles found so far
        int candidate = 0;
        while (! detected[rayIdx] && (candidate < collisions.count)) {
            int triangleID = collisions.hits[candidate++];
            checkRayTriangleIntersection(vertices, triangles, rayFrom, rayTo,
                                         detected, rayIdx, triangleID);
        }
    }
    while ((detected[rayIdx] == 0) && (nextNode != NULL));
}

// This version checks all candidates for the nearest intersection
__device__ void bvhFindCollisions(const float* vertices,
                                  const int* triangles,
                                  const float* rayFrom,
                                  const float* rayTo,
                                  const AABB* rayBox,
                                  const NodePtr bvhRoot,
                                  CollisionList &collisions,
                                  int* intersectTriangle,
                                  float* baryT,
                                  float* baryU,
                                  float* baryV,
                                  float* debug,
                                  int rayIdx,
                                  int queryRayIdx)
{
    NodePtr stack[MAX_STACK_PTRS];
    NodePtr* stackPtr = stack;
    *stackPtr++ = NULL;
    NodePtr nextNode(bvhRoot);
    do {
        collisions.count = 0;
        bvhTraverse(rayBox[rayIdx], nextNode, stackPtr, collisions);

        int candidate = 0;
        while (candidate < collisions.count) {
            int triangleID = collisions.hits[candidate];
            checkRayTriangleIntersection(vertices, triangles, rayFrom, rayTo,
                                         intersectTriangle, baryT, baryU, baryV,
                                         debug, rayIdx, triangleID, candidate, queryRayIdx);
            candidate++;
        }
    }
    while (nextNode != NULL);
}

// This version attempts to count unique ray-surface intersections
__device__ void bvhFindCollisions(const float* vertices,
                                  const int* triangles,
                                  const float* rayFrom,
                                  const float* rayTo,
                                  const AABB* rayBox,
                                  const NodePtr bvhRoot,
                                  CollisionList &collisions,
                                  InterceptDistances &interceptDists,
                                  int* detected,
                                  int rayIdx)
{
    NodePtr stack[MAX_STACK_PTRS];
    NodePtr* stackPtr = stack;
    *stackPtr++ = NULL;
    NodePtr nextNode(bvhRoot);

    interceptDists.count = 0;
    for (int i = 0; i < MAX_INTERSECTIONS; i++) {
        interceptDists.t[i] = -1;
    }
    do {
        collisions.count = 0;
        bvhTraverse(rayBox[rayIdx], nextNode, stackPtr, collisions);

        int candidate = 0;
        while (candidate < collisions.count) {
            int triangleID = collisions.hits[candidate++];
            checkRayTriangleIntersection(vertices, triangles, rayFrom, rayTo,
                                         interceptDists, detected, rayIdx, triangleID);
        }
    }
    while (nextNode != NULL);
}

__global__ void bvhResetKernel(const float* __restrict__ vertices,
                               const int* __restrict__ triangles,
                               BVHNode* __restrict__ internalNodes,
                               BVHNode* __restrict__ leafNodes,
                               int* __restrict__ sortedObjectIDs, int nNodes)
{
   //reset parameters for internal and leaf nodes
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nNodes)
        return;
    //set triangle attributes in leaf
    int t;
    float triangleVerts[9];
    leafNodes[i].triangleID = t = sortedObjectIDs[i];
    for(int j = 0; j < 3; j++) {
        int v = triangles[3*t+j];
        for (int k = 0; k < 3; k++) {
            triangleVerts[3*j+k] = vertices[3*v+k];
        }
    }
    computeTriangleBounds(triangleVerts, leafNodes[i].bounds);

    leafNodes[i].atomic = 1; //this allows the next thread to process
    leafNodes[i].rangeLeft = i;
    leafNodes[i].rangeRight = i;
#ifdef COMPILE_NON_ESSENTIAL
    leafNodes[i].idxSelf = i;
    internalNodes[i].parent = NULL;
    internalNodes[i].idxSelf = i;
#endif
    internalNodes[i].triangleID = -1;
    internalNodes[i].atomic = 0;//first thread passes through
    internalNodes[i].childLeft = internalNodes[i].childRight = NULL;
    internalNodes[i].rangeLeft = internalNodes[i].rangeRight = -1;
    if (nNodes == 1)
    {
        internalNodes[0].bounds = leafNodes[0].bounds;
        internalNodes[0].childLeft = &leafNodes[0];
    }
}

template <typename M>
__global__ void bvhConstruct(BVHNode *internalNodes, BVHNode *leafNodes,
                             M *morton, int nNodes)
{   //construct binary radix tree (Apetrei, 2014)
    //select and update current node's parent in a bottom-up manner
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nNodes)
        return;
    //do this only for leaf nodes, information will propagate upward
    bvhUpdateParent<M>(&leafNodes[i], internalNodes, leafNodes, morton, nNodes);
}

// This version returns ray-surface intersection results via `detected`
__global__ void bvhIntersectionKernel(const float* __restrict__ vertices,
                                      const int* __restrict__ triangles,
                                      const float* __restrict__ rayFrom,
                                      const float* __restrict__ rayTo,
                                      const BVHNode* __restrict__ internalNodes,
                                      const AABB* __restrict__ rayBox,
                                      CollisionList* __restrict__ raytriBoxHitIDs,
                                      int* __restrict__ detected,
                                      int numTriangles, int numRays)
{
    __shared__ NodePtr bvhRoot;
    __shared__ int stride;
    if (threadIdx.x == 0) {
        bvhRoot = internalNodes[numTriangles-1].childLeft;
        stride = gridDim.x * blockDim.x;
    }
    __syncthreads();

    int threadStartIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int bufferIdx = threadStartIdx;
    //iterate if numRays exceeds dimension of thread-block
    for (int idx = threadStartIdx; idx < numRays; idx += stride) {
        if (idx < numRays) {
            //access thread-specific collision array
            CollisionList &collisions = raytriBoxHitIDs[bufferIdx];
            bvhFindCollisions(vertices, triangles, rayFrom, rayTo, rayBox,
                              bvhRoot, collisions, detected, idx);
        }
    }
}

// This version returns results via `intersectTriangle` and barycentric coordinates
__global__ void bvhIntersectionKernel(const float* __restrict__ vertices,
                                      const int* __restrict__ triangles,
                                      const float* __restrict__ rayFrom,
                                      const float* __restrict__ rayTo,
                                      const BVHNode* __restrict__ internalNodes,
                                      const AABB* __restrict__ rayBox,
                                      CollisionList* __restrict__ raytriBoxHitIDs,
                                      int* __restrict__ intersectTriangle,
                                      float* __restrict__ baryT,
                                      float* __restrict__ baryU,
                                      float* __restrict__ baryV,
                                      float* __restrict__ debug,
                                      int numTriangles, int numRays, int queryRayIdx)
{
    __shared__ NodePtr bvhRoot;
    __shared__ int stride;
    if (threadIdx.x == 0) {
        bvhRoot = internalNodes[numTriangles-1].childLeft;
        stride = gridDim.x * blockDim.x;
    }
    __syncthreads();

    int threadStartIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int bufferIdx = threadStartIdx;

    for (int idx = threadStartIdx; idx < numRays; idx += stride) {
        if (idx < numRays) {
            CollisionList &collisions = raytriBoxHitIDs[bufferIdx];
            bvhFindCollisions(vertices, triangles, rayFrom, rayTo, rayBox,
                              bvhRoot, collisions, intersectTriangle,
                              baryT, baryU, baryV, debug, idx, queryRayIdx);
        }
    }
}

// This version counts number of unique surface intersections (limited to < 32)
__global__ void bvhIntersectionKernel(const float* __restrict__ vertices,
                                      const int* __restrict__ triangles,
                                      const float* __restrict__ rayFrom,
                                      const float* __restrict__ rayTo,
                                      const BVHNode* __restrict__ internalNodes,
                                      const AABB* __restrict__ rayBox,
                                      CollisionList* __restrict__ raytriBoxHitIDs,
                                      InterceptDistances* __restrict__ rayInterceptDists,
                                      int* __restrict__ detected,
                                      int numTriangles, int numRays)
{
    __shared__ NodePtr bvhRoot;
    __shared__ int stride;
    if (threadIdx.x == 0) {
        bvhRoot = internalNodes[numTriangles-1].childLeft;
        stride = gridDim.x * blockDim.x;
    }
    __syncthreads();

    int threadStartIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int bufferIdx = threadStartIdx;

    for (int idx = threadStartIdx; idx < numRays; idx += stride) {
        if (idx < numRays) {
            CollisionList &collisions = raytriBoxHitIDs[bufferIdx];
            InterceptDistances &interceptDists = rayInterceptDists[bufferIdx];
            bvhFindCollisions(vertices, triangles, rayFrom, rayTo, rayBox,
                              bvhRoot, collisions, interceptDists, detected, idx);
        }
    }    
}

#if COMPILE_NON_ESSENTIAL
BVHNode* testDecipherDescendent(const BVHNode &node,
                               int which,
                               vector<BVHNode> &internalNodes,
                               vector<BVHNode> &leafNodes)
{   if (which == LEFT) {
        return node.isLeafChildL? &leafNodes[node.idxChildL] :
                                  &internalNodes[node.idxChildL];
    }
    else {
        return node.isLeafChildR? &leafNodes[node.idxChildR] :
                                  &internalNodes[node.idxChildR];
    }
}

void inline testSimulateTreeExpansion(const BVHNode &node, int levels,
                                          vector<BVHNode> &internalNodes,
                                          vector<BVHNode> &leafNodes)
{   testPrintNode(node, "root ");
    queue<BVHNode*> bvhQueue;
    bvhQueue.push(testDecipherDescendent(node, LEFT, internalNodes, leafNodes));
    bvhQueue.push(testDecipherDescendent(node, RIGHT, internalNodes, leafNodes));
    for (int i = 1; i <= levels; i++){
        queue<BVHNode*> bvhNext;
        stringstream buf;
        buf << i;
        while (!bvhQueue.empty()) {
            BVHNode *current = bvhQueue.front();
            testPrintNode(*current, "left(" + buf.str() + ")");
            if (current->triangleID < 0) { //not leaf node
                bvhNext.push(testDecipherDescendent(*current, LEFT, internalNodes, leafNodes));
                bvhNext.push(testDecipherDescendent(*current, RIGHT, internalNodes, leafNodes));
            }
            bvhQueue.pop();
            current = bvhQueue.front();
            testPrintNode(*current, "right(" + buf.str() + ")");
            if (current->triangleID < 0) {
                bvhNext.push(testDecipherDescendent(*current, LEFT, internalNodes, leafNodes));
                bvhNext.push(testDecipherDescendent(*current, RIGHT, internalNodes, leafNodes));
            }
            bvhQueue.pop();
        }
        bvhQueue.swap(bvhNext);
    }
}

void inline testPrintNode(const BVHNode &node, string desc){
    const AABB &b = node.bounds;
    string mkL = (node.isLeafChildL == 0)? "" : "*";
    string mkR = (node.isLeafChildR == 0)? "" : "*";
    cout << desc.c_str() << "node: " << node.self << endl;
    cout << "- childL: " << node.childLeft
         << " [" << node.idxChildL << "]" << mkL
         << ", childR: " << node.childRight
         << " [" << node.idxChildR << "]" << mkR << endl;
    cout << "- parent: " << node.parent << endl;
    cout << "- bounds: (" << b.xMin << "," << b.yMin << "," << b.zMin << "), "
         << "(" << b.xMax << "," << b.yMax << "," << b.zMax << ")" << endl;
    cout << "- triangleID:" << node.triangleID << ", "
         << "atomic: " << node.atomic << ", "
         << "range: [" << node.rangeLeft << ", " 
         << node.rangeRight << "]" << endl;
}
#endif

}