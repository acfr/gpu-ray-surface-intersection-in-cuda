//Copyright (c) 2022, Raymond Leung
//All rights reserved.
//
//This source code is licensed under the BSD-3-clause license found
//in the LICENSE.md file in the root directory of this source tree.
//
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <typeinfo>

/*
Usage: ./sort_rays input/rayFrom_f32 input/rayTo_f32 input/sorted_ray
*/

#include <stdint.h>
#include "bvh_structure.h"

using namespace std;
using namespace lib_bvh;


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
    vector<float> h_rayFrom;
    vector<float> h_rayTo;
    int nRays;
    auto start = chrono::steady_clock::now();

    if (argc == 2 && strcmp(argv[1], "--help")==0)
    {
        cout << "Preprocessor to organise rays according to Z-order scan\n"
             << "Optional args:\n"
             << "[1] segment start points, (nRays,3) as binary float32[]\n"
             << "[2] segment end points, (nRays,3) as binary float32[]\n"
             << "[3] file prefix for output rays (default: \"input/sorted_ray\")\n\n";
        return 0;
    }
    //optional arguments
    std::string fileFrom(argc > 1? argv[1]: "input/rayFrom_f32"),
                fileTo(argc > 2? argv[2]: "input/rayTo_f32"),
                filePrefix(argc > 3? argv[3]: "input/sorted_ray");
    std::string fileOutFrom(filePrefix + "From_f32"),
                fileOutTo(filePrefix + "To_f32"),
                fileOutPermute(filePrefix + "Permutation_i32");

    //read input data
    nRays = readData(fileFrom, h_rayFrom, 3);
    assert(readData(fileTo, h_rayTo, 3) == nRays);

    float minval[3], maxval[3], half_delta[3], inv_delta[3];
    vector<uint64_t> h_morton;
    vector<int> h_sortedRayIDs(nRays);

    //- convert ray endpoints to integer coords
    getMinMaxExtentOfRays<float>(h_rayFrom, h_rayTo, minval, maxval,
                                 half_delta, inv_delta, nRays);
    //- map line segment midpoints to morton code to produce
    //  keys that preserve spatial locality
    createMortonCode<float, uint64_t>(h_rayFrom, h_rayTo, minval,
                                      half_delta, inv_delta, h_morton, nRays);

    thrust::host_vector<uint64_t> thrust_morton(h_morton);
    thrust::host_vector<int> thrust_sortedRayIDs(nRays);
    //- get scan order, this will group objects in 3D space
    //  note: cost of sort is quite substantial, OK if only done once
    //        reduce cost using thrust's device parallelised execution
    sortMortonCodeThrust<uint64_t>(thrust_morton, thrust_sortedRayIDs);

    //- reorder the rays
    thrust::copy(thrust_sortedRayIDs.begin(), thrust_sortedRayIDs.end(),
                 h_sortedRayIDs.begin());
    reorderRays<float>(h_rayFrom, h_rayTo, h_sortedRayIDs);

    writeData(fileOutFrom, h_rayFrom);
    writeData(fileOutTo, h_rayTo);
    writeData(fileOutPermute, h_sortedRayIDs);

    auto end = chrono::steady_clock::now();
    cout << "Elapsed time: "
         << chrono::duration_cast<chrono::nanoseconds>(end - start).count() / 1000000.0
         << " ms" << endl;
} 
