#Copyright (c) 2022, Raymond Leung
#All rights reserved.
#
#This source code is licensed under the BSD-3-clause license found
#in the LICENSE.md file in the root directory of this source tree.
#
#Verify the scrambled results from sorted rays are correct
import numpy as np

#(a) gpu_ray_surface_intersect result given unsorted rays
results_unsorted = np.fromfile('results_i32', dtype=np.int32)
#(b) gpu_ray_surface_intersect result given sorted rays (Z-order scan using Morton code)
results_sorted = np.fromfile('sorted_results_i32', dtype=np.int32)
#rearrange indices in (b) to obtain the same ordering as (a)
perm = np.fromfile('sorted_rayPermutation_i32', dtype=np.int32)

results_sorted_rearranged = np.empty(len(results_unsorted), dtype=np.int32)
results_sorted_rearranged[perm] = results_sorted

print('unsorted:          ' + ','.join([str(x) for x in results_unsorted[:20]]))
print('sorted(reordered): ' + ','.join([str(x) for x in results_sorted_rearranged[:20]]))
all(results_unsorted == results_sorted_rearranged)
