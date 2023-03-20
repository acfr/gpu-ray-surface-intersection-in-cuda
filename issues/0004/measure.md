### Distance measure

The first measure `d_orthog` uses `compute_perpendicular_distance_from_ray` to compute the orthogonal distance between a point X and line segment PQ. When the estimated location is free from numerical errors, X corresponds exactly to the ray-surface intersecting point.

Since a point close to the extended ray, but strictly outside the interval PQ, can also have a small distance based on this measure, the `find_minimum_distance_from_ray` method checks if the projection of X onto the ray is within the interval PQ. When `proj_PQ(PX)` is inside interval `[0,1]`, `d_orthog` is indeed the nearest distance to the line segment. Otherwise, `d_endpts` is taken as the nearest distance to the line segment. `d_endpts` is computed using `compute_endpoint_distance` and it represents the minimum distance of X with respect to the endpoints P and Q.

```python
inner = lambda x,y : np.sum(x * y, axis=1)
    
def compute_perpendicular_distance_from_ray(vPQ, vPX):
    # find orthogonal distance between point X and line segment PQ
    # d = |cross_product(PQ, PX)| / |PQ|
    cross = np.c_[vPQ[:,1] * vPX[:,2] - vPQ[:,2] * vPX[:,1],
                  vPQ[:,2] * vPX[:,0] - vPQ[:,0] * vPX[:,2],
                  vPQ[:,0] * vPX[:,1] - vPQ[:,1] * vPX[:,0]]
    return np.sqrt(inner(cross, cross)/inner(vPQ, vPQ))

def compute_endpoint_distance(vPX, vQX):
    # compute minimum distance from either end of the line segment
    dPX = np.sqrt(vPX[:,0]**2 + vPX[:,1]**2 + vPX[:,2]**2)
    dQX = np.sqrt(vQX[:,0]**2 + vQX[:,1]**2 + vQX[:,2]**2)
    return np.min(np.c_[dPX, dQX], axis=1)

def find_minimum_distance_from_ray(rayIDs, intersection_points):
    p = raysFrom[rayIDs]
    q = raysTo[rayIDs]
    vPQ = q - p
    vPX = intersection_points - p
    vQX = intersection_points - q
    # check if projection of X onto PQ is within the line segment
    projection = inner(vPQ, vPX) / inner(vPQ, vPQ)
    within = np.logical_and(projection >= 0, projection <= 1).astype(float)
    d_orthog = compute_perpendicular_distance_from_ray(vPQ, vPX)
    d_endpts = compute_endpoint_distance(vPX, vQX)
    return within * d_orthog + (1 - within) * d_endpts
```

For a given `delta`, `numpy.where(d >= delta)` is used to find the rays for which the associated intersecting point (location estimate X) deviates by more than `delta` from the line segment. The distance array `d` returned by `find_minimum_distance_from_ray()` has the same dimensions as `rayIDs`.