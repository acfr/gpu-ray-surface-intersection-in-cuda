## Issues: GPU implementation of a ray-surface intersection algorithm in CUDA

Issue number, followed by the commit where issue is found
- Description
- Resolution
<hr>

1. SHA 2175b498a14bcb32
- Numerical error due to internal float32 representation may be significant
  when working with vertices and rays expressed in [UTM coordinates](
  https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system).
  This may produce unexpected results due to rounding. Refer to details given
  in ```scripts/gpu_ray_surface_intersect.py```, see ```PyGpuRSI.translate_data```
  doc string.
- For the CUDA command line API (see ```gpu_ray_surface_intersect``` usage in
  [sec. 2.3](doc/gpu-rsi-doc.pdf#subsection.2.3)), the caller is responsible
  for centering the input data to minimise its effective range. For the python
  API (see ```PyGpuRSI.test``` method), translation will be performed
  automatically. The minimum spatial coordinates with respect to the first
  supplied surface will be subtracted from the vertices and rays. This offset
  remains unchanged in subsequent calls and the same adjustments will be made
  internally each time, until the PyGpuRSI object goes out of scope.
