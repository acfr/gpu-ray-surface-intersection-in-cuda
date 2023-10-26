# Docker files

This directory contains two docker files which were tested on the following systems.

`$[rhel7-d1] nvidia-container-cli info`
- `NVRM version:   460.27.04`
- `CUDA version:   11.2`
- `Model:          GeForce GTX TITAN X`
- `GPU UUID:       GPU-95f91607-ec6f-9b5f-9f94-184500a14878`
- `Architecture:   5.2`

`$[ubuntu-d3] nvidia-container-cli info`
- `NVRM version:   525.125.06`
- `CUDA version:   12.0`
- `Model:          NVIDIA GeForce RTX 3090`
- `GPU UUID:       GPU-d60362f6-9af3-c781-4684-e40064407ca6`
- `Architecture:   8.6`

Which dockerfile to use is dependent on the CUDA toolkit version on the host machine, which must be at least as recent as the CPU runtime version deployed in the docker container.

<b>[dockerfile11](dockerfile11.txt)</b> installs the required python 3.8 packages <b>excluding</b> `pycuda`. It is built on top of the `nvidia/cuda:11.2.2-devel-ubuntu18.04` image. Therefore, it can be tested on the `rhel7-d1` host which uses a TITAN X GPU (with CUDA version 11.2).

<b>[dockerfile12](dockerfile12.txt)</b> installs all required python packages including `pycuda`. It is built on top of the `nvidia/cuda:12.0.1-devel-ubuntu20.04` image. Therefore, it can only be tested on the `ubuntu-d3` host which uses a RTX 3090 GPU (with CUDA version 12.0).

| Compatibility    | rhel7-d1 (CUDA 11.2) | ubutntu-d3 (CUDA 12.0) |
|------------------|----------------------|------------------------|
| dockerfile11.txt |          Yes         |          Yes           |
| dockerfile12.txt |          No          |          Yes           |

## Comments
It is possible to run the GPU code in Python without PyCUDA support, simply by importing the `../scripts/gpu_ray_surface_intersect.py` module. In return, it requires binary input files to be explicitly created (read/written) in the background, we also lose the ability to (a) generate GPU code at runtime using Python scripting; (b) configure and perform certain algebraic operations at double precision. We note that the general user may not care about these features. The use case without PyCUDA is shown in `test/demo_cuda.py`.

## Building docker image
`docker build - < ${DOCKER_FILE_NAME}`

where `${DOCKER_FILE_NAME}` is either `dockerfile11.txt` or `dockerfile12.txt`.

## Running docker container
We will mount the `~/wrk/test` local directory, assuming it contains the scripts in `<repo>/docker/test`. To find the SHA for the docker image just built, the `docker image ls` command may be used. For instance, `DOCKER_IMAGE_SHA` might read `ba9173e8d7d9`.

`docker run --gpus all -it -v ~/wrk/test:/home/wrk --rm DOCKER_IMAGE_SHA`

## Testing CUDA code in docker container

`cd /home/wrk`

For <b>`dockerfile11.txt`</b>,
- `cp /gpu-ray-surface-intersection-in-cuda-main/scripts/*.py ./`
- `cp /gpu-ray-surface-intersection-in-cuda-main/pycuda/*.py ./`
- `python3 -m demo_cuda`
- `python3 -m experimental_feature`

For <b>`dockerfile12.txt`</b>, the PyCUDA tests can also be run.
- `cp /home/scripts/* ./`
- `python3 -m demo_pycuda`
- `python3 -m demo_cuda`
- `python3 -m experimental_feature`
