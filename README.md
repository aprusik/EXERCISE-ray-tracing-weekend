# My Ray Tracing Weekend
My implementation of Peter Shirley's "[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)"

This was a personal project practice ray tracing, C++, and learn CUDA programming. 
The master branch closely follows Peter Shirley's book: "Ray Tracing in One Weekend".
The "cuda" branch reimplements the original code in CUDA to utilize GPU acceleration and is based off of the article by Rodger Allen: 
"[Accelerated Ray Tracing in One Weekend in CUDA](https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/)"

Developed on an Intel i9 9900k an NVIDIA GeForce RTX 2080 ti.

note: the CPU and CUDA versions use different random number generators, so the images are different, but functionally incorperate the 
same principles.

## CPU Version (single-threaded)
### Image
* Render Time: 351 seconds
* Resolution: 1200x800
* Number of Random Samples per pixel: 10

![cpu-10-pass.png](https://github.com/aprusik/ray-tracing-weekend/blob/cuda/renders/cpu-10-pass.png)

## CUDA version
### Low-Quality Image
* Render Time: ~14 seconds
* Resolution: 1200x800
* Number of Random Samples per pixel: 100

![cuda-100-pass.png](https://github.com/aprusik/ray-tracing-weekend/blob/cuda/renders/cuda-100-pass.png)

### High-Quality Image
* Renter Time: ~134 seconds
* Resolution: 1200x800
* Number of Random Samples per pixel: 1000

![cuda-1000-pass.png](https://github.com/aprusik/ray-tracing-weekend/blob/cuda/renders/cuda-1000-pass.png)

## Implementation Details
Each version of the application creates a .ppm file as the render output.

The CUDA version includes functionality to minimize intersecting spheres.
The CPU version does this too, but it was part of the book.
The CUDA version is implemented differently and wasn't included in Rodger Allen's implementation.

The CPU (plain C++) version will theoretically compile for any machine, but the CUDA version requires it to be run on
a machine with a CUDA capeable GPU (most NVIDIA GPUs)

Images were created as .ppm (portable pixel map) files and converted to .png files using Photoshop.
