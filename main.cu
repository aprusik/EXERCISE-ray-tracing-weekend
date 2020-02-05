#include "ray.h"
#include "hitablelist.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

#include <iostream>
#include <fstream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {  // is a hit
            ray scattered;
            vec3 attenuation;
            // calculate color
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else { // no hit, so render background
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(2020, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(2020, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam,
                       hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || j >= (max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    // sum of all AA samples
    vec3 col(0,0,0);
    for (int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u,v,&local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    
    // average color from all AA samples
    col = col/float(ns);
    fb[pixel_index] = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera,
                             int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        
        int missed = 0;
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = curand_uniform(&local_rand_state);
                vec3 center(a+0.9*curand_uniform(&local_rand_state),0.2,b+0.9*curand_uniform(&local_rand_state));
                if ((center-vec3(4,0.2,0)).length() > 0.9) {
                    if (choose_mat < 0.8f) { // diffuse
                        d_list[i++] = new sphere(center, 0.2,
                            new lambertian(vec3(curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state),
                                                curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state),
                                                curand_uniform(&local_rand_state)*curand_uniform(&local_rand_state))
                            )
                        );
                    }
                    else if (choose_mat < 0.95f) { // metal
                        d_list[i++] = new sphere(center, 0.2,
                            new metal(vec3(0.5f*(1.0f + curand_uniform(&local_rand_state)),
                                            0.5f*(1.0f + curand_uniform(&local_rand_state)),
                                            0.5f*(1.0f + curand_uniform(&local_rand_state))),
                                            0.5f*curand_uniform(&local_rand_state)));
                    }
                    else { // glass
                        d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                    }
                }
                else {
                    missed++;
                }
            }
        }
    
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitablelist(d_list, 22*22+1+3 - missed);

        // define camera
        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        // float dist_to_focus = (lookfrom-lookat).length();
        float dist_to_focus = 10.0;
        float aperture = 0.1;

        *d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 20,
                               float(nx)/float(ny), aperture, dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for (int i=0; i < 22*22+1+3; i++) {
        if (d_list[i] != nullptr) {
            delete ((sphere *)d_list[i])->mat_ptr;
            delete d_list[i];
        }
    }
    delete *d_world;
    delete *d_camera;
}

int main() {
    // determine size of image
    int nx = 1200; // width
    int ny = 800; // height
    int ns = 100; // number of AA samples per pixel

    int tx = 16;
    int ty = 16;

    vec3 lower_left_corner(-2.0, -1.0, -1.0);
    vec3 horizontal(4.0, 0.0, 0.0);
    vec3 vertical(0.0, 2.0, 0.0);
    vec3 origin(0.0, 0.0, 0.0);

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // initialize random states for rays
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    // initialize random state for random scene generation
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, sizeof(curandState)));
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate and create world on GPU
    int num_hitables = 22*22+1+3;
    hitable **d_list = {nullptr};
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render our buffer
    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // open file and add header
    std::ofstream outfile;
    outfile.open ("render.ppm");
    outfile << "P3\n" << nx << " " << ny << "\n255\n";

    // print image to outfile
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            outfile << ir << " " << ig << " " << ib << "\n";
        }
    }

    // Free allocated GPU memory
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}