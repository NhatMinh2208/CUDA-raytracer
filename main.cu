#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.cuh"
#include "ray.cuh"
#include "sphere.cuh"
#include "hitable_list.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "box.cuh"
#include "transform.cuh"
#include "aarect.cuh"
#include "texture.cuh"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"



// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
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

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    vec3 background = vec3(255/255.0f,192/255.0f,203/255.0f);
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    vec3 emitted = vec3(0.0, 0.0, 0.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if(!((*world)->hit(cur_ray,0.001f,FLT_MAX,rec))){
            return background*cur_attenuation + emitted;
        }
        else{
            ray scattered;
            vec3 attenuation;
            emitted = rec.mat_ptr->emitted(0,0,rec.p);
            if(!(rec.mat_ptr->scatter(r, rec, attenuation, scattered,local_rand_state))){
                return emitted*cur_attenuation;
            }
            cur_attenuation *= attenuation;
            cur_ray = scattered;
        }
        // if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
        //     ray scattered;
        //     vec3 attenuation;
        //     if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
        //         cur_attenuation *= attenuation;
        //         cur_ray = scattered;
        //     }
        //     else {
        //         return vec3(0.0,0.0,0.0);
        //     }
        // }
        // else {
        //     vec3 unit_direction = unit_vector(cur_ray.direction());
        //     float t = 0.5f*(unit_direction.y() + 1.0f);
        //     vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
        //     return cur_attenuation * c;
        // }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))
__global__ void texture_init(unsigned char* tex_data, int nx, int ny, image_texture** tex){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        // #if __CUDA_ARCH__>=200
        //     printf("%d \n", tex_data[0][0]);
        // #endif
        // for(int i = 0; i< 9 ; i++){
        //     tex[i] = new image_texture(tex_data[i], nx[i], ny[i]);
        // }
        *tex = new image_texture(tex_data, nx, ny);
    }
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, image_texture** tex, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    // d_list[i++] = new sphere(center, 0.2,
                    //                          new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(*tex));
                    
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new diffuse_light(vec3(15, 15, 15)));
        // d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(*tex));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void create_universe(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, image_texture** tex, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.1f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(tex[0]));
                    
                }
                else if(choose_mat < 0.2f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(tex[1]));
                }
                else if(choose_mat < 0.3f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(tex[2]));
                }
                else if(choose_mat < 0.4f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(tex[3]));
                }
                else if(choose_mat < 0.5f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(tex[4]));
                }
                else if(choose_mat < 0.6f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(tex[5]));
                }
                else if(choose_mat < 0.7f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(tex[6]));
                }
                else if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(tex[7]));
                }
                else if(choose_mat < 0.9f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(tex[8]));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new diffuse_light(vec3(15, 15, 15)));
        // d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(*tex));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, 22*22+1+3);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void create_cornell_box(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        // auto red   = new lambertian(vec3(.65, .05, .05));
        // auto white = new lambertian(vec3(.73, .73, .73));
        // auto green = new lambertian(vec3(.12, .45, .15));
        //auto light = new diffuse_light(vec3(15, 15, 15));

        int i = 0;
        d_list[0] = new yz_rect(0, 555, 0, 555, 555, new lambertian(vec3(.12, .45, .15)));
        d_list[1] = new yz_rect(0, 555, 0, 555, 0, new lambertian(vec3(.65, .05, .05)));
        d_list[2] = new xz_rect(0, 555, 0, 555, 0,  new lambertian(vec3(.73, .73, .73)));
        d_list[3] = new xz_rect(0, 555, 0, 555, 555,  new lambertian(vec3(.73, .73, .73)));
        d_list[4] = new xy_rect(0, 555, 0, 555, 555,  new lambertian(vec3(.73, .73, .73)));
        d_list[5] = new xz_rect(213, 343, 227, 332, 554, new diffuse_light(vec3(15, 15, 15)));

        hitable* box1 = new box(point3(0, 0, 0), point3(165, 330, 165), new lambertian(vec3(.73, .73, .73)));
        box1 = new rotate_y(box1, 15);
        box1 = new translate(box1, vec3(265,0,295));
        d_list[6] = box1;

        hitable* box2 = new box(point3(0,0,0), point3(165,165,165), new lambertian(vec3(.73, .73, .73)));
        box2 =  new rotate_y(box2, -18);
        box2 = new translate(box2, vec3(130,0,65));
        d_list[7] = box2;

        *d_world  = new hitable_list(d_list, 8);
        auto lookfrom = point3(278, 278, -800);
        auto lookat = point3(278, 278, 0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 40.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);

    }
}

__global__ void free_cornell_box(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 6; i++) {
        delete d_list[i]->mat_ptr;
        delete d_list[i];
    }
    for(int i=6; i < 8; i++){
        delete ((rotate_y*)(((translate*)d_list[i])->ptr))->ptr->mat_ptr;
        delete ((rotate_y*)(((translate*)d_list[i])->ptr))->ptr;
        delete ((translate*)d_list[i])->ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main() {
    int nx = 1200;
    int ny = 800;
    int ns = 16;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    
    int tex_x, tex_y, tex_n;
    int tex_x1, tex_y1, tex_n1;
    unsigned char* tex_data_host;
    
    tex_data_host = stbi_load("image/mercury.jpg", &tex_x, &tex_y, &tex_n, 0);
    
    // stbi_image_free(tex_data_host);
    // tex_data_host = NULL;
    // tex_data_host = stbi_load("image/mars.jpg", &tex_x, &tex_y, &tex_n, 0);
    // tex_data_host[1] = stbi_load("image/mars.jpg", &tex_x, &tex_y, &tex_n, 0);
    //unsigned char* tex_data_host1  = stbi_load("image/jovivan.jpg", &tex_x1, &tex_y1, &tex_n1, 0);
    
    unsigned char **tex_data;
    checkCudaErrors(cudaMallocManaged(tex_data, tex_x * tex_y * tex_n * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(tex_data[0], tex_data_host, tex_x * tex_y * tex_n * sizeof(unsigned char), cudaMemcpyHostToDevice));
   

    //unsigned char *tex_data_host1 = stbi_load("image/mars.jpg", &tex_x1, &tex_y1, &tex_n1, 0);
    // checkCudaErrors(cudaMallocManaged(tex_data+1, tex_x * tex_y * tex_n * sizeof(unsigned char)));
    // checkCudaErrors(cudaMemcpy(tex_data[1], tex_data_host, tex_x * tex_y * tex_n * sizeof(unsigned char), cudaMemcpyHostToDevice));

    image_texture **texture;
    checkCudaErrors(cudaMalloc((void **)&texture, sizeof(image_texture*)));
    texture_init<<<1, 1>>>(tex_data[0], tex_x, tex_y, texture);
    //allocate texture
    // int tex_x[9], tex_y[9], tex_n[9];
    // unsigned char** tex_data = new unsigned char*[9];
    // unsigned char *tex_data_host[9];
    // char* image_address[9] = {"image/earthmap.jpg","image/greenearth.jpg","image/jovian.jpg","image/mars.jpg", "image/mercury.jpg",
    //             "image/moon.jpg", "image/neptune.jpg", "image/tenis.jpg", "image/vanilla.jpg"};
    // for(int i = 0; i < 9; i++){
    //     tex_data_host[i] = stbi_load(image_address[i],tex_x+i,tex_y+i,tex_n+i, 0);
    //     printf("hello %d", tex_x[i]);
    //     checkCudaErrors(cudaMallocManaged(tex_data+i,tex_x[i]*tex_y[i]*tex_n[i]*sizeof(unsigned char)));
    //     printf("h");
    //     std::cerr << tex_data_host[i][0];
    //     checkCudaErrors(cudaMemcpy(tex_data[i], tex_data_host,tex_x[i]*tex_y[i]*tex_n[i]*sizeof(unsigned char),cudaMemcpyHostToDevice));
        
    //     printf("hi");
    // }
    // unsigned char *tex_data_host = stbi_load("image/earthmap.jpg",tex_x,tex_y,tex_n, 0);
    // checkCudaErrors(cudaMallocManaged(&tex_data,tex_x[0]*tex_y[0]*tex_n[0]*sizeof(unsigned char)));
    // checkCudaErrors(cudaMemcpy(tex_data, tex_data_host,tex_x*tex_y*tex_n*sizeof(unsigned char),cudaMemcpyHostToDevice));
    // image_texture** image_tex;
    // checkCudaErrors(cudaMalloc((void**)&image_tex, 9*sizeof(image_texture*)));
    // //texture_init<<<1,1>>>(tex_data);//,tex_x,tex_y,image_tex);
    // checkCudaErrors(cudaGetLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
    // make our world of hitables & the camera
    hitable **d_list;
    int num_hitables = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    //create_cornell_box<<<1,1>>>(d_list, d_world, d_camera, nx, ny);
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny,texture,d_rand_state2);
    //create_universe<<<1,1>>>(d_list, d_world, d_camera, nx, ny,image_tex,d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    //free_cornell_box<<<1,1>>>(d_list,d_world,d_camera);
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(texture));
    // for(int i = 0; i < 9; i++){
    //     //checkCudaErrors(cudaFree(&tex_data + i));
    // }
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
