#ifndef TEXTURE_H
#define TEXTURE_H


//#include "perlin.h"
#include "ray.cuh"
#include "vec3.cuh"
#include <iostream>
class Texture{
    public:
        __device__ virtual vec3 value(float u, float v, const point3& p) const = 0;  
};

class solid_color : public Texture {
    public:
        __device__ solid_color() {}
        __device__ solid_color(vec3 c) : color_value(c) {}

        __device__ solid_color(float red, float green, float blue)
          : solid_color(vec3(red,green,blue)) {}

        __device__ virtual vec3 value(float u, float v, const vec3& p) const override {
            return color_value;
        }

    private:
        vec3 color_value;
};

class checker_texture : public Texture {
    public:
        __device__ checker_texture(){}
        __device__ checker_texture(Texture* _even, Texture* _odd) :
            even(_even), odd(_odd) {}
        __device__ checker_texture(vec3 c1, vec3 c2): even(new solid_color(c1)),
        odd(new solid_color(c2)) {}
        __device__ virtual vec3 value(float u, float v, const vec3& p) const override {
            auto sine = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
            if(sine < 0 ){
                return odd->value(u,v,p);
            }
            else{
                return even->value(u,v,p);
            }
        }
    public: 
        Texture* odd;
        Texture* even;
};

// class noise_texture : public texture {
//     public:
//         noise_texture() {}
//         noise_texture(float sc) : scale(sc) {}
//         virtual color value(float u, float v, const point3& p) const override {
//             return color(1,1,1) * 0.5 * (1 + sin(scale*p.z() + 10*noise.turb(p)));
//         }

//     public:
//         perlin noise;
//         float scale;
// };

class image_texture : public Texture {
public:
    __device__ image_texture() {}
    __device__ image_texture(unsigned char *pixels, int A, int B) : data(pixels), nx(A), ny(B) {}
    __device__ virtual vec3 value(float u, float v, const vec3& p) const;
    
    unsigned char *data;
    int nx, ny;
};

__device__ vec3 image_texture::value(float u, float v, const vec3& p) const {
    int i = u * nx;
    int j = (1 - v) * ny - 0.001;
    if (i < 0) i = 0;
    if (j < 0) j = 0;
    if (i > nx-1) i = nx-1;
    if (j > ny-1) j = ny-1;
    float r = int(data[3 * i + 3 * nx * j ]) / 255.0;
    float g = int(data[3 * i + 3 * nx * j + 1]) / 255.0;
    float b = int(data[3 * i + 3 * nx * j + 2]) / 255.0;
    return vec3(r, g, b);
}

#endif