#ifndef HITABLEH
#define HITABLEH

#include "ray.cuh"
#include "aabb.cuh"

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
    float u,v;
};

class hitable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
        __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const = 0;
        
    public:
        material *mat_ptr;

};

__device__ vec3 set_face_normal(const ray& r, const vec3& outward_normal){
        bool front_face = dot(r.direction(), outward_normal) < 0;
        vec3 normal = front_face ? outward_normal : -outward_normal;
        return normal;
    }
#endif
