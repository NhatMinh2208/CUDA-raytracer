#ifndef TRANSFORM_H
#define TRANSFORM_H
#include "hitable.cuh"
#include "vec3.cuh"
#include "material.cuh"
#include "ray.cuh"

__device__ __constant__ float infinity = std::numeric_limits<float>::infinity();
__device__ __constant__ float pi = 3.1415926535897932385;
__device__ inline float degrees_to_radians(float degrees){
    return (degrees * pi)/ 180.0;
}
class translate : public hitable {
    public:
        __device__ translate(hitable* p, const vec3& displacement)
            : ptr(p), offset(displacement) {}
        __device__ virtual bool hit(
            const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;
    public:
        hitable* ptr;
        vec3 offset;
};

__device__ bool translate::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    ray moved_r = ray(r.origin() - offset, r.direction());
    if(!ptr->hit(moved_r,t_min,t_max,rec)){
        return false;
    }

    rec.p += offset;
    rec.normal = set_face_normal(moved_r, rec.normal);

    return true;
}

__device__ bool translate::bounding_box(float time0, float time1, aabb& output_box) const {
    if(!bounding_box(time0,time1,output_box)) return false;
    output_box = aabb(output_box.min() + offset, output_box.max()+offset);
    return true;
}

class rotate_y : public hitable {
    public:
        __device__ rotate_y(hitable* p, float angle);

        __device__ virtual bool hit(
            const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
            output_box = bbox;
            return hasbox;
        }

    public:
        hitable* ptr;
        float sin_theta;
        float cos_theta;
        bool hasbox;
        aabb bbox;
};
__device__ rotate_y::rotate_y(hitable* p, float angle) : ptr(p) {
    auto radian = degrees_to_radians(angle);
    sin_theta = sin(radian);
    cos_theta = cos(radian);
    hasbox = ptr->bounding_box(0,1,bbox);

    point3 min(infinity,infinity,infinity);
    point3 max(-infinity,-infinity,-infinity);

    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            for(int k = 0; k < 2; k++){
                auto x = i*bbox.max().x() + (1-i)*bbox.min().x();
                auto y = j*bbox.max().y() + (1-j)*bbox.min().y();
                auto z = k*bbox.max().z() + (1-k)*bbox.min().z();

                auto newx =  cos_theta*x + sin_theta*z;
                auto newz = -sin_theta*x + cos_theta*z;

                vec3 tester(newx,y,newz);

                for(int c = 0; c < 3; c++){
                    min[c] = fmin(min[c],tester[c]);
                    max[c] = fmax(max[c],tester[c]);
                }
            }
        }
    }
    bbox = aabb(min,max);
}

__device__ bool rotate_y::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto origin = r.origin();
    auto direction = r.direction();

    origin[0] = cos_theta * origin[0] - sin_theta * origin[2];
    origin[2] = sin_theta * origin[0] + cos_theta * origin[2];

    direction[0] = cos_theta * direction[0] - sin_theta * direction[2];
    direction[2] = sin_theta * direction[0] + cos_theta * direction[2];

    ray new_ray = ray(origin,direction);
    if(!ptr->hit(new_ray,t_min,t_max,rec)) return false;

    auto p = rec.p;
    auto normal = rec.normal;

    p[0] = cos_theta * p[0] + sin_theta * p[2];
    p[2] = -sin_theta * p[0] + cos_theta * p[2];

    normal[0] = cos_theta * normal[0] + sin_theta * normal[2];
    normal[2] = -sin_theta * normal[0] + cos_theta * normal[2];

    rec.p = p;
    rec.normal = set_face_normal(new_ray,normal);
    return true;
}

#endif