#ifndef AARECT_H
#define AARECT_H

#include "hitable.cuh"
#include "ray.cuh"
#include "vec3.cuh"
#include "aabb.cuh"
class xy_rect : public hitable {
    public:
    __device__ xy_rect() {}
    __device__ xy_rect(float _x0,float _x1,float _y0,float _y1, float _k,material* mat) 
       : x0(_x0),x1(_x1),y0(_y0),y1(_y1),k(_k) {
           mat_ptr = mat;
       };
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record &rec) const override;
    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override{
        // The bounding box must have non-zero width in each dimension, so pad the Z
            // dimension a small amount.
            output_box = aabb(point3(x0,y0, k-0.0001), point3(x1, y1, k+0.0001));
            return true;
    };
    public:
        //material* mp;
        float x0, x1, y0, y1, k;
};
class xz_rect : public hitable {
    public:
        __device__ xz_rect() {}

        __device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _k,
            material* mat)
            : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k){mat_ptr = mat;};

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the Y
            // dimension a small amount.
            output_box = aabb(point3(x0,k-0.0001,z0), point3(x1, k+0.0001, z1));
            return true;
        }

    public:
        //material* mp;
        float x0, x1, z0, z1, k;
};

class yz_rect : public hitable {
    public:
        __device__ yz_rect() {}

        __device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _k,
            material* mat)
            : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k) {mat_ptr = mat;};

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the X
            // dimension a small amount.
            output_box = aabb(point3(k-0.0001, y0, z0), point3(k+0.0001, y1, z1));
            return true;
        }

    public:
        //material* mp;
        float y0, y1, z0, z1, k;
};
__device__ bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record &rec) const {
    auto t = (k-r.origin().z()) / r.direction().z();
    if(t < t_min || t > t_max){
        return false;
    }
    auto x = r.origin().x() + t * r.direction().x();
    auto y = r.origin().y() + t * r.direction().y();
    if(x < x0 || x > x1 || y < y0 || y > y1){
        return false;
    }
    rec.t = t;
    auto outward_normal = vec3(0,0,1);
    rec.normal = set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.point_at_parameter(t);
    return true;
}
__device__ bool xz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k-r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin().x() + t*r.direction().x();
    auto z = r.origin().z() + t*r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.t = t;
    auto outward_normal = vec3(0, 1, 0);
    rec.normal = set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.point_at_parameter(t);
    return true;
}

__device__ bool yz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k-r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max)
        return false;
    auto y = r.origin().y() + t*r.direction().y();
    auto z = r.origin().z() + t*r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    rec.t = t;
    auto outward_normal = vec3(1, 0, 0);
    rec.normal = set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.point_at_parameter(t);
    return true;
}
#endif