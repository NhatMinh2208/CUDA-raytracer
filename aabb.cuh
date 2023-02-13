#ifndef AABB_H
#define AABB_H
#include "ray.cuh"
#include "vec3.cuh"
class aabb{
    public:
        __device__  aabb(){}
        __device__  aabb(const point3& a, const point3& b){minimum = a; maximum =b;}

        __device__ point3 min() const {return minimum;}
        __device__ point3 max() const {return maximum;}

        __device__ inline bool hit(const ray& r, float t_min, float t_max) const {
            for(int a = 0; a < 3; a++){
                auto inv = 1.0f / r.direction()[a];
                auto t0 = (minimum[a] - r.origin()[a]) * inv;
                auto t1 = (maximum[a] - r.origin()[a]) * inv;
                if (inv < 0.0f){
                    std::swap(t0, t1);
                }
                t_min = (t_min > t0)? t_min : t0;
                t_max = (t_max < t1)? t_max : t1;
                if(t_max <= t_min) return false;
            }
            return true;
        }
    public:
        point3 minimum;
        point3 maximum;
};
__device__ aabb surrounding_box(aabb box0, aabb box1){
    point3 small(fmin(box0.min().x(),box1.min().x()),
                 fmin(box0.min().y(),box1.min().y()),
                 fmin(box0.min().z(),box1.min().z()));
    point3 big(fmax(box0.max().x(), box1.max().x()),
               fmax(box0.max().y(), box1.max().y()),
               fmax(box0.max().z(), box1.max().z()));
    return aabb(small,big);
}
#endif