#include <optix.h>

#include "structs.hh"
#include "random.h"

#include <sutil/vec_math.h>


extern "C" __device__ float rng(unsigned int &seed) {
  return rnd(seed) * 2.0f - 1.0f;
}

extern "C" __device__ float3 shoot_ray_hemisphere(const float3 &normal, unsigned int &seed) {
  float3 random_dir = normalize(make_float3(rng(seed), rng(seed), rng(seed)));
  
  return faceforward(random_dir, normal, random_dir);
}

extern "C" __device__ float3 get_refract_dir(const float3 &ray_dir,
                                             const float3 &normal_,
                                             const float  &n_1)
{
  double n1, n2, n;
  double cosI = dot(ray_dir,normal_);
  float3 normal = normal_;
  if(cosI > 0.0)
  {
      n1 = n_1;
      n2 = 1.0f;
      normal = -normal_;//invert
  }
  else
  {
      n1 = 1.0f;
      n2 = n_1;
      cosI = -cosI;
  }
  n = n1/n2;
  double sinT2 = n*n * (1.0 - cosI * cosI);
  double cosT = sqrt(1.0 - sinT2);

  if(n == 1.0)
    return ray_dir;
  if(cosT*cosT < 0.0)//tot inner refl
    return reflect(ray_dir, normal);
  return n * ray_dir + (n * cosI - cosT) * normal;
}

extern "C" __device__ float fresnel (const float3 &ray_dir, const float3 &normal, const float &n) {
  const float cost = dot(ray_dir, normal);
  const float R0 = pow((1 - n) / (1 + n), 2);
  return R0 + (1 - R0) * pow (1-cost, 5);
}


extern "C" __device__ float3 barycentric_normal(const float3 &hit_point,
                                                const float3 &n1,
                                                const float3 &n2,
                                                const float3 &n3,
                                                const float3 &v1,
                                                const float3 &v2,
                                                const float3 &v3)
{
  const float3 edge1 = v2 - v1;
  const float3 edge2 = v3 - v1;
  const float3 i = hit_point - v1;
  const float d00 = dot(edge1, edge1);
  const float d01 = dot(edge1, edge2);
  const float d11 = dot(edge2, edge2);
  const float d20 = dot(i, edge1);
  const float d21 = dot(i, edge2);
  const float denom = d00 * d11 - d01 * d01;

  const float w = (d00 * d21 - d01 * d20) / denom; 
  const float v = (d11 * d20 - d01 * d21) / denom;
  const float u = 1 - v - w;

  return u*n1 + v*n2 + w*n3;
}

extern "C" __device__ float3 get_barycentric_normal(const float3 &hit_point, const HitGroupData* rt_data) {
  const int    prim_idx        = optixGetPrimitiveIndex();
  const int    vert_idx_offset = prim_idx*3;

  const float3 v1      = rt_data->vertices[vert_idx_offset + 0];
  const float3 v2      = rt_data->vertices[vert_idx_offset + 1];
  const float3 v3      = rt_data->vertices[vert_idx_offset + 2];
  const float3 n1      = rt_data->normals[vert_idx_offset + 0];
  const float3 n2      = rt_data->normals[vert_idx_offset + 1];
  const float3 n3      = rt_data->normals[vert_idx_offset + 2];
  return barycentric_normal(hit_point, n1, n2, n3, v1, v2, v3);
}