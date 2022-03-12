#include "structs.hh"
#include "random.h"

#include <sutil/vec_math.h>

static __forceinline__ __device__ float rng(unsigned int &seed) {
  return rnd(seed) * 2.0f - 1.0f;
}

static __forceinline__ __device__ float3 shoot_ray_hemisphere(const float3 &normal,
                                                              unsigned int &seed)
{
  float3 random_dir = normalize(make_float3(rng(seed), rng(seed), rng(seed)));
  return faceforward(random_dir, normal, random_dir);
}

static __forceinline__ __device__ float fresnel (const float cosT, const float &eta) {
  const float R0 = pow((1 - eta) / (1 + eta), 2);
  return R0 + (1 - R0) * pow (1-cosT, 5);
}

static __forceinline__ __device__ float3 refract(const float &cosI,
                                                 const float3 &ray_dir,
                                                 const float3 &N,
                                                 const float &eta)
{
  float cost2 = 1.0f - eta * eta * (1.0f - cosI*cosI);
  float3 t = eta*ray_dir + ((eta*cosI - sqrt(abs(cost2))) * N);
  return t * (cost2 > 0);
}


static __forceinline__ __device__ float3 barycentric_normal(const float3 &hit_point,
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

  return normalize(u*n1 + v*n2 + w*n3);
}

static __forceinline__ __device__ float3 get_barycentric_normal(const float3 &hit_point,
                                                                const HitGroupData* rt_data)
{
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
