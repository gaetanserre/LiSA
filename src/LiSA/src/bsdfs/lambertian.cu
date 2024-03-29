#include "structs.hh"
#include "../maths.cu"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

static __forceinline__ __device__ float3 bounce(const float3 &ray_dir,
                                                const float3 &N,
                                                unsigned int &seed,
                                                const Material* mat)
{
  return lerp(reflect(ray_dir, N), shoot_ray_hemisphere(N, seed), mat->roughness);
}

static __forceinline__ __device__ float BRDF(const float3 &N,
                                             const float3 &L,
                                             const Material* mat)
{
  float NdotL            = clamp(dot(N, L), 0.0f, 1.0f);
  const float sampleProb = NdotL / M_PI;
  return NdotL* sampleProb;
}

static __forceinline__ __device__ float BTDF(const float &cosI, const float &eta)
{
  return fresnel(cosI, eta);
}