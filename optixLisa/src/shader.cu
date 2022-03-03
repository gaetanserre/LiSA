#include <optix.h>

#include "structs.hh"
#include "maths.cu"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include <vector>

static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}


static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

/***** SHADER *****/

extern "C" {
__constant__ Params params;
}

struct PixelState
{
  Material material;
  float3 normal;
  float3 xyz;
  float3 mask_color  = make_float3(1.0f);
  float3 accum_color = make_float3(0.0f);
  float3 direction;
  bool hit = false;
  bool need_reflection_ray = false;
  bool done = false;
  unsigned int seed;
};

static __forceinline__ __device__ PixelState* get_pstate() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<PixelState*>(unpackPointer(u0, u1));
}


/**** TRACE FUNCTIONS ****/

static __forceinline__ __device__ void trace_occlusion(OptixTraversableHandle handle,
                                                      float3 ray_origin,
                                                      float3 ray_direction,
                                                      float  tmin,
                                                      float  tmax,
                                                      PixelState* inter)
{
    unsigned int u0, u1;
    packPointer(inter, u0, u1);
    optixTrace(handle,
              ray_origin,
              ray_direction,
              tmin,
              tmax,
              0.0f,                    // rayTime
              OptixVisibilityMask(1),
              OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
              RAY_TYPE_OCCLUSION,      // SBT offset
              RAY_TYPE_COUNT,          // SBT stride
              RAY_TYPE_OCCLUSION,      // missSBTIndex
              u0, u1);
}


static __forceinline__ __device__ void trace_radiance(OptixTraversableHandle handle,
                                                      float3 ray_origin,
                                                      float3 ray_direction,
                                                      float  tmin,
                                                      float  tmax,
                                                      PixelState* inter)
{
    unsigned int u0, u1;
    packPointer(inter, u0, u1);
    optixTrace(handle,
              ray_origin,
              ray_direction,
              tmin,
              tmax,
              0.0f,                // rayTime
              OptixVisibilityMask(1),
              OPTIX_RAY_FLAG_NONE,
              RAY_TYPE_RADIANCE,        // SBT offset
              RAY_TYPE_COUNT,           // SBT stride
              RAY_TYPE_RADIANCE,        // missSBTIndex
              u0, u1);
}

extern "C" __device__ float3 trace(float3 ray_origin,
                                   float3 ray_direction,
                                   const int &bounces,
                                   unsigned int &seed)
{

  PixelState pstate;
  pstate.seed = seed;

  for (int i = 0; i < bounces; i++) {
    if (pstate.done) break;

    trace_radiance(params.handle,
                ray_origin,
                ray_direction,
                1e-6f,
                1e16f,
                &pstate);
    

    if (pstate.need_reflection_ray) {
      const float fresnel_factor  = fresnel(-ray_direction, pstate.normal, pstate.material.n);
      const float3 reflection_dir = reflect(ray_direction, pstate.normal);
      Material original = pstate.material;
      PixelState pstate_reflection;
      pstate_reflection.seed = seed;
      trace_radiance(params.handle,
                  ray_origin,
                  reflection_dir,
                  1e-6f,
                  1e16f,
                  &pstate_reflection);
      pstate.accum_color = (fresnel_factor * pstate_reflection.accum_color
                            + (1 - fresnel_factor) * pstate.accum_color * (1 - original.alpha))
                           * original.diffuse_color;
      pstate.need_reflection_ray = false;
    }

    ray_direction = pstate.direction;
    ray_origin    = pstate.xyz;
  }
  return pstate.accum_color;
}

extern "C" __global__ void __raygen__rg() {
  const float2 size = make_float2(params.width, params.height);
  const float3 eye  = params.eye;
  const float3 U    = params.U;
  const float3 V    = params.V;
  const float3 W    = params.W;
  const uint3  idx  = optixGetLaunchIndex();
  const float2  idx2 = make_float2(idx.x, idx.y);

  const int subframe_index = params.subframe_index;
  unsigned int seed        = tea<4>(idx.y*size.x + idx.x, subframe_index);

  const int samples_per_launch = params.samples_per_launch;
  const int nb_bounces = 5;

  float3 accum_color = make_float3(0.0f);

  for (int i = 0; i < samples_per_launch; i++) {
    /* Builds ray direction/origin */
    const float2 antialiasing_jitter = normalize(make_float2(rng(seed), rng(seed)));
    const float3 d                   = make_float3((2.0f * idx2 + antialiasing_jitter) / size - 1.0f, 1.0f);
    float3 ray_direction             = normalize(d.x*U + d.y*V + W);
    float3 ray_origin                = eye;

    accum_color += trace(ray_origin, ray_direction, nb_bounces, seed);
  }
  const uint3 launch_index       = optixGetLaunchIndex();
  const unsigned int image_index = launch_index.y * params.width + launch_index.x;
  accum_color                    = accum_color / static_cast<float>(samples_per_launch);

  if( subframe_index > 0 ) {
      const float a                  = 1.0f / static_cast<float>(subframe_index + 1);
      const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
      accum_color = lerp(accum_color_prev, accum_color, a);
  }
  params.accum_buffer[ image_index ] = make_float4(accum_color, 1.0f);
  params.frame_buffer[ image_index ] = make_color (accum_color);
}


/**** OCCLUSION ****/

extern "C" __global__ void __miss__occlusion() {
  PixelState* pstate = get_pstate();
  pstate->hit = false;
}

extern "C" __global__ void __closesthit__occlusion() {
  const HitGroupData* rt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

  if (rt_data->material.alpha == 0.0f) {
    float3 ray_dir          = optixGetWorldRayDirection();
    const float3 ray_origin = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    const float3 normal     = get_barycentric_normal(ray_origin, rt_data);
    ray_dir                 = get_refract_dir(ray_dir, normal, rt_data->material.n);

    PixelState* pstate = get_pstate();
    trace_occlusion(params.handle, ray_origin, ray_dir, 1e-6f, 1e16f, pstate);
  } else if (rt_data->material.emit) {
    PixelState* pstate = get_pstate();
    pstate->material = rt_data->material;
    pstate->hit = true;
  }
}


/**** RADIANCE ****/

extern "C" __global__ void __miss__radiance() {
  MissData* rt_data  = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
  PixelState* pstate = get_pstate();

  pstate->done        = true;
  pstate->accum_color += make_float3(rt_data->bg_color) * pstate->mask_color;
}

extern "C" __device__ float3 shoot_ray_to_light(PixelState* pstate) {
  const unsigned int count = 10u;
  for (int i = 0; i < count; i++) {
    float3 dir = shoot_ray_hemisphere(pstate->normal, pstate->seed);
    trace_occlusion(params.handle, pstate->xyz, dir, 1e-6f, 1e16f, pstate);

    if (pstate->hit) {
      const float d = clamp(dot(pstate->normal, dir), 0.0f, 1.0f);
      return d * pstate->material.emission_color;
    }
  }
  return make_float3(0.0f);
}

extern "C" __global__ void __closesthit__radiance() {

  HitGroupData* rt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  PixelState* pstate = get_pstate();
  
  if (rt_data->material.emit) {
    pstate->accum_color += rt_data->material.emission_color * pstate->mask_color;
    pstate->material    = rt_data->material;
    pstate->done        = true;
  } else {
    const float3 ray_dir = optixGetWorldRayDirection();
    pstate->xyz          = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
    pstate->normal       = get_barycentric_normal(pstate->xyz, rt_data);

    if (rt_data->material.alpha < 1.0f) {
      pstate->direction        = get_refract_dir(ray_dir, pstate->normal, rt_data->material.n);
      pstate->need_reflection_ray = dot(-ray_dir, pstate->normal) > 0;
    } else {
      pstate->mask_color *= rt_data->material.diffuse_color;
      pstate->accum_color += shoot_ray_to_light(pstate) * pstate->mask_color;

      pstate->direction = lerp(reflect(ray_dir, pstate->normal),
                               shoot_ray_hemisphere(pstate->normal, pstate->seed),
                               rt_data->material.roughness);  
    }
  }
}