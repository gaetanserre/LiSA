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

struct RayState
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

static __forceinline__ __device__ RayState* get_ray_state() {
  const unsigned int u0 = optixGetPayload_0();
  const unsigned int u1 = optixGetPayload_1();
  return reinterpret_cast<RayState*>(unpackPointer(u0, u1));
}


/**** TRACE FUNCTIONS ****/

static __forceinline__ __device__ void trace_occlusion(OptixTraversableHandle handle,
                                                      float3 ray_origin,
                                                      float3 ray_direction,
                                                      float  tmin,
                                                      float  tmax,
                                                      RayState* inter)
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
                                                      RayState* inter)
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

  RayState ray_state;
  ray_state.seed = seed;

  for (int i = 0; i < bounces; i++) {
    if (ray_state.done) break;

    trace_radiance(params.handle,
                ray_origin,
                ray_direction,
                1e-6f,
                1e16f,
                &ray_state);
    

    if (ray_state.need_reflection_ray) {
/*       const float fresnel_factor  = fresnel(-ray_direction, ray_state.normal, ray_state.material.n);
      const float3 reflection_dir = reflect(ray_direction, ray_state.normal);
      Material original = ray_state.material;
      RayState pstate_reflection;
      pstate_reflection.seed = seed;
      trace_radiance(params.handle,
                  ray_origin,
                  reflection_dir,
                  1e-6f,
                  1e16f,
                  &pstate_reflection);
      ray_state.accum_color = (fresnel_factor * pstate_reflection.accum_color
                            + (1 - fresnel_factor) * ray_state.accum_color * (1 - original.alpha))
                           * original.diffuse_color;
      ray_state.need_reflection_ray = false; */
    }

    ray_direction = ray_state.direction;
    ray_origin    = ray_state.xyz;
  }
  return ray_state.accum_color;
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
  RayState* ray_state = get_ray_state();
  ray_state->hit = false;
}

extern "C" __global__ void __closesthit__occlusion() {
  const HitGroupData* rt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  RayState* ray_state = get_ray_state();

  if (rt_data->material.alpha < 1.0f) {
    float3 ray_dir          = optixGetWorldRayDirection();
    const float3 ray_origin = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    const float3 normal     = get_barycentric_normal(ray_origin, rt_data);
    ray_dir                 = get_refract_dir(ray_dir, normal, rt_data->material.n);

    trace_occlusion(params.handle, ray_origin, ray_dir, 1e-6f, 1e16f, ray_state);
  } else if (rt_data->material.emit) {
    ray_state->material = rt_data->material;
    ray_state->hit = true;
  }
}


/**** RADIANCE ****/

extern "C" __global__ void __miss__radiance() {
  MissData* rt_data  = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
  RayState* ray_state = get_ray_state();

  ray_state->done        = true;
  ray_state->accum_color += make_float3(rt_data->bg_color) * ray_state->mask_color;
}

extern "C" __device__ float3 shoot_ray_to_light(RayState* ray_state) {
  const unsigned int count = 10u;
  for (int i = 0; i < count; i++) {
    float3 dir = shoot_ray_hemisphere(ray_state->normal, ray_state->seed);
    trace_occlusion(params.handle, ray_state->xyz, dir, 1e-6f, 1e16f, ray_state);

    if (ray_state->hit) {
      const float d = clamp(dot(ray_state->normal, dir), 0.0f, 1.0f);
      return d * ray_state->material.emission_color;
    }
  }
  return make_float3(0.0f);
}

extern "C" __global__ void __closesthit__radiance() {

  HitGroupData* rt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  RayState* ray_state = get_ray_state();
  
  if (rt_data->material.emit) {
    ray_state->accum_color += rt_data->material.emission_color * ray_state->mask_color;
    ray_state->done        = true;
  } else {
    const float3 ray_dir = optixGetWorldRayDirection();
    ray_state->xyz          = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
    ray_state->normal       = get_barycentric_normal(ray_state->xyz, rt_data);

    if (rt_data->material.alpha < 1.0f) {
      ray_state->direction           = get_refract_dir(ray_dir,
                                                      ray_state->normal,
                                                      rt_data->material.n);
      ray_state->need_reflection_ray = false;//dot(-ray_dir, ray_state->normal) > 0;
    } else {
      ray_state->mask_color *= rt_data->material.diffuse_color;
      ray_state->accum_color += shoot_ray_to_light(ray_state) * ray_state->mask_color;

      ray_state->direction = lerp(reflect(ray_dir, ray_state->normal),
                               shoot_ray_hemisphere(ray_state->normal, ray_state->seed),
                               rt_data->material.roughness);  
    }
  }
}