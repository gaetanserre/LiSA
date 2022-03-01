#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sutil/vec_math.h>

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

struct Material {
  float3 diffuse_color;
  bool emit;
  float3 emission_color;
};

static Material mk_material(float3 color, bool emit) {
  Material material;
  if (emit) {
    material.emit = true;
    material.emission_color = color;
    material.diffuse_color = make_float3(0.0f);
  } else {
    material.emit = false;
    material.emission_color = make_float3(0.0f);
    material.diffuse_color = color;
  }
  return material;
}

struct ParallelogramLight {
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
};

struct Params {
    unsigned int subframe_index;
    float4*      accum_buffer;
    uchar4*      frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    OptixTraversableHandle handle;
};

struct RendererState {
  OptixDeviceContext context = 0;

  CUstream stream = 0;

  CUdeviceptr d_vertices = 0;
  CUdeviceptr d_gas_handler = 0;

  OptixPipeline               pipeline = 0;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixModule                 ptx_module = 0;

  OptixProgramGroup raygen_prog_group    = 0;
  OptixProgramGroup radiance_miss_group  = 0;
  OptixProgramGroup occlusion_miss_group = 0;
  OptixProgramGroup radiance_hit_group   = 0;
  OptixProgramGroup occlusion_hit_group  = 0;

  OptixShaderBindingTable sbt = {};

  Params  params;
  Params* d_params;
};


struct MissData {
  float4 bg_color;
};


struct HitGroupData {
  Material material;
  float4* vertices;
};

struct RayGenData {};