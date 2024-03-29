#ifndef STRUCTS_HH
#define STRUCTS_HH

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
  float roughness;
  float alpha;
  float n; // refractive index
  float3 diffuse_color;
  bool emit;
  float3 emission_color;
};

inline Material mk_material_emit(float3 color) {
  Material material;
  material.emit           = true;
  material.emission_color = color;
  material.alpha          = 1.0f;
  return material;
};

/**
 * @brief Smart constructor for diffuse material
 * @param color Diffuse color
 * @param alpha Transparency level
 * @param param If alpha = 0, param is consider as the refractive index.
 *              Otherwise, it is considered as the roughness value.
 * @return Material 
*/
inline Material mk_material_diffuse(float3 color, float alpha, float param) {
  Material material;
  if (alpha < 1.0f)
    material.n = param;
  else
    material.roughness = param;

  material.alpha         = alpha;
  material.diffuse_color = color;
  material.emit          = false;
  return material;
};

struct Camera {
  float3 eye;
  float3 look_at;
  float fov;
};

struct OptixParams {
    unsigned int subframe_index;
    float4*      accum_buffer;
    uchar4*      frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;
    unsigned int num_bounces;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    OptixTraversableHandle handle;
};

struct MissData {
  float4 bg_color;
};


struct HitGroupData {
  Material* material;
  float3* vertices;
  float3* normals;
};

struct RayGenData {};

struct RendererParams {
  const float3* vertices;
  const float3* normals;
  const Material* materials;
  const int* mat_indices;
  int num_vertices;
  int num_materials;

  unsigned int width, height;
  Camera camera;
  unsigned int num_samples;
  unsigned int num_bounces;
  const char* output_image;
};

struct RendererState {
  OptixDeviceContext context = 0;

  CUstream stream = 0;

  CUdeviceptr d_vertices = 0;
  CUdeviceptr d_normals  = 0;
  CUdeviceptr d_gas_handler = 0;

  OptixPipeline               pipeline = 0;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixModule                 ptx_module = 0;

  OptixProgramGroup raygen_prog_group    = 0;
  OptixProgramGroup radiance_miss_group  = 0;
  OptixProgramGroup occlusion_miss_group = 0;
  OptixProgramGroup radiance_hit_group   = 0;
  OptixProgramGroup occlusion_hit_group  = 0;

  CUdeviceptr d_materials     = 0;
  OptixShaderBindingTable sbt = {};

  OptixParams  params;
  OptixParams* d_params;
};

#endif // STRUCTS_HH