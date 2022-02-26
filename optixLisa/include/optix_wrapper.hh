#include "structs.hh"

namespace optix_wrapper {
  void init_optix(RendererState &state);

  void create_mesh_handler(RendererState &state,
                            const float4* vertices,
                            const int* idx_material,
                            const int num_materials,
                            const int num_vertices);

  void create_module(RendererState &state);

  void create_programs(RendererState &state);

  void create_pipeline(RendererState &state);

  void create_shaders_binding_table(RendererState &state,
                                    const float3* diffuse_colors,
                                    const float3* emission_colors,
                                    const int num_materials);

  void init_params(RendererState & state, int samples_per_launch, int width, int height);

  void clean_state(RendererState &state);
};