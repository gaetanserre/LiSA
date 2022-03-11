#include "structs.hh"

class OptixWrapper {
  public:
    OptixWrapper(const RendererParams &params);
    ~OptixWrapper();
    RendererState* get_pstate() { return &(this->state); }
  private:
    void init_optix();
    void create_mesh_handler();
    void create_module();
    void create_programs();
    void create_pipeline();
    void create_shaders_binding_table();
    void init_params();
    void clean_state();

    RendererParams params;
    RendererState state;
};


namespace optix_wrapper {
  void init_optix(RendererState &state);

  void create_mesh_handler(RendererState &state,
                            const float3* vertices,
                            const float3* normals,
                            const int* mat_indices,
                            const int num_materials,
                            const int num_vertices);

  void create_module(RendererState &state);

  void create_programs(RendererState &state);

  void create_pipeline(RendererState &state);

  void create_shaders_binding_table(RendererState &state,
                                    const Material* materials,
                                    const int num_materials);

  void init_params(RendererState &state,
                   int samples_per_launch,
                   int width,
                   int height,
                   Camera camera);

  void clean_state(RendererState &state);
};