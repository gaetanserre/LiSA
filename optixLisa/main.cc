#include <iostream>
#include "optix_wrapper.hh"
#include "display.hh"
#include "data.hh"

int main(int argc, char** argv) {

  RendererState state;
  optix_wrapper::init_optix(state);
  optix_wrapper::create_mesh_handler(state, vertices, mat_indices, MAT_COUNT, VERTICE_COUNT);
  optix_wrapper::create_module(state);
  optix_wrapper::create_programs(state);
  optix_wrapper::create_pipeline(state);
  optix_wrapper::create_shaders_binding_table(state, materials, MAT_COUNT);
  optix_wrapper::init_params(state, samples_per_launch, width, height);

  display::render(state);

  optix_wrapper::clean_state(state);
  return 0;
}