#include <iostream>
#include "optix_wrapper.hh"
#include "display.hh"
#include "scene_parser.hh"

int main(int argc, char** argv) {

  const int samples_per_launch = 16;
  int width, height;
  SceneParser parser("assets/cornell_box.lisa", width, height);
  const float3* vertices    = parser.vertices.data();
  const float3* normals     = parser.normals.data();
  const Material* materials = parser.materials.data();
  const int* mat_indices    = parser.mat_indices.data();
  const int NUM_VERTICES    = parser.vertices.size();
  const int NUM_MATERIALS   = parser.materials.size();

  RendererState state;
  optix_wrapper::init_optix(state);
  optix_wrapper::create_mesh_handler(state, vertices, normals, mat_indices, NUM_MATERIALS, NUM_VERTICES);
  optix_wrapper::create_module(state);
  optix_wrapper::create_programs(state);
  optix_wrapper::create_pipeline(state);
  optix_wrapper::create_shaders_binding_table(state, materials, NUM_MATERIALS);
  optix_wrapper::init_params(state, samples_per_launch, width, height, parser.camera);

  display::render(state);

  optix_wrapper::clean_state(state);
  return 0;
}