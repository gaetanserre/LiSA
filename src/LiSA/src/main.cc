#include <iostream>
#include "render.hh"
#include "scene_parser.hh"
#include "parse_args.hh"

int main(int argc, char** argv) {
  char* scene_path;
  if (cmdOptionExists(argv, argv + argc, "-s")) {
    scene_path = getCmdOption(argv, argv + argc, "-s");
  } else {
    cerr << "Missing scene path." << endl;
    cerr << "Usage: " << argv[0] << " -s scene_path" << endl;
    exit(1);
  }

  SceneParser parser(scene_path);
  RendererParams params = parser.get_params();
  OptixWrapper wrapper(params);

  printf("Starting rendering...\n");
  if (cmdOptionExists(argv, argv + argc, "-d")) {
    display(*wrapper.get_pstate(), params);
  } else {
    render(*wrapper.get_pstate(), params);
  }

  return 0;
}