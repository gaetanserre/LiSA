#include "headers/engine.h"
#include <stdio.h>

int main(int argc, char** argv) {

	const char *fshader_path = "../shaders/vf_shaders/fragment_shader.glsl";
	const char *vshader_path = "../shaders/vf_shaders/vertex_shader.glsl";
	const char *cshader_path = "../shaders/compute_shader/compute_shader_MC.glsl";

	if (argc < 3 || argc == 4) {
		printf("Not Enough arguments.\nUsage : ./LiSA scene_file nbPasses [output_file.ppm nbSample=3]\n");
		return -1;
	}

	RayTracingEngine engine(argv[1], "Ray tracing engine OpengGL", vshader_path, fshader_path, cshader_path);

	int nbFrames = atoi(argv[2]);
	int sample = 3;
	char* output_path = NULL;
	if (argc == 5) {
		output_path = argv[3];
		sample = atoi(argv[4]);
	}

	engine.run(nbFrames, output_path, sample);


	return 0;

}