#include "headers/engine.h"
#include <stdio.h>

int main(int argc, char** argv) {

	const char *fshader_path = "/home/gaetan/Documents/Projets/Ray Tracing/Shaders/vf_shaders/fragment_shader.glsl";
	const char *vshader_path = "/home/gaetan/Documents/Projets/Ray Tracing/Shaders/vf_shaders/vertex_shader.glsl";
	const char *cshader_path = "/home/gaetan/Documents/Projets/Ray Tracing/Shaders/compute_shader/compute_shader_MC.glsl";

	int WIDTH = 1280, HEIGTH = 720;

	if (argc < 2) {
		printf("Not Enough args.\nUsage : ./ray_tracing nbFrames");
		return -1;
	}

	int nbFrames = atoi(argv[1]);
	
	RayTracingEngine engine = RayTracingEngine("Ray tracing engine OpengGL", WIDTH, HEIGTH, vshader_path, fshader_path, cshader_path);

	engine.run(nbFrames);

	return 0;

}
