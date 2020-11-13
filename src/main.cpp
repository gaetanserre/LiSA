#include "headers/engine.h"
#include <stdio.h>

int main(int argc, char** argv) {

	const char *fshader_path = "/home/gaetan/Documents/Projets/Ray-Tracing-OpenGL/Shaders/vf_shaders/fragment_shader.glsl";
	const char *vshader_path = "/home/gaetan/Documents/Projets/Ray-Tracing-OpenGL/Shaders/vf_shaders/vertex_shader.glsl";
	const char *cshader_path = "/home/gaetan/Documents/Projets/Ray-Tracing-OpenGL/Shaders/compute_shader/compute_shader_MC.glsl";

	int WIDTH = 1280, HEIGTH = 720;

	if (argc < 3) {
		printf("Not Enough arguments.\nUsage : ./ray_tracing scene_file nbFrames\n");
		return -1;
	}

	int nbFrames = atoi(argv[2]);
	
	RayTracingEngine engine(argv[1], "Ray tracing engine OpengGL", WIDTH, HEIGTH, vshader_path, fshader_path, cshader_path);

	if (argc == 4) engine.run(nbFrames, argv[3]);
	else engine.run(nbFrames, NULL);

	return 0;

}
