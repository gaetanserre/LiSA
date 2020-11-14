#include "headers/engine.h"
#include <stdio.h>

int main(int argc, char** argv) {

	const char *cshader_path = "/home/gaetan/Documents/Projets/Ray-Tracing-OpenGL/shaders/compute_shader/compute_shader_MC.glsl";

	if (argc < 4) {
		printf("Not Enough arguments.\nUsage : ./ray_tracing scene_file nbFrames output_image.ppm\n");
		return -1;
	}

	int nbFrames = atoi(argv[2]);
	
	RayTracingEngine engine(argv[1], cshader_path);
	engine.run(nbFrames, argv[3]);

	return 0;

}
