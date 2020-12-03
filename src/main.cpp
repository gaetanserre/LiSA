#include "headers/engine.hpp"

int main(int argc, char** argv) {
	if (argc < 4) {
		printf("Not Enough arguments.\nUsage : ./LiSA scene_file nbPasses output_file.ppm [nbSample=3]\n");
		return -1;
	}

	RayTracingEngine engine(argv[1]);

	int nb_passe = atoi(argv[2]);
	int sample = 3;
	char* output_path = argv[3];
	
	if (argc == 5)
		sample = atoi(argv[4]);

	engine.run(nb_passe, output_path, sample);


	return 0;

}
