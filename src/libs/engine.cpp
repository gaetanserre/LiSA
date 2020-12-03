#include "../headers/engine.hpp"

RayTracingEngine::RayTracingEngine(char* scene_file_path) {	
	this->scene_builder = SceneBuilder (scene_file_path, &this->WIDTH, &this->HEIGTH);
}

void RayTracingEngine::run(int nb_passe, char* output_path, int nb_sample) {
	this->scene_builder.sendDataToCuda(&this->cudaEngine, this->WIDTH, this->HEIGTH);

	void (*export_img) (int, int, glm::vec3*, char*) = &exportImage;

	this->cudaEngine.run(this->WIDTH, this->HEIGTH, nb_passe, nb_sample, output_path, export_img);
}
