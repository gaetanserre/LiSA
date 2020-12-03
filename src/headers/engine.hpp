#include "scene_builder.hpp"
#include "export_image.hpp"

class RayTracingEngine {
    public:
        RayTracingEngine(char* scene_file_path);

        void run(int nb_passe, char* output_path, int nb_sample);

    private:
        SceneBuilder scene_builder;
        CudaEngine cudaEngine;
        int WIDTH;
        int HEIGTH;
};
