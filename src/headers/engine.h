#include "opengl_utils.h"
#include "scene_builder.h"
#include "export_image.h"

class RayTracingEngine {
    public:
        RayTracingEngine(char* scene_file_path, const char* cshader_path);
        void run(int nbFrames, char* output_path);

    private:
        SceneBuilder scene_builder;
        GLuint quadTex;
        GLuint Compute_Prog;
        int WIDTH;
        int HEIGTH;
};