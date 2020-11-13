#include "opengl_utils.h"
#include "scene_builder.h"
#include "export_image.h"

class RayTracingEngine {
    public:
        RayTracingEngine(
                    char* scene_file_path,
                    const char* window_name,
                    const char* vshader_path,
                    const char* fshader_path,
                    const char* cshader_path
        );

        void run(int nbFrames, char* output_path);

    private:
        SceneBuilder scene_builder;
        GLuint quad_Tex;
        GLuint Display_Prog;
        GLuint Compute_Prog;
        GLuint quad_VAO;
        GLFWwindow* window;
        int WIDTH;
        int HEIGTH;
};