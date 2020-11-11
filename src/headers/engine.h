#include "opengl_utils.h"
#include "scene_builder.h"

class RayTracingEngine {
    public:
        RayTracingEngine(
                    const char* window_name,
                    int width,
                    int heigth,
                    const char* vshader_path,
                    const char* fshader_path,
                    const char* cshader_path
        );

        void run(int nbFrames, string scene_file_path);

    private:
        GLuint quad_Tex;
        GLuint Display_Prog;
        GLuint Compute_Prog;
        GLuint quad_VAO;
        GLFWwindow* window;
        int WIDTH;
        int HEIGTH;
};