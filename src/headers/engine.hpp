#include "opengl_utils.hpp"
#include "scene_builder.hpp"
#include "export_image.hpp"

class RayTracingEngine {
    public:
        RayTracingEngine(
                    char* scene_file_path,
                    const char* window_name,
                    const char* vshader_path,
                    const char* fshader_path,
                    const char* cshader_path
        );

        void run(int nbFrames, char* output_path, int nb_sample);

    private:
        SceneBuilder scene_builder;
        GLuint quadTex;
        GLuint Display_Prog;
        GLuint Compute_Prog;
        GLuint quad_VAO;
        GLFWwindow* window;
        int WIDTH;
        int HEIGTH;
};
