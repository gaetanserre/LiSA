#include "dependencies.h"

GLuint createTex(int TEX_WIDTH, int TEX_HEIGTH);
GLuint LoadVFShaders(const char* vs_file_path, const char* fs_file_path);
GLuint LoadComputeShader(const char* cs_file_path);
GLFWwindow* createWindow(const int width, const int heigth, const char* name);
GLuint getQuadVao();