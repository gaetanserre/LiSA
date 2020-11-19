#include "dependencies.hpp"
#include "parse_obj.hpp"
#include <regex>


struct Camera {
        glm::vec3 pos;
        glm::vec3 look_at;
        float fov;
};

class SceneBuilder {

    public:
        SceneBuilder();
        SceneBuilder(char* path, int* WIDTH, int* HEIGTH);
        void sendDataToShader(GLuint ComputeShaderProgram, int width, int heigth);
    
    private:
        vector<glm::vec4> spheres;
        vector<glm::vec3> meshes_vertices;
        vector<glm::vec3> meshes_normals;
        vector<glm::vec4> materials;
        vector<int> materials_idx;
        Camera camera;

        vector<string> materials_name;
        vector<bool> matIsLight;
        vector<glm::vec4> materials_temp;

        vector<string> matchReg(string str, regex r);
        void searchDim(string str, int* WIDTH, int* HEIGTH);
        void buildMaterials(vector<string> materials_str);
        void buildSpheres(vector<string> spheres_str);
        void buildMeshes(vector<string> meshes_str);
        void buildCamera(vector<string> camera_str);
};
