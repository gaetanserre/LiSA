#include "dependencies.h"
#include <regex>
#include <vector>


struct Camera {
        glm::vec3 pos;
        glm::vec3 look_at;
};

class SceneBuilder {

    public:
        SceneBuilder(string path);
        void sendDataToShader(GLuint ComputeShaderProgram, glm::mat4 projection_matrix);
    
    private:
        vector<glm::vec4> spheres;
        vector<glm::vec4> materials;
        Camera camera;

        vector<string> materials_name;
        vector<bool> matIsLight; 
        vector<string> matchReg(string str, regex r);
        void buildMaterials(vector<string> materials_str);
        void buildSpheres(vector<string> spheres_str);
        void buildCamera(vector<string> camera);
};