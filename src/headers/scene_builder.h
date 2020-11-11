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
        vector<glm::vec4> spheres;
        vector<glm::vec4> materials;
        Camera camera;
        int isLight;
        int nb_spheres;
    
    
    private:
        vector<string> materials_name;
        vector<bool> matIsLight; 
        string readFile(string path);
        vector<string> matchReg(string str, regex r);
        void buildMaterials(vector<string> materials_str);
        void buildSpheres(vector<string> spheres_str);
        void buildCamera(vector<string> camera);
};