#include "dependencies.hpp"
#include "parse_obj.hpp"
#include "cuda/cudaEngine.hpp"
#include <regex>


struct Camera {
        glm::vec3 pos;
        glm::vec3 look_at;
        float fov;
        glm::vec2 focal_plane; //focal_plane.x is the focal plane, if focal_plane.y = 0, focal plane is not used
};

class SceneBuilder {

    public:
        SceneBuilder();
        SceneBuilder(char* path, int* WIDTH, int* HEIGTH);
        void sendDataToCuda(CudaEngine *cudaKernel, int width, int heigth);
    
    private:
        string mat_name = "([a-zA-Z0-9]|_)+";

        vector<Sphere> spheres;
        vector<Triangle> triangles;
        vector<glm::vec3> meshes_vertices;
        vector<glm::vec3> meshes_normals;
        vector<Material> materials;
        int idxLight = -1;
        Camera camera;

        vector<string> materials_name;
        
        void buildMaterials(vector<string> materials_str);
        void buildSpheres(vector<string> spheres_str);
        void buildMeshes(vector<string> meshes_str);
        void buildCamera(vector<string> camera_str);
};
