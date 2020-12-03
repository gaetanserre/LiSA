#include "../dependencies.hpp"
#include "render_thread.hpp"

class CudaEngine {
    public:
        CudaEngine(){};
        void init(vector<Material> materials,
                  vector<Sphere> spheres,
                  vector<Triangle> triangles,
                  vector<glm::vec3> meshes_vertices,
                  vector<glm::vec3> meshes_normals,
                  int idxLight,
                  glm::mat4 PVmatrix,
                  glm::vec3 cameraPos, glm::vec2 focal_plane);

        void run(int width, int heigth, int nb_passe, int nb_sample,
                 char* output_path,
                 void (*export_image) (int, int, glm::vec3*, char*));

    private:
        Material* materials;
        Sphere* spheres;
        Triangle* triangles;
        glm::vec3* meshes_vertices;
        glm::vec3* meshes_normals;
        int nb_triangle;
        int nb_sphere;
        int idxLight;
        glm::mat4 PVMatrix;
        glm::vec3 cameraPos;
        glm::vec2 focal_plane;
};

clock_t startChrono();

double stopChrono(clock_t start);