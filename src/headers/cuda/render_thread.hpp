#include "../dependencies.hpp"
#include "3Dstructs.hpp"

class CudaPool {
    public:
        CudaPool(
            glm::vec3* image,
            Material* materials,
            Sphere *spheres,
            Triangle* triangles,
            glm::vec3* meshes_vertices,
            glm::vec3* meshes_normals,
            int nb_sphere,
            int nb_triangle,
            int idxLight,
            glm::mat4 PVMatrix,
            glm::vec3 cameraPos,
            glm::vec2 focal_plane,
            glm::vec3 dimGridV,
            glm::vec3 dimBlockV,
            int nb_passe,
            int nb_sample
        );
};