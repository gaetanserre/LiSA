#include "../../headers/cuda/cudaEngine.hpp"

clock_t startChrono() { 
    return clock();
}

double stopChrono(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}


void CudaEngine::init(
    vector<Material> materials,
    vector<Sphere> spheres,
    vector<Triangle> triangles,
    vector<glm::vec3> meshes_vertices,
    vector<glm::vec3> meshes_normals,
    int idxLight,
    glm::mat4 PVmatrix,
    glm::vec3 cameraPos, glm::vec2 focal_plane
)
{   
    Material* materials_a;
    cudaMallocManaged(&materials_a, materials.size() * sizeof(Material));
    for (int i = 0; i<materials.size(); i++)
        materials_a[i] = materials[i];
    
    Sphere* spheres_a;
    cudaMallocManaged(&spheres_a, spheres.size() * sizeof(Sphere));
    for (int i = 0; i<spheres.size(); i++)
        spheres_a[i] = spheres[i];

    Triangle* triangles_a;
    cudaMallocManaged(&triangles_a, triangles.size() * sizeof(Triangle));
    for (int i = 0; i<triangles.size(); i++)
        triangles_a[i] = triangles[i];

    glm::vec3* meshes_vertices_a;
    cudaMallocManaged(&meshes_vertices_a, meshes_vertices.size() * sizeof(glm::vec3));
    for(int i = 0; i<meshes_vertices.size(); i++)
        meshes_vertices_a[i] = meshes_vertices[i]
    ;

    glm::vec3* meshes_normals_a;
    cudaMallocManaged(&meshes_normals_a, meshes_normals.size() * sizeof(glm::vec3));
    for(int i = 0; i<meshes_normals.size(); i++)
        meshes_normals_a[i] = meshes_normals[i]
    ;

    this->spheres = spheres_a;
    this->materials = materials_a;
    this->triangles = triangles_a;
    this->meshes_vertices = meshes_vertices_a;
    this->meshes_normals = meshes_normals_a;
    this->nb_triangle = triangles.size();
    this->nb_sphere = spheres.size();
    this->idxLight = idxLight;
    this->PVMatrix = PVmatrix;
    this->cameraPos = cameraPos;
    this->focal_plane = focal_plane;

}

void CudaEngine::run(int width, int heigth, int nb_passe, int nb_sample,
                     char* output_path, void (*export_image) (int, int, glm::vec3*, char*)) {

    glm::vec3 *image;
    int size = width * heigth;
    cudaMallocManaged(&image, size*sizeof(glm::vec3));


    glm::vec3 dimGrid(width/20, heigth/20, 1);
    glm::vec3 dimBlock(20, 20, 1);

    cout << "Starting rendering on " << nb_passe << " pass(es) and " << nb_sample <<" sample(s)..." << endl;
    clock_t start = startChrono();
    CudaThread thread(image,
                     this->materials,
                     this->spheres, this->triangles,
                     this->meshes_vertices,
                     this->meshes_normals,
                     this->nb_sphere,
                     this->nb_triangle,
                     this->idxLight,
                     this->PVMatrix, this->cameraPos, this->focal_plane,
                     dimGrid, dimBlock, nb_passe, nb_sample
    );

    double dur = stopChrono(start) / 60.0;
    printf("Redering finished in %.3f minutes.\n", dur);
    export_image(width, heigth, image, output_path);
    cout << "Exporting finished." << endl;

    cudaFree(this->materials);
    cudaFree(this->spheres);
    cudaFree(this->triangles);
    cudaFree(this->meshes_vertices);
    cudaFree(this->meshes_normals);
    cudaFree(image);
    cudaDeviceReset();
}
