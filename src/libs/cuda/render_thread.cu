#include "../../headers/cuda/render_thread.hpp"

__device__
Intersection buildIntersection() {
    Intersection i = {
        -1,
        glm::vec3(-1),
        glm::vec3(-1),
        buildMaterial(glm::vec3(-1), -1),
        false
    };
    return i;
}

__device__
float random(glm::vec2 *seed, glm::vec2 randomVector){
    *seed -= randomVector;
    return glm::fract(glm::sin(glm::dot(*seed, glm::vec2(12.9898,78.233))) * 43758.5453);
}

__device__
glm::vec3 Rand3Normal(glm::vec2 *seed, glm::vec2 randomVector) {
    float r1 = random(seed, randomVector) * 2 - 1;
    float r2 = random(seed, randomVector) * 2 - 1;
    float r3 = random(seed, randomVector) * 2 - 1;
    return normalize(glm::vec3(r1, r2, r3));
}

__device__
bool intersectSphere(Ray ray, Sphere s, Intersection *i, Material *materials) { 
    glm::vec3 l = s.center - ray.origin; 
    float tca = glm::dot(l, ray.dir); 
    if (tca < 0) return false;
    float d2 = glm::dot(l, l) - tca * tca;
    float radius2 = s.radius * s.radius;
    if (d2 > radius2) return false;
    float thc = glm::sqrt(radius2 - d2); 
    float t0 = tca - thc; 
    float t1 = tca + thc;

    (*i).hitPoint = ray.origin + t0 * ray.dir;
    (*i).normal = glm::normalize((*i).hitPoint - s.center);
    (*i).t = t0;
    (*i).material = materials[s.materialIdx];
    (*i).hit = true;
    
    return true; 
}

__device__
float intersectSpheres(Ray ray, Intersection *intersection, Sphere* spheres, int nb_sphere, Material* materials) {
    float minDist = 10e30;
    Intersection temp = buildIntersection();

    for(int i = 0; i<nb_sphere; i++) {
        bool inter = intersectSphere(ray, spheres[i], &temp, materials);

        if(inter && temp.t < minDist) {
            minDist = temp.t;
            *intersection = temp;
        }
        temp = buildIntersection();
    }
    return minDist;
}

__device__
bool intersectTriangle(Ray ray, Triangle triangle, Intersection *i,
                       glm::vec3* meshes_vertices, glm::vec3* meshes_normals,
                       Material *materials)
{
    const float EPSILON = 0.0000001;
    glm::vec3 p1 = meshes_vertices[triangle.p1Idx];
    glm::vec3 p2 = meshes_vertices[triangle.p2Idx];
    glm::vec3 p3 = meshes_vertices[triangle.p3Idx];
    glm::vec3 edge1, edge2, h, s, q;
    float a,f,u,v;
    edge1 = p2 - p1;
    edge2 = p3 - p1;
    h = glm::cross(ray.dir, edge2);
    a = glm::dot(edge1, h);

    if (a > - EPSILON && a < EPSILON) return false; //Rayon parallèle

    f = 1.0/a;
    s = ray.origin - p1;
    u = f * (glm::dot(s, h));

    if (u < 0.0 || u > 1.0) return false;
    q = glm::cross(s, edge1);
    v = f *glm:: dot(ray.dir, q);
    if (v < 0.0 || u + v > 1.0) return false;

    float t = f * glm::dot(edge2, q);
    if (t > EPSILON) {


        (*i).hitPoint = ray.origin + t * ray.dir;

        /***** Barycentric coordinates *****/
        glm::vec3 v0 = edge1; glm::vec3 v1 = edge2; glm::vec3 v2 = (*i).hitPoint - p1;
        float d00 = glm::dot(v0, v0);
        float d01 = glm::dot(v0, v1);
        float d11 = glm::dot(v1,v1);
        float d20 = glm::dot(v2,v0);
        float d21 = glm::dot(v2,v1);
        float denom = d00 * d11 - d01 * d01;
	
	    float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom; 
        float u = 1 - v - w;

        glm::vec3 normalHit = w * meshes_normals[triangle.n3Idx] +
                            v * meshes_normals[triangle.n2Idx] + u * meshes_normals[triangle.n1Idx];

        (*i).normal = normalHit;
        (*i).t = t;
        (*i).material = materials[triangle.materialIdx];
        (*i).hit = true;

        return true;

    } else return false;
}

__device__
float intersectTriangles(Ray ray, Intersection *intersection, Triangle* triangles, int nb_triangle,
                         glm::vec3* meshes_vertices, glm::vec3* meshes_normals,
                         Material* materials)
{
    float minDist = 10e30;
    Intersection temp = buildIntersection();

    for(int i = 0; i<nb_triangle; i++) {
        bool inter = intersectTriangle(ray, triangles[i], &temp, meshes_vertices, meshes_normals, materials);

        if(inter && temp.t < minDist) {
            minDist = temp.t;
            *intersection = temp;
        }
        temp = buildIntersection();
    }
    return minDist;
}

__device__
Intersection intersectObjects(Ray ray, Sphere* spheres, Triangle* triangles,
                              glm::vec3* meshes_vertices, glm::vec3* meshes_normals,
                              int nb_sphere, int nb_triangle, Material* materials)
{
    Intersection intersection_spheres = buildIntersection();
    Intersection intersection_triangles = buildIntersection();

    float dist_sphere = intersectSpheres(ray, &intersection_spheres, spheres, nb_sphere, materials);
    float dist_triangle = intersectTriangles(ray, &intersection_triangles, triangles, nb_triangle,
                                             meshes_vertices, meshes_normals, materials);

    float m = glm::min(dist_sphere, dist_triangle);
    if (m == dist_sphere) return intersection_spheres;
    else return intersection_triangles;
}


__device__
glm::vec3 shootRayHemisphere(glm::vec3 normal, glm::vec2* seed, glm::vec2 randomVector) {
    glm::vec3 randomRay = Rand3Normal(seed, randomVector);
    if (glm::dot(normal, randomRay) < 0)
        randomRay *= -1;
    return randomRay;
}


__device__
glm::vec3 getLightPoint(glm::vec3 mask, Intersection i,
                        Sphere* spheres, Triangle* triangles,
                        glm::vec3* meshes_vertices, glm::vec3* meshes_normals,
                        int nb_sphere, int nb_triangle,
                        Material* materials,
                        int idxLight, int nb_sample,
                        glm::vec2* seed, glm::vec2 randomVector)
{
    int temp = nb_sample / 2;
    int count = idxLight == -1 ? nb_sample + glm::ceil((float) temp) : 1;

    for (int j = 0; j<count; j++) {
        glm::vec3 lray;
        if (idxLight == -1) {
            lray = shootRayHemisphere(i.normal, seed, randomVector);
        } else {
            glm::vec3 lp = spheres[idxLight].center + Rand3Normal(seed, randomVector) * spheres[idxLight].radius;
            lray = glm::normalize(lp - i.hitPoint);
        }

        Ray shadow_ray = {
            i.hitPoint + 0.0001f * lray,
            lray
        };

        Intersection temp = intersectObjects(
                                            shadow_ray,
                                            spheres,
                                            triangles,
                                            meshes_vertices,
                                            meshes_normals,
                                            nb_sphere,
                                            nb_triangle,
                                            materials
                                            );

        if(temp.material.emit) {
            float d = glm::clamp(glm::dot(i.normal, shadow_ray.dir), 0.0f, 1.0f);
            return d * temp.material.emit_intensity * temp.material.color * mask;
        }
    }
    return glm::vec3(0);
}

__global__
void trace(glm::mat4 PVMatrix, glm::vec3 cameraPos, glm::vec2 focal_plane,
           glm::vec3* image,
           Material* materials,
           Sphere *spheres, Triangle* triangles,
           glm::vec3* meshes_vertices, glm::vec3* meshes_normals,
           int nb_sphere, int nb_triangle,
           int idxLight, glm::vec2 randomVector,
           int nb_passe, int nb_sample) 
{

    glm::vec3 ambient_color(1, 1, 1);
    float ambient_intensity = 0.05;



    glm::vec2 pixel(threadIdx.x + (blockIdx.x * blockDim.x), threadIdx.y + blockIdx.y * blockDim.y);
    glm::vec2 size(gridDim.x * blockDim.x, gridDim.y * blockDim.y);

    glm::vec2 seed = pixel;

    glm::vec3 antialiasing = Rand3Normal(&seed, randomVector);
    glm::vec3 dir( (2.f * pixel + glm::vec2(antialiasing)) / size - 1.f, 1);
    dir = glm::normalize(PVMatrix * glm::vec4(dir,1));

    if (focal_plane.y == 1) {
        float t_fp = (focal_plane.x - cameraPos.z)/dir.z;
        glm::vec3 p_fp = cameraPos + t_fp * dir;
        glm::vec3 randUnif = Rand3Normal(&seed, randomVector);
        cameraPos = cameraPos + 0.0005f * randUnif;
        dir = p_fp - cameraPos;
    }

    Ray ray = {cameraPos, dir};



    glm::vec3 mask(1);
    glm::vec3 accumulator(0);

    for(int i = 0; i<nb_sample; i++) {
        Intersection intersection = intersectObjects(
                                                    ray,
                                                    spheres,
                                                    triangles,
                                                    meshes_vertices,
                                                    meshes_normals,
                                                    nb_sphere,
                                                    nb_triangle,
                                                    materials
                                                );

        if(!intersection.hit) {
            accumulator += ambient_color * ambient_intensity * mask;
            break;
        } else {
            if (intersection.material.emit) {
                accumulator += intersection.material.color * intersection.material.emit_intensity * mask;
                break;
            } else {

                mask *= intersection.material.color;

                accumulator += getLightPoint(mask, intersection,
                                            spheres, triangles,
                                            meshes_vertices, meshes_normals,
                                            nb_sphere, nb_triangle, materials,
                                            idxLight, nb_sample,
                                            &seed, randomVector);
                
                glm::vec3 ndir = glm::normalize(
                    glm::mix(
                        glm::reflect(ray.dir, intersection.normal),
                        shootRayHemisphere(intersection.normal, &seed, randomVector),
                        intersection.material.roughness
                    )
                );
                Ray t_ray = {
                    intersection.hitPoint + 0.0001f * ndir,
                    ndir
                };
                ray = t_ray;
            }
        }
    }
    int num_row = threadIdx.x + blockIdx.x * blockDim.x;
    int num_line = threadIdx.y + blockIdx.y * blockDim.y;
    int size_row = blockDim.x * gridDim.x;
    int idx = num_row + num_line * size_row;
    image[idx] = image[idx] + (accumulator/float(nb_passe));
}

CudaPool::CudaPool(
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
)
{
    srand (time(NULL));
    glm::vec2 randomVector;

    dim3 dimGrid(dimGridV.x, dimGridV.y, dimGridV.z);
    dim3 dimBlock(dimBlockV.x, dimBlockV.y, dimBlockV.z);
    for (int i = 0; i<nb_passe; i++) {
        randomVector.x = float(rand())/float((RAND_MAX));
        randomVector.y = float(rand())/float((RAND_MAX));
        trace<<<dimGrid, dimBlock>>>(PVMatrix, cameraPos, focal_plane,
                                    image,
                                    materials,
                                    spheres, triangles,
                                    meshes_vertices, meshes_normals,
                                    nb_sphere, nb_triangle,
                                    idxLight, randomVector, nb_passe, nb_sample
                                );
    }
    cudaDeviceSynchronize();
}