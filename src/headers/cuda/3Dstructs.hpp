#pragma once

#include "../dependencies.hpp"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

struct Material {
    glm::vec3 color;
    float roughness;
    bool emit;
    float emit_intensity;
};

struct Intersection {
    float t;
    glm::vec3 hitPoint;
    glm::vec3 normal;
    Material material;
    bool hit;
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 dir;
};

struct Sphere {
    glm::vec3 center;
    float radius;
    int materialIdx;
};

struct Plane {
    glm::vec3 position;
    glm::vec3 normal;
    int materialIdx;
};

struct Triangle {
    int p1Idx;
    int p2Idx;
    int p3Idx;
    
    int n1Idx;
    int n2Idx;
    int n3Idx;
    int materialIdx;
};

CUDA_HOSTDEV
Material buildMaterial(glm::vec3 color, float roughness);

CUDA_HOSTDEV
Material buildLight(glm::vec3 emit_color, float intensity);