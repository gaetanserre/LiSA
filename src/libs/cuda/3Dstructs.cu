#include "../../headers/cuda/3Dstructs.hpp"

Material buildMaterial(glm::vec3 color, float roughness) {
    Material m = {
        color,
        roughness,
        false,
        0
    };
    return m;
}

Material buildLight(glm::vec3 emit_color, float intensity) {
    Material m = {
        emit_color,
        0,
        true,
        intensity
    };
    return m;
}