#pragma once
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "sceneStructs.h"

// Sample an RGB texture (ignore alpha)
__device__ glm::vec3 sampleTexture3(const Texture& tex, float u, float v);

// Sample an RGBA texture (with alpha support)
__device__ glm::vec4 sampleTexture4(const Texture& tex, const glm::vec2& uv);

// Sample metallic/roughness/occlusion maps
__device__ void sampleMetallicRoughness(
    const Material& m,
    const Texture* textures,
    const glm::vec2& uv,
    float& outMetallic,
    float& outRoughness,
    float& outOcclusion);
