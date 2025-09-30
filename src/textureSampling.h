#pragma once
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "sceneStructs.h"

__device__ glm::vec3 sampleTexture(const Texture& tex, float u, float v);