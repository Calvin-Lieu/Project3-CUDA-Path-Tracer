
#pragma once
#include <stdint.h>
#include "sceneStructs.h" 

// Keep class ids contiguous for a single partition boundary
enum ShadingClass : int { SHADING_EMISSIVE = 0, SHADING_DIFFUSE = 1, SHADING_REFRACTIVE = 2 };

// Material-id key per active path
__global__ void buildMaterialKeys(int n,
    const ShadeableIntersection* __restrict__ isects,
    uint32_t* __restrict__ keys);

