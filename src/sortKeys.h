
#pragma once
#include <stdint.h>
#include "sceneStructs.h" 

// Keep class ids contiguous for a single partition boundary
enum ShadingClass : int { SHADING_EMISSIVE = 0, SHADING_DIFFUSE = 1 };

// Material-id key per active path
__global__ void buildMaterialKeys(int n,
    const ShadeableIntersection* __restrict__ isects,
    uint32_t* __restrict__ keys);

// Shading-class key per active path
__global__ void buildTypeKeys(int n,
    const ShadeableIntersection* __restrict__ isects,
    const Material* __restrict__ materials,
    int* __restrict__ typeKeys);
