
#pragma once
#include <stdint.h>
#include "sceneStructs.h" 

// Material-id key per active path
__global__ void buildMaterialKeys(int n,
    const ShadeableIntersection* __restrict__ isects,
    uint32_t* __restrict__ keys);

