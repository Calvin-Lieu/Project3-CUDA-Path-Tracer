#include "sortKeys.h"

__global__ void buildMaterialKeys(int n,
    const ShadeableIntersection* __restrict__ isects,
    uint32_t* __restrict__ keys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    keys[i] = static_cast<uint32_t>(isects[i].materialId);
}
