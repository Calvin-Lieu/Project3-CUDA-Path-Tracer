#include "sortKeys.h"

__global__ void buildMaterialKeys(int n,
    const ShadeableIntersection* __restrict__ isects,
    uint32_t* __restrict__ keys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    keys[i] = static_cast<uint32_t>(isects[i].materialId);
}

__global__ void buildTypeKeys(int n,
    const ShadeableIntersection* __restrict__ isects,
    const Material* __restrict__ materials,
    int* __restrict__ typeKeys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Material& m = materials[isects[i].materialId];
    typeKeys[i] = (m.emittance > 0.0f) ? SHADING_EMISSIVE : SHADING_DIFFUSE;
}
