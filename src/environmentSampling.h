#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "sceneStructs.h"
#include "utilities.h"

// Binary search for CDF lookup
__device__ inline int binarySearchCDF(const float* cdf, int size, float u) {
    int left = 0, right = size - 1;
    while (left < right) {
        int mid = (left + right) / 2;
        if (cdf[mid] < u) left = mid + 1;
        else right = mid;
    }
    return left;
}

__device__ inline glm::vec3 sampleEnvironmentMap(const glm::vec3& direction,
    const EnvironmentMap& envMap) {
    float theta = acosf(glm::clamp(direction.y, -1.0f, 1.0f));
    float phi = atan2f(direction.z, direction.x);
    if (phi < 0.0f) phi += 2.0f * PI;

    float u = phi / (2.0f * PI);
    float v = theta / PI;

    float4 texel = tex2D<float4>(envMap.texture, u, v);
    return glm::vec3(texel.x, texel.y, texel.z);
}

__device__ inline glm::vec3 sampleEnvironmentMapImportance(
    const EnvironmentMap& envMap,
    float u1, float u2,
    glm::vec3& direction,
    float& pdf)
{
    // Sample row using marginal CDF
    int y = binarySearchCDF(envMap.marginalCDF, envMap.height, u1);
    float marginalPdf = (y == 0) ? envMap.marginalCDF[0] :
        (envMap.marginalCDF[y] - envMap.marginalCDF[y - 1]);

    // Sample column using conditional CDF for this row
    const float* rowCDF = envMap.conditionalCDF + y * envMap.width;
    int x = binarySearchCDF(rowCDF, envMap.width, u2);
    float conditionalPdf = (x == 0) ? rowCDF[0] : (rowCDF[x] - rowCDF[x - 1]);

    // Convert pixel coordinates to direction
    float u = (float(x) + 0.5f) / float(envMap.width);
    float v = (float(y) + 0.5f) / float(envMap.height);

    float theta = v * PI;
    float phi = u * 2.0f * PI;

    float sinTheta = sinf(theta);
    float cosTheta = cosf(theta);
    float sinPhi = sinf(phi);
    float cosPhi = cosf(phi);

    direction = glm::vec3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);

    // Compute PDF
    float piSquared = PI * PI;
    pdf = (marginalPdf * conditionalPdf * float(envMap.height) * float(envMap.width)) /
        (2.0f * piSquared * sinTheta);

    // Sample the texture - HDR is already linear, no sRGB conversion needed
    float4 texel = tex2D<float4>(envMap.texture, u, v);
    return glm::vec3(texel.x, texel.y, texel.z);
}

__device__ inline float environmentMapPdf(const EnvironmentMap& envMap,
    const glm::vec3& direction) {
    float theta = acosf(glm::clamp(direction.y, -1.0f, 1.0f));
    float phi = atan2f(direction.z, direction.x);
    if (phi < 0.0f) phi += 2.0f * PI;

    float u = phi / (2.0f * PI);
    float v = theta / PI;

    int x = glm::min(int(u * float(envMap.width)), envMap.width - 1);
    int y = glm::min(int(v * float(envMap.height)), envMap.height - 1);

    float marginalPdf = (y == 0) ? envMap.marginalCDF[0] :
        (envMap.marginalCDF[y] - envMap.marginalCDF[y - 1]);

    const float* rowCDF = envMap.conditionalCDF + y * envMap.width;
    float conditionalPdf = (x == 0) ? rowCDF[0] : (rowCDF[x] - rowCDF[x - 1]);

    float sinTheta = sinf(theta);
    if (sinTheta == 0.0f) return 0.0f;

    float piSquared = PI * PI;
    return (marginalPdf * conditionalPdf * float(envMap.height) * float(envMap.width)) /
        (2.0f * piSquared * sinTheta);
}