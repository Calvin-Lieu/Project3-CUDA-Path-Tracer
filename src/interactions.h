#pragma once

#include <math_constants.h>

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */

__device__ float fresnelDielectric(float cosThetaI, float etaI, float etaT);

__device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng);

__host__ __device__ inline bool isDiffuse(const Material& m) {
    return (m.hasReflective < 0.5f) && (m.hasRefractive < 0.5f);
}
__host__ __device__ inline bool isDielectric(const Material& m) {
    return (m.hasRefractive > 0.5f);
}
__host__ __device__ inline bool isMirrorLike(const Material& m) {
    return (m.hasReflective > 0.5f) && (m.roughness <= 0.0f);
}

// ----- Microfacet GGX helpers (reflection) -----

__device__ inline void makeONB(const glm::vec3& n, glm::vec3& t, glm::vec3& b) {
    if (fabsf(n.z) < 0.999f) {
        t = glm::normalize(glm::vec3(-n.y, n.x, 0.0f));
    }
    else {
        t = glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f));
    }
    b = glm::cross(n, t);
}

__device__ inline float D_GGX(float NoH, float alpha) {
    // Trowbridge-Reitz GGX NDF
    float a2 = alpha * alpha;
    float d = NoH * NoH * (a2 - 1.0f) + 1.0f;
    return a2 / (CUDART_PI_F * d * d);
}

__device__ inline float Lambda_GGX(float cosTheta, float alpha) {
    // Smith's lambda term for GGX
    float a = alpha;
    float cos2 = cosTheta * cosTheta;
    float sin2 = fmaxf(0.0f, 1.0f - cos2);
    float tan2 = (cos2 > 0.0f) ? (sin2 / cos2) : 1e20f;
    return (-1.0f + sqrtf(1.0f + a * a * tan2)) * 0.5f;
}

__device__ inline float G_SmithGGX(float NoV, float NoL, float alpha) {
    float a2 = alpha * alpha;
    float GGXV = NoV * sqrtf(a2 + (1.0f - a2) * NoL * NoL);
    float GGXL = NoL * sqrtf(a2 + (1.0f - a2) * NoV * NoV);
    return 2.0f * NoL * NoV / (GGXV + GGXL + 1e-6f);
}

// --- Vec3 version (For metals/rough GGX) ---
__host__ __device__ inline glm::vec3 Fresnel_Schlick(
    float cosTheta,
    const glm::vec3& F0)
{
    float m = fminf(fmaxf(1.0f - cosTheta, 0.0f), 1.0f);
    float m2 = m * m;
    float m5 = m2 * m2 * m;
    return F0 + (glm::vec3(1.0f) - F0) * m5;
}

// --- Float version (Ffor glass/dielectrics) ---
__host__ __device__ inline float Fresnel_Schlick(
    float cosTheta,
    float etaI,
    float etaT)
{
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
}

// Sample GGX half-vector (basic Trowbridge-Reitz)
__device__ inline glm::vec3 sampleGGX_H(const glm::vec3& n, float alpha,
    thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    float u1 = u01(rng);
    float u2 = u01(rng);

    float phi = 2.0f * CUDART_PI_F * u1;
    float a2 = alpha * alpha;
    float cosTheta = sqrtf((1.0f - u2) / (1.0f + (a2 - 1.0f) * u2));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    glm::vec3 t, b;
    makeONB(n, t, b);
    // local -> world
    return glm::normalize(
        sinTheta * cosf(phi) * t +
        sinTheta * sinf(phi) * b +
        cosTheta * n);
}


