#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>
#include <math_constants.h>
#include <glm/gtx/norm.hpp>


// ----- Microfacet GGX helpers (reflection) -----
__device__ inline float saturate(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

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
    float lambdaV = Lambda_GGX(fabsf(NoV), alpha);
    float lambdaL = Lambda_GGX(fabsf(NoL), alpha);
    return 1.0f / (1.0f + lambdaV + lambdaL);
}

__device__ inline glm::vec3 Fresnel_Schlick(float cosTheta, const glm::vec3& F0) {
    float m = saturate(1.0f - cosTheta);
    float m2 = m * m;
    float m5 = m2 * m2 * m;
    return F0 + (glm::vec3(1.0f) - F0) * m5;
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


__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__device__ float fresnelDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
        cosThetaI = fabsf(cosThetaI);
    }

    // Compute cosThetaT using Snell's law
    float sinThetaI = sqrtf(fmaxf(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1.f) return 1.f;

    float cosThetaT = sqrtf(fmaxf(0.f, 1.f - sinThetaT * sinThetaT));

    // Fresnel equations
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));

    return (Rparl * Rparl + Rperp * Rperp) / 2.f;
}

__device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    const float eps = 1e-3f;

    // ---------- DIELECTRIC (GLASS-LIKE) ----------
    if (m.hasRefractive > 0.5f) {
        const float ior = (m.indexOfRefraction > 0.f) ? m.indexOfRefraction : 1.5f;

        glm::vec3 I = glm::normalize(pathSegment.ray.direction);
        glm::vec3 n = glm::normalize(normal);

        // Determine if entering or exiting based on ray-normal dot product
        bool entering = (glm::dot(I, n) < 0.0f);
        if (!entering) n = -n;  // Flip normal to face incoming ray

        float etaI = entering ? 1.0f : ior;
        float etaT = entering ? ior : 1.0f;
        float eta = etaI / etaT;

        float cosI = fmaxf(0.f, -glm::dot(I, n));
        float Fr = fresnelDielectric(cosI, etaI, etaT);

        thrust::uniform_real_distribution<float> u01(0.f, 1.f);
        float xi = u01(rng);

        if (xi < Fr) {
            // Specular reflection
            glm::vec3 R = glm::reflect(I, n);
            pathSegment.ray.origin = intersect + n * eps;
            pathSegment.ray.direction = glm::normalize(R);
            pathSegment.prevBsdfPdf = 0.0f; // Delta event
            pathSegment.prevWasDelta = 1;   // Mark as delta for MIS
            // No color change for reflection in dielectrics
        }
        else {
            // Refraction/Transmission
            glm::vec3 T = glm::refract(I, n, eta);

            // Check for total internal reflection
            if (glm::length2(T) < 1e-10f) {
                // Fall back to reflection
                glm::vec3 R = glm::reflect(I, n);
                pathSegment.ray.origin = intersect + n * eps;
                pathSegment.ray.direction = glm::normalize(R);
            }
            else {
                // Successful refraction
                pathSegment.ray.origin = intersect - n * eps; // Move to other side
                pathSegment.ray.direction = glm::normalize(T);

                // Apply material tint only to transmitted light
                pathSegment.color *= m.color;
            }

            pathSegment.prevBsdfPdf = 0.0f; // Delta event
            pathSegment.prevWasDelta = 1;   // Mark as delta for MIS
        }

        pathSegment.remainingBounces--;
        return;
    }

    // ---------- SPECULAR / METALLIC (MICROFACET GGX REFLECTION) ----------
    if (m.hasReflective > 0.5f || m.metallic > 0.0f) {
        glm::vec3 n = glm::normalize(normal);
        glm::vec3 wo = -glm::normalize(pathSegment.ray.direction); // view/outgoing

        // PBR-style base reflectance: non-metals ~0.04, metals use base color
        float metallic = fminf(fmaxf(m.metallic, 0.f), 1.f);
        glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.color, metallic);

        float rough = fminf(fmaxf(m.roughness, 0.f), 1.f);
        float alpha = fmaxf(rough * rough, 1e-3f);

        // If extremely smooth, treat as ideal mirror delta
        if (alpha < 1e-4f) {
            glm::vec3 wi = glm::reflect(-wo, n);
            float NoV = fmaxf(1e-6f, fabsf(glm::dot(n, wo)));
            glm::vec3 F = Fresnel_Schlick(NoV, F0);

            pathSegment.ray.origin = intersect + n * eps;
            pathSegment.ray.direction = glm::normalize(wi);

            // Delta: mark pdf=0; scale throughput by Fresnel reflectance
            pathSegment.prevBsdfPdf = 0.0f;
            pathSegment.color *= F;

            pathSegment.remainingBounces--;
            return;
        }

        // Rough GGX: sample half-vector h, then reflect
        glm::vec3 h = sampleGGX_H(n, alpha, rng);
        glm::vec3 wi = glm::reflect(-wo, h);

        float NoV = fmaxf(1e-6f, fabsf(glm::dot(n, wo)));
        float NoL = fmaxf(0.0f, glm::dot(n, wi));
        if (NoL <= 0.0f) {
            // Fell below the surface; terminate this path cheaply
            pathSegment.remainingBounces = 0;
            return;
        }
        float NoH = fmaxf(1e-6f, fabsf(glm::dot(n, h)));
        float HoV = fmaxf(1e-6f, fabsf(glm::dot(h, wo)));

        float    D = D_GGX(NoH, alpha);
        float    G = G_SmithGGX(NoV, NoL, alpha);
        glm::vec3 F = Fresnel_Schlick(HoV, F0);

        // Cook-Torrance BRDF (reflection only)
        glm::vec3 f = (D * G) * F / (4.0f * NoV * NoL + 1e-6f);

        // PDF for sampling wi via GGX half-vector sampling:
        // pdf(wi) = D(h) * NoH / (4 * HoV)
        float pdf = (D * NoH) / (4.0f * HoV + 1e-6f);
        pdf = fmaxf(pdf, 1e-6f);

        // Throughput update: beta *= f * cos / pdf
        pathSegment.color *= f * (NoL / pdf);
        pathSegment.prevBsdfPdf = pdf;

        pathSegment.ray.origin = intersect + n * eps;
        pathSegment.ray.direction = glm::normalize(wi);
        pathSegment.remainingBounces--;
        return;
    }

    // ---------- DIFFUSE ----------
    {
        glm::vec3 n = glm::normalize(normal);
        glm::vec3 wi = calculateRandomDirectionInHemisphere(n, rng);

        float cosTheta = fmaxf(0.f, glm::dot(wi, n));
        float pdf = (cosTheta > 0.f) ? (cosTheta / CUDART_PI_F) : 1e-6f;

        // Lambert: f = albedo / pi; beta *= f * cos / pdf = albedo
        pathSegment.color *= m.color;
        pathSegment.prevBsdfPdf = pdf;

        pathSegment.ray.origin = intersect + n * eps;
        pathSegment.ray.direction = wi;
        pathSegment.remainingBounces--;
    }
}
