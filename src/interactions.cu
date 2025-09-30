#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>
#include <thrust/swap.h>
#include <math_constants.h>
#include <glm/gtx/norm.hpp>


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


// ------------------------------------------------------------
// BRDFs
// ------------------------------------------------------------

// Diffuse (Lambertian)
__host__ __device__ void diffuseBRDF(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    glm::vec3 intersect)
{
    glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng);

    float cosTheta = fmaxf(0.f, glm::dot(wi, normal));
    float pdf = (cosTheta > 0.f) ? (cosTheta / CUDART_PI_F) : 1e-6f;

    pathSegment.color *= m.color;
    pathSegment.prevBsdfPdf = pdf;

    pathSegment.ray.origin = intersect + normal * 1e-3f;
    pathSegment.ray.direction = wi;

    
}

// GGX Specular / Metallic
__device__ void ggxSpecularBRDF(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    glm::vec3 intersect)
{
    float eps = 1e-3f;
    glm::vec3 n = glm::normalize(normal);
    glm::vec3 wo = -glm::normalize(pathSegment.ray.direction); // outgoing/view

    float metallic = fminf(fmaxf(m.metallic, 0.f), 1.f);
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.color, metallic);

    float rough = fminf(fmaxf(m.roughness, 0.f), 1.f);
    float alpha = fmaxf(rough * rough, 1e-3f);

    // Ideal mirror branch
    if (alpha < 1e-4f) {
        glm::vec3 wi = glm::reflect(-wo, n);
        float NoV = fmaxf(1e-6f, fabsf(glm::dot(n, wo)));
        glm::vec3 F = Fresnel_Schlick(NoV, F0);

        pathSegment.ray.origin = intersect + n * eps;
        pathSegment.ray.direction = glm::normalize(wi);

        pathSegment.prevBsdfPdf = 0.0f;
        pathSegment.prevWasDelta = 1;
        pathSegment.color *= F;
        return;
    }

    // Rough GGX
    glm::vec3 h = sampleGGX_H(n, alpha, rng);
    glm::vec3 wi = glm::reflect(-wo, h);

    float NoV = fmaxf(1e-6f, fabsf(glm::dot(n, wo)));
    float NoL = fmaxf(0.0f, glm::dot(n, wi));
    if (NoL <= 0.0f) {
        pathSegment.color *= m.color;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(n, rng);
        pathSegment.ray.origin = intersect + n * eps;
        pathSegment.prevBsdfPdf = CUDART_PI_F;
        return;
    }
    float NoH = fmaxf(1e-6f, fabsf(glm::dot(n, h)));
    float HoV = fmaxf(1e-6f, fabsf(glm::dot(h, wo)));

    float D = D_GGX(NoH, alpha);
    float G = G_SmithGGX(NoV, NoL, alpha);
    glm::vec3 F = Fresnel_Schlick(HoV, F0);

    glm::vec3 f = (D * G) * F / (4.0f * NoV * NoL + 1e-6f);

    float pdf = (D * NoH) / (4.0f * HoV + 1e-6f);
    pdf = fmaxf(pdf, 1e-6f);

    pathSegment.color *= F * G * HoV / (NoV * NoH);
    pathSegment.prevBsdfPdf = pdf;

    pathSegment.ray.origin = intersect + n * eps;
    pathSegment.ray.direction = glm::normalize(wi);
}

// Dielectric (Glass)
__host__ __device__ void dielectricBRDF(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    glm::vec3 intersect)
{
    glm::vec3 incidentDir = glm::normalize(pathSegment.ray.direction);

    // Check if ray is entering or exiting
    bool isInside = glm::dot(incidentDir, normal) > 0.0f;
    float iorFrom = isInside ? m.indexOfRefraction : 1.0f;   // medium we are coming from
    float iorTo = isInside ? 1.0f : m.indexOfRefraction;   // medium we are going into
    float eta = iorFrom / iorTo;                                // relative index of refraction

    glm::vec3 orientedNormal = isInside ? -normal : normal;

    float cosIncident = glm::clamp(-glm::dot(incidentDir, orientedNormal), 0.f, 1.f);

    // Reflection and refraction directions
    glm::vec3 reflectDir = glm::reflect(incidentDir, orientedNormal);
    glm::vec3 refractDir = glm::refract(incidentDir, orientedNormal, eta);

    // Fresnel reflectance (probability of reflection)
    float reflectance = Fresnel_Schlick(cosIncident, iorFrom, iorTo);

    thrust::uniform_real_distribution<float> u01(0, 1);
    float xi = u01(rng);

    if (xi < reflectance  || glm::length2(refractDir) < 1e-10f) {
        pathSegment.ray.direction = glm::normalize(reflectDir);
        pathSegment.ray.origin = intersect + orientedNormal * 1e-3f;
    }
    else {
        pathSegment.ray.direction = glm::normalize(refractDir);
        pathSegment.ray.origin = intersect - orientedNormal * 1e-3f;
        pathSegment.color *= m.color;
    }

    pathSegment.prevBsdfPdf = 0.0f;
    pathSegment.prevWasDelta = 0;
}

// ------------------------------------------------------------
// Scatter Dispatcher
// ------------------------------------------------------------
__device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    if (m.hasReflective > 0.0f || m.metallic > 0.0f) {
        // GGX specular/metallic
		//printf("GGX Specular/Metallic\n");
        ggxSpecularBRDF(pathSegment, normal, m, rng, intersect);
    }
    else if (m.hasRefractive > 0.0f) {
        // Dielectric glass
		//printf("Dielectric Glass\n");
        dielectricBRDF(pathSegment, normal, m, rng, intersect);
    }
    else {
        // Diffuse
		//printf("Diffuse\n");
        diffuseBRDF(pathSegment, normal, m, rng, intersect);
    }

    pathSegment.remainingBounces--;
}
