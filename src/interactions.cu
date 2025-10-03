#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>
#include <thrust/swap.h>
#include <math_constants.h>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/component_wise.hpp>

// Cosine-weighted hemisphere sampling
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find orthogonal vector to normal using component magnitude test
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

    // Build tangent frame
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

// Lambertian diffuse BRDF
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

// GGX microfacet BRDF with importance sampling
__device__ void ggxSpecularBRDF(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    glm::vec3 intersect)
{
    float eps = 1e-3f;
    glm::vec3 n = glm::normalize(normal);
    glm::vec3 wo = -glm::normalize(pathSegment.ray.direction); 

    float metallic = fminf(fmaxf(m.metallic, 0.f), 1.f);
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.color, metallic);

    float rough = fminf(fmaxf(m.roughness, 0.f), 1.f);
    float alpha = fmaxf(rough * rough, 1e-3f);

    // Perfect mirror for alpha near zero
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

    // Sample GGX halfvector and reflect
    glm::vec3 h = sampleGGX_H(n, alpha, rng);
    glm::vec3 wi = glm::reflect(-wo, h);

    float NoV = fmaxf(1e-6f, fabsf(glm::dot(n, wo)));
    float NoL = fmaxf(0.0f, glm::dot(n, wi));
    
    // Fallback to diffuse if reflected below surface
    if (NoL <= 0.0f) {
        pathSegment.color *= m.color;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(n, rng);
        pathSegment.ray.origin = intersect + n * eps;
        pathSegment.prevBsdfPdf = CUDART_PI_F;
        return;
    }
    
    float NoH = fmaxf(1e-6f, fabsf(glm::dot(n, h)));
    float HoV = fmaxf(1e-6f, fabsf(glm::dot(h, wo)));

    // Evaluate GGX terms
    float D = D_GGX(NoH, alpha);
    float G = G_SmithGGX(NoV, NoL, alpha);
    glm::vec3 F = Fresnel_Schlick(HoV, F0);

    // PDF is D * NoH / (4 * HoV)
    float pdf = (D * NoH) / (4.0f * HoV + 1e-6f);
    pdf = fmaxf(pdf, 1e-6f);

    // Throughput update with MIS weight implicit
    pathSegment.color *= F * G * HoV / (NoV * NoH);
    pathSegment.prevBsdfPdf = pdf;

    pathSegment.ray.origin = intersect + n * eps;
    pathSegment.ray.direction = glm::normalize(wi);
}

// Dielectric BTDF with Fresnel reflection/refraction split
__host__ __device__ void dielectricBRDF(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    glm::vec3 intersect)
{
    glm::vec3 incidentDir = glm::normalize(pathSegment.ray.direction);

    // Determine if entering or exiting medium
    bool isInside = glm::dot(incidentDir, normal) > 0.0f;
    float iorFrom = isInside ? m.indexOfRefraction : 1.0f;
    float iorTo = isInside ? 1.0f : m.indexOfRefraction;
    float eta = iorFrom / iorTo;

    glm::vec3 orientedNormal = isInside ? -normal : normal;
    float cosIncident = glm::clamp(-glm::dot(incidentDir, orientedNormal), 0.f, 1.f);

    glm::vec3 reflectDir = glm::reflect(incidentDir, orientedNormal);
    glm::vec3 refractDir = glm::refract(incidentDir, orientedNormal, eta);

    // Schlick approximation for dielectric Fresnel
    float reflectance = Fresnel_Schlick(cosIncident, iorFrom, iorTo);

    thrust::uniform_real_distribution<float> u01(0, 1);
    float xi = u01(rng);

    // Stochastically choose reflection or refraction
    if (xi < reflectance || glm::length2(refractDir) < 1e-10f) {
        // Total internal reflection or reflected ray
        pathSegment.ray.direction = glm::normalize(reflectDir);
        pathSegment.ray.origin = intersect + orientedNormal * 1e-3f;
    }
    else {
        // Transmitted ray
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

// Routes to appropriate BRDF based on material properties
__device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 P,
    glm::vec3 N,
    const Material& m,
    thrust::default_random_engine& rng)
{
    // Refractive materials (glass, water, etc)
    if (m.hasRefractive > 0.5f || m.transmission > 0.0f) {
        dielectricBRDF(pathSegment, N, m, rng, P);
        pathSegment.remainingBounces--;
        return;
    }

    // Opaque PBR: metallic-roughness workflow
    float metallic  = glm::clamp(m.metallic,  0.0f, 1.0f);
    float roughness = glm::clamp(m.roughness, 0.04f, 1.0f);

    // Dielectric base (0.04) vs metallic base (albedo)
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), m.color, metallic);

    // Energy balance: metals have no diffuse lobe
    float Favg = 0.3333f * (F0.x + F0.y + F0.z);
    float diffuseWeight = (1.0f - metallic) * (1.0f - Favg);
    diffuseWeight = glm::clamp(diffuseWeight, 0.0f, 1.0f);
    float specWeight = 1.0f - diffuseWeight;

    // Stochastically sample diffuse vs specular lobe
    thrust::uniform_real_distribution<float> u01(0,1);
    float xi = u01(rng);

    if (xi < specWeight) {
        ggxSpecularBRDF(pathSegment, N, m, rng, P);
    } else {
        diffuseBRDF(pathSegment, N, m, rng, P);
    }

    pathSegment.remainingBounces--;
}