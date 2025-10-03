#include "directLighting.h"

#include "environmentSampling.h"
#include "glm/gtx/norm.hpp"
#include "interactions.h"

// Lambertian
__device__ float lambert_pdf(const glm::vec3& n, const glm::vec3& wi) {
    float c = fmaxf(0.f, glm::dot(n, wi));
    return c > 0.f ? c / CUDART_PI_F : 0.f;
}
__device__ glm::vec3 lambert_f(const glm::vec3& albedo) {
    return albedo / CUDART_PI_F;
}

// Atomic add to image
__device__ void atomicAddVec3(glm::vec3* img, int pix, const glm::vec3& v) {
    atomicAdd(&img[pix].x, v.x);
    atomicAdd(&img[pix].y, v.y);
    atomicAdd(&img[pix].z, v.z);
}

// Sample a sphere light
__device__ void sampleSphereLight(const Geom& g,
    thrust::default_random_engine& rng,
    glm::vec3& Pl, glm::vec3& Nl, float& area)
{
    thrust::uniform_real_distribution<float> uni(0.f, 1.f);
    float u = uni(rng), v = uni(rng);
    float z = 1.f - 2.f * u;
    float r = sqrtf(fmaxf(0.f, 1.f - z * z));
    float phi = 2.f * CUDART_PI_F * v;
    glm::vec3 dir = glm::vec3(r * cosf(phi), r * sinf(phi), z);

    glm::vec3 cx = glm::vec3(g.transform * glm::vec4(0, 0, 0, 1));
    float rx = glm::length(glm::vec3(g.transform * glm::vec4(0.5f, 0, 0, 0)));
    if (rx <= 0.f) rx = 0.5f;

    Pl = cx + rx * dir;
    Nl = glm::normalize(glm::mat3(g.invTranspose) * dir);
    area = 4.f * CUDART_PI_F * rx * rx;
}

// Sample a cube light
__device__ void sampleCubeLight(const Geom& g,
    thrust::default_random_engine& rng,
    glm::vec3& Pl, glm::vec3& Nl, float& area)
{
    const glm::vec3 faceN[6] = { { 1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1} };
    const glm::vec3 faceU[6] = { {0,1,0},{0,1,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0} };
    const glm::vec3 faceV[6] = { {0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,1,0},{0,1,0} };

    float areas[6]; float sumA = 0.f;
    for (int f = 0; f < 6; ++f) {
        glm::vec3 U = glm::vec3(g.transform * glm::vec4(0.5f * faceU[f], 0.0f));
        glm::vec3 V = glm::vec3(g.transform * glm::vec4(0.5f * faceV[f], 0.0f));
        areas[f] = 4.f * glm::length(glm::cross(U, V));
        sumA += areas[f];
    }

    thrust::uniform_real_distribution<float> uni(0.f, 1.f);
    float pick = uni(rng) * sumA; int f = 0;
    for (; f < 6; ++f) { if (pick <= areas[f]) break; pick -= areas[f]; }
    if (f == 6) f = 5;

    float u = uni(rng) - 0.5f, v = uni(rng) - 0.5f;
    glm::vec3 Pobj = 0.5f * faceN[f] + u * faceU[f] + v * faceV[f];
    glm::vec3 Nobj = faceN[f];

    Pl = glm::vec3(g.transform * glm::vec4(Pobj, 1.0f));
    Nl = glm::normalize(glm::mat3(g.invTranspose) * Nobj);
    area = sumA;
}


__device__ bool visible(const glm::vec3& P, const glm::vec3& Q,
    const glm::vec3& N,
    const Geom* geoms, int ngeoms)
{
    glm::vec3 d = Q - P;
    float maxT = glm::length(d);
    if (maxT <= 1e-6f) return false;

    glm::vec3 dir = d / maxT;

    // Offset proportional to distance - scales with scene
    float offsetEpsilon = maxT * 1e-4f;
    glm::vec3 O = P + N * offsetEpsilon;

    // Recalculate distance from offset origin
    float adjustedMaxT = glm::length(Q - O);

    glm::vec3 I_tmp, N_tmp;
    bool outside;

    for (int i = 0; i < ngeoms; ++i) {
        const Geom& g = geoms[i];
        Ray r;
        r.origin = O;
        r.direction = dir;

        float t = -1.0f;
        if (g.type == CUBE)
            t = boxIntersectionTest(g, r, I_tmp, N_tmp, outside);
        else if (g.type == SPHERE)
            t = sphereIntersectionTest(g, r, I_tmp, N_tmp, outside);

        // Accept hits with relative tolerance
        if (t > 0.0f && t < adjustedMaxT * 0.999f) {
            return false;
        }
    }
    return true;
}

__device__ void addDirectLightingNEE(
    const glm::vec3& P,
    const glm::vec3& N,
    const glm::vec3& wo,
    const Material* __restrict__ materials,
    const Geom* __restrict__ geoms, int ngeoms,
    const int* __restrict__ lightIdx, int numLights,
    const glm::vec3& albedoTimesThroughput,
    float metallic, float roughness,
    int pixelIndex,
    glm::vec3* __restrict__ image,
    thrust::default_random_engine& rng,
    const EnvironmentMap* __restrict__ envMap)
{
    // --- Diffuse BRDF ---
    const glm::vec3 f_diff = albedoTimesThroughput / CUDART_PI_F;

    // --- Specular F0 ---
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedoTimesThroughput, metallic);
    float alpha = roughness * roughness;

    // 1) Sample an emissive light (area lights)
    if (numLights > 0) {
        thrust::uniform_int_distribution<int> pick(0, numLights - 1);
        const int li = pick(rng);
        const Geom& Lg = geoms[lightIdx[li]];
        const Material& Lm = materials[Lg.materialid];
        if (Lm.emittance > 0.f) {
            glm::vec3 Pl, Nl; float area = 0.f;
            if (Lg.type == SPHERE) sampleSphereLight(Lg, rng, Pl, Nl, area);
            else                   sampleCubeLight(Lg, rng, Pl, Nl, area);

            glm::vec3 wi = glm::normalize(Pl - P);
            float d2 = glm::length2(Pl - P);
            float cosS = fmaxf(0.f, glm::dot(N, wi));
            float cosL = fmaxf(0.f, glm::dot(Nl, -wi));

            if (cosS > 0.f && cosL > 0.f && visible(P, Pl, N, geoms, ngeoms)) {
                // Microfacet specular eval
                glm::vec3 H = glm::normalize(wi + wo);
                float NoV = fmaxf(0.f, glm::dot(N, wo));
                float NoL = fmaxf(0.f, glm::dot(N, wi));
                float NoH = fmaxf(0.f, glm::dot(N, H));
                float VoH = fmaxf(0.f, glm::dot(wo, H));

                glm::vec3 F = Fresnel_Schlick(VoH, F0);
                float D = D_GGX(NoH, alpha);
                float G = G_SmithGGX(NoV, NoL, alpha);

                glm::vec3 f_spec = (D * F * G) / fmaxf(4.f * NoV * NoL, 1e-4f);
                glm::vec3 f = f_diff * (1.0f - metallic) + f_spec;

                glm::vec3 Le = Lm.color * Lm.emittance;
                float pmfL = 1.f / float(numLights);
                float p_l = pmfL * (d2 / (cosL * fmaxf(1e-8f, area)));
                float p_b = lambert_pdf(N, wi);

                if (p_l > 0.f && p_b > 0.f) {
                    float w_l = (p_l * p_l) / (p_l * p_l + p_b * p_b);
                    glm::vec3 contrib = f * Le * cosS * (w_l / p_l);
                    atomicAddVec3(image, pixelIndex, contrib);
                }
            }
        }
    }

    // 2) Environment map sampling
    if (envMap) {
        thrust::uniform_real_distribution<float> u01(0.f, 1.f);
        float u1 = u01(rng), u2 = u01(rng);

        glm::vec3 wi_env;
        float pdf_env = 0.f;
        glm::vec3 Le_env = sampleEnvironmentMapImportance(*envMap, u1, u2, wi_env, pdf_env);

        if (pdf_env > 1e-6f) {
            float cosS = fmaxf(0.f, glm::dot(N, wi_env));
            if (cosS > 0.f && visible(P, P + wi_env * 1e6f, N, geoms, ngeoms)) {
                glm::vec3 H = glm::normalize(wi_env + wo);
                float NoV = fmaxf(0.f, glm::dot(N, wo));
                float NoL = fmaxf(0.f, glm::dot(N, wi_env));
                float NoH = fmaxf(0.f, glm::dot(N, H));
                float VoH = fmaxf(0.f, glm::dot(wo, H));

                glm::vec3 F = Fresnel_Schlick(VoH, F0);
                float D = D_GGX(NoH, alpha);
                float G = G_SmithGGX(NoV, NoL, alpha);

                glm::vec3 f_spec = (D * F * G) / fmaxf(4.f * NoV * NoL, 1e-4f);
                glm::vec3 f = f_diff * (1.0f - metallic) + f_spec;

                float p_b = lambert_pdf(N, wi_env);
                float w_l = (pdf_env * pdf_env) / (pdf_env * pdf_env + p_b * p_b + 1e-16f);
                glm::vec3 contrib = f * Le_env * cosS * (w_l / pdf_env);
                atomicAddVec3(image, pixelIndex, contrib);
            }
        }
    }
}

__device__ float computeLightPdf(
    const glm::vec3& P,
    const glm::vec3& lightP,
    const glm::vec3& lightN,
    float lightArea,
    int numLights)
{
    const glm::vec3 wi = lightP - P;
    const float d2 = glm::length2(wi);
    const float cosL = fmaxf(0.f, glm::dot(lightN, -glm::normalize(wi)));
    if (cosL <= 0.f) return 0.f;

    const float pmfLight = 1.f / float(numLights);
    return pmfLight * (d2 / (cosL * fmaxf(1e-8f, lightArea)));
}

__device__ glm::vec3 evalEmissiveWithMIS(
    const PathSegment& path,
    const ShadeableIntersection& isect,
    const glm::vec3& Le,
    int depth,
    const Geom* geoms,
    const int* lightIdx,
    int numLights)
{
    // First bounce or no previous BSDF sample: no MIS
    /*if (depth == 1 || path.prevBsdfPdf <= 0.0f) {
        return path.color * Le;
    }*/

    if (depth == 1 || path.prevWasDelta || path.prevBsdfPdf <= 0.0f)
    {
        return path.color * Le;
    }
    const int hitGeomIdx = isect.geomId;
    
    // Find light in list
    int lightListIdx = -1;
    for (int j = 0; j < numLights; ++j) {
        if (lightIdx[j] == hitGeomIdx) {
            lightListIdx = j;
            break;
        }
    }
    
    if (lightListIdx < 0) return path.color * Le;
    
    // Compute light area
    const Geom& lightGeom = geoms[hitGeomIdx];
    float lightArea;
    if (lightGeom.type == SPHERE) {
        float r = glm::length(glm::vec3(lightGeom.transform * glm::vec4(0.5f, 0, 0, 0)));
        lightArea = 4.f * CUDART_PI_F * fmaxf(r, 0.5f) * fmaxf(r, 0.5f);
    } else {
        glm::vec3 U = glm::vec3(lightGeom.transform * glm::vec4(0.5f, 0, 0, 0));
        glm::vec3 V = glm::vec3(lightGeom.transform * glm::vec4(0, 0.5f, 0, 0));
        glm::vec3 W = glm::vec3(lightGeom.transform * glm::vec4(0, 0, 0.5f, 0));
        lightArea = 2.f * (4.f * glm::length(glm::cross(V, W)) + 
                          4.f * glm::length(glm::cross(U, W)) + 
                          4.f * glm::length(glm::cross(U, V)));
    }
    
    // Compute light PDF
    float d2 = isect.t * isect.t;
    float cosL = fmaxf(0.f, glm::dot(isect.surfaceNormal, -glm::normalize(path.ray.direction)));
    float p_l = (cosL > 0.f && lightArea > 0.f) 
        ? (1.f / float(numLights)) * (d2 / (cosL * lightArea)) 
        : 0.f;
    
    float p_b = path.prevBsdfPdf;
    float w_b = (p_l > 0.f && p_b > 0.f) 
        ? (p_b * p_b) / (p_b * p_b + p_l * p_l) 
        : 1.0f;
    return path.color * Le * w_b;
}