#include "directLighting.h"
#include "glm/gtx/norm.hpp"

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

// Visibility check
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

// NEE for diffuse with MIS (power heuristic)
__device__ void addDirectLighting_NEEDiffuse(
    const glm::vec3& P,
    const glm::vec3& N,
    const glm::vec3& wo,
    const Material* __restrict__ materials,
    const Geom* __restrict__ geoms, int ngeoms,
    const int* __restrict__ lightIdx, int numLights,
    const glm::vec3& albedo,
    int pixelIndex,
    glm::vec3* __restrict__ image,
    thrust::default_random_engine& rng)
{
    if (numLights <= 0) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("NEE skipped: numLights = %d\n", numLights);
        }
        return;
    }

    thrust::uniform_int_distribution<int> pick(0, numLights - 1);
    const int li = pick(rng);
    const Geom& Lg = geoms[lightIdx[li]];
    const Material& Lm = materials[Lg.materialid];
    if (Lm.emittance <= 0.f) return;

    glm::vec3 Pl, Nl; float area = 0.f;
    if (Lg.type == SPHERE) sampleSphereLight(Lg, rng, Pl, Nl, area);
    else                   sampleCubeLight(Lg, rng, Pl, Nl, area);

    const glm::vec3 wi = glm::normalize(Pl - P);
    const float d2 = glm::length2(Pl - P);
    const float cosS = fmaxf(0.f, glm::dot(N, wi));
    const float cosL = fmaxf(0.f, glm::dot(Nl, -wi));
    if (cosS <= 0.f || cosL <= 0.f) return;

    const float pmfL = 1.f / float(numLights);
    const float p_l = pmfL * (d2 / (cosL * fmaxf(1e-8f, area)));

    const glm::vec3 f = lambert_f(albedo);
    const float     p_b = lambert_pdf(N, wi);
    if (p_l <= 0.f || p_b <= 0.f) return;

    if (!visible(P, Pl, N, geoms, ngeoms)) return;

    const glm::vec3 Le = Lm.color * Lm.emittance;
    const float w_l = (p_l * p_l) / (p_l * p_l + p_b * p_b);

    const glm::vec3 contrib = f * Le * cosS * (w_l / p_l);
    if (pixelIndex == 320000 && contrib.x > 0.001f) {
        printf("NEE adding: (%f, %f, %f) to pixel %d\n", contrib.x, contrib.y, contrib.z, pixelIndex);
    }

    atomicAddVec3(image, pixelIndex, contrib);
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

    if (depth == 1 || path.prevWasDelta) {
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