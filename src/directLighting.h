#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/random.h>
#include "glm/glm.hpp"
#include "sceneStructs.h"
#include "intersections.h"

// Lambertian helpers
__device__ float     lambert_pdf(const glm::vec3& n, const glm::vec3& wi);
__device__ glm::vec3 lambert_f(const glm::vec3& albedo);

// Atomic accumulate to image
__device__ void atomicAddVec3(glm::vec3* img, int pix, const glm::vec3& v);

// Light sampling (sphere / cube)
__device__ void sampleSphereLight(const Geom& g,
    thrust::default_random_engine& rng,
    glm::vec3& Pl, glm::vec3& Nl, float& area);

__device__ void sampleCubeLight(const Geom& g,
    thrust::default_random_engine& rng,
    glm::vec3& Pl, glm::vec3& Nl, float& area);

// Shadow ray visibility test
__device__ bool visible(const glm::vec3& P, const glm::vec3& Q,
    const glm::vec3& N, const Geom* geoms, int ngeoms);

// One-sample NEE for diffuse (MIS power heuristic)
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
    thrust::default_random_engine& rng);

__device__ float computeLightPdf(
    const glm::vec3& P,
    const glm::vec3& lightP,
    const glm::vec3& lightN,
    float lightArea,
    int numLights);

__device__ glm::vec3 evalEmissiveWithMIS(
    const PathSegment& path,
    const ShadeableIntersection& isect,
    const glm::vec3& Le,
    int depth,
    const Geom* geoms,
    const int* lightIdx,
    int numLights);