#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <climits>
#include <iostream>
#include <utility>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <stb_image.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "pathHelpers.h"
#include "sortKeys.h"
#include "directLighting.h"
#include "bvh.h"
#include "textureSampling.h"
#include "environmentSampling.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif
    exit(EXIT_FAILURE);
#endif
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// Sorting/reorder buffers
static uint32_t* dev_matKeys = nullptr;
static int* dev_indices = nullptr;
static PathSegment* dev_paths_alt = nullptr;
static ShadeableIntersection* dev_intersections_alt = nullptr;

static int* dev_lightGeomIdx = nullptr;
static int hst_numLights = 0;

static BVH dev_bvh;
static bool bvhBuilt = false;

static TriangleMeshData* dev_meshes = nullptr;
static int numMeshes = 0;

static Texture* dev_textures = nullptr;
static int numTextures = 0;

static EnvironmentMap* dev_envMap = nullptr;  
static cudaArray_t envMapArray = nullptr;     
static bool hasEnvironmentMap = false;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void buildEnvironmentCDFsFromFloat(const std::vector<float>& imageData,
    int width, int height,
    EnvironmentMap& envMap)
{
    std::vector<float> luminance(width * height);

    // Compute luminance for each pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 4;
            float r = imageData[idx];
            float g = imageData[idx + 1];
            float b = imageData[idx + 2];

            // Weight by solid angle
            float theta = PI * (y + 0.5f) / height;
            float sinTheta = sinf(theta);

            luminance[y * width + x] = (0.2126f * r + 0.7152f * g + 0.0722f * b) * sinTheta;
        }
    }

    // Build conditional CDFs (one per row)
    std::vector<float> conditionalCDF(width * height);
    std::vector<float> rowIntegrals(height);

    for (int y = 0; y < height; y++) {
        float sum = 0.0f;
        for (int x = 0; x < width; x++) {
            sum += luminance[y * width + x];
            conditionalCDF[y * width + x] = sum;
        }

        rowIntegrals[y] = sum;

        // Normalize this row's CDF
        if (sum > 0.0f) {
            for (int x = 0; x < width; x++) {
                conditionalCDF[y * width + x] /= sum;
            }
        }
    }

    // Build marginal CDF (over rows)
    std::vector<float> marginalCDF(height);
    float totalSum = 0.0f;
    for (int y = 0; y < height; y++) {
        totalSum += rowIntegrals[y];
        marginalCDF[y] = totalSum;
    }

    // Normalize marginal CDF
    if (totalSum > 0.0f) {
        for (int y = 0; y < height; y++) {
            marginalCDF[y] /= totalSum;
        }
    }

    envMap.totalLuminance = totalSum;

    // Copy to device
    cudaMalloc(&envMap.marginalCDF, height * sizeof(float));
    cudaMemcpy(envMap.marginalCDF, marginalCDF.data(),
        height * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&envMap.conditionalCDF, width * height * sizeof(float));
    cudaMemcpy(envMap.conditionalCDF, conditionalCDF.data(),
        width * height * sizeof(float), cudaMemcpyHostToDevice);
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // For sort
    cudaMalloc(&dev_matKeys, pixelcount * sizeof(uint32_t));
    cudaMalloc(&dev_indices, pixelcount * sizeof(int));
    cudaMalloc(&dev_paths_alt, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersections_alt, pixelcount * sizeof(ShadeableIntersection));

     if (!scene->meshes.empty()) {
        numMeshes = scene->meshes.size();
        cudaMalloc(&dev_meshes, numMeshes * sizeof(TriangleMeshData));

        std::vector<TriangleMeshData> hostMeshes(numMeshes);

        for (int i = 0; i < numMeshes; ++i) {
            const auto& hostMesh = scene->meshes[i];

            // Vertices
            cudaMalloc(&hostMeshes[i].vertices, hostMesh.vertices.size() * sizeof(float));
            cudaMemcpy(hostMeshes[i].vertices, hostMesh.vertices.data(),
                hostMesh.vertices.size() * sizeof(float), cudaMemcpyHostToDevice);

            // Normals
            cudaMalloc(&hostMeshes[i].normals, hostMesh.normals.size() * sizeof(float));
            cudaMemcpy(hostMeshes[i].normals, hostMesh.normals.data(),
                hostMesh.normals.size() * sizeof(float), cudaMemcpyHostToDevice);

            // Texture coordinates
            cudaMalloc(&hostMeshes[i].texcoords, hostMesh.texcoords.size() * sizeof(float));
            cudaMemcpy(hostMeshes[i].texcoords, hostMesh.texcoords.data(),
                hostMesh.texcoords.size() * sizeof(float), cudaMemcpyHostToDevice);

            if (!hostMesh.tangents.empty()) {
                cudaMalloc(&hostMeshes[i].tangents, hostMesh.tangents.size() * sizeof(float));
                cudaMemcpy(hostMeshes[i].tangents, hostMesh.tangents.data(),
                    hostMesh.tangents.size() * sizeof(float), cudaMemcpyHostToDevice);
            }
            else {
                hostMeshes[i].tangents = nullptr;
            }

            // Indices
            cudaMalloc(&hostMeshes[i].indices, hostMesh.indices.size() * sizeof(unsigned int));
            cudaMemcpy(hostMeshes[i].indices, hostMesh.indices.data(),
                hostMesh.indices.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
            
            hostMeshes[i].triangleCount = hostMesh.indices.size() / 3;
        }

        cudaMemcpy(dev_meshes, hostMeshes.data(), numMeshes * sizeof(TriangleMeshData), cudaMemcpyHostToDevice);
    }
    else {
        // No meshes to load
        dev_meshes = nullptr;
        numMeshes = 0;
    }

    // Build emissive geoms list and copy to device
    
    std::vector<int> lightIdx;
    lightIdx.reserve(scene->geoms.size());
    for (int i = 0; i < (int)scene->geoms.size(); ++i) {
        int mid = scene->geoms[i].materialid;
        if (scene->materials[mid].emittance > 0.0f)
            lightIdx.push_back(i);
    }
    hst_numLights = (int)lightIdx.size();
    if (hst_numLights > 0) {
        cudaMalloc(&dev_lightGeomIdx, hst_numLights * sizeof(int));
        cudaMemcpy(dev_lightGeomIdx, lightIdx.data(),
            hst_numLights * sizeof(int), cudaMemcpyHostToDevice);
    }
    //printf("Found %d lights\n", hst_numLights);
    //for (int i = 0; i < hst_numLights; i++) {
    //    printf("  Light %d: geom index %d\n", i, lightIdx[i]);
    //}

    if (!scene->textures.empty()) {
        numTextures = scene->textures.size();
        std::cout << "Uploading " << numTextures << " textures to GPU...\n";

        std::vector<Texture> hostTextures(numTextures);

        for (int i = 0; i < numTextures; i++) {
            auto& texPair = scene->textures[i];
            Texture& tex = hostTextures[i];

            tex.width = texPair.second.width;
            tex.height = texPair.second.height;
            tex.channels = texPair.second.channels;

            size_t imageSize = tex.width * tex.height * tex.channels;

            // Allocate GPU memory for texture data
            cudaMalloc(&tex.data, imageSize);
            cudaMemcpy(tex.data, texPair.first.data(), imageSize, cudaMemcpyHostToDevice);

            std::cout << "  Texture " << i << ": " << tex.width << "x" << tex.height
                << " (" << tex.channels << " channels)\n";
        }

        // Copy texture array to device
        cudaMalloc(&dev_textures, numTextures * sizeof(Texture));
        cudaMemcpy(dev_textures, hostTextures.data(),
            numTextures * sizeof(Texture), cudaMemcpyHostToDevice);
    }


    if (!hst_scene->environmentMapPath.empty()) {
        std::cout << "Loading environment map: " << hst_scene->environmentMapPath << "\n";

        int width, height, channels;
        float* data = stbi_loadf(hst_scene->environmentMapPath.c_str(),
            &width, &height, &channels, 4);

        if (data) {
            // Create CUDA texture
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32,
                cudaChannelFormatKindFloat);
            cudaMallocArray(&envMapArray, &channelDesc, width, height);
            cudaMemcpy2DToArray(envMapArray, 0, 0, data, width * 4 * sizeof(float),
                width * 4 * sizeof(float), height, cudaMemcpyHostToDevice);

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = envMapArray;

            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 1;

            // Host-side struct to build
            EnvironmentMap hostEnvMap;
            cudaCreateTextureObject(&hostEnvMap.texture, &resDesc, &texDesc, nullptr);
            hostEnvMap.width = width;
            hostEnvMap.height = height;

            // Build CDFs
            std::vector<float> floatVec(data, data + width * height * 4);
            buildEnvironmentCDFsFromFloat(floatVec, width, height, hostEnvMap);

            // Allocate struct on device and copy
            cudaMalloc(&dev_envMap, sizeof(EnvironmentMap));
            cudaMemcpy(dev_envMap, &hostEnvMap, sizeof(EnvironmentMap), cudaMemcpyHostToDevice);

            hasEnvironmentMap = true;
            stbi_image_free(data);

            std::cout << "Environment map loaded: " << width << "x" << height << "\n";
            std::cout << "Total luminance: " << hostEnvMap.totalLuminance << "\n";
        }
    }
    // Build BVH
    dev_bvh = BVHBuilder::build(scene->geoms, scene->meshes);
    bvhBuilt = true;

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    cudaFree(dev_matKeys);
    cudaFree(dev_indices);
    cudaFree(dev_paths_alt);
    cudaFree(dev_intersections_alt);
    cudaFree(dev_lightGeomIdx);

    if (dev_meshes) {
        TriangleMeshData* hostMeshes = new TriangleMeshData[numMeshes];
        cudaMemcpy(hostMeshes, dev_meshes, numMeshes * sizeof(TriangleMeshData), cudaMemcpyDeviceToHost);

        for (int i = 0; i < numMeshes; ++i) {
            if (hostMeshes[i].vertices) cudaFree(hostMeshes[i].vertices);
            if (hostMeshes[i].normals) cudaFree(hostMeshes[i].normals);
            if (hostMeshes[i].tangents) cudaFree(hostMeshes[i].tangents);
            if (hostMeshes[i].indices) cudaFree(hostMeshes[i].indices);
        }

        delete[] hostMeshes;
        cudaFree(dev_meshes);
        dev_meshes = nullptr;
        numMeshes = 0;
    }

    if (dev_textures) {
        Texture* hostTextures = new Texture[numTextures];
        cudaMemcpy(hostTextures, dev_textures, numTextures * sizeof(Texture),
            cudaMemcpyDeviceToHost);

        for (int i = 0; i < numTextures; i++) {
            if (hostTextures[i].data) cudaFree(hostTextures[i].data);
        }
        delete[] hostTextures;
        cudaFree(dev_textures);
        dev_textures = nullptr;
    }

    // Free environment map
    if (hasEnvironmentMap) {
        EnvironmentMap hostEnvMap;
        cudaMemcpy(&hostEnvMap, dev_envMap, sizeof(EnvironmentMap),
            cudaMemcpyDeviceToHost);

        cudaDestroyTextureObject(hostEnvMap.texture);
        cudaFreeArray(envMapArray);
        cudaFree(hostEnvMap.marginalCDF);
        cudaFree(hostEnvMap.conditionalCDF);
        cudaFree(dev_envMap);

        dev_envMap = nullptr;
        envMapArray = nullptr;
        hasEnvironmentMap = false;
    }
    BVHBuilder::free(dev_bvh);
    checkCUDAError("pathtraceFree");
}

// Generate PathSegments with rays from the camera into the scene
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.prevBsdfPdf = 0.0f;
        segment.prevWasDelta = 0;

        // 4x4 stratified jitter per iteration
        const int S = 4;
        unsigned s = (iter - 1) % (S * S);
        int sx = s % S, sy = s / S;

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jx = (sx + u01(rng)) / S - 0.5f;
        float jy = (sy + u01(rng)) / S - 0.5f;

        segment.ray.direction = glm::normalize(
            cam.view
            - cam.right * cam.pixelLength.x * ((x + jx) - cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((y + jy) - cam.resolution.y * 0.5f));

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}


// Slightly optimized intersection kernel
__global__ void computeIntersections(
    int depth,
    int num_paths,
    const PathSegment* __restrict__ pathSegments,
    const Geom* __restrict__ geoms,
    int geoms_size,
    const TriangleMeshData* __restrict__ meshes,
    ShadeableIntersection* __restrict__ intersections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    const Ray ray = pathSegments[idx].ray;

    float t_min = 1e20f;
    int hit_i = -1;
    glm::vec3 n_best(0.0f);
    glm::vec2 uv_best(0.0f);
    glm::vec4 tangent_best(0.0f);

    glm::vec3 I_tmp, N_tmp;
    glm::vec2 uv_tmp;
    glm::vec4 tangent_tmp;
    bool outside;

    for (int i = 0; i < geoms_size; ++i) {
        const Geom& g = geoms[i];
        float t = -1.0f;

        if (g.type == CUBE) {
            t = boxIntersectionTest(g, ray, I_tmp, N_tmp, outside);
            uv_tmp = glm::vec2(0.0f);
            tangent_tmp = glm::vec4(0, 0, 0, 1);
        }
        else if (g.type == SPHERE) {
            t = sphereIntersectionTest(g, ray, I_tmp, N_tmp, outside);
            uv_tmp = glm::vec2(0.0f);
            tangent_tmp = glm::vec4(0, 0, 0, 1);
        }
        else if (g.type == TRIANGLE_MESH && g.meshIndex >= 0) {
            t = meshIntersectionTest(
                g, meshes[g.meshIndex], ray,
                I_tmp, N_tmp, outside,
                uv_tmp, tangent_tmp);
        }

        if (t > 0.0f && t < t_min) {
            t_min = t;
            hit_i = i;
            n_best = N_tmp;
            uv_best = uv_tmp;
            tangent_best = tangent_tmp;
        }
    }

    ShadeableIntersection out;
    out.t = (hit_i < 0) ? -1.0f : t_min;
    if (hit_i >= 0) {
        out.materialId = geoms[hit_i].materialid;
        out.surfaceNormal = n_best;
        out.geomId = hit_i;
        out.uv = uv_best;
        out.tangent = tangent_best;
    }
    intersections[idx] = out;
}


__global__ void computeIntersectionsBVH(
    int depth, int num_paths,
    const PathSegment* __restrict__ pathSegments,
    const Geom* __restrict__ geoms,
    const BVHNode* __restrict__ bvhNodes,
    const BVHPrimitive* __restrict__ primitives,
    const TriangleMeshData* __restrict__ meshes,
    ShadeableIntersection* __restrict__ intersections)
{ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    const Ray ray = pathSegments[idx].ray;

    float t_min = 1e20f;
    int hit_i = -1;
    glm::vec3 n_best(0.0f);
    glm::vec2 uv_best(0.0f);
    glm::vec4 tangent_best(0.0f);

    glm::vec3 I_tmp, N_tmp;
    glm::vec2 uv_tmp;
    glm::vec4 tangent_tmp;
    bool outside;

    // Stack for iterative BVH traversal
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const BVHNode& node = bvhNodes[nodeIdx];

        // Test AABB
        if (!intersectAABB(ray, node.aabbMin, node.aabbMax)) continue;

        if (node.leftChild == -1) {
            // Leaf node
            for (int i = 0; i < node.primCount; i++) {
                const BVHPrimitive& prim = primitives[node.primStart + i];
                const Geom& g = geoms[prim.geomIndex];

                float t = -1.0f;

                if (prim.type == PRIM_GEOM) {
                    if (g.type == CUBE) {
                        t = boxIntersectionTest(g, ray, I_tmp, N_tmp, outside);
                        uv_tmp = glm::vec2(0.0f);
                        tangent_tmp = glm::vec4(0, 0, 0, 1);
                    }
                    else if (g.type == SPHERE) {
                        t = sphereIntersectionTest(g, ray, I_tmp, N_tmp, outside);
                        uv_tmp = glm::vec2(0.0f);
                        tangent_tmp = glm::vec4(0, 0, 0, 1);
                    }
                }
                else {
                    t = singleTriangleIntersectionTest(
                        g, meshes[g.meshIndex], prim.triangleIndex, ray,
                        I_tmp, N_tmp, outside,
                        uv_tmp, tangent_tmp);
                }

                if (t > 0.0f && t < t_min) {
                    t_min = t;
                    hit_i = prim.geomIndex;
                    n_best = N_tmp;
                    uv_best = uv_tmp;
                    tangent_best = tangent_tmp;
                }
            }
        }
        else {
            // Interior node - add children to stack
            if (stackPtr < 62) { // Leave room for both children
                stack[stackPtr++] = node.leftChild;
                stack[stackPtr++] = node.rightChild;
            }
        }
    }

    ShadeableIntersection out;
    out.t = (hit_i < 0) ? -1.0f : t_min;
    if (hit_i >= 0) {
        out.materialId = geoms[hit_i].materialid;
        out.surfaceNormal = n_best;
        out.geomId = hit_i;
        out.uv = uv_best;
        out.tangent = tangent_best;
    }
    intersections[idx] = out;
}


// Per-class shading over contiguous spans
__global__ void shadeEmissiveRange(
    int n, int depth,
    const ShadeableIntersection* __restrict__ isects,
    const Material* __restrict__ materials,
    PathSegment* paths,
    const Geom* __restrict__ geoms,
    const int* __restrict__ lightIdx,
    int numLights,
    glm::vec3* __restrict__ image)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const ShadeableIntersection& isect = isects[i];
    PathSegment& path = paths[i];
    const Material& m = materials[isect.materialId];

    glm::vec3 Le = m.color * m.emittance;
    glm::vec3 contrib = evalEmissiveWithMIS(path, isect, Le, depth, geoms, lightIdx, numLights);

    atomicAddVec3(image, path.pixelIndex, contrib);

    // Terminate path
    path.color = glm::vec3(0.0f);
    path.remainingBounces = 0;
}
//__global__ void shadeDiffuseRange(
//    int iter, int n, int depth, int useRR, int useNEE,
//    const ShadeableIntersection* __restrict__ isects,
//    const Material* __restrict__ materials,
//    PathSegment* paths,
//    const Geom* __restrict__ geoms, int ngeoms,
//    const int* __restrict__ lightIdx, int numLights,
//    glm::vec3* __restrict__ image)
//{
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i >= n) return;
//
//    const ShadeableIntersection isect = isects[i];
//    PathSegment& ps = paths[i];
//    if (ps.remainingBounces <= 0 || isect.t < 0.f) return;
//
//    thrust::default_random_engine rng =
//        makeSeededRandomEngine(iter, paths[i].pixelIndex, depth);
//
//    const glm::vec3 P = ps.ray.origin + ps.ray.direction * isect.t;
//    const glm::vec3 N = isect.surfaceNormal;
//    const glm::vec3 wo = -ps.ray.direction;
//    const Material& m = materials[isect.materialId];
//
//    // NEE only for diffuse surfaces
//   if (useNEE && (numLights > 0 || envMap) && isDiffuse(m)) {
//        const glm::vec3 albedoTimesThroughput = m.color * ps.color;
//        addDirectLighting_NEEDiffuse(
//            P, N, wo,
//            materials,
//            geoms, ngeoms,
//            lightIdx, numLights,
//            albedoTimesThroughput,
//            ps.pixelIndex,
//            image,
//            rng,
//            envMap);
//    }
//
//
//    scatterRay(ps, P, N, m, rng);
//
//    if (useRR) applyRussianRoulette(ps, depth, 3, rng);
//}

__global__ void gatherTerminated(int n, glm::vec3* image, PathSegment* paths)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (paths[i].remainingBounces <= 0) {
        image[paths[i].pixelIndex] += paths[i].color;
        paths[i].color = glm::vec3(0.0f);
    }
}

__device__ glm::vec3 debugUVColor(const glm::vec2& uv) {
    // Wrap to [0,1]
    float u = uv.x - floorf(uv.x);
    float v = uv.y - floorf(uv.y);

    return glm::vec3(u, v, 0.0f);

    //int checkU = int(floorf(u * 8)) % 2;
    //int checkV = int(floorf(v * 8)) % 2;
    //float c = (checkU ^ checkV) ? 1.0f : 0.0f;
    //return glm::vec3(c, u, v); 
    
}

__global__ void shadeMaterials(
    int iter, int num_paths, int depth, int useRR, int useNEE,
    ShadeableIntersection* __restrict__ isects,
    PathSegment* paths,
    Material* __restrict__ materials,
    const Texture* __restrict__ textures,
    const EnvironmentMap* envMap,
    const Geom* __restrict__ geoms, int ngeoms,
    const int* __restrict__ lightIdx, int numLights,
    glm::vec3* __restrict__ image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    const ShadeableIntersection isect = isects[idx];
    PathSegment& ps = paths[idx];

    if (ps.remainingBounces <= 0) {
        ps.color = glm::vec3(0);
        return;
    }

    if (isect.t < 0.0f) {
        // Ray missed - sample environment map
        if (envMap) {
            glm::vec3 envColor = sampleEnvironmentMap(ps.ray.direction, *envMap);
            atomicAddVec3(image, ps.pixelIndex, ps.color * envColor);
        }
        ps.color = glm::vec3(0);
        ps.remainingBounces = 0;
        return;
    }

    Material m = materials[isect.materialId];

    // --- Base color ---
    glm::vec3 albedo = m.color;
    if (textures && m.baseColorTexture >= 0) {
        albedo *= sampleTexture(textures[m.baseColorTexture], isect.uv.x, isect.uv.y);
    }
    m.color = albedo;

    // --- Metallic-Roughness ---
    if (textures && m.metallicRoughnessTexture >= 0) {
        glm::vec3 mr = sampleTexture(textures[m.metallicRoughnessTexture], isect.uv.x, isect.uv.y);
        m.metallic *= mr.b;
        m.roughness *= mr.g;
    }

    // --- Normal mapping ---
    glm::vec3 shadingNormal = glm::normalize(isect.surfaceNormal);
    if (textures && m.normalTexture >= 0) {
        glm::vec3 nSample = sampleTexture(textures[m.normalTexture], isect.uv.x, isect.uv.y);
        nSample = glm::normalize(nSample * 2.0f - 1.0f);  // [0,1] -> [-1,1]

        glm::vec3 T = glm::normalize(glm::vec3(isect.tangent));
        glm::vec3 N = shadingNormal;
        glm::vec3 B = glm::normalize(glm::cross(N, T) * isect.tangent.w);

        glm::mat3 TBN(T, B, N);
        shadingNormal = glm::normalize(TBN * nSample);
    }

    // --- Occlusion ---
    float ao = 1.0f;
    if (textures && m.occlusionTexture >= 0) {
        glm::vec3 occ = sampleTexture(textures[m.occlusionTexture], isect.uv.x, isect.uv.y);
        ao = 1.0f + (occ.r - 1.0f) * m.occlusionStrength;
    }
    m.color *= ao;

    // --- Emissive (additive) ---
    glm::vec3 Le_tex(0.0f);
    if (textures && m.emissiveTexture >= 0) {
        Le_tex = sampleTexture(textures[m.emissiveTexture], isect.uv.x, isect.uv.y);
    }
    glm::vec3 Le_gltf = m.emissiveFactor * Le_tex;

    // Treat glTF emissive as self-emission (no NEE sampling)
    if (Le_gltf.x > 0.0f || Le_gltf.y > 0.0f || Le_gltf.z > 0.0f) {
        glm::vec3 contrib = ps.color * Le_gltf;
        atomicAddVec3(image, ps.pixelIndex, contrib);
        ps.color = glm::vec3(0);
        ps.remainingBounces = 0;
        return;
    }

    // --- Emissive ---
    if (m.emittance > 0.0f) {
        glm::vec3 Le = m.color * m.emittance;
        glm::vec3 contrib;
        if (useNEE) {
            // With NEE: use MIS to balance light sampling vs BRDF sampling
            contrib = evalEmissiveWithMIS(ps, isect, Le, depth, geoms, lightIdx, numLights);
        }
        else {
            // Without NEE: only BRDF sampling is active, no MIS needed
            contrib = ps.color * Le;
        }

        atomicAddVec3(image, ps.pixelIndex, contrib);
        ps.color = glm::vec3(0);
        ps.remainingBounces = 0;
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, ps.pixelIndex, depth);

    const glm::vec3 P = ps.ray.origin + ps.ray.direction * isect.t;
    const glm::vec3 wo = -ps.ray.direction;


    // NEE for diffuse surfaces
    if (useNEE && numLights > 0 && isDiffuse(m)) {
        addDirectLighting_NEEDiffuse(
            P, shadingNormal, wo,
            materials,
            geoms, ngeoms,
            lightIdx, numLights,
            m.color,     
            ps.color,    
            ps.pixelIndex,
            image,
            rng,
            envMap);
    }

    scatterRay(ps, P, shadingNormal, m, rng);

    if (useRR) applyRussianRoulette(ps, depth, 3, rng);
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = cam.resolution.x * cam.resolution.y;

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        const int useBVH = (guiData && guiData->UseBVH) ? 1 : 0;
        if (useBVH && bvhBuilt) {
            computeIntersectionsBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth, num_paths, dev_paths, dev_geoms,
                dev_bvh.nodes, dev_bvh.primitives, dev_meshes, dev_intersections);
        }
        else {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth, num_paths, dev_paths, dev_geoms,
                hst_scene->geoms.size(), dev_meshes, dev_intersections);
        }
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        {
            auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections));
            auto zip_end = zip_begin + num_paths;

            thrust::for_each(thrust::device, zip_begin, zip_end, MarkMissDead{});
            auto zip_new_end = thrust::remove_if(thrust::device, zip_begin, zip_end, IsDeadTuple{});
            num_paths = static_cast<int>(zip_new_end - zip_begin);
        }

        depth++;

        if (num_paths == 0) {
            iterationComplete = true;
            if (guiData) guiData->TracedDepth = depth;
            break;
        }

        const int useRR = (guiData && guiData->UseRussianRoulette) ? 1 : 0;
        const int useNEE = (guiData && guiData->UseDirectLighting) ? 1 : 0;

        bool doSort = !guiData ? true : guiData->SortByMaterial;
        if (doSort) {
            buildMaterialKeys << <numblocksPathSegmentTracing, blockSize1d >> > (
                num_paths, dev_intersections, dev_matKeys);
            checkCUDAError("build material keys");

            thrust::sequence(thrust::device, dev_indices, dev_indices + num_paths);

            thrust::sort_by_key(thrust::device,
                dev_matKeys, dev_matKeys + num_paths,
                dev_indices);

            thrust::gather(thrust::device, dev_indices, dev_indices + num_paths,
                dev_paths, dev_paths_alt);
            thrust::gather(thrust::device, dev_indices, dev_indices + num_paths,
                dev_intersections, dev_intersections_alt);

            std::swap(dev_paths, dev_paths_alt);
            std::swap(dev_intersections, dev_intersections_alt);
            checkCUDAError("reorder by material id");
        }
        EnvironmentMap* dev_envMapPtr = hasEnvironmentMap ? dev_envMap : nullptr;
        shadeMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter, num_paths, depth, useRR, useNEE,
            dev_intersections, dev_paths, dev_materials,
            dev_textures, dev_envMapPtr, 
            dev_geoms, (int)hst_scene->geoms.size(),
            dev_lightGeomIdx, hst_numLights,
            dev_image);
        checkCUDAError("mega shading");
        

        gatherTerminated << <numblocksPathSegmentTracing, blockSize1d >> > (
            num_paths, dev_image, dev_paths);

        auto newEnd = thrust::remove_if(
            thrust::device, dev_paths, dev_paths + num_paths, IsDeadPath{});

        num_paths = static_cast<int>(newEnd - dev_paths);
        iterationComplete = (num_paths == 0) || (depth >= traceDepth);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }
      
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
