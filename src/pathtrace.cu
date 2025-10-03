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
#include <OpenImageDenoise/oidn.h>

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

void checkOIDNError(OIDNDevice device)
{
#if ERRORCHECK
    const char* errorMessage;
    if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
    {
        fprintf(stderr, "OIDN error: %s\n", errorMessage);
    }
#endif
}

__device__ __host__ inline glm::vec3 ReinhardToneMap(const glm::vec3& x)
{
    return x / (glm::vec3(1.0f) + x);
}

__device__ __host__ inline glm::vec3 ACESToneMap(const glm::vec3& x)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return glm::clamp((x * (a * x + b)) / (x * (c * x + d) + e),
        glm::vec3(0.0f), glm::vec3(1.0f));
}

// Converts accumulated HDR buffer to LDR and writes to PBO for display
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image, int toneMappingMode, float exposure, float gamma)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;

    int idx = x + y * resolution.x;

    // Average accumulated samples
    glm::vec3 color = image[idx] / (float)iter;
    color *= powf(2.0f, exposure);

    if (toneMappingMode == 1) {
        color = ReinhardToneMap(color);
    }
    else if (toneMappingMode == 2) {
        color = ACESToneMap(color);
    }

    color = pow(color, glm::vec3(1.0f / gamma));

    pbo[idx] = make_uchar4(
        (unsigned char)(fminf(color.r, 1.0f) * 255.0f),
        (unsigned char)(fminf(color.g, 1.0f) * 255.0f),
        (unsigned char)(fminf(color.b, 1.0f) * 255.0f),
        255);
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;

// Separate buffers for raw accumulation and denoised output
static glm::vec3* dev_image_raw = NULL;
static glm::vec3* dev_image_denoised = NULL;

static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// Material sorting scratch buffers
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

// Denoiser auxiliary buffers (accumulated + averaged per-iter)
static glm::vec3* dev_albedo_accum = nullptr;
static glm::vec3* dev_albedo = nullptr;
static glm::vec3* dev_normal_accum = nullptr;
static glm::vec3* dev_normal = nullptr;

// CUDA OIDN device and filter (runs on GPU, no CPU copy)
static OIDNDevice oidnDevice;
static OIDNFilter oidnFilter;
static OIDNBuffer oidnColorBuf;
static OIDNBuffer oidnAlbedoBuf;
static OIDNBuffer oidnNormalBuf;
static OIDNBuffer oidnOutputBuf;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

// Builds CDF for importance sampling environment map
void buildEnvironmentCDFsFromFloat(const std::vector<float>& imageData,
    int width, int height,
    EnvironmentMap& envMap)
{
    std::vector<float> luminance(width * height);

    // Compute luminance weighted by solid angle (sin(theta) for sphere)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 4;
            float r = imageData[idx];
            float g = imageData[idx + 1];
            float b = imageData[idx + 2];

            float theta = PI * (y + 0.5f) / height;
            float sinTheta = sinf(theta);

            luminance[y * width + x] = (0.2126f * r + 0.7152f * g + 0.0722f * b) * sinTheta;
        }
    }

    // Build conditional CDF (per-row)
    std::vector<float> conditionalCDF(width * height);
    std::vector<float> rowIntegrals(height);

    for (int y = 0; y < height; y++) {
        float sum = 0.0f;
        for (int x = 0; x < width; x++) {
            sum += luminance[y * width + x];
            conditionalCDF[y * width + x] = sum;
        }

        rowIntegrals[y] = sum;

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

    if (totalSum > 0.0f) {
        for (int y = 0; y < height; y++) {
            marginalCDF[y] /= totalSum;
        }
    }

    envMap.totalLuminance = totalSum;

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
    const int imageSizeBytes = pixelcount * sizeof(glm::vec3);

    cudaMalloc(&dev_image_raw, imageSizeBytes);
    cudaMemset(dev_image_raw, 0, imageSizeBytes);

    cudaMalloc(&dev_image_denoised, imageSizeBytes);
    cudaMemset(dev_image_denoised, 0, imageSizeBytes);

    cudaMalloc(&dev_albedo_accum, imageSizeBytes);
    cudaMemset(dev_albedo_accum, 0, imageSizeBytes);
    cudaMalloc(&dev_albedo, imageSizeBytes);
    cudaMemset(dev_albedo, 0, imageSizeBytes);

    cudaMalloc(&dev_normal_accum, imageSizeBytes);
    cudaMemset(dev_normal_accum, 0, imageSizeBytes);
    cudaMalloc(&dev_normal, imageSizeBytes);
    cudaMemset(dev_normal, 0, imageSizeBytes);

    // Setup CUDA OIDN device (GPU-based denoising)
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    cudaStream_t stream = 0;

    oidnDevice = oidnNewCUDADevice(&deviceId, &stream, 1);
    oidnCommitDevice(oidnDevice);

    // Wrap device pointers as shared buffers (zero-copy)
    oidnColorBuf = oidnNewSharedBuffer(oidnDevice, dev_image_raw, imageSizeBytes);
    oidnAlbedoBuf = oidnNewSharedBuffer(oidnDevice, dev_albedo, imageSizeBytes);
    oidnNormalBuf = oidnNewSharedBuffer(oidnDevice, dev_normal, imageSizeBytes);
    oidnOutputBuf = oidnNewSharedBuffer(oidnDevice, dev_image_denoised, imageSizeBytes);

    oidnFilter = oidnNewFilter(oidnDevice, "RT");
    oidnSetFilterImage(oidnFilter, "color", oidnColorBuf, OIDN_FORMAT_FLOAT3,
        cam.resolution.x, cam.resolution.y, 0, 0, 0);
    oidnSetFilterImage(oidnFilter, "albedo", oidnAlbedoBuf, OIDN_FORMAT_FLOAT3,
        cam.resolution.x, cam.resolution.y, 0, 0, 0);
    oidnSetFilterImage(oidnFilter, "normal", oidnNormalBuf, OIDN_FORMAT_FLOAT3,
        cam.resolution.x, cam.resolution.y, 0, 0, 0);
    oidnSetFilterImage(oidnFilter, "output", oidnOutputBuf, OIDN_FORMAT_FLOAT3,
        cam.resolution.x, cam.resolution.y, 0, 0, 0);
    oidnSetFilterBool(oidnFilter, "hdr", true);
    oidnCommitFilter(oidnFilter);

    checkOIDNError(oidnDevice);

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

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

            cudaMalloc(&hostMeshes[i].vertices, hostMesh.vertices.size() * sizeof(float));
            cudaMemcpy(hostMeshes[i].vertices, hostMesh.vertices.data(),
                hostMesh.vertices.size() * sizeof(float), cudaMemcpyHostToDevice);

            cudaMalloc(&hostMeshes[i].normals, hostMesh.normals.size() * sizeof(float));
            cudaMemcpy(hostMeshes[i].normals, hostMesh.normals.data(),
                hostMesh.normals.size() * sizeof(float), cudaMemcpyHostToDevice);

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

            cudaMalloc(&hostMeshes[i].indices, hostMesh.indices.size() * sizeof(unsigned int));
            cudaMemcpy(hostMeshes[i].indices, hostMesh.indices.data(),
                hostMesh.indices.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

            hostMeshes[i].triangleCount = hostMesh.indices.size() / 3;
        }

        cudaMemcpy(dev_meshes, hostMeshes.data(), numMeshes * sizeof(TriangleMeshData), cudaMemcpyHostToDevice);
    }
    else {
        dev_meshes = nullptr;
        numMeshes = 0;
    }

    // Collect emissive geometry indices for direct lighting
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

    if (!scene->textures.empty()) {
        numTextures = scene->textures.size();
        //std::cout << "Uploading " << numTextures << " textures to GPU...\n";

        std::vector<Texture> hostTextures(numTextures);

        for (int i = 0; i < numTextures; i++) {
            auto& texPair = scene->textures[i];
            Texture& tex = hostTextures[i];

            tex.width = texPair.second.width;
            tex.height = texPair.second.height;
            tex.channels = texPair.second.channels;

            size_t imageSize = tex.width * tex.height * tex.channels;

            cudaMalloc(&tex.data, imageSize);
            cudaMemcpy(tex.data, texPair.first.data(), imageSize, cudaMemcpyHostToDevice);

            std::cout << "  Texture " << i << ": " << tex.width << "x" << tex.height
                << " (" << tex.channels << " channels)\n";
        }

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
            // create CUDA texture for environment map
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

            EnvironmentMap hostEnvMap;
            cudaCreateTextureObject(&hostEnvMap.texture, &resDesc, &texDesc, nullptr);
            hostEnvMap.width = width;
            hostEnvMap.height = height;

            std::vector<float> floatVec(data, data + width * height * 4);
            buildEnvironmentCDFsFromFloat(floatVec, width, height, hostEnvMap);

            cudaMalloc(&dev_envMap, sizeof(EnvironmentMap));
            cudaMemcpy(dev_envMap, &hostEnvMap, sizeof(EnvironmentMap), cudaMemcpyHostToDevice);

            hasEnvironmentMap = true;
            stbi_image_free(data);

            std::cout << "Environment map loaded: " << width << "x" << height << "\n";
            std::cout << "Total luminance: " << hostEnvMap.totalLuminance << "\n";
        }
    }

    dev_bvh = BVHBuilder::build(scene->geoms, scene->meshes);
    bvhBuilt = true;

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_albedo_accum);
    cudaFree(dev_albedo);
    cudaFree(dev_normal_accum);
    cudaFree(dev_normal);

    cudaFree(dev_image_raw);
    cudaFree(dev_image_denoised);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    cudaFree(dev_matKeys);
    cudaFree(dev_indices);
    cudaFree(dev_paths_alt);
    cudaFree(dev_intersections_alt);
    cudaFree(dev_lightGeomIdx);

    // Free nested mesh data
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

    oidnReleaseBuffer(oidnColorBuf);
    oidnReleaseBuffer(oidnAlbedoBuf);
    oidnReleaseBuffer(oidnNormalBuf);
    oidnReleaseBuffer(oidnOutputBuf);
    oidnReleaseFilter(oidnFilter);
    oidnReleaseDevice(oidnDevice);

    BVHBuilder::free(dev_bvh);
    checkCUDAError("pathtraceFree");
}

// Generates primary rays with 4x4 stratified sampling
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

        // Stratified jitter (4x4 grid cycles every 16 iterations)
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

// Naive intersection (tests all geometry)
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

// BVH-accelerated intersection (world-space ray traversal)
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

    // Stack-based BVH traversal
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const BVHNode& node = bvhNodes[nodeIdx];

        if (!intersectAABB(ray, node.aabbMin, node.aabbMax)) continue;

        if (node.leftChild == -1) {
            // Leaf: test all primitives
            for (int i = 0; i < node.primCount; i++) {
                const BVHPrimitive& prim = primitives[node.primStart + i];
                const Geom& g = geoms[prim.geomIndex];

                float t = -1.0f;
                glm::vec3 I_tmp, N_tmp;
                glm::vec2 uv_tmp;
                glm::vec4 tangent_tmp;
                bool outside;

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
                    t = singleTriangleIntersectionTestWorldSpace(
                        g, meshes[g.meshIndex], prim.triangleIndex, ray,
                        I_tmp, N_tmp, outside, uv_tmp, tangent_tmp);
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
            // Internal: push children
            if (stackPtr < 62) {
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

__global__ void gatherTerminated(int n, glm::vec3* image, PathSegment* paths)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (paths[i].remainingBounces <= 0) {
        image[paths[i].pixelIndex] += paths[i].color;
        paths[i].color = glm::vec3(0.0f);
    }
}

// Shades materials, handles textures, emissive, NEE, BRDF sampling
__global__ void shadeMaterials(
    int iter, int num_paths, int depth, int useRR, int useNEE,
    ShadeableIntersection* __restrict__ isects,
    PathSegment* paths,
    Material* __restrict__ materials,
    const Texture* __restrict__ textures,
    const EnvironmentMap* envMap,
    const Geom* __restrict__ geoms, int ngeoms,
    const int* __restrict__ lightIdx, int numLights,
    glm::vec3* __restrict__ image,
    glm::vec3* __restrict__ albedoAccum,
    glm::vec3* __restrict__ albedoBuffer,
    glm::vec3* __restrict__ normalAccum,
    glm::vec3* __restrict__ normalBuffer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    const ShadeableIntersection isect = isects[idx];
    PathSegment& ps = paths[idx];

    if (ps.remainingBounces <= 0) {
        ps.color = glm::vec3(0);
        return;
    }

    // Miss: sample environment if present
    if (isect.t < 0.0f) {
        if (envMap) {
            glm::vec3 envColor = sampleEnvironmentMap(ps.ray.direction, *envMap);
            atomicAddVec3(image, ps.pixelIndex, ps.color * envColor);
        }
        ps.color = glm::vec3(0);
        ps.remainingBounces = 0;
        return;
    }

    Material m = materials[isect.materialId];

    // Sample base color texture
    glm::vec3 albedo = m.color;
    if (textures && m.baseColorTexture >= 0) {
        glm::vec4 base = sampleTexture4(textures[m.baseColorTexture], isect.uv);
        albedo *= glm::vec3(base);
    }
    m.color = albedo;

    // Accumulate and average albedo for denoiser (first hit only)
    if (depth == 0) {
        glm::vec3 clamped_albedo = glm::clamp(albedo, 0.0f, 1.0f);
        glm::vec3 accum_albedo = albedoAccum[ps.pixelIndex] + clamped_albedo;
        albedoAccum[ps.pixelIndex] = accum_albedo;
        albedoBuffer[ps.pixelIndex] = accum_albedo / (float)iter;
    }

    // Sample metallic-roughness texture
    if (textures && m.metallicRoughnessTexture >= 0) {
        float metallic, roughness, occlusion;
        sampleMetallicRoughness(m, textures, isect.uv, metallic, roughness, occlusion);

        m.metallic = metallic;
        m.roughness = roughness;
        m.color *= 1.0f + (occlusion - 1.0f) * m.occlusionStrength;
    }

    // Normal mapping
    glm::vec3 shadingNormal = glm::normalize(isect.surfaceNormal);
    if (textures && m.normalTexture >= 0) {
        glm::vec3 nSample = sampleTexture3(textures[m.normalTexture], isect.uv.x, isect.uv.y);
        nSample = glm::normalize(nSample * 2.0f - 1.0f);

        glm::vec3 T = glm::normalize(glm::vec3(isect.tangent));
        glm::vec3 N = shadingNormal;
        glm::vec3 B = glm::normalize(glm::cross(N, T) * isect.tangent.w);

        glm::mat3 TBN(T, B, N);
        shadingNormal = glm::normalize(TBN * nSample);
    }

    // Accumulate and average normals for denoiser (first hit only)
    if (depth == 0) {
        glm::vec3 nWorld = glm::normalize(shadingNormal);
        glm::vec3 accum_normal = normalAccum[ps.pixelIndex] + nWorld;
        normalAccum[ps.pixelIndex] = accum_normal;
        normalBuffer[ps.pixelIndex] = accum_normal / (float)iter;
    }

    // Ambient occlusion
    float ao = 1.0f;
    if (textures && m.occlusionTexture >= 0) {
        glm::vec3 occ = sampleTexture3(textures[m.occlusionTexture], isect.uv.x, isect.uv.y);
        ao = 1.0f + (occ.r - 1.0f) * m.occlusionStrength;
    }
    m.color *= ao;

    // gltf emissive texture (not sampled by NEE)
    glm::vec3 Le_tex(0.0f);
    if (textures && m.emissiveTexture >= 0) {
        Le_tex = sampleTexture3(textures[m.emissiveTexture], isect.uv.x, isect.uv.y);
    }
    glm::vec3 Le_gltf = m.emissiveFactor * Le_tex;

    if (Le_gltf.x > 0.0f || Le_gltf.y > 0.0f || Le_gltf.z > 0.0f) {
        glm::vec3 contrib = ps.color * Le_gltf;
        atomicAddVec3(image, ps.pixelIndex, contrib);
        ps.color = glm::vec3(0);
        ps.remainingBounces = 0;
        return;
    }

    // Explicit light sources (sampled by NEE)
    if (m.emittance > 0.0f) {
        glm::vec3 Le = m.color * m.emittance;
        glm::vec3 contrib;
        if (useNEE) {
            // MIS weight for hitting light via BRDF sampling
            contrib = evalEmissiveWithMIS(ps, isect, Le, depth, geoms, lightIdx, numLights);
        }
        else {
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

    // Next event estimation (skip dielectrics)
    if (useNEE && numLights > 0 && !isDielectric(m)) {
        addDirectLightingNEE(
            P, shadingNormal, wo,
            materials,
            geoms, ngeoms,
            lightIdx, numLights,
            m.color * ps.color,
            m.metallic, m.roughness,
            ps.pixelIndex,
            image,
            rng,
            envMap);
    }

    // BRDF sampling
    scatterRay(ps, P, shadingNormal, m, rng);

    // Russian roulette path termination
    if (useRR) applyRussianRoulette(ps, depth, 3, rng);
}

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

        // Compact: remove rays that missed
        {
            auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections));
            auto zip_end = zip_begin + num_paths;

            thrust::for_each(thrust::device, zip_begin, zip_end, MarkMissDead{});
            auto zip_new_end = thrust::remove_if(thrust::device, zip_begin, zip_end, IsDeadTuple{});
            num_paths = static_cast<int>(zip_new_end - zip_begin);
        }

        if (num_paths == 0) {
            iterationComplete = true;
            if (guiData) guiData->TracedDepth = depth;
            break;
        }

        const int useRR = (guiData && guiData->UseRussianRoulette) ? 1 : 0;
        const int useNEE = (guiData && guiData->UseDirectLighting) ? 1 : 0;

        // Sort by material for better coherence
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
            dev_image_raw, dev_albedo_accum, dev_albedo,
            dev_normal_accum, dev_normal);
        checkCUDAError("mega shading");

        depth++;

        gatherTerminated << <numblocksPathSegmentTracing, blockSize1d >> > (
            num_paths, dev_image_raw, dev_paths);

        // Compact: remove terminated paths
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
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image_raw, dev_paths);

    // Denoise on GPU (no CPU copy)
    glm::vec3* displayImage = dev_image_raw;
    if (guiData && guiData->UseDenoiser) {
        oidnExecuteFilter(oidnFilter);
        checkOIDNError(oidnDevice);
        displayImage = dev_image_denoised;
    }

    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, displayImage,
        guiData ? guiData->ToneMappingMode : 0, guiData ? guiData->Exposure : 0, guiData ? guiData->Gamma : 1);

    cudaMemcpy(hst_scene->state.image.data(), displayImage,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}