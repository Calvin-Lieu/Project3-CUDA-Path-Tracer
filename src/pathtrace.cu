#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <climits>
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




#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

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
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

struct IsDeadPath {
    __host__ __device__
        bool operator()(const PathSegment& p) const {
        return p.remainingBounces <= 0;
    }
};

// Mark t<0 misses as dead before sorting/shading
struct MarkMissDead {
    __host__ __device__
        void operator()(thrust::tuple<PathSegment&, const ShadeableIntersection&> t) const {
        PathSegment& p = thrust::get<0>(t);
        const ShadeableIntersection& isect = thrust::get<1>(t);
        if (isect.t < 0.0f) {
            p.color = glm::vec3(0.0f);
            p.remainingBounces = 0;
        }
    }
};

// Zip predicate: compact paths+intersections together
struct IsDeadTuple {
    __host__ __device__
        bool operator()(const thrust::tuple<PathSegment, ShadeableIntersection>& t) const {
        return thrust::get<0>(t).remainingBounces <= 0;
    }
};

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
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

        // Each thread writes one pixel location in the texture (textel)
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
// TODO: static variables for device memory, any extra info you need, etc
// ...
// --- Sorting/reorder buffers ---
static uint32_t* dev_matKeys = nullptr;                 // material id per active path
static int* dev_typeKeys = nullptr;                     // shading class per active path
static int* dev_indices = nullptr;                      // permutation
static PathSegment* dev_paths_alt = nullptr;            // ping-pong buffer
static ShadeableIntersection* dev_intersections_alt = nullptr;
static int* dev_typeKeys_alt = nullptr;                 // ping-pong for class keys


void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
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

    // TODO: initialize any extra device memeory you need
    // For sort
    cudaMalloc(&dev_matKeys, pixelcount * sizeof(uint32_t));
    cudaMalloc(&dev_typeKeys, pixelcount * sizeof(int));
    cudaMalloc(&dev_indices, pixelcount * sizeof(int));
    cudaMalloc(&dev_paths_alt, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersections_alt, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_typeKeys_alt, pixelcount * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_matKeys);
    cudaFree(dev_typeKeys);
    cudaFree(dev_indices);
    cudaFree(dev_paths_alt);
    cudaFree(dev_intersections_alt);
    cudaFree(dev_typeKeys_alt);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

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

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersectionsUnopt(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}
   
// Slightly optimized intersection kernel
__global__ void computeIntersections(
    int depth,
    int num_paths,
    const PathSegment* __restrict__ pathSegments,
    const Geom* __restrict__ geoms,
    int geoms_size,
    ShadeableIntersection* __restrict__ intersections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    const Ray ray = pathSegments[idx].ray;

    float t_min = 1e20f;
    int   hit_i = -1;
    glm::vec3 n_best = glm::vec3(0.0f);

    glm::vec3 I_tmp, N_tmp;
    bool outside;

	#pragma unroll 1
    for (int i = 0; i < geoms_size; ++i)
    {
        const Geom& g = geoms[i];
        float t = -1.0f;
        if (g.type == CUBE)        t = boxIntersectionTest(g, ray, I_tmp, N_tmp, outside);
        else if (g.type == SPHERE) t = sphereIntersectionTest(g, ray, I_tmp, N_tmp, outside);

        if (t > 0.0f && t < t_min) { t_min = t; hit_i = i; n_best = N_tmp; }
    }

    ShadeableIntersection out;
    out.t = (hit_i < 0) ? -1.0f : t_min;
    if (hit_i >= 0) {
        out.materialId = geoms[hit_i].materialid;
        out.surfaceNormal = n_best;
    }
    intersections[idx] = out;
}



// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

__global__ void buildMaterialKeys(int n,
    const ShadeableIntersection* __restrict__ isects,
    uint32_t* __restrict__ keys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    keys[i] = static_cast<uint32_t>(isects[i].materialId);
}

enum ShadingClass : int { SHADING_EMISSIVE = 0, SHADING_DIFFUSE = 1 };

__global__ void buildTypeKeys(
    int n,
    const ShadeableIntersection* __restrict__ isects,
    const Material* __restrict__ materials,
    int* __restrict__ typeKeys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Material& m = materials[isects[i].materialId];
    typeKeys[i] = (m.emittance > 0.0f) ? SHADING_EMISSIVE : SHADING_DIFFUSE;
}

// Per-class shading over contiguous spans (branch-free)
__global__ void shadeEmissiveRange(int n,
    const ShadeableIntersection* __restrict__ isects,
    const Material* __restrict__ materials,
    PathSegment* paths)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Material& m = materials[isects[i].materialId];
    //if (m.emittance > 0.0f) {
    paths[i].color *= (m.color * m.emittance);
    paths[i].remainingBounces = 0;
    //}

}

__global__ void shadeDiffuseRange(
    int iter, int n,
    const ShadeableIntersection* __restrict__ isects,
    const Material* __restrict__ materials,
    PathSegment* paths)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    ShadeableIntersection isect = isects[i];
    PathSegment& ps = paths[i];
    if (ps.remainingBounces <= 0) return;

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, i, ps.remainingBounces);
    const glm::vec3 P = ps.ray.origin + ps.ray.direction * isect.t;
    scatterRay(ps, P, isect.surfaceNormal, materials[isect.materialId], rng);
}

__global__ void gatherTerminated(int n, glm::vec3* image, PathSegment* paths)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Paths with remainingBounces == 0 are done this bounce.
    if (paths[i].remainingBounces <= 0) {
        image[paths[i].pixelIndex] += paths[i].color;
        paths[i].color = glm::vec3(0.0f); // avoid double add if it lingers
    }
}


// BSDF shading: handles miss, emissive, and diffuse via scatterRay
__global__ void shadeMaterials(
    int iter,
    int num_paths,
    ShadeableIntersection* isects,
    PathSegment* paths,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    ShadeableIntersection isect = isects[idx];
    PathSegment& ps = paths[idx];

    if (ps.remainingBounces <= 0) { ps.color = glm::vec3(0); return; }
    if (isect.t < 0.0f) { ps.color = glm::vec3(0); ps.remainingBounces = 0; return; }

    const Material& m = materials[isect.materialId];

    if (m.emittance > 0.0f) {
        ps.color *= (m.color * m.emittance);
        ps.remainingBounces = 0;
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, ps.remainingBounces);
    const glm::vec3 hitP = ps.ray.origin + ps.ray.direction * isect.t;
    scatterRay(ps, hitP, isect.surfaceNormal, m, rng);
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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    /*int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;*/
    int depth = 0;
    int num_paths = cam.resolution.x * cam.resolution.y;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // Kill misses and compact (keep paths+isects aligned)
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


        // Optionally sort-by-material
        bool doSort = !guiData ? true : guiData->SortByMaterial;
        if (doSort) {
            // Build material keys from current intersections
            buildMaterialKeys << <numblocksPathSegmentTracing, blockSize1d >> > (
                num_paths, dev_intersections, dev_matKeys);
            thrust::sequence(thrust::device, dev_indices, dev_indices + num_paths);

            // Sort by material ID
            thrust::sort_by_key(thrust::device,
                dev_matKeys, dev_matKeys + num_paths,
                dev_indices);

            // Reorder paths and intersections using the permutation
            thrust::gather(thrust::device, dev_indices, dev_indices + num_paths,
                dev_paths, dev_paths_alt);
            thrust::gather(thrust::device, dev_indices, dev_indices + num_paths,
                dev_intersections, dev_intersections_alt);

            std::swap(dev_paths, dev_paths_alt);
            std::swap(dev_intersections, dev_intersections_alt);

            // Build typeKeys from the REORDERED intersections
            buildTypeKeys << <numblocksPathSegmentTracing, blockSize1d >> > (
                num_paths, dev_intersections, dev_materials, dev_typeKeys);
            checkCUDAError("build type keys");

            // Create indices for secondary sort
            thrust::sequence(thrust::device, dev_indices, dev_indices + num_paths);

            // Stable sort by type to separate emissive from diffuse
            thrust::stable_sort_by_key(thrust::device,
                dev_typeKeys, dev_typeKeys + num_paths,
                dev_indices);

            // Reorder everything by type
            thrust::gather(thrust::device, dev_indices, dev_indices + num_paths,
                dev_paths, dev_paths_alt);
            thrust::gather(thrust::device, dev_indices, dev_indices + num_paths,
                dev_intersections, dev_intersections_alt);
            thrust::gather(thrust::device, dev_indices, dev_indices + num_paths,
                dev_typeKeys, dev_typeKeys_alt);

            std::swap(dev_paths, dev_paths_alt);
            std::swap(dev_intersections, dev_intersections_alt);
            std::swap(dev_typeKeys, dev_typeKeys_alt);

            checkCUDAError("reorder by type");

            // Find boundary between emissive and diffuse
            thrust::device_ptr<int> kb(dev_typeKeys);
            const int nEmissive = (int)(thrust::upper_bound(thrust::device, kb, kb + num_paths, SHADING_EMISSIVE) - kb);
            const int nDiffuse = num_paths - nEmissive;

            auto blocks = [&](int n) { return (n + blockSize1d - 1) / blockSize1d; };

            // Shade emissive materials
            if (nEmissive > 0) {
                shadeEmissiveRange << <blocks(nEmissive), blockSize1d >> > (
                    nEmissive, dev_intersections, dev_materials, dev_paths);
                checkCUDAError("emissive shading");
            }

            // Shade diffuse materials  
            if (nDiffuse > 0) {
                shadeDiffuseRange << <blocks(nDiffuse), blockSize1d >> > (
                    iter, nDiffuse,
                    dev_intersections + nEmissive,
                    dev_materials,
                    dev_paths + nEmissive);
                checkCUDAError("diffuse shading");
            }
        }
        else {
            shadeMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
                iter, num_paths, dev_intersections, dev_paths, dev_materials);
            checkCUDAError("mega shading");
        }

        // Accumulate finished paths for this bounce
        gatherTerminated << <numblocksPathSegmentTracing, blockSize1d >> > (
            num_paths, dev_image, dev_paths);

		// Stream compact away terminated paths
        auto newEnd = thrust::remove_if(
            thrust::device, dev_paths, dev_paths + num_paths, IsDeadPath{});

        num_paths = static_cast<int>(newEnd - dev_paths);
        iterationComplete = (num_paths == 0) || (depth >= traceDepth);


        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
