#include "pathHelpers.h"
#include <thrust/random.h>
#include "intersections.h"


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

__device__ float rr_luminance(const glm::vec3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ void applyRussianRoulette(PathSegment& ps,
    int depth,
    int rrStartDepth,
    thrust::default_random_engine& rng)
{
    if (depth < rrStartDepth) return;

    thrust::uniform_real_distribution<float> u01(0.f, 1.f);

    // Termination probability (increases as throughput dims)
    float q = fmaxf(0.05f, 1.0f - rr_luminance(ps.color));

    if (u01(rng) < q) {
        ps.color = glm::vec3(0.0f);
        ps.remainingBounces = 0;
    }
    else {
        ps.color /= (1.0f - q);
    }
}
