#pragma once

#include <thrust/random.h>
#include <thrust/tuple.h>
#include "glm/glm.hpp"

#include "utilities.h"        
#include "sceneStructs.h"   

// RNG with deterministic seeding
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth);


// ---- Stream-compaction functors 
struct IsDeadPath {
    __host__ __device__ bool operator()(const PathSegment& p) const {
        return p.remainingBounces <= 0;
    }
};

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

struct IsDeadTuple {
    __host__ __device__
        bool operator()(const thrust::tuple<PathSegment, ShadeableIntersection>& t) const {
        return thrust::get<0>(t).remainingBounces <= 0;
    }
};

__device__ float rr_luminance(const glm::vec3& c);

__device__ void applyRussianRoulette(PathSegment& ps,
    int depth,
    int rrStartDepth,  
    float rrMinP,       
    thrust::default_random_engine& rng);
