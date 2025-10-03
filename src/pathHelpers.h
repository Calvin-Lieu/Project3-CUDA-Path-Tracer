#pragma once

#include <thrust/random.h>
#include <thrust/tuple.h>
#include "glm/glm.hpp"

#include "utilities.h"        
#include "sceneStructs.h"   

// RNG with deterministic seeding
__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth);


// Stream-compaction functors 
struct IsDeadPath {
    __host__ __device__ bool operator()(const PathSegment& p) const {
        return p.remainingBounces <= 0;
    }
};

struct MarkMissDead {
    __host__ __device__
        void operator()(const thrust::tuple<PathSegment&, const ShadeableIntersection&>& t) const {
        PathSegment& ps = thrust::get<0>(t);
        if (ps.remainingBounces <= 0) {
            ps.color = glm::vec3(0.0f);
        }
    }
};

struct IsDeadTuple {
    __host__ __device__
        bool operator()(const thrust::tuple<PathSegment, ShadeableIntersection>& t) const {
        const PathSegment& ps = thrust::get<0>(t);
        return ps.remainingBounces <= 0;
    }
};

__device__ float rr_luminance(const glm::vec3& c);

__device__ void applyRussianRoulette(PathSegment& ps,
    int depth,
    int rrStartDepth,  
    thrust::default_random_engine& rng);
