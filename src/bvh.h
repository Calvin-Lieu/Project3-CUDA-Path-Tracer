#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "sceneStructs.h"

struct BVHNode {
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
    int leftChild;   // index of left child, or -1 if leaf
    int rightChild;  // index of right child
    int primStart;   // first primitive index (for leaves)
    int primCount;   // number of primitives (for leaves)
};

struct BVH {
    BVHNode* nodes;
    int* primIndices;
    int nodeCount;
    int maxDepth;
};

class BVHBuilder {
public:
    static BVH build(const std::vector<Geom>& geoms);
    static void free(BVH& bvh);

private:
    static int buildRecursive(
        const std::vector<Geom>& geoms,
        std::vector<int>& indices,
        int start, int end,
        std::vector<BVHNode>& nodes,
        int depth,
        int& maxDepth);

    static void computeGeomAABB(const Geom& g, glm::vec3& min, glm::vec3& max);
    static glm::vec3 getGeomCenter(const Geom& g);
};

// Device functions for GPU traversal
__device__ bool intersectAABB(const Ray& ray, const glm::vec3& aabbMin, const glm::vec3& aabbMax);