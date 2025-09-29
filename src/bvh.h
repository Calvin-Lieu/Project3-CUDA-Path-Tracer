#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "sceneStructs.h"

enum PrimitiveType {
    PRIM_GEOM,      // Sphere or Cube (whole geometry)
    PRIM_TRIANGLE   // Individual triangle from mesh
};

struct BVHPrimitive {
    PrimitiveType type;
    int geomIndex;      // Index into geoms array
    int triangleIndex;  // Only valid if type == PRIM_TRIANGLE (-1 otherwise)
};

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
    BVHPrimitive* primitives;
    int nodeCount;
    int maxDepth;
};

class BVHBuilder {
public:
    static BVH build(const std::vector<Geom>& geoms, const std::vector<MeshData>& meshes);
    static void free(BVH& bvh);

private:
    static int buildRecursive(
        const std::vector<Geom>& geoms,
        const std::vector<MeshData>& meshes,
        const std::vector<BVHPrimitive>& primitives,
        std::vector<int>& indices,
        int start, int end,
        std::vector<BVHNode>& nodes,
        int depth,
        int& maxDepth);

    static void computePrimitiveAABB(
        const Geom& g,
        const std::vector<MeshData>& meshes,
        const BVHPrimitive& prim,
        glm::vec3& min,
        glm::vec3& max);

    static glm::vec3 getPrimitiveCenter(
        const Geom& g,
        const std::vector<MeshData>& meshes,
        const BVHPrimitive& prim);
};

