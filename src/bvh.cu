#include "bvh.h"
#include <cfloat>

BVH BVHBuilder::build(const std::vector<Geom>& geoms) {
    BVH bvh;
    bvh.maxDepth = 0;

    if (geoms.empty()) {
        bvh.nodes = nullptr;
        bvh.primIndices = nullptr;
        bvh.nodeCount = 0;
        return bvh;
    }

    std::vector<BVHNode> nodes;
    std::vector<int> indices(geoms.size());
    std::iota(indices.begin(), indices.end(), 0);

    buildRecursive(geoms, indices, 0, geoms.size(), nodes, 0, bvh.maxDepth);

    // Allocate GPU memory
    cudaMalloc(&bvh.nodes, nodes.size() * sizeof(BVHNode));
    cudaMemcpy(bvh.nodes, nodes.data(),
        nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&bvh.primIndices, indices.size() * sizeof(int));
    cudaMemcpy(bvh.primIndices, indices.data(),
        indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    bvh.nodeCount = (int)nodes.size();

    printf("BVH built: %d nodes, max depth %d\n", bvh.nodeCount, bvh.maxDepth);

    return bvh;
}

void BVHBuilder::free(BVH& bvh) {
    if (bvh.nodes) cudaFree(bvh.nodes);
    if (bvh.primIndices) cudaFree(bvh.primIndices);
    bvh.nodes = nullptr;
    bvh.primIndices = nullptr;
    bvh.nodeCount = 0;
}

int BVHBuilder::buildRecursive(
    const std::vector<Geom>& geoms,
    std::vector<int>& indices,
    int start, int end,
    std::vector<BVHNode>& nodes,
    int depth,
    int& maxDepth)
{
    if (depth > maxDepth) maxDepth = depth;

    BVHNode node;

    // Compute AABB for this range
    node.aabbMin = glm::vec3(FLT_MAX);
    node.aabbMax = glm::vec3(-FLT_MAX);
    for (int i = start; i < end; i++) {
        glm::vec3 gMin, gMax;
        computeGeomAABB(geoms[indices[i]], gMin, gMax);
        node.aabbMin = glm::min(node.aabbMin, gMin);
        node.aabbMax = glm::max(node.aabbMax, gMax);
    }

    int nodeIdx = (int)nodes.size();
    nodes.push_back(node);

    // Leaf node if few primitives or max depth
    if (end - start <= 4 || depth >= 20) {
        nodes[nodeIdx].leftChild = -1;
        nodes[nodeIdx].rightChild = -1;
        nodes[nodeIdx].primStart = start;
        nodes[nodeIdx].primCount = end - start;
        return nodeIdx;
    }

    // Split along longest axis using median
    glm::vec3 extent = node.aabbMax - node.aabbMin;
    int axis = (extent.x > extent.y && extent.x > extent.z) ? 0 :
        (extent.y > extent.z) ? 1 : 2;

    int mid = (start + end) / 2;
    std::nth_element(indices.begin() + start,
        indices.begin() + mid,
        indices.begin() + end,
        [&](int a, int b) {
            return getGeomCenter(geoms[a])[axis] <
                getGeomCenter(geoms[b])[axis];
        });

    // Build children
    nodes[nodeIdx].primStart = -1;
    nodes[nodeIdx].primCount = 0;
    nodes[nodeIdx].leftChild = buildRecursive(geoms, indices, start, mid, nodes, depth + 1, maxDepth);
    nodes[nodeIdx].rightChild = buildRecursive(geoms, indices, mid, end, nodes, depth + 1, maxDepth);

    return nodeIdx;
}

void BVHBuilder::computeGeomAABB(const Geom& g, glm::vec3& min, glm::vec3& max) {
    glm::vec3 corners[8] = {
        {-0.5f,-0.5f,-0.5f}, {0.5f,-0.5f,-0.5f},
        {-0.5f,0.5f,-0.5f},  {0.5f,0.5f,-0.5f},
        {-0.5f,-0.5f,0.5f},  {0.5f,-0.5f,0.5f},
        {-0.5f,0.5f,0.5f},   {0.5f,0.5f,0.5f}
    };

    min = glm::vec3(FLT_MAX);
    max = glm::vec3(-FLT_MAX);
    for (const auto& corner : corners) {
        glm::vec3 p = glm::vec3(g.transform * glm::vec4(corner, 1.0f));
        min = glm::min(min, p);
        max = glm::max(max, p);
    }
}

glm::vec3 BVHBuilder::getGeomCenter(const Geom& g) {
    return glm::vec3(g.transform * glm::vec4(0, 0, 0, 1));
}