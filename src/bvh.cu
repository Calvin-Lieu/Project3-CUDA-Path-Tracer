#include "bvh.h"
#include <cfloat>
#include <iostream>

BVH BVHBuilder::build(const std::vector<Geom>& geoms, const std::vector<MeshData>& meshes) {
    BVH bvh;
    bvh.maxDepth = 0;

    std::vector<BVHPrimitive> primitives;

    // Build primitive list: individual triangles for meshes, whole geoms for others
    for (int g = 0; g < geoms.size(); g++) {
        if (geoms[g].type == TRIANGLE_MESH) {
            int meshIdx = geoms[g].meshIndex;
            if (meshIdx >= 0 && meshIdx < meshes.size()) {
                int triCount = meshes[meshIdx].indices.size() / 3;
                for (int t = 0; t < triCount; t++) {
                    primitives.push_back({ PRIM_TRIANGLE, g, t });
                }
            }
        }
        else {
            // Sphere or Cube
            primitives.push_back({ PRIM_GEOM, g, -1 });
        }
    }

    if (primitives.empty()) {
        bvh.nodes = nullptr;
        bvh.primitives = nullptr;
        bvh.nodeCount = 0;
        return bvh;
    }

    std::cout << "Building BVH with " << primitives.size() << " primitives...\n";

    std::vector<BVHNode> nodes;
    std::vector<int> indices(primitives.size());
    std::iota(indices.begin(), indices.end(), 0);

    buildRecursive(geoms, meshes, primitives, indices, 0, primitives.size(), nodes, 0, bvh.maxDepth);

    // Reorder primitives according to indices
    std::vector<BVHPrimitive> orderedPrimitives(primitives.size());
    for (int i = 0; i < indices.size(); i++) {
        orderedPrimitives[i] = primitives[indices[i]];
    }

    // Allocate GPU memory
    cudaMalloc(&bvh.nodes, nodes.size() * sizeof(BVHNode));
    cudaMemcpy(bvh.nodes, nodes.data(), nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&bvh.primitives, orderedPrimitives.size() * sizeof(BVHPrimitive));
    cudaMemcpy(bvh.primitives, orderedPrimitives.data(),
        orderedPrimitives.size() * sizeof(BVHPrimitive), cudaMemcpyHostToDevice);

    bvh.nodeCount = (int)nodes.size();

    printf("BVH built: %d nodes, max depth %d\n", bvh.nodeCount, bvh.maxDepth);

    return bvh;
}

void BVHBuilder::free(BVH& bvh) {
    if (bvh.nodes) cudaFree(bvh.nodes);
    if (bvh.primitives) cudaFree(bvh.primitives);
    bvh.nodes = nullptr;
    bvh.primitives = nullptr;
    bvh.nodeCount = 0;
}

int BVHBuilder::buildRecursive(
    const std::vector<Geom>& geoms,
    const std::vector<MeshData>& meshes,
    const std::vector<BVHPrimitive>& primitives,
    std::vector<int>& indices,
    int start, int end,
    std::vector<BVHNode>& nodes,
    int depth,
    int& maxDepth)
{
    if (depth > maxDepth) maxDepth = depth;

    BVHNode node;
    node.aabbMin = glm::vec3(FLT_MAX);
    node.aabbMax = glm::vec3(-FLT_MAX);

    // Compute AABB for all primitives in range
    for (int i = start; i < end; i++) {
        glm::vec3 pMin, pMax;
        const BVHPrimitive& prim = primitives[indices[i]];
        computePrimitiveAABB(geoms[prim.geomIndex], meshes, prim, pMin, pMax);
        node.aabbMin = glm::min(node.aabbMin, pMin);
        node.aabbMax = glm::max(node.aabbMax, pMax);
    }

    int nodeIdx = (int)nodes.size();
    nodes.push_back(node);

    // Leaf node: small number of primitives or max depth
    if (end - start <= 4 || depth >= 30) {
        nodes[nodeIdx].leftChild = -1;
        nodes[nodeIdx].rightChild = -1;
        nodes[nodeIdx].primStart = start;
        nodes[nodeIdx].primCount = end - start;
        return nodeIdx;
    }

    // Split along longest axis
    glm::vec3 extent = node.aabbMax - node.aabbMin;
    int axis = (extent.x > extent.y && extent.x > extent.z) ? 0 :
        (extent.y > extent.z) ? 1 : 2;

    int mid = (start + end) / 2;
    std::nth_element(indices.begin() + start,
        indices.begin() + mid,
        indices.begin() + end,
        [&](int a, int b) {
            const BVHPrimitive& primA = primitives[a];
            const BVHPrimitive& primB = primitives[b];
            const Geom& geomA = geoms[primA.geomIndex];
            const Geom& geomB = geoms[primB.geomIndex];
            glm::vec3 centerA = getPrimitiveCenter(geomA, meshes, primA);
            glm::vec3 centerB = getPrimitiveCenter(geomB, meshes, primB);
            return centerA[axis] < centerB[axis];
        });

    // Build children
    nodes[nodeIdx].primStart = -1;
    nodes[nodeIdx].primCount = 0;
    nodes[nodeIdx].leftChild = buildRecursive(geoms, meshes, primitives, indices, start, mid, nodes, depth + 1, maxDepth);
    nodes[nodeIdx].rightChild = buildRecursive(geoms, meshes, primitives, indices, mid, end, nodes, depth + 1, maxDepth);

    return nodeIdx;
}

void BVHBuilder::computePrimitiveAABB(
    const Geom& g,
    const std::vector<MeshData>& meshes,
    const BVHPrimitive& prim,
    glm::vec3& min,
    glm::vec3& max)
{
    if (prim.type == PRIM_GEOM) {
        // Compute AABB for whole geometry (sphere/cube)
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
    else {
        const MeshData& mesh = meshes[g.meshIndex];
        unsigned int i0 = mesh.indices[prim.triangleIndex * 3];
        unsigned int i1 = mesh.indices[prim.triangleIndex * 3 + 1];
        unsigned int i2 = mesh.indices[prim.triangleIndex * 3 + 2];

        glm::vec3 v0(mesh.vertices[i0 * 3], mesh.vertices[i0 * 3 + 1], mesh.vertices[i0 * 3 + 2]);
        glm::vec3 v1(mesh.vertices[i1 * 3], mesh.vertices[i1 * 3 + 1], mesh.vertices[i1 * 3 + 2]);
        glm::vec3 v2(mesh.vertices[i2 * 3], mesh.vertices[i2 * 3 + 1], mesh.vertices[i2 * 3 + 2]);

        v0 = glm::vec3(g.transform * glm::vec4(v0, 1.0f));
        v1 = glm::vec3(g.transform * glm::vec4(v1, 1.0f));
        v2 = glm::vec3(g.transform * glm::vec4(v2, 1.0f));

        min = glm::min(v0, glm::min(v1, v2));
        max = glm::max(v0, glm::max(v1, v2));

        // Account for inverse transform precision errors
        // Use relative epsilon based on coordinate magnitude
        glm::vec3 center = (min + max) * 0.5f;
        float coordMagnitude = glm::length(center);
        float eps = fmaxf(1e-3f, coordMagnitude * 1e-5f);  // Scale epsilon with distance from origin

        min -= glm::vec3(eps);
        max += glm::vec3(eps);
    }
    //if (prim.triangleIndex < 5) {  // Print first 5 triangles
    //    std::cout << "Triangle " << prim.triangleIndex
    //        << " AABB: min=(" << min.x << "," << min.y << "," << min.z << ") "
    //        << " max=(" << max.x << "," << max.y << "," << max.z << ")\n";
    //}
}

glm::vec3 BVHBuilder::getPrimitiveCenter(
    const Geom& g,
    const std::vector<MeshData>& meshes,
    const BVHPrimitive& prim)
{
    if (prim.type == PRIM_GEOM) {
        return glm::vec3(g.transform * glm::vec4(0, 0, 0, 1));
    }
    else {
        const MeshData& mesh = meshes[g.meshIndex];
        unsigned int i0 = mesh.indices[prim.triangleIndex * 3];
        unsigned int i1 = mesh.indices[prim.triangleIndex * 3 + 1];
        unsigned int i2 = mesh.indices[prim.triangleIndex * 3 + 2];

        glm::vec3 v0(mesh.vertices[i0 * 3], mesh.vertices[i0 * 3 + 1], mesh.vertices[i0 * 3 + 2]);
        glm::vec3 v1(mesh.vertices[i1 * 3], mesh.vertices[i1 * 3 + 1], mesh.vertices[i1 * 3 + 2]);
        glm::vec3 v2(mesh.vertices[i2 * 3], mesh.vertices[i2 * 3 + 1], mesh.vertices[i2 * 3 + 2]);

        glm::vec3 center = (v0 + v1 + v2) / 3.0f;
        return glm::vec3(g.transform * glm::vec4(center, 1.0f));
    }
}