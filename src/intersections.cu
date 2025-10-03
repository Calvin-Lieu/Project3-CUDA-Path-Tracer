#include "intersections.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/intersect.hpp>

// AABB intersection using slab method
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // Transform ray into object space
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;

    // Test each axis slab
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

// Analytic sphere intersection
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    constexpr float radius = 0.5f;

    // Transform to object space
    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    // Quadratic formula coefficients
    float a = glm::dot(rd, rd);
    float b = 2.0f * glm::dot(ro, rd);
    float c = glm::dot(ro, ro) - radius * radius;

    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) {
        return -1.0f;
    }

    float sqrtDiscriminant = sqrtf(discriminant);
    float t1 = (-b - sqrtDiscriminant) / (2.0f * a);
    float t2 = (-b + sqrtDiscriminant) / (2.0f * a);

    // Choose nearest positive intersection
    float t = (t1 > 0.0f) ? t1 : (t2 > 0.0f ? t2 : -1.0f);
    if (t < 0.0f) {
        return -1.0f;
    }

    outside = (t1 > 0.0f);

    glm::vec3 objSpaceIntersection = ro + t * rd;

    // Transform back to world space
    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objSpaceIntersection, 1.0f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objSpaceIntersection, 0.0f)));

    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

// AABB test for BVH traversal
__device__ bool intersectAABB(const Ray& ray, const glm::vec3& aabbMin, const glm::vec3& aabbMax) {
    glm::vec3 invDir = 1.0f / ray.direction;
    glm::vec3 t0 = (aabbMin - ray.origin) * invDir;
    glm::vec3 t1 = (aabbMax - ray.origin) * invDir;

    glm::vec3 tmin = glm::min(t0, t1);
    glm::vec3 tmax = glm::max(t0, t1);

    float tNear = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
    float tFar = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

    return tNear <= tFar && tFar > 0.0f;
}

// Moller-Trumbore triangle intersection with attribute interpolation
__device__ bool intersectTriangle(
    const glm::vec3& ro, const glm::vec3& rd,
    const TriangleMeshData& mesh, int triIdx,
    float& t_out,
    glm::vec3& normal_out,
    glm::vec2& uv_out,
    glm::vec4& tangent_out)
{
    const float EPS = 1e-7f;

    const unsigned int i0 = mesh.indices[triIdx * 3 + 0];
    const unsigned int i1 = mesh.indices[triIdx * 3 + 1];
    const unsigned int i2 = mesh.indices[triIdx * 3 + 2];

    const glm::vec3 v0(mesh.vertices[i0 * 3 + 0], mesh.vertices[i0 * 3 + 1], mesh.vertices[i0 * 3 + 2]);
    const glm::vec3 v1(mesh.vertices[i1 * 3 + 0], mesh.vertices[i1 * 3 + 1], mesh.vertices[i1 * 3 + 2]);
    const glm::vec3 v2(mesh.vertices[i2 * 3 + 0], mesh.vertices[i2 * 3 + 1], mesh.vertices[i2 * 3 + 2]);

    // Moller-Trumbore
    const glm::vec3 e1 = v1 - v0;
    const glm::vec3 e2 = v2 - v0;
    const glm::vec3 pvec = glm::cross(rd, e2);
    const float det = glm::dot(e1, pvec);
    if (fabsf(det) < EPS) return false;

    const float invDet = 1.0f / det;
    const glm::vec3 tvec = ro - v0;
    const float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    const glm::vec3 qvec = glm::cross(tvec, e1);
    const float v = glm::dot(rd, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) return false;

    const float t = glm::dot(e2, qvec) * invDet;
    if (t <= EPS) return false;

    const float w = 1.0f - u - v;

    // Barycentric interpolation of normals
    if (mesh.normals) {
        glm::vec3 n0(mesh.normals[i0 * 3 + 0], mesh.normals[i0 * 3 + 1], mesh.normals[i0 * 3 + 2]);
        glm::vec3 n1(mesh.normals[i1 * 3 + 0], mesh.normals[i1 * 3 + 1], mesh.normals[i1 * 3 + 2]);
        glm::vec3 n2(mesh.normals[i2 * 3 + 0], mesh.normals[i2 * 3 + 1], mesh.normals[i2 * 3 + 2]);
        normal_out = glm::normalize(w * n0 + u * n1 + v * n2);
    }
    else {
        normal_out = glm::normalize(glm::cross(e1, e2));
    }

    // Barycentric interpolation of UVs
    if (mesh.texcoords) {
        glm::vec2 uv0(mesh.texcoords[i0 * 2], mesh.texcoords[i0 * 2 + 1]);
        glm::vec2 uv1(mesh.texcoords[i1 * 2], mesh.texcoords[i1 * 2 + 1]);
        glm::vec2 uv2(mesh.texcoords[i2 * 2], mesh.texcoords[i2 * 2 + 1]);
        uv_out = w * uv0 + u * uv1 + v * uv2;
    }
    else {
        uv_out = glm::vec2(0.0f);
    }

    // Barycentric interpolation of tangents
    if (mesh.tangents) {
        glm::vec4 t0(mesh.tangents[i0 * 4 + 0], mesh.tangents[i0 * 4 + 1], mesh.tangents[i0 * 4 + 2], mesh.tangents[i0 * 4 + 3]);
        glm::vec4 t1(mesh.tangents[i1 * 4 + 0], mesh.tangents[i1 * 4 + 1], mesh.tangents[i1 * 4 + 2], mesh.tangents[i1 * 4 + 3]);
        glm::vec4 t2(mesh.tangents[i2 * 4 + 0], mesh.tangents[i2 * 4 + 1], mesh.tangents[i2 * 4 + 2], mesh.tangents[i2 * 4 + 3]);
        tangent_out = glm::normalize(w * t0 + u * t1 + v * t2);
    }
    else {
        tangent_out = glm::vec4(1, 0, 0, 1);
    }

    t_out = t;
    return true;
}

// Naive mesh intersection (tests all triangles)
__device__ float meshIntersectionTest(
    const Geom& geom,
    const TriangleMeshData& mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    glm::vec2& uv,
    glm::vec4& tangent)
{
    const glm::vec3 ro = glm::vec3(geom.inverseTransform * glm::vec4(r.origin, 1.0f));
    const glm::vec3 rd = glm::normalize(glm::vec3(geom.inverseTransform * glm::vec4(r.direction, 0.0f)));

    float tClosest = 1e30f;
    glm::vec3 bestNormal(0.0f);
    glm::vec2 bestUV(0.0f);
    glm::vec4 bestTangent(1, 0, 0, 1);
    bool hit = false;

    for (int i = 0; i < mesh.triangleCount; ++i) {
        float t_tmp;
        glm::vec3 n_tmp;
        glm::vec2 uv_tmp;
        glm::vec4 tangent_tmp;

        if (intersectTriangle(ro, rd, mesh, i, t_tmp, n_tmp, uv_tmp, tangent_tmp)) {
            if (t_tmp < tClosest) {
                tClosest = t_tmp;
                bestNormal = n_tmp;
                bestUV = uv_tmp;
                bestTangent = tangent_tmp;
                hit = true;
            }
        }
    }

    if (!hit) return -1.0f;

    // Transform from object to world space
    const glm::vec3 Pobj = ro + rd * tClosest;
    intersectionPoint = glm::vec3(geom.transform * glm::vec4(Pobj, 1.0f));
    normal = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(bestNormal, 0.0f)));

    // Transform tangent to world space and orthogonalize
    glm::vec3 T_world = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(glm::vec3(bestTangent), 0.0f)));
    T_world = glm::normalize(T_world - normal * glm::dot(normal, T_world));
    tangent = glm::vec4(T_world, bestTangent.w);

    uv = bestUV;

    // Determine front/back facing
    outside = glm::dot(r.direction, normal) < 0.0f;
    if (!outside) normal = -normal;

    return glm::length(intersectionPoint - r.origin);
}

// Single triangle intersection in world space (used by BVH)
__device__ float singleTriangleIntersectionTestWorldSpace(
    const Geom& geom,
    const TriangleMeshData& mesh,
    int triangleIndex,
    const Ray& worldRay,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    glm::vec2& uv,
    glm::vec4& tangent)
{
    const float EPS = 1e-7f;

    const unsigned int i0 = mesh.indices[triangleIndex * 3];
    const unsigned int i1 = mesh.indices[triangleIndex * 3 + 1];
    const unsigned int i2 = mesh.indices[triangleIndex * 3 + 2];

    // Load object-space vertices
    glm::vec3 v0_obj(mesh.vertices[i0 * 3], mesh.vertices[i0 * 3 + 1], mesh.vertices[i0 * 3 + 2]);
    glm::vec3 v1_obj(mesh.vertices[i1 * 3], mesh.vertices[i1 * 3 + 1], mesh.vertices[i1 * 3 + 2]);
    glm::vec3 v2_obj(mesh.vertices[i2 * 3], mesh.vertices[i2 * 3 + 1], mesh.vertices[i2 * 3 + 2]);

    // Transform to world space
    glm::vec3 v0 = glm::vec3(geom.transform * glm::vec4(v0_obj, 1.0f));
    glm::vec3 v1 = glm::vec3(geom.transform * glm::vec4(v1_obj, 1.0f));
    glm::vec3 v2 = glm::vec3(geom.transform * glm::vec4(v2_obj, 1.0f));

    // Moller-Trumbore in world space
    const glm::vec3 e1 = v1 - v0;
    const glm::vec3 e2 = v2 - v0;
    const glm::vec3 pvec = glm::cross(worldRay.direction, e2);
    const float det = glm::dot(e1, pvec);

    if (fabsf(det) < EPS) return -1.0f;

    const float invDet = 1.0f / det;
    const glm::vec3 tvec = worldRay.origin - v0;
    const float u = glm::dot(tvec, pvec) * invDet;

    if (u < 0.0f || u > 1.0f) return -1.0f;

    const glm::vec3 qvec = glm::cross(tvec, e1);
    const float v = glm::dot(worldRay.direction, qvec) * invDet;

    if (v < 0.0f || (u + v) > 1.0f) return -1.0f;

    const float t = glm::dot(e2, qvec) * invDet;

    if (t <= EPS) return -1.0f;

    const float w = 1.0f - u - v;

    intersectionPoint = worldRay.origin + t * worldRay.direction;

    // Interpolate and transform normal
    if (mesh.normals) {
        glm::vec3 n0(mesh.normals[i0 * 3], mesh.normals[i0 * 3 + 1], mesh.normals[i0 * 3 + 2]);
        glm::vec3 n1(mesh.normals[i1 * 3], mesh.normals[i1 * 3 + 1], mesh.normals[i1 * 3 + 2]);
        glm::vec3 n2(mesh.normals[i2 * 3], mesh.normals[i2 * 3 + 1], mesh.normals[i2 * 3 + 2]);
        glm::vec3 n_obj = glm::normalize(w * n0 + u * n1 + v * n2);
        normal = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(n_obj, 0.0f)));
    }
    else {
        normal = glm::normalize(glm::cross(e1, e2));
    }

    // Interpolate UVs
    if (mesh.texcoords) {
        glm::vec2 uv0(mesh.texcoords[i0 * 2], mesh.texcoords[i0 * 2 + 1]);
        glm::vec2 uv1(mesh.texcoords[i1 * 2], mesh.texcoords[i1 * 2 + 1]);
        glm::vec2 uv2(mesh.texcoords[i2 * 2], mesh.texcoords[i2 * 2 + 1]);
        uv = w * uv0 + u * uv1 + v * uv2;
    }
    else {
        uv = glm::vec2(0.0f);
    }

    // Interpolate and transform tangents
    if (mesh.tangents) {
        glm::vec4 t0(mesh.tangents[i0 * 4], mesh.tangents[i0 * 4 + 1], mesh.tangents[i0 * 4 + 2], mesh.tangents[i0 * 4 + 3]);
        glm::vec4 t1(mesh.tangents[i1 * 4], mesh.tangents[i1 * 4 + 1], mesh.tangents[i1 * 4 + 2], mesh.tangents[i1 * 4 + 3]);
        glm::vec4 t2(mesh.tangents[i2 * 4], mesh.tangents[i2 * 4 + 1], mesh.tangents[i2 * 4 + 2], mesh.tangents[i2 * 4 + 3]);
        glm::vec4 t_obj = w * t0 + u * t1 + v * t2;
        glm::vec3 T_world = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(glm::vec3(t_obj), 0.0f)));
        T_world = glm::normalize(T_world - normal * glm::dot(normal, T_world));
        tangent = glm::vec4(T_world, t_obj.w);
    }
    else {
        tangent = glm::vec4(1, 0, 0, 1);
    }

    outside = glm::dot(worldRay.direction, normal) < 0.0f;
    if (!outside) normal = -normal;

    return t;
}