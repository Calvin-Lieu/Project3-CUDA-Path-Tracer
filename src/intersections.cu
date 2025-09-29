#include "intersections.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/intersect.hpp>

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
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

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

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

__device__ float singleTriangleIntersectionTest(
    const Geom& geom,
    const TriangleMeshData& mesh,
    int triangleIndex,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // Transform ray to object space
    glm::vec3 ro = glm::vec3(geom.inverseTransform * glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(glm::vec3(geom.inverseTransform * glm::vec4(r.direction, 0.0f)));

    const float EPS = 1e-7f;

    const unsigned int i0 = mesh.indices[triangleIndex * 3 + 0];
    const unsigned int i1 = mesh.indices[triangleIndex * 3 + 1];
    const unsigned int i2 = mesh.indices[triangleIndex * 3 + 2];

    const glm::vec3 v0(mesh.vertices[i0 * 3], mesh.vertices[i0 * 3 + 1], mesh.vertices[i0 * 3 + 2]);
    const glm::vec3 v1(mesh.vertices[i1 * 3], mesh.vertices[i1 * 3 + 1], mesh.vertices[i1 * 3 + 2]);
    const glm::vec3 v2(mesh.vertices[i2 * 3], mesh.vertices[i2 * 3 + 1], mesh.vertices[i2 * 3 + 2]);

    // Moller-Trumbore
    const glm::vec3 e1 = v1 - v0;
    const glm::vec3 e2 = v2 - v0;
    const glm::vec3 pvec = glm::cross(rd, e2);
    const float det = glm::dot(e1, pvec);

    if (fabsf(det) < EPS) return -1.0f;

    const float invDet = 1.0f / det;
    const glm::vec3 tvec = ro - v0;
    const float u = glm::dot(tvec, pvec) * invDet;

    if (u < 0.0f || u > 1.0f) return -1.0f;

    const glm::vec3 qvec = glm::cross(tvec, e1);
    const float v = glm::dot(rd, qvec) * invDet;

    if (v < 0.0f || (u + v) > 1.0f) return -1.0f;

    const float t = glm::dot(e2, qvec) * invDet;

    if (t <= EPS) return -1.0f;

    // Interpolate normals
    glm::vec3 objNormal;
    if (mesh.normals) {
        const glm::vec3 n0(mesh.normals[i0 * 3], mesh.normals[i0 * 3 + 1], mesh.normals[i0 * 3 + 2]);
        const glm::vec3 n1(mesh.normals[i1 * 3], mesh.normals[i1 * 3 + 1], mesh.normals[i1 * 3 + 2]);
        const glm::vec3 n2(mesh.normals[i2 * 3], mesh.normals[i2 * 3 + 1], mesh.normals[i2 * 3 + 2]);
        const float w = 1.0f - u - v;
        objNormal = glm::normalize(w * n0 + u * n1 + v * n2);
    }
    else {
        objNormal = glm::normalize(glm::cross(e1, e2));
    }

    // Transform back to world space
    const glm::vec3 Pobj = ro + rd * t;
    intersectionPoint = glm::vec3(geom.transform * glm::vec4(Pobj, 1.0f));
    normal = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(objNormal, 0.0f)));

    outside = glm::dot(r.direction, normal) < 0.0f;
    if (!outside) normal = -normal;

    return glm::length(intersectionPoint - r.origin);
}

__device__ float meshIntersectionTest(
    const Geom& geom,
    const TriangleMeshData& mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // Ray to object space
    const glm::vec3 ro = glm::vec3(geom.inverseTransform * glm::vec4(r.origin, 1.0f));
    const glm::vec3 rd = glm::normalize(glm::vec3(geom.inverseTransform * glm::vec4(r.direction, 0.0f)));

    const float EPS = 1e-7f;
    float tClosest = 1e30f;
    glm::vec3 bestNormal(0.0f);
    bool hit = false;

    for (int i = 0; i < mesh.triangleCount; ++i) {
        const unsigned int i0 = mesh.indices[i * 3 + 0];
        const unsigned int i1 = mesh.indices[i * 3 + 1];
        const unsigned int i2 = mesh.indices[i * 3 + 2];

        const glm::vec3 v0(mesh.vertices[i0 * 3 + 0], mesh.vertices[i0 * 3 + 1], mesh.vertices[i0 * 3 + 2]);
        const glm::vec3 v1(mesh.vertices[i1 * 3 + 0], mesh.vertices[i1 * 3 + 1], mesh.vertices[i1 * 3 + 2]);
        const glm::vec3 v2(mesh.vertices[i2 * 3 + 0], mesh.vertices[i2 * 3 + 1], mesh.vertices[i2 * 3 + 2]);

        const glm::vec3 e1 = v1 - v0;
        const glm::vec3 e2 = v2 - v0;

        // Moller-Trumbore
        const glm::vec3 pvec = glm::cross(rd, e2);
        const float det = glm::dot(e1, pvec);
        if (fabsf(det) < EPS) continue;
        const float invDet = 1.0f / det;

        const glm::vec3 tvec = ro - v0;
        const float u = glm::dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f) continue;

        const glm::vec3 qvec = glm::cross(tvec, e1);
        const float v = glm::dot(rd, qvec) * invDet;
        if (v < 0.0f || (u + v) > 1.0f) continue;

        const float t = glm::dot(e2, qvec) * invDet;
        if (t <= EPS) continue;

        if (t < tClosest) {
            tClosest = t;
            hit = true;

            // Interpolate normals if available; otherwise face normal
            if (mesh.normals) {
                const glm::vec3 n0(mesh.normals[i0 * 3 + 0], mesh.normals[i0 * 3 + 1], mesh.normals[i0 * 3 + 2]);
                const glm::vec3 n1(mesh.normals[i1 * 3 + 0], mesh.normals[i1 * 3 + 1], mesh.normals[i1 * 3 + 2]);
                const glm::vec3 n2(mesh.normals[i2 * 3 + 0], mesh.normals[i2 * 3 + 1], mesh.normals[i2 * 3 + 2]);
                const float w = 1.0f - u - v;
                bestNormal = glm::normalize(w * n0 + u * n1 + v * n2);
            } else {
                bestNormal = glm::normalize(glm::cross(e1, e2));
            }
        }
    }

    if (!hit) return -1.0f;

    // Object to world
    const glm::vec3 Pobj = ro + rd * tClosest;
    intersectionPoint = glm::vec3(geom.transform * glm::vec4(Pobj, 1.0f));
    normal = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(bestNormal, 0.0f)));

    // Orient and set outside
    outside = glm::dot(r.direction, normal) < 0.0f;
    if (!outside) normal = -normal;

    // Return world-space distance like your sphere/box tests
    return glm::length(intersectionPoint - r.origin);
}
