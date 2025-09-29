#pragma once


#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    TRIANGLE_MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    // For triangle meshes loaded from glTF
    int meshIndex = -1;  // Index into tinygltf::Model::meshes
};

struct TriangleMeshData {
    float* vertices;        // Device pointer: [x,y,z, x,y,z, ...]
    float* normals;         // Device pointer: [nx,ny,nz, nx,ny,nz, ...]  
    unsigned int* indices;  // Device pointer: [i0,i1,i2, i0,i1,i2, ...]
    int triangleCount;      // Number of triangles (indices.size() / 3)
};

struct MeshData {
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<unsigned int> indices;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    float roughness;      // 0 = perfect smooth, 1 = very rough
    float metallic;       // 0 = dielectric, 1 = metallic
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    float prevBsdfPdf;
    unsigned char prevWasDelta;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  int geomId;
};
