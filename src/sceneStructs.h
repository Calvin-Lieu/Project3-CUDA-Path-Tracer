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
    float* vertices;        
    float* normals;         
    float* texcoords;
    unsigned int* indices;  
    int triangleCount;    
};

struct MeshData {
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> texcoords; 
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

    float roughness;
    float metallic;

    // Texture support
    int baseColorTexture;
    int metallicRoughnessTexture;
    int normalTexture;
    int emissiveTexture;

    float transmission = 1.0f;          // like KHR_materials_transmission
    float thickness = 0.0f;             // volume thickness
    float attenuationDistance = 1e6f;   // volume absorption distance
    glm::vec3 attenuationColor = glm::vec3(1.0f); // absorption tint
};

struct Texture {
    unsigned char* data;  // Device pointer to texture data
    int width;
    int height;
    int channels;
};

struct EnvironmentMap {
    cudaTextureObject_t texture;
    int width;
    int height;
    float* marginalCDF;      
    float* conditionalCDF;  
    float totalLuminance;
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
  glm::vec2 uv;
  int materialId;
  int geomId;
};
