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
    int meshIndex = -1; 
};

struct TriangleMeshData {
    float* vertices;        
    float* normals;         
    float* texcoords;
    float* tangents;
    unsigned int* indices;  
    int triangleCount;    
};

struct MeshData {
    std::vector<float> vertices;   
    std::vector<float> normals;    
    std::vector<float> texcoords; 
    std::vector<float> tangents; 
    std::vector<unsigned int> indices;
};

struct Material
{
    // Base properties
    glm::vec3 color = glm::vec3(1.0f);   
    struct
    {
        float exponent = 0.0f;
        glm::vec3 color = glm::vec3(0.0f);
    } specular;

    float hasReflective = 0.0f;
    float hasRefractive = 0.0f;
    float indexOfRefraction = 1.5f;
    float emittance = 0.0f;

    // PBR factors
    float roughness = 1.0f;          
    float metallic = 0.0f;        

    int baseColorTexture = -1;
    int metallicRoughnessTexture = -1;
    int normalTexture = -1;
    int emissiveTexture = -1; 
	int occlusionTexture = -1;

    float occlusionStrength = 1.0f;
    float transmission = 0.0f;        
    float thickness = 0.0f;           
    float attenuationDistance = 1e6f;  
    glm::vec3 attenuationColor = glm::vec3(1.0f);

    glm::vec3 emissiveFactor = glm::vec3(0.0f);

    float alphaCutoff = 0.5f; // default for MASK mode
    int alphaMode = 0;        // 0=OPAQUE, 1=MASK, 2=BLEND
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
  glm::vec4 tangent;
  int materialId;
  int geomId;
};
