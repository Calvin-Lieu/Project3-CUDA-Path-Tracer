#pragma once

#include "sceneStructs.h"
#include "tiny_gltf.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfPath);
    void processGLTFNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parentTransform);
    void processGLTFMesh(const tinygltf::Model& model, const tinygltf::Mesh& mesh, const glm::mat4& transform);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<MeshData> meshes;
    std::vector<std::pair<std::vector<unsigned char>, Texture>> textures;
    RenderState state;
};

