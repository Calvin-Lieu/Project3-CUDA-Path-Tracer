#pragma once

#include "sceneStructs.h"
#include "tiny_gltf.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfPath);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<MeshData> meshes;
    std::vector<std::pair<std::vector<unsigned char>, Texture>> textures;
    std::string environmentMapPath;
    RenderState state;
};

