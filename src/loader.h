#pragma once

#include "sceneStructs.h"
#include "json.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <tiny_gltf.h>

// Aliases
using json = nlohmann::json;

// Forward declarations
struct RenderState;
struct Camera;
struct MeshData;
struct Geom;
struct Material;

namespace MaterialLoader {
    void loadFromJSON(const json& materialsData,
        std::vector<Material>& materials,
        std::unordered_map<std::string, uint32_t>& MatNameToID);

    void loadFromGLTF(const tinygltf::Model& model,
        std::vector<Material>& materials,
        std::vector<std::pair<std::vector<unsigned char>, Texture>>& textures);

    void appendFromGLTF(const tinygltf::Model& model,
        std::vector<Material>& materials,
        std::vector<std::pair<std::vector<unsigned char>, Texture>>& textures,
        size_t textureOffset);
}

namespace GeometryLoader {
    void loadFromJSON(const json& objectsData,
        std::vector<Geom>& geoms,
        std::vector<MeshData>& meshes,
        const std::unordered_map<std::string, uint32_t>& MatNameToID,
        std::vector<Material>& materials,
        std::vector<std::pair<std::vector<unsigned char>, Texture>>& textures);

    void processGLTFNode(const tinygltf::Model& model, int nodeIndex,
        const glm::mat4& parentTransform,
        std::vector<Geom>& geoms,
        std::vector<MeshData>& meshes,
        size_t materialOffset);

    void processGLTFMesh(const tinygltf::Model& model,
        const tinygltf::Mesh& mesh,
        const glm::mat4& transform,
        size_t materialOffset,
        std::vector<Geom>& geoms,
        std::vector<MeshData>& meshes);
}

namespace CameraLoader {
    void loadFromJSON(const json& cameraData,
        Camera& camera,
        RenderState& state);

    void setDefault(Camera& camera, RenderState& state);
}

namespace EnvironmentLoader {
    void loadFromJSON(const json& backgroundData,
        std::string& environmentMapPath);
}

namespace GltfLoader {
    void loadFile(const std::string& gltfPath,
        std::vector<Geom>& geoms,
        std::vector<MeshData>& meshes,
        std::vector<Material>& materials,
        std::vector<std::pair<std::vector<unsigned char>, Texture>>& textures,
        const glm::mat4& baseTransform);
}
