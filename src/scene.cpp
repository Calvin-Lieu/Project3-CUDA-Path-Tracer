#include "scene.h"
#include "loader.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    auto ext = filename.substr(filename.find_last_of('.'));

    if (ext == ".json") {
        loadFromJSON(filename);
    }
    else if (ext == ".gltf" || ext == ".glb") {
        loadFromGLTF(filename);
    }
    else {
        cout << "Unsupported file format: " << ext << endl;
        exit(-1);
    }
}

void Scene::loadFromGLTF(const std::string& gltfPath) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    if (!loader.LoadASCIIFromFile(&model, &err, &warn, gltfPath)) {
        std::cerr << "Failed to load glTF: " << err << std::endl;
        exit(-1);
    }

    // Load materials
    MaterialLoader::loadFromGLTF(model, materials, textures);

    // Load scene graph
    const tinygltf::Scene& scene = model.scenes[model.defaultScene];
    for (int nodeIndex : scene.nodes) {
        GeometryLoader::processGLTFNode(model, nodeIndex,
            glm::mat4(1.0f),
            geoms, meshes, 0);
    }

    cout << "Loaded " << geoms.size() << " geometries, "
        << meshes.size() << " meshes" << endl;

    // Default camera setup
    CameraLoader::setDefault(state.camera, state);
}

void Scene::loadFromJSON(const std::string& jsonName) {
    std::ifstream f(jsonName);
    json data = json::parse(f);

    // Load materials
    std::unordered_map<std::string, uint32_t> MatNameToID;
    MaterialLoader::loadFromJSON(data["Materials"], materials, MatNameToID);

    // Load objects (cubes, spheres, GLTFs)
    GeometryLoader::loadFromJSON(data["Objects"], geoms, meshes,
        MatNameToID, materials, textures);

    // Load camera
    CameraLoader::loadFromJSON(data["Camera"], state.camera, state);

    // Load environment map
    if (data.contains("Background")) {
        const auto& backgroundData = data["Background"];
        if (backgroundData["TYPE"] == "skybox" && backgroundData.contains("PATH")) {
            environmentMapPath = backgroundData["PATH"];
            cout << "Environment map path set to: " << environmentMapPath << "\n";
        }
    }
}
