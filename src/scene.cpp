#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <glm/gtx/transform.hpp>

using namespace std;
using json = nlohmann::json;

//Scene::Scene(string filename)
//{
//    cout << "Reading scene from " << filename << " ..." << endl;
//    cout << " " << endl;
//    auto ext = filename.substr(filename.find_last_of('.'));
//    if (ext == ".json")
//    {
//        loadFromJSON(filename);
//        return;
//    }
//    else
//    {
//        cout << "Couldn't read from " << filename << endl;
//        exit(-1);
//    }
//}
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

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfPath);
    if (!ret) {
        std::cerr << "Failed to load glTF: " << err << std::endl;
        exit(-1);
    }

    // Process materials first
    for (const auto& mat : model.materials) {
        Material newMat{};

        // Get base color
        if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4) {
            newMat.color = glm::vec3(
                mat.pbrMetallicRoughness.baseColorFactor[0],
                mat.pbrMetallicRoughness.baseColorFactor[1],
                mat.pbrMetallicRoughness.baseColorFactor[2]
            );
        }
        else {
            newMat.color = glm::vec3(0.8f);
        }

        newMat.metallic = mat.pbrMetallicRoughness.metallicFactor;
        newMat.roughness = mat.pbrMetallicRoughness.roughnessFactor;

        materials.push_back(newMat);
    }

    // Process scene nodes
    const tinygltf::Scene& scene = model.scenes[model.defaultScene];
    for (int nodeIndex : scene.nodes) {
        processGLTFNode(model, nodeIndex, glm::mat4(1.0f));
    }
}

void Scene::processGLTFNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parentTransform) {
    const tinygltf::Node& node = model.nodes[nodeIndex];

    // Calculate node transform
    glm::mat4 nodeTransform = parentTransform;

    if (node.matrix.size() == 16) {
        // Node has a transformation matrix
        glm::mat4 m;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[i][j] = static_cast<float>(node.matrix[j * 4 + i]);
            }
        }
        nodeTransform = parentTransform * m;
    }
    else {
        // Build transform from TRS
        glm::vec3 translation(0.0f);
        glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale(1.0f);

        if (node.translation.size() == 3) {
            translation = glm::vec3(
                static_cast<float>(node.translation[0]),
                static_cast<float>(node.translation[1]),
                static_cast<float>(node.translation[2])
            );
        }
        if (node.rotation.size() == 4) {
            rotation = glm::quat(
                static_cast<float>(node.rotation[3]),
                static_cast<float>(node.rotation[0]),
                static_cast<float>(node.rotation[1]),
                static_cast<float>(node.rotation[2])
            );
        }
        if (node.scale.size() == 3) {
            scale = glm::vec3(
                static_cast<float>(node.scale[0]),
                static_cast<float>(node.scale[1]),
                static_cast<float>(node.scale[2])
            );
        }

        glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
        glm::mat4 R = glm::mat4_cast(rotation);
        glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
        nodeTransform = parentTransform * T * R * S;
    }

    // Process mesh if present
    if (node.mesh >= 0) {
        processGLTFMesh(model, model.meshes[node.mesh], nodeTransform);
    }

    // Process children
    for (int childIndex : node.children) {
        processGLTFNode(model, childIndex, nodeTransform);
    }
}

void Scene::processGLTFMesh(const tinygltf::Model& model, const tinygltf::Mesh& mesh, const glm::mat4& transform) {
    for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx) {
        const auto& primitive = mesh.primitives[primIdx];

        if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
            continue; // Skip non-triangle primitives
        }

        // Extract vertex data
        std::vector<float> vertices;
        std::vector<float> normals;
        std::vector<unsigned int> indices;

        // Get position data
        auto posIt = primitive.attributes.find("POSITION");
        if (posIt != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[posIt->second];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

            const float* data = reinterpret_cast<const float*>(
                &buffer.data[bufferView.byteOffset + accessor.byteOffset]
                );

            vertices.resize(accessor.count * 3);
            for (size_t i = 0; i < accessor.count; ++i) {
                vertices[i * 3] = data[i * 3];
                vertices[i * 3 + 1] = data[i * 3 + 1];
                vertices[i * 3 + 2] = data[i * 3 + 2];
            }
        }

        // Get normal data
        auto normalIt = primitive.attributes.find("NORMAL");
        if (normalIt != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[normalIt->second];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

            const float* data = reinterpret_cast<const float*>(
                &buffer.data[bufferView.byteOffset + accessor.byteOffset]
                );

            normals.resize(accessor.count * 3);
            for (size_t i = 0; i < accessor.count; ++i) {
                normals[i * 3] = data[i * 3];
                normals[i * 3 + 1] = data[i * 3 + 1];
                normals[i * 3 + 2] = data[i * 3 + 2];
            }
        }
        else {
            // Compute face normals if not provided
            normals.resize(vertices.size());
            for (size_t i = 0; i < vertices.size(); i += 9) { // Every 3 vertices
                glm::vec3 v0(vertices[i], vertices[i + 1], vertices[i + 2]);
                glm::vec3 v1(vertices[i + 3], vertices[i + 4], vertices[i + 5]);
                glm::vec3 v2(vertices[i + 6], vertices[i + 7], vertices[i + 8]);

                glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

                // Set the same normal for all three vertices of this triangle
                for (int j = 0; j < 3; ++j) {
                    normals[i + j * 3] = normal.x;
                    normals[i + j * 3 + 1] = normal.y;
                    normals[i + j * 3 + 2] = normal.z;
                }
            }
        }

        // Get indices
        if (primitive.indices >= 0) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

            indices.resize(accessor.count);

            if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                const unsigned short* data = reinterpret_cast<const unsigned short*>(
                    &buffer.data[bufferView.byteOffset + accessor.byteOffset]
                    );
                for (size_t i = 0; i < accessor.count; ++i) {
                    indices[i] = static_cast<unsigned int>(data[i]);
                }
            }
            else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                const unsigned int* data = reinterpret_cast<const unsigned int*>(
                    &buffer.data[bufferView.byteOffset + accessor.byteOffset]
                    );
                for (size_t i = 0; i < accessor.count; ++i) {
                    indices[i] = data[i];
                }
            }
            else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                const unsigned char* data = reinterpret_cast<const unsigned char*>(
                    &buffer.data[bufferView.byteOffset + accessor.byteOffset]
                    );
                for (size_t i = 0; i < accessor.count; ++i) {
                    indices[i] = static_cast<unsigned int>(data[i]);
                }
            }
        }

        // Create geometry for this mesh
        Geom newGeom;
        newGeom.type = TRIANGLE_MESH;
        newGeom.materialid = (primitive.material >= 0) ? primitive.material : 0;
        newGeom.meshIndex = meshes.size(); // Index into our mesh array

        // Set transform
        newGeom.transform = transform;
        newGeom.inverseTransform = glm::inverse(transform);
        newGeom.invTranspose = glm::inverseTranspose(transform);

        geoms.push_back(newGeom);

        // Store mesh data for later GPU upload
        MeshData meshData;
        meshData.vertices = vertices;
        meshData.normals = normals;
        meshData.indices = indices;

        meshes.push_back(meshData);
    }
}





void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            if (p.contains("IOR")) {
                newMaterial.indexOfRefraction = p["IOR"];
            }
            else {
                newMaterial.indexOfRefraction = 1.5f; // Default glass IOR
            }
            if (p.contains("ROUGHNESS")) {
                newMaterial.roughness = p["ROUGHNESS"];
            }
            std::cout << "Loaded refractive material '" << name << "': "
                << "hasRefractive=" << newMaterial.hasRefractive
                << ", IOR=" << newMaterial.indexOfRefraction << "\n";
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
