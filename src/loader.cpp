#include "loader.h"
#include "utilities.h"
#include <iostream>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_inverse.hpp>

// =======================================================
// Material Loader
// =======================================================
void MaterialLoader::loadFromJSON(const json& materialsData,
    std::vector<Material>& materials,
    std::unordered_map<std::string, uint32_t>& MatNameToID) {
    for (const auto& item : materialsData.items()) {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};

        if (p["TYPE"] == "Diffuse") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
            newMaterial.metallic = p.value("METALLIC", 0.0f);
            newMaterial.roughness = p.value("ROUGHNESS", 0.0f);
        }
        else if (p["TYPE"] == "Refractive") {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
            newMaterial.roughness = p.value("ROUGHNESS", 0.0f);
            newMaterial.transmission = p.value("TRANSMISSION", 1.0f);
            newMaterial.thickness = p.value("THICKNESS", 0.0f);
            newMaterial.attenuationDistance = p.value("ATTENUATION_DISTANCE", 1e6f);

            if (p.contains("ATTENUATION_COLOR")) {
                const auto& ac = p["ATTENUATION_COLOR"];
                newMaterial.attenuationColor = glm::vec3(ac[0], ac[1], ac[2]);
            }
            else {
                newMaterial.attenuationColor = glm::vec3(1.0f);
            }

            std::cout << "Loaded refractive material '" << name << "': "
                << "IOR=" << newMaterial.indexOfRefraction
                << " transmission=" << newMaterial.transmission
                << " thickness=" << newMaterial.thickness
                << " attenDist=" << newMaterial.attenuationDistance
                << " attenColor=(" << newMaterial.attenuationColor.r << ","
                << newMaterial.attenuationColor.g << ","
                << newMaterial.attenuationColor.b << ")\n";
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
}

void MaterialLoader::loadFromGLTF(const tinygltf::Model& model,
    std::vector<Material>& materials,
    std::vector<std::pair<std::vector<unsigned char>, Texture>>& textures) {
    // Load textures
    for (const auto& tex : model.textures) {
        const auto& image = model.images[tex.source];
        Texture newTex;
        newTex.width = image.width;
        newTex.height = image.height;
        newTex.channels = image.component;
        newTex.data = nullptr;
        textures.push_back({ image.image, newTex });
    }

    // Default fallback material
    Material defaultMat{};
    defaultMat.color = glm::vec3(0.7f);
    defaultMat.metallic = 0.5f;
    defaultMat.roughness = 0.5f;
    materials.push_back(defaultMat);

    // Load glTF materials
    for (const auto& mat : model.materials) {
        Material newMat{};
        if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4) {
            newMat.color = glm::vec3(
                mat.pbrMetallicRoughness.baseColorFactor[0],
                mat.pbrMetallicRoughness.baseColorFactor[1],
                mat.pbrMetallicRoughness.baseColorFactor[2]);
        }
        newMat.metallic = mat.pbrMetallicRoughness.metallicFactor;
        newMat.roughness = mat.pbrMetallicRoughness.roughnessFactor;
        newMat.baseColorTexture = mat.pbrMetallicRoughness.baseColorTexture.index;
        materials.push_back(newMat);
    }
}

void MaterialLoader::appendFromGLTF(const tinygltf::Model& model,
    std::vector<Material>& materials,
    std::vector<std::pair<std::vector<unsigned char>, Texture>>& textures,
    size_t textureOffset) {
    // Append textures
    for (const auto& tex : model.textures) {
        const auto& image = model.images[tex.source];
        Texture newTex;
        newTex.width = image.width;
        newTex.height = image.height;
        newTex.channels = image.component;
        newTex.data = nullptr;
        textures.push_back({ image.image, newTex });
    }

    // Append materials
    for (const auto& mat : model.materials) {
        Material newMat{};
        if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4) {
            newMat.color = glm::vec3(
                mat.pbrMetallicRoughness.baseColorFactor[0],
                mat.pbrMetallicRoughness.baseColorFactor[1],
                mat.pbrMetallicRoughness.baseColorFactor[2]);
        }
        newMat.metallic = mat.pbrMetallicRoughness.metallicFactor;
        newMat.roughness = mat.pbrMetallicRoughness.roughnessFactor;
        newMat.baseColorTexture = (mat.pbrMetallicRoughness.baseColorTexture.index >= 0)
            ? mat.pbrMetallicRoughness.baseColorTexture.index + textureOffset
            : -1;
        materials.push_back(newMat);
    }
}
// =======================================================
// Geometry Loader (GLTF)
// =======================================================

void GeometryLoader::loadFromJSON(const json& objectsData,
    std::vector<Geom>& geoms,
    std::vector<MeshData>& meshes,
    const std::unordered_map<std::string, uint32_t>& MatNameToID,
    std::vector<Material>& materials,
    std::vector<std::pair<std::vector<unsigned char>, Texture>>& textures)
{
    for (const auto& p : objectsData) {
        const auto& type = p["TYPE"];

        if (type == "gltf") {
            // Load GLTF file specified
            std::string gltfFile = p["FILE"];
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];

            glm::vec3 translation(trans[0], trans[1], trans[2]);
            glm::vec3 rotation(rotat[0], rotat[1], rotat[2]);
            glm::vec3 scaleVec(scale[0], scale[1], scale[2]);

            glm::mat4 transform = utilityCore::buildTransformationMatrix(
                translation, rotation, scaleVec);

            // Reuse existing loader for GLTF into scene
            GltfLoader::loadFile(gltfFile, geoms, meshes, materials, textures, transform);
            continue;
        }

        Geom newGeom;
        if (type == "cube") newGeom.type = CUBE;
        else newGeom.type = SPHERE;

        newGeom.materialid = MatNameToID.at(p["MATERIAL"]);

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
}

void GeometryLoader::processGLTFNode(const tinygltf::Model& model,
    int nodeIndex,
    const glm::mat4& parentTransform,
    std::vector<Geom>& geoms,
    std::vector<MeshData>& meshes,
    size_t materialOffset) {
    const tinygltf::Node& node = model.nodes[nodeIndex];

    glm::mat4 nodeTransform = parentTransform;

    if (node.matrix.size() == 16) {
        glm::mat4 m;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[i][j] = static_cast<float>(node.matrix[j * 4 + i]);
            }
        }
        nodeTransform = parentTransform * m;
    }
    else {
        glm::vec3 translation(0.0f);
        glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 scale(1.0f);

        if (node.translation.size() == 3) {
            translation = glm::vec3(node.translation[0],
                node.translation[1],
                node.translation[2]);
        }
        if (node.rotation.size() == 4) {
            rotation = glm::quat(node.rotation[3],
                node.rotation[0],
                node.rotation[1],
                node.rotation[2]);
        }
        if (node.scale.size() == 3) {
            scale = glm::vec3(node.scale[0],
                node.scale[1],
                node.scale[2]);
        }

        glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
        glm::mat4 R = glm::mat4_cast(rotation);
        glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
        nodeTransform = parentTransform * T * R * S;
    }

    if (node.mesh >= 0) {
        processGLTFMesh(model, model.meshes[node.mesh],
            nodeTransform, materialOffset, geoms, meshes);
    }

    for (int childIndex : node.children) {
        processGLTFNode(model, childIndex, nodeTransform,
            geoms, meshes, materialOffset);
    }
}

void GeometryLoader::processGLTFMesh(const tinygltf::Model& model,
    const tinygltf::Mesh& mesh,
    const glm::mat4& transform,
    size_t materialOffset,
    std::vector<Geom>& geoms,
    std::vector<MeshData>& meshes) {
    for (const auto& primitive : mesh.primitives) {
        if (primitive.mode != TINYGLTF_MODE_TRIANGLES) continue;

        std::vector<float> vertices, normals, texcoords;
        std::vector<unsigned int> indices;

        // POSITION
        auto posIt = primitive.attributes.find("POSITION");
        if (posIt != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[posIt->second];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            const float* data = reinterpret_cast<const float*>(
                &buffer.data[bufferView.byteOffset + accessor.byteOffset]);

            vertices.resize(accessor.count * 3);
            for (size_t i = 0; i < accessor.count; ++i) {
                vertices[i * 3 + 0] = data[i * 3 + 0];
                vertices[i * 3 + 1] = data[i * 3 + 1];
                vertices[i * 3 + 2] = data[i * 3 + 2];
            }
        }

        // NORMAL
        auto normalIt = primitive.attributes.find("NORMAL");
        if (normalIt != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[normalIt->second];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            const float* data = reinterpret_cast<const float*>(
                &buffer.data[bufferView.byteOffset + accessor.byteOffset]);

            normals.resize(accessor.count * 3);
            for (size_t i = 0; i < accessor.count; ++i) {
                normals[i * 3 + 0] = data[i * 3 + 0];
                normals[i * 3 + 1] = data[i * 3 + 1];
                normals[i * 3 + 2] = data[i * 3 + 2];
            }
        }
        else {
            normals.resize(vertices.size());
            for (size_t i = 0; i < vertices.size(); i += 9) {
                glm::vec3 v0(vertices[i], vertices[i + 1], vertices[i + 2]);
                glm::vec3 v1(vertices[i + 3], vertices[i + 4], vertices[i + 5]);
                glm::vec3 v2(vertices[i + 6], vertices[i + 7], vertices[i + 8]);
                glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                for (int j = 0; j < 3; ++j) {
                    normals[i + j * 3 + 0] = normal.x;
                    normals[i + j * 3 + 1] = normal.y;
                    normals[i + j * 3 + 2] = normal.z;
                }
            }
        }

        // INDICES
        if (primitive.indices >= 0) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

            indices.resize(accessor.count);

            if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                const unsigned short* data =
                    reinterpret_cast<const unsigned short*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; i++) indices[i] = data[i];
            }
            else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                const unsigned int* data =
                    reinterpret_cast<const unsigned int*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
                for (size_t i = 0; i < accessor.count; i++) indices[i] = data[i];
            }
            else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                const unsigned char* data =
                    &buffer.data[bufferView.byteOffset + accessor.byteOffset];
                for (size_t i = 0; i < accessor.count; i++) indices[i] = data[i];
            }
        }

        auto texIt = primitive.attributes.find("TEXCOORD_0");
        if (texIt != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[texIt->second];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            const float* data = reinterpret_cast<const float*>(
                &buffer.data[bufferView.byteOffset + accessor.byteOffset]);

            texcoords.resize(accessor.count * 2);
            for (size_t i = 0; i < accessor.count; ++i) {
                texcoords[i * 2 + 0] = data[i * 2 + 0];
                texcoords[i * 2 + 1] = data[i * 2 + 1];
            }
        }
        else {
            texcoords.resize((vertices.size() / 3) * 2, 0.0f);
        }

        // Create geometry
        Geom newGeom;
        newGeom.type = TRIANGLE_MESH;
        newGeom.materialid = (primitive.material >= 0)
            ? primitive.material + materialOffset : 0;
        newGeom.meshIndex = meshes.size();
        newGeom.transform = transform;
        newGeom.inverseTransform = glm::inverse(transform);
        newGeom.invTranspose = glm::inverseTranspose(transform);
        geoms.push_back(newGeom);

        // Store mesh
        MeshData meshData;
        meshData.vertices = vertices;
        meshData.normals = normals;
        meshData.texcoords = texcoords;
        meshData.indices = indices;
        meshes.push_back(meshData);
    }
}


// =======================================================
// Camera Loader
// =======================================================
void CameraLoader::loadFromJSON(const json& cameraData,
    Camera& camera,
    RenderState& state) {
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

    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void CameraLoader::setDefault(Camera& camera, RenderState& state) {
    camera.resolution = glm::ivec2(800, 800);
    camera.position = glm::vec3(0, 0.5f, 3.0f);
    camera.lookAt = glm::vec3(0, 0.3f, 0);
    camera.up = glm::vec3(0, 1, 0);

    float fovy = 45.0f;
    state.iterations = 5000;
    state.traceDepth = 8;
    state.imageName = "gltf_render";

    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(
        2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

// =======================================================
// Environment Loader
// =======================================================
void EnvironmentLoader::loadFromJSON(const json& backgroundData,
    std::string& environmentMapPath) {
    if (backgroundData["TYPE"] == "skybox" && backgroundData.contains("PATH")) {
        environmentMapPath = backgroundData["PATH"];
        std::cout << "Environment map path set to: " << environmentMapPath << "\n";
    }
}

// =======================================================
// Gltf Loader
// =======================================================
void GltfLoader::loadFile(const std::string& gltfPath,
    std::vector<Geom>& geoms,
    std::vector<MeshData>& meshes,
    std::vector<Material>& materials,
    std::vector<std::pair<std::vector<unsigned char>, Texture>>& textures,
    const glm::mat4& baseTransform)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    if (!loader.LoadASCIIFromFile(&model, &err, &warn, gltfPath)) {
        std::cerr << "Failed to load glTF: " << err << std::endl;
        return;
    }

    // Append materials + textures
    size_t textureOffset = textures.size();
    size_t materialOffset = materials.size();
    MaterialLoader::appendFromGLTF(model, materials, textures,
        textureOffset);

    // Process scene nodes
    const tinygltf::Scene& scene = model.scenes[model.defaultScene];
    for (int nodeIndex : scene.nodes) {
        GeometryLoader::processGLTFNode(model, nodeIndex,
            baseTransform,
            geoms, meshes,
            materialOffset);
    }
}


