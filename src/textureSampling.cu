#include "textureSampling.h"

// RGB Sampler (no alpha)
__device__ glm::vec3 sampleTexture3(const Texture& tex, float u, float v)
{
    if (!tex.data) return glm::vec3(1.0f);

    // Wrap UVs into [0,1]
    u = u - floorf(u);
    v = v - floorf(v);

    // Convert to pixel coords
    float x = u * (tex.width - 1);
    float y = v * (tex.height - 1);

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = (x0 + 1) % tex.width;
    int y1 = (y0 + 1) % tex.height;

    float fx = x - x0;
    float fy = y - y0;

    // Helper
    auto getPixel = [&](int px, int py) -> glm::vec3 {
        int idx = (py * tex.width + px) * tex.channels;
        if (tex.channels >= 3) {
            return glm::vec3(
                tex.data[idx] / 255.0f,
                tex.data[idx + 1] / 255.0f,
                tex.data[idx + 2] / 255.0f
            );
        }
        else if (tex.channels == 1) {
            float g = tex.data[idx] / 255.0f;
            return glm::vec3(g);
        }
        return glm::vec3(1.0f);
        };

    glm::vec3 c00 = getPixel(x0, y0);
    glm::vec3 c10 = getPixel(x1, y0);
    glm::vec3 c01 = getPixel(x0, y1);
    glm::vec3 c11 = getPixel(x1, y1);

    glm::vec3 c0 = c00 * (1.0f - fx) + c10 * fx;
    glm::vec3 c1 = c01 * (1.0f - fx) + c11 * fx;
    return c0 * (1.0f - fy) + c1 * fy;
}

// RGBA Sampler (with alpha)
__device__ glm::vec4 sampleTexture4(const Texture& tex, const glm::vec2& uv)
{
    if (!tex.data) return glm::vec4(1.0f);

    float u = uv.x - floorf(uv.x);
    float v = uv.y - floorf(uv.y);

    float x = u * (tex.width - 1);
    float y = v * (tex.height - 1);

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = (x0 + 1) % tex.width;
    int y1 = (y0 + 1) % tex.height;

    float fx = x - x0;
    float fy = y - y0;

    auto getPixel = [&](int px, int py) -> glm::vec4 {
        int idx = (py * tex.width + px) * tex.channels;
        float r = tex.data[idx] / 255.0f;
        float g = (tex.channels > 1) ? tex.data[idx + 1] / 255.0f : r;
        float b = (tex.channels > 2) ? tex.data[idx + 2] / 255.0f : r;
        float a = (tex.channels > 3) ? tex.data[idx + 3] / 255.0f : 1.0f;
        return glm::vec4(r, g, b, a);
        };

    glm::vec4 c00 = getPixel(x0, y0);
    glm::vec4 c10 = getPixel(x1, y0);
    glm::vec4 c01 = getPixel(x0, y1);
    glm::vec4 c11 = getPixel(x1, y1);

    glm::vec4 c0 = c00 * (1.0f - fx) + c10 * fx;
    glm::vec4 c1 = c01 * (1.0f - fx) + c11 * fx;
    return c0 * (1.0f - fy) + c1 * fy;
}

// Metallic/Roughness/Occlusion Sampler
// (ORM: R=Occlusion, G=Roughness, B=Metallic)
__device__ void sampleMetallicRoughness(
    const Material& m,
    const Texture* textures,
    const glm::vec2& uv,
    float& outMetallic,
    float& outRoughness,
    float& outOcclusion)
{
    outMetallic = m.metallic;
    outRoughness = m.roughness;
    outOcclusion = 1.0f;

    if (m.metallicRoughnessTexture >= 0) {
        glm::vec4 orm = sampleTexture4(textures[m.metallicRoughnessTexture], uv);
        outOcclusion = orm.r;
        outRoughness = orm.g;
        outMetallic = orm.b;
    }
}
