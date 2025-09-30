#include "textureSampling.h"

__device__ glm::vec3 sampleTexture(const Texture& tex, float u, float v)
{
    if (!tex.data) return glm::vec3(1.0f);

    // Wrap UVs to [0,1]
    u = u - floorf(u);
    v = v - floorf(v);

    // Convert to pixel coordinates
    float x = u * (tex.width - 1);
    float y = v * (tex.height - 1);

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = (x0 + 1) % tex.width;
    int y1 = (y0 + 1) % tex.height;

    float fx = x - x0;
    float fy = y - y0;

    // Get pixel color helper
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
            float gray = tex.data[idx] / 255.0f;
            return glm::vec3(gray);
        }
        return glm::vec3(1.0f);
        };

    // Bilinear interpolation
    glm::vec3 c00 = getPixel(x0, y0);
    glm::vec3 c10 = getPixel(x1, y0);
    glm::vec3 c01 = getPixel(x0, y1);
    glm::vec3 c11 = getPixel(x1, y1);

    glm::vec3 c0 = c00 * (1.0f - fx) + c10 * fx;
    glm::vec3 c1 = c01 * (1.0f - fx) + c11 * fx;
    return c0 * (1.0f - fy) + c1 * fy;
}