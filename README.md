CUDA Path Tracer
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Calvin Lieu
  * [LinkedIn](www.linkedin.com/in/calvin-lieu-91912927b)
* Tested on: Windows 11, i5-13450HX @ 2.40GHz 16GB, NVIDIA GeForce RTX 5050 Laptop GPU 8GB (Personal)

---
## Overview:

# CUDA Path Tracer

A GPU-accelerated Monte Carlo path tracer built with CUDA, featuring physically-based rendering, advanced material models, and real-time denoising.

## Showcase

|       Master Chief       |
| :----------------------: |
| ![](./img/chief_cover.png) |

|             Dark Knight             |
| :---------------------------------: |
| ![](./img/dark_knight.png) |

|          Goliath            |
| :-------------------------: |
| ![](./img/goliath.png) |

|                 BXDFs                  |
| :------------------------------------: |
|         ![](./img/bsdf.png)          |

|   Master Chief in OG Green   |
| :--------------------------: |
| ![](./img/chief.png) |
---
## Feature List
- Diffuse BSDF, Specular/GGC Microfacet BRDF
- Dielectric BTDF for refraction
- Depth of Field
- Direct Lighting with Next Event Estimation
- Environment Map Sampling (HDR)
- Texture mapping
- Tone Mapping (Reinhard, ACES)
- glTF Loading
- Material Sort
- Antialiasing
- Russian Roulette Path Termination
- BVH Acceleration
- Denoising
- Some glTF material extensions

## Controls

**Camera Movement:**
- `Left Mouse Button`: Rotate camera around look-at point
- `Right Mouse Button`: Zoom in/out (vertical drag)
- `Middle Mouse Button`: Pan look-at point in XZ plane
- `Space`: Reset camera to original position

**Rendering:**
- `S`: Save current image (auto-generates filename with timestamp and sample count)
- `ESC`: Save final image and exit application

**GUI Controls:**
- Material sorting toggle
- Russian Roulette path
- Direct Lighting
- BVH Acceleration
- Denoiser
- Exposure/Gamma Sliders
- Tone Mapping Mode
- Aperture
- Focal Distance

## Scene File Format

Scenes are defined in JSON format with the following structure:

### Camera Configuration
```json
{
    "Background": {
        "TYPE": "skybox",
        "PATH": "../scenes/environments/brown_photostudio_02_4k.hdr"
    },
    "Materials": {
        "default": {
            "TYPE": "Diffuse",
            "RGB": [ 1.0, 1.0, 1.0 ]
        }
    },
    "Camera": {
        "RES": [ 1280, 720 ],
        "FOVY": 35.0,
        "ITERATIONS": 5000,
        "DEPTH": 12,
        "FILE": "helmet_studio",
        "EYE": [ 0.0, 0.4, 1.5 ],
        "LOOKAT": [ 0.0, 0.3, 0.0 ],
        "UP": [ 0.0, 1.0, 0.0 ]
    },
    "Objects": [
        {
            "TYPE": "sphere",
            "MATERIAL": "default",
            "TRANS": [ -2.5, 0.0, 0.0 ],
            "ROTAT": [ 0.0, 0.0, 0.0 ],
            "SCALE": [ 1.0, 1.0, 1.0 ]
        },
        {
            "TYPE": "gltf",
            "FILE": "../scenes/models/blue_chief/scene.gltf",
            "MATERIAL": "white",
            "TRANS": [ 0.0, -12.5, 0.0 ],
            "ROTAT": [ 90.0, 180.0, 60.0 ],
            "SCALE": [ 10.0, 10.0, 10.0 ]
        }
    ]
}
```
---
### Core Path Tracing

#### Physically-Based Materials
![Materials Showcase](img/materials.png)

Multiple BSDF implementations:
- **Lambertian Diffuse**: Cosine-weighted hemisphere sampling for perfectly diffuse surfaces
- **GGX Microfacet**: Cook-Torrance specular BRDF with importance sampling
- **Dielectric Transmission**: Fresnel-based refraction using Schlick's approximation for glass, water, and other transmissive materials with configurable index of refraction

#### Stochastic Antialiasing
![Antialiasing Comparison](img/antialiasing.png)
*Left: No antialiasing | Right: 4x4 stratified sampling*

Implements 4x4 stratified jitter sampling that cycles every 16 iterations. Distributes samples across a grid pattern within each pixel, reducing aliasing artifacts while maintaining unbiased Monte Carlo integration.

### Lighting & Sampling

#### Next Event Estimation (Direct Lighting)
![NEE Comparison](img/nee_comparison.png)
*Left: Path tracing only (500 spp) | Right: With NEE (500 spp)*

Explicit direct lighting using multiple importance sampling. Combines BRDF importance sampling with light source sampling, using the balance heuristic to weight contributions. 

#### Environment Map Importance Sampling
![Environment Lighting](img/environment.png)

HDR environment map lighting with importance sampling via precomputed CDFs. Builds marginal and conditional probability distributions for sampling bright regions of the skybox preferentially.

### Material System

#### PBR Texture Mapping
![Texture Mapping](img/textures.png)

Full glTF 2.0 PBR texture support:
- **Base Color Textures**: Albedo/diffuse color maps
- **Metallic-Roughness Textures**: Packed metallic (B) and roughness (G) channels
- **Normal Mapping**: Tangent-space normal maps with TBN matrix construction
- **Emissive Textures**: Self-illuminating texture maps
- **Occlusion Textures**: Ambient occlusion maps with configurable strength

Texture fetches add overhead but dramatically improve visual quality.

#### glTF 2.0 Mesh Loading
![glTF Model](img/gltf_model.png)

Complete glTF 2.0 scene loader supporting:
- Hierarchical scene graph traversal
- Node transformations (TRS decomposition)
- Triangle mesh primitives with indexed geometry
- Material property mapping
- Texture coordinate interpolation
- Tangent space calculation for normal mapping

Supports glTF extensions:
- `KHR_materials_transmission` - Refractive materials
- `KHR_materials_volume` - Subsurface parameters
- `KHR_materials_ior` - Custom index of refraction

### Performance Analysis //TODO

#### Material Sorting
![Material Sorting Performance](img/material_sorting.png)

Stream compaction by material type using Thrust sorting primitives. Groups rays intersecting the same material contiguously in memory before shading, improving warp coherence.

**Performance Impact:**

## Third-Party Code & Libraries

This project uses the following third-party libraries:

- **[tinygltf](https://github.com/syoyo/tinygltf)** - glTF 2.0 file loading and parsing
- **[stb_image](https://github.com/nothings/stb)** - Image loading for textures and HDR environment maps  
- **[Intel Open Image Denoise](https://www.openimagedenoise.org/)** - AI-accelerated denoising
- **[Dear ImGui](https://github.com/ocornut/imgui)** - Immediate mode GUI for controls
- **[GLM](https://github.com/g-truc/glm)** - OpenGL mathematics library
- **[CUDA Thrust](https://thrust.github.io/)** - Parallel algorithms and primitives
- **[GLFW](https://www.glfw.org/)** - Window and input handling
- **[GLEW](http://glew.sourceforge.net/)** - OpenGL extension loading

### Third-Party Assets

- HDR environment maps from [Poly Haven](https://polyhaven.com/)
- 3D models from [Sketchfab](https://sketchfab.com/) (see individual model credits)

All third-party code and assets are used in accordance with their respective licenses.
