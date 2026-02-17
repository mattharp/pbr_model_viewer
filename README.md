# PBR Model Viewer

A desktop 3D model viewer built with Python, PyQt5, and OpenGL. Supports FBX, GLB/glTF, and OBJ formats with physically-based rendering, real-time lighting controls, and HDR environment mapping.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

![PBR Model Viewer](https://mattgraphics.com/pics/pbr_viewer.png)

---

## Features

- **Format Support** — Wavefront OBJ (with MTL materials), GLB/glTF (with embedded textures and PBR materials), and Autodesk FBX (with PBR materials via ufbx)
- **PBR Rendering** — Cook-Torrance BRDF with base color, metallic/roughness, normal, occlusion, and emissive maps
- **Display Modes** — Smooth shaded, flat shaded, wireframe, backface-culled wireframe, and solid + wireframe overlay
- **Real-Time Lighting** — Three-point lighting system with configurable color and brightness for both directional and ambient lights
- **HDR Environment Maps** — Image-based lighting from HDR/EXR environment maps with adjustable intensity and rotation
- **Material Controls** — Live sliders for metallic, roughness, normal strength, AO, and emissive intensity
- **Debug Visualization** — UV checker map overlay, bounding box display, and vertex normal visualization
- **Interactive Camera** — Trackball rotation, pan, zoom, and WASD keyboard navigation

---

## Installation

### Requirements

- Python 3.8 or later
- OpenGL 3.3+ compatible GPU
- A C compiler (gcc/MinGW) for building FBX support (optional)

### Setup

```bash
# Clone or download the project
cd modelviewer

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| PyQt5 | Application window and UI |
| PyOpenGL | OpenGL rendering |
| numpy | Math and array operations |
| Pillow | Image loading for OBJ textures |
| opencv-contrib-python | HDR/EXR environment map loading |
| trimesh | GLB/glTF file parsing |

---

## Building FBX Support

FBX loading is handled by [ufbx](https://github.com/ufbx/ufbx), a single-file C library, wrapped with a thin C bridge layer. The bridge compiles into a shared library that `fbx_loader.py` loads at runtime via ctypes. If the library is not present, the viewer still works — FBX support is simply disabled.

### Prerequisites

You need a C compiler. On Windows, [MinGW-w64](https://www.mingw-w64.org/) (gcc) or MSVC will work. On Linux and macOS, gcc or clang should already be available.

### Build Steps

From the `extern/ufbx/` directory:

**Windows (MinGW)**
```bash
gcc -shared -O2 -o ../../bin/ufbx.dll ufbx.c ufbx_bridge.c -lm
```

**Windows (MSVC)**
```bash
cl /LD /O2 ufbx.c ufbx_bridge.c /Fe:../../bin/ufbx.dll
```

**Linux**
```bash
gcc -shared -fPIC -O2 -o ../../bin/ufbx.so ufbx.c ufbx_bridge.c -lm
```

**macOS**
```bash
gcc -shared -fPIC -O2 -o ../../bin/ufbx.dylib ufbx.c ufbx_bridge.c -lm
```

Alternatively, use the provided build scripts:

```bash
# Windows
extern\ufbx\build.bat

# Linux / macOS
./extern/ufbx/build.sh
```

The compiled library is placed in `bin/` where `fbx_loader.py` expects to find it.

---

## Usage

### Launch with a model

```bash
python viewer.py path/to/model.obj
python viewer.py path/to/model.glb
python viewer.py path/to/model.fbx
```

### Launch empty

```bash
python viewer.py
```

Use the **Open Model** button at the top of the control panel to load a file.

### Mouse Controls

| Input | Action |
|---|---|
| Right Click + Drag | Rotate model |
| Middle Click + Drag | Pan camera |
| Mouse Wheel | Zoom in/out |

### Keyboard Controls

| Key | Action |
|---|---|
| W / S | Zoom in / out |
| A / D | Pan left / right |
| Q / E | Pan up / down |
| R | Reset camera |

---

## Control Panel

The floating control panel on the right side of the viewport provides access to all rendering options. Click and drag the header bar to reposition it. Click the arrow button to collapse or expand.

- **Open Model** — Browse for and load a new OBJ, GLB, or FBX file at any time
- **Display Mode** — Switch between shading and wireframe modes
- **Enable Lighting** — Toggle the three-point light system
- **Enable Textures** — Toggle texture rendering (available when model has textures)
- **PBR Rendering** — Toggle physically-based rendering (available for GLB and FBX models with PBR materials)
- **Directional Light** — Color picker and brightness slider for the main light
- **Ambient Light** — Color picker and strength slider for ambient fill
- **Material Properties** — Metallic, roughness, normal strength, AO, and emissive intensity sliders (active in PBR mode)
- **HDR Environment** — Load an HDR or EXR file for image-based lighting, with intensity and rotation controls
- **Debug** — UV checker map, bounding box, and vertex normal overlays

---

## Project Structure

```
modelviewer/
├── viewer.py            # Entry point
├── main_window.py       # Application window and file open dialog
├── gl_widget.py         # OpenGL viewport and rendering pipeline
├── control_panel.py     # Floating UI panel
├── obj_loader.py        # Wavefront OBJ/MTL loader
├── glb_loader.py        # GLB/glTF loader with PBR materials
├── fbx_loader.py        # FBX loader with PBR materials (via ufbx bridge)
├── vbo_renderer.py      # VBO/VAO management for shader rendering
├── shader_manager.py    # GLSL shader compilation and uniforms
├── trackball.py         # Trackball camera rotation
├── shaders/
│   ├── pbr.vert         # PBR vertex shader
│   └── pbr.frag         # PBR fragment shader
├── bin/                 # Compiled native libraries (gitignored)
│   └── ufbx.dll/.so/.dylib
├── extern/
│   └── ufbx/            # ufbx library and bridge source
│       ├── README.md    # Version info and build notes
│       ├── build.bat    # Windows build script
│       ├── build.sh     # Linux/macOS build script
│       ├── ufbx.c       # ufbx library (upstream, unmodified)
│       ├── ufbx.h       # ufbx header (upstream, unmodified)
│       ├── ufbx_bridge.c  # Bridge implementation
│       └── ufbx_bridge.h  # Bridge API header
└── requirements.txt
```

---

## Supported Formats

### OBJ (Wavefront)

- Vertex positions, normals, and texture coordinates
- MTL material files with diffuse textures and colors
- Rendered with fixed-function OpenGL pipeline
- Triangles and n-gon faces

### GLB / glTF

- Full PBR material support (metallic-roughness workflow)
- Embedded textures: base color, metallic/roughness, normal, occlusion, emissive
- Parsed via trimesh with automatic material extraction
- Rendered with custom GLSL shaders when PBR mode is enabled

### FBX (Autodesk)

- Full PBR material support (metallic-roughness workflow)
- Embedded and external textures: base color, metallic/roughness, normal, occlusion, emissive
- Automatic texture path resolution for absolute paths, relative paths, and common subdirectory conventions
- Loaded via [ufbx](https://github.com/ufbx/ufbx) through a lightweight C bridge with ctypes bindings
- Rendered with custom GLSL shaders when PBR mode is enabled
- Requires building the ufbx shared library (see [Building FBX Support](#building-fbx-support))
