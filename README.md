# PBR Model Viewer

A desktop 3D model viewer built with Python, PyQt5, and OpenGL. Supports OBJ and GLB/glTF formats with physically-based rendering, real-time lighting controls, and HDR environment mapping.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

![PBR Model Viewer](https://mattgraphics.com/pics/pbr_viewer.png)

---

## Features

- **Format Support** — Wavefront OBJ (with MTL materials) and GLB/glTF (with embedded textures and PBR materials)
- **PBR Rendering** — Cook-Torrance BRDF with base color, metallic/roughness, normal, occlusion, and emissive maps
- **Display Modes** — Smooth shaded, flat shaded, wireframe, backface-culled wireframe, and solid + wireframe overlay
- **Real-Time Lighting** — Three-point lighting system with configurable color and brightness for both directional and ambient lights
- **HDR Environment Maps** — Image-based lighting from HDR/EXR environment maps with adjustable intensity and rotation
- **Material Controls** — Live sliders for metallic, roughness, normal strength, AO, and emissive intensity
- **Debug Visualization** — UV checker map overlay, bounding box display, and vertex normal visualization
- **Interactive Camera** — Trackball rotation, pan, zoom, and WASD keyboard navigation
- **Draggable Control Panel** — Floating UI panel with collapsible sections that remembers its position

---

## Installation

### Requirements

- Python 3.8 or later
- OpenGL 3.3+ compatible GPU

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

## Usage

### Launch with a model

```bash
python viewer.py path/to/model.obj
python viewer.py path/to/model.glb
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

- **Open Model** — Browse for and load a new OBJ or GLB file at any time
- **Display Mode** — Switch between shading and wireframe modes
- **Enable Lighting** — Toggle the three-point light system
- **Enable Textures** — Toggle texture rendering (available when model has textures)
- **PBR Rendering** — Toggle physically-based rendering (available for GLB models with PBR materials)
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
├── vbo_renderer.py      # VBO/VAO management for shader rendering
├── shader_manager.py    # GLSL shader compilation and uniforms
├── trackball.py         # Trackball camera rotation
├── shaders/
│   ├── pbr.vert         # PBR vertex shader
│   └── pbr.frag         # PBR fragment shader
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
