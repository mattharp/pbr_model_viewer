"""
FBX Model Loader Module
Handles loading and rendering of FBX files using ufbx via a C bridge.
Supports PBR materials with embedded and external textures.
"""

import io
import math
import sys
from pathlib import Path
from typing import Tuple

import ctypes
from ctypes import (
    c_void_p, c_char_p, c_int, c_uint32, c_float,
    Structure, POINTER, byref, create_string_buffer
)

import numpy as np
from PIL import Image
from OpenGL.GL import *

# ---------------------------------------------------------------------------
# ufbx bridge ctypes bindings
# ---------------------------------------------------------------------------

# Map type constants (must match ufbx_bridge.h)
FBX_MAP_BASE_COLOR = 0
FBX_MAP_METALLIC   = 1
FBX_MAP_ROUGHNESS  = 2
FBX_MAP_NORMAL     = 3
FBX_MAP_EMISSIVE   = 4
FBX_MAP_OCCLUSION  = 5
FBX_MAP_COUNT      = 6

MAP_TYPE_NAMES = {
    FBX_MAP_BASE_COLOR: 'base_color',
    FBX_MAP_METALLIC:   'metallic_roughness',
    FBX_MAP_ROUGHNESS:  'roughness',
    FBX_MAP_NORMAL:     'normal',
    FBX_MAP_EMISSIVE:   'emissive',
    FBX_MAP_OCCLUSION:  'occlusion',
}


class FBXMaterialInfo(Structure):
    """Mirrors fbx_material_info in ufbx_bridge.h."""
    _fields_ = [
        ('base_color', c_float * 4),
        ('metallic',   c_float),
        ('roughness',  c_float),
        ('emissive',   c_float * 3),
        ('has_texture', c_int * FBX_MAP_COUNT),
    ]


class FBXTextureInfo(Structure):
    """Mirrors fbx_texture_info in ufbx_bridge.h."""
    _fields_ = [
        ('data',     c_void_p),
        ('size',     c_uint32),
        ('filename', ctypes.c_char * 512),
    ]


def _load_bridge_library():
    """Locate and load the ufbx bridge shared library."""
    lib_dir = Path(__file__).parent / 'bin'

    if sys.platform == 'win32':
        lib_name = 'ufbx.dll'
    elif sys.platform == 'darwin':
        lib_name = 'ufbx.dylib'
    else:
        lib_name = 'ufbx.so'

    lib_path = lib_dir / lib_name
    if not lib_path.exists():
        return None

    try:
        lib = ctypes.cdll.LoadLibrary(str(lib_path))
    except OSError as e:
        print(f"Warning: Failed to load ufbx library: {e}")
        return None

    # Define function signatures
    lib.fbx_load.argtypes = [c_char_p]
    lib.fbx_load.restype = c_void_p

    lib.fbx_free.argtypes = [c_void_p]
    lib.fbx_free.restype = None

    lib.fbx_mesh_count.argtypes = [c_void_p]
    lib.fbx_mesh_count.restype = c_int

    lib.fbx_material_count.argtypes = [c_void_p]
    lib.fbx_material_count.restype = c_int

    lib.fbx_mesh_get_counts.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_int)]
    lib.fbx_mesh_get_counts.restype = None

    lib.fbx_mesh_extract.argtypes = [
        c_void_p, c_int,
        POINTER(c_float), POINTER(c_uint32),
        c_int, c_int
    ]
    lib.fbx_mesh_extract.restype = c_int

    lib.fbx_mesh_material_index.argtypes = [c_void_p, c_int]
    lib.fbx_mesh_material_index.restype = c_int

    lib.fbx_material_get_info.argtypes = [c_void_p, c_int, POINTER(FBXMaterialInfo)]
    lib.fbx_material_get_info.restype = None

    lib.fbx_material_get_texture.argtypes = [
        c_void_p, c_int, c_int, POINTER(FBXTextureInfo)
    ]
    lib.fbx_material_get_texture.restype = c_int

    lib.fbx_material_get_texture_path.argtypes = [c_void_p, c_int, c_int, c_char_p, c_int]
    lib.fbx_material_get_texture_path.restype = c_int

    return lib


# Try to load the library at import time
_ufbx_lib = _load_bridge_library()
UFBX_AVAILABLE = _ufbx_lib is not None

if not UFBX_AVAILABLE:
    print("Warning: ufbx library not found in bin/. FBX support disabled.")


# ---------------------------------------------------------------------------
# FBX Model class
# ---------------------------------------------------------------------------

class FBXModel:
    """FBX file loader with display lists and PBR material support."""

    def __init__(self, filename: str, flip_yz: bool = False,
                 default_color: Tuple[float, float, float] = (0.7, 0.6, 0.5)):

        if not UFBX_AVAILABLE:
            raise ImportError(
                "ufbx library not found. Place the compiled ufbx shared library "
                "in the bin/ directory."
            )

        print(f"\nLoading FBX: {filename}")

        self.verts = []
        self.normals = []
        self.texcoords = []
        self.tangents = []
        self.faces = []
        self.materials = {}
        self.textures = {}
        self.default_color = default_color
        self.center = [0, 0, 0]
        self.scale_factor = 1.0
        self.flip_yz = flip_yz

        self.dl_smooth = None
        self.dl_flat = None
        self.dl_wireframe = None

        # VBO/VAO for shader rendering
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.vertex_count = 0
        self.use_shader_rendering = False

        # Store raw VBO data for compilation after GL context is ready
        self._vbo_vertex_data = None
        self._vbo_index_data = None

        self._scene = None
        self._fbx_dir = Path(filename).parent  # For resolving external textures

        self._load_fbx(filename, flip_yz)
        self._analyze_model()
        self._calculate_tangents()

    def _load_fbx(self, filename: str, flip_yz: bool):
        """Load FBX file via ufbx bridge."""
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"FBX file not found: {filename}")

        # Load scene through bridge
        filepath_bytes = str(filepath).encode('utf-8')
        self._scene = _ufbx_lib.fbx_load(filepath_bytes)
        if not self._scene:
            raise RuntimeError(f"Failed to load FBX file: {filename}")

        mesh_count = _ufbx_lib.fbx_mesh_count(self._scene)
        mat_count = _ufbx_lib.fbx_material_count(self._scene)
        print(f"  Scene: {mesh_count} meshes, {mat_count} materials")

        # Extract materials first
        self._extract_materials(mat_count)

        # Extract all meshes, combining into unified vertex lists
        vertex_offset = 0
        for mi in range(mesh_count):
            self._extract_mesh(mi, vertex_offset, flip_yz)
            vertex_offset = len(self.verts)

        print(f"  FBX loaded: {len(self.verts)} vertices, {len(self.faces)} faces")

    def _extract_materials(self, mat_count: int):
        """Extract PBR material properties and texture references."""
        for mi in range(mat_count):
            info = FBXMaterialInfo()
            _ufbx_lib.fbx_material_get_info(self._scene, mi, byref(info))

            mat_dict = {}

            # Base color factor
            bc = [info.base_color[i] for i in range(4)]
            mat_dict['baseColorFactor'] = bc
            mat_dict['diffuse'] = tuple(bc[:3])

            # PBR factors
            mat_dict['metallicFactor'] = info.metallic
            mat_dict['roughnessFactor'] = info.roughness

            # Emissive
            em = [info.emissive[i] for i in range(3)]
            if any(v > 0 for v in em):
                mat_dict['emissiveFactor'] = em

            # Extract textures (stored as PIL images until GL context)
            for map_type in range(FBX_MAP_COUNT):
                if info.has_texture[map_type]:
                    tex_info = FBXTextureInfo()
                    has_embedded = _ufbx_lib.fbx_material_get_texture(
                        self._scene, mi, map_type, byref(tex_info)
                    )

                    image = None
                    map_name = MAP_TYPE_NAMES.get(map_type, f'unknown_{map_type}')

                    if has_embedded and tex_info.size > 0:
                        # Decode embedded texture data
                        image = self._decode_embedded_texture(tex_info, map_name)
                    else:
                        # Try loading from external file
                        path_buf = create_string_buffer(1024)
                        path_len = _ufbx_lib.fbx_material_get_texture_path(
                            self._scene, mi, map_type, path_buf, 1024
                        )
                        if path_len > 0:
                            tex_path = path_buf.value.decode('utf-8', errors='replace')
                            image = self._load_external_texture(tex_path, map_name)

                    if image is not None:
                        mat_dict[f'image_{map_name}'] = image
                        print(f"    Found {map_name} texture")

            textures_found = [k.replace('image_', '') for k in mat_dict if k.startswith('image_')]
            if textures_found:
                print(f"    Material {mi}: textures = {', '.join(textures_found)}")

            self.materials[mi] = mat_dict

    def _decode_embedded_texture(self, tex_info, map_name: str):
        """Decode embedded texture bytes into a PIL Image."""
        try:
            raw_data = ctypes.string_at(tex_info.data, tex_info.size)
            image = Image.open(io.BytesIO(raw_data))
            print(f"    Decoded embedded {map_name}: {image.size[0]}x{image.size[1]}")
            return image
        except Exception as e:
            print(f"    Failed to decode embedded {map_name}: {e}")
            return None

    def _load_external_texture(self, tex_path: str, map_name: str):
        """Load texture from an external file path."""
        tex_path_obj = Path(tex_path)
        filename = tex_path_obj.name

        candidates = []

        # Absolute path as-is (if file exists at original location)
        if tex_path_obj.is_absolute():
            candidates.append(tex_path_obj)

        # Full relative path from FBX directory
        candidates.append(self._fbx_dir / tex_path)

        # Just the filename in the FBX directory
        candidates.append(self._fbx_dir / filename)

        # Try reconstructing relative subdirectories from absolute path
        parts = tex_path_obj.parts
        for i in range(1, len(parts)):
            sub_path = Path(*parts[i:])
            candidates.append(self._fbx_dir / sub_path)

        # Search common texture subdirectory names
        for subdir in ['textures', 'Textures', 'tex', 'maps', 'Materials']:
            candidates.append(self._fbx_dir / subdir / filename)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for p in candidates:
            resolved = str(p)
            if resolved not in seen:
                seen.add(resolved)
                unique.append(p)

        for path in unique:
            if path.exists():
                try:
                    image = Image.open(str(path))
                    print(f"    Loaded external {map_name}: {path.name}")
                    return image
                except Exception as e:
                    print(f"    Failed to load {path}: {e}")

        print(f"    External texture not found: {tex_path}")
        return None

    def _extract_mesh(self, mesh_idx: int, vertex_offset: int, flip_yz: bool):
        """Extract a single mesh's geometry using the bridge."""
        vert_count = c_int(0)
        idx_count = c_int(0)
        _ufbx_lib.fbx_mesh_get_counts(self._scene, mesh_idx,
                                       byref(vert_count), byref(idx_count))

        if vert_count.value == 0:
            return

        # Allocate buffers
        max_verts = vert_count.value
        max_indices = idx_count.value

        vert_buf = (c_float * (max_verts * 11))()   # 11 floats per vertex
        idx_buf = (c_uint32 * max_indices)()

        unique_count = _ufbx_lib.fbx_mesh_extract(
            self._scene, mesh_idx,
            vert_buf, idx_buf,
            max_verts, max_indices
        )

        # Get material assignment
        mat_idx = _ufbx_lib.fbx_mesh_material_index(self._scene, mesh_idx)
        if mat_idx < 0:
            mat_idx = None

        # Unpack vertices into our lists
        base = len(self.verts)

        for vi in range(unique_count):
            off = vi * 11
            px, py, pz = vert_buf[off], vert_buf[off+1], vert_buf[off+2]
            nx, ny, nz = vert_buf[off+3], vert_buf[off+4], vert_buf[off+5]
            u, v = vert_buf[off+6], vert_buf[off+7]
            tx, ty, tz = vert_buf[off+8], vert_buf[off+9], vert_buf[off+10]

            if flip_yz:
                px, py, pz = px, pz, py
                nx, ny, nz = nx, nz, ny
                tx, ty, tz = tx, tz, ty

            self.verts.append((px, py, pz))
            self.normals.append((nx, ny, nz))
            self.texcoords.append((u, v))
            self.tangents.append((tx, ty, tz))

        # Unpack indices into face triples (1-indexed to match OBJ/GLB convention)
        for ti in range(0, max_indices, 3):
            if ti + 2 >= max_indices:
                break
            i0, i1, i2 = idx_buf[ti], idx_buf[ti+1], idx_buf[ti+2]

            # Convert to 1-indexed and offset
            face_verts = [int(i0) + base + 1, int(i1) + base + 1, int(i2) + base + 1]
            face_normals = list(face_verts)
            face_uvs = list(face_verts)

            self.faces.append((face_verts, face_normals, face_uvs, mat_idx))

        print(f"  Mesh {mesh_idx}: {unique_count} unique verts, "
              f"{max_indices // 3} triangles, material={mat_idx}")

    def _load_texture_from_image(self, image):
        """Load texture from PIL Image into OpenGL."""
        try:
            if image.mode == 'RGBA':
                gl_format = GL_RGBA
                internal = GL_RGBA
            elif image.mode == 'RGB':
                gl_format = GL_RGB
                internal = GL_RGB
            elif image.mode == 'L':
                gl_format = GL_RED
                internal = GL_RED
            else:
                image = image.convert('RGB')
                gl_format = GL_RGB
                internal = GL_RGB

            # Flip for OpenGL
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

            image_data = np.frombuffer(image.tobytes(), dtype=np.uint8)
            width, height = image.size

            texname = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texname)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(GL_TEXTURE_2D, 0, internal, width, height, 0,
                         gl_format, GL_UNSIGNED_BYTE, image_data)

            return texname
        except Exception as e:
            print(f"    Error loading texture: {e}")
            return None

    def _analyze_model(self):
        """Compute bounding box, center, and scale factor."""
        if not self.verts:
            print("  WARNING: No vertices found!")
            return

        xs = [v[0] for v in self.verts]
        ys = [v[1] for v in self.verts]
        zs = [v[2] for v in self.verts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        self.center = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]

        max_size = max(max_x - min_x, max_y - min_y, max_z - min_z)
        if max_size > 0:
            self.scale_factor = 5.0 / max_size

        print(f"  Bounding box: {max_x - min_x:.2f} x {max_y - min_y:.2f} x {max_z - min_z:.2f}")
        print(f"  Scale factor: {self.scale_factor:.4f}")

    def _calculate_tangents(self):
        """Tangents are already extracted from ufbx, this just validates them."""
        if len(self.tangents) != len(self.verts):
            print("  Tangent count mismatch, padding...")
            while len(self.tangents) < len(self.verts):
                self.tangents.append((1.0, 0.0, 0.0))

        print(f"  Tangents: {len(self.tangents)} vertices")

    def compile_display_lists(self):
        """Compile display lists and upload textures (call after GL context ready)."""
        print("Compiling FBX display lists...")

        # Upload textures from PIL images to GL
        for mat_idx, mat in self.materials.items():
            image_keys = [k for k in mat.keys() if k.startswith('image_')]
            for img_key in image_keys:
                texture_id = self._load_texture_from_image(mat[img_key])
                if texture_id:
                    tex_key = img_key.replace('image_', 'texture_')
                    mat[tex_key] = texture_id
                    print(f"  Created {tex_key.replace('texture_', '')} map")
                del mat[img_key]

        # Compile display lists
        self.dl_smooth = glGenLists(1)
        glNewList(self.dl_smooth, GL_COMPILE)
        self._render_geometry('smooth')
        glEndList()

        self.dl_flat = glGenLists(1)
        glNewList(self.dl_flat, GL_COMPILE)
        self._render_geometry('flat')
        glEndList()

        self.dl_wireframe = glGenLists(1)
        glNewList(self.dl_wireframe, GL_COMPILE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self._render_geometry('smooth')
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEndList()

        # Free the ufbx scene now that we've extracted everything
        if self._scene:
            _ufbx_lib.fbx_free(self._scene)
            self._scene = None

        print("  FBX display lists ready")

    def compile_vbos(self):
        """Compile vertex buffer objects for PBR shader rendering."""
        print("Compiling FBX VBOs for shader rendering...")

        vertices = []
        indices = []
        current_index = 0

        for face in self.faces:
            face_verts, face_normals, face_uvs, material_idx = face

            for i in range(len(face_verts)):
                v_idx = face_verts[i] - 1
                n_idx = face_normals[i] - 1
                uv_idx = face_uvs[i] - 1

                # Position
                if v_idx < len(self.verts):
                    vertices.extend(self.verts[v_idx])
                else:
                    vertices.extend([0.0, 0.0, 0.0])

                # Normal
                if 0 <= n_idx < len(self.normals):
                    vertices.extend(self.normals[n_idx])
                else:
                    vertices.extend([0.0, 1.0, 0.0])

                # UV
                if 0 <= uv_idx < len(self.texcoords):
                    vertices.extend(self.texcoords[uv_idx])
                else:
                    vertices.extend([0.0, 0.0])

                # Tangent
                t_idx = v_idx
                if self.tangents and 0 <= t_idx < len(self.tangents):
                    vertices.extend(self.tangents[t_idx])
                else:
                    vertices.extend([1.0, 0.0, 0.0])

            # Triangle (faces are already triangulated)
            if len(face_verts) == 3:
                indices.extend([current_index, current_index + 1, current_index + 2])
                current_index += 3
            elif len(face_verts) == 4:
                indices.extend([current_index, current_index + 1, current_index + 2])
                indices.extend([current_index, current_index + 2, current_index + 3])
                current_index += 4
            else:
                for i in range(1, len(face_verts) - 1):
                    indices.extend([current_index, current_index + i, current_index + i + 1])
                current_index += len(face_verts)

        vertex_data = np.array(vertices, dtype=np.float32)
        index_data = np.array(indices, dtype=np.uint32)
        self.vertex_count = len(indices)

        # Create VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        # EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, GL_STATIC_DRAW)

        # Vertex attributes: pos(3) + normal(3) + uv(2) + tangent(3) = 11 floats
        # each float is 4 bytes, stride = 11 * 4 = 44 bytes
        stride = 44

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, c_void_p(24))

        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, c_void_p(32))

        glBindVertexArray(0)

        print(f"  VBOs ready: {self.vertex_count} indices, {len(vertices)//11} vertices")

    def _render_geometry(self, mode, enable_textures=True):
        """Render geometry for display list compilation."""
        glPushMatrix()
        glScalef(self.scale_factor, self.scale_factor, self.scale_factor)
        glTranslatef(-self.center[0], -self.center[1], -self.center[2])

        for face in self.faces:
            vertices, normals, texture_coords, material_idx = face

            material_applied = False

            if material_idx is not None and material_idx in self.materials:
                mat = self.materials[material_idx]

                if 'texture_base_color' in mat and enable_textures:
                    glBindTexture(GL_TEXTURE_2D, mat['texture_base_color'])
                    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                    glColor3f(1.0, 1.0, 1.0)
                    material_applied = True
                elif 'diffuse' in mat:
                    glColor(*mat['diffuse'])
                    material_applied = True

            if not material_applied:
                glColor(*self.default_color)

            # Flat shading normal
            if mode == 'flat' or not self.normals or not any(normals):
                if len(vertices) >= 3:
                    v1 = self.verts[vertices[0] - 1]
                    v2 = self.verts[vertices[1] - 1]
                    v3 = self.verts[vertices[2] - 1]
                    u = [v2[i] - v1[i] for i in range(3)]
                    v = [v3[i] - v1[i] for i in range(3)]
                    n = [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]]
                    length = math.sqrt(sum(x*x for x in n))
                    if length > 0:
                        n = [x/length for x in n]
                        glNormal3fv(n)

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if mode == 'smooth' and self.normals and normals[i] > 0 and normals[i] <= len(self.normals):
                    glNormal3fv(self.normals[normals[i] - 1])
                if self.texcoords and texture_coords[i] > 0 and texture_coords[i] <= len(self.texcoords):
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                if vertices[i] > 0 and vertices[i] <= len(self.verts):
                    glVertex3fv(self.verts[vertices[i] - 1])
            glEnd()

        glPopMatrix()

    def render(self, mode='smooth', enable_textures=True):
        """Render using pre-compiled display list."""
        if self.has_textures() and enable_textures:
            glEnable(GL_TEXTURE_2D)
        else:
            glDisable(GL_TEXTURE_2D)

        if mode == 'smooth' and self.dl_smooth:
            glCallList(self.dl_smooth)
        elif mode == 'flat' and self.dl_flat:
            glCallList(self.dl_flat)
        elif mode == 'wireframe' and self.dl_wireframe:
            glCallList(self.dl_wireframe)

        glDisable(GL_TEXTURE_2D)

    def has_textures(self):
        """Check if model has any textures."""
        for mat in self.materials.values():
            if any(k.startswith('texture_') for k in mat.keys()):
                return True
            if any(k.startswith('image_') for k in mat.keys()):
                return True
        return False

    def get_available_maps(self):
        """Get list of available texture map types across all materials."""
        map_types = set()
        for mat in self.materials.values():
            for key in mat.keys():
                if key.startswith('texture_'):
                    map_types.add(key.replace('texture_', ''))
        return sorted(list(map_types))

    def has_map_type(self, map_type):
        """Check if model has a specific texture map type."""
        texture_key = f'texture_{map_type}'
        return any(texture_key in mat for mat in self.materials.values())

    def __del__(self):
        """Clean up ufbx scene if still loaded."""
        if hasattr(self, '_scene') and self._scene and UFBX_AVAILABLE:
            _ufbx_lib.fbx_free(self._scene)
            self._scene = None


def is_fbx_file(filename: str) -> bool:
    """Check if file is FBX format."""
    return Path(filename).suffix.lower() == '.fbx'
