"""
GLB Model Loader Module
Handles loading and rendering of GLB/glTF files with embedded textures and materials.
GLB (GL Transmission Format Binary) is a single-file format that includes geometry,
materials, and textures all in one file.
"""

import math
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
from OpenGL.GL import *
import ctypes

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("⚠️  Warning: trimesh not installed. GLB support disabled.")
    print("   Install with: pip install trimesh[easy]")


class GLBModel:
    """GLB/glTF file loader with display lists and material support."""
    
    def __init__(self, filename: str, flip_yz: bool = False,
                 default_color: Tuple[float, float, float] = (0.7, 0.6, 0.5)):
        
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for GLB loading. Install with: pip install trimesh[easy]")
        
        print(f"\nLoading GLB: {filename}")
        
        self.verts = []
        self.normals = []
        self.texcoords = []
        self.tangents = []  # For normal mapping
        self.faces = []
        self.materials = {}
        self.textures = {}
        self.texture_images = {}  # Store PIL images until OpenGL context ready
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
        self.use_shader_rendering = False  # Toggle between display lists and shaders
        
        self._load_glb(filename, flip_yz)
        self._analyze_model()
        self._calculate_tangents()
    
    def _load_glb(self, filename: str, flip_yz: bool):
        """Load GLB file using trimesh."""
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"GLB file not found: {filename}")
        
        try:
            # Load with trimesh
            print("  Loading with trimesh...")
            scene = trimesh.load(str(filename))
            
            # GLB files are loaded as scenes
            if isinstance(scene, trimesh.Scene):
                print(f"  Scene with {len(scene.geometry)} geometries")
                
                # Process each mesh in the scene
                vertex_offset = 0
                face_offset = 0
                
                for geom_name, mesh in scene.geometry.items():
                    print(f"  Processing: {geom_name}")
                    self._process_mesh(mesh, vertex_offset, face_offset, flip_yz)
                    vertex_offset += len(mesh.vertices)
                    face_offset += len(mesh.faces)
            else:
                # Single mesh
                print(f"  Single mesh")
                self._process_mesh(scene, 0, 0, flip_yz)
            
            print(f"✓ GLB loaded: {len(self.verts)} vertices, {len(self.faces)} faces")
            
        except Exception as e:
            raise RuntimeError(f"Error loading GLB file: {e}")
    
    def _process_mesh(self, mesh, vertex_offset: int, face_offset: int, flip_yz: bool):
        """Process a single mesh from the scene."""
        # Get vertices
        for vertex in mesh.vertices:
            v = [float(vertex[0]), float(vertex[1]), float(vertex[2])]
            if flip_yz:
                v = [v[0], v[2], v[1]]
            self.verts.append(tuple(v))
        
        # Get vertex normals
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            for normal in mesh.vertex_normals:
                n = [float(normal[0]), float(normal[1]), float(normal[2])]
                if flip_yz:
                    n = [n[0], n[2], n[1]]
                self.normals.append(tuple(n))
        
        # Get texture coordinates
        has_uvs = False
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uv_offset = len(self.texcoords)
            for uv in mesh.visual.uv:
                self.texcoords.append((float(uv[0]), float(uv[1])))
            has_uvs = len(mesh.visual.uv) > 0
        
        # Process material and texture
        material_idx = None
        if hasattr(mesh.visual, 'material'):
            material = mesh.visual.material
            mat_dict = {}
            
            # Get base color
            if hasattr(material, 'main_color'):
                color = material.main_color
                if len(color) >= 3:
                    mat_dict['diffuse'] = (color[0]/255.0, color[1]/255.0, color[2]/255.0)
            elif hasattr(material, 'diffuse'):
                mat_dict['diffuse'] = tuple(material.diffuse[:3])
            
            # Get PBR material properties
            if hasattr(material, 'baseColorFactor'):
                mat_dict['baseColorFactor'] = material.baseColorFactor
            if hasattr(material, 'metallicFactor'):
                mat_dict['metallicFactor'] = material.metallicFactor
            if hasattr(material, 'roughnessFactor'):
                mat_dict['roughnessFactor'] = material.roughnessFactor
            if hasattr(material, 'emissiveFactor'):
                mat_dict['emissiveFactor'] = material.emissiveFactor
            
            # Debug: Show all material attributes
            print(f"    [MATERIAL DEBUG] Attributes: {[a for a in dir(material) if not a.startswith('_') and 'Texture' in a or a in ['baseColorFactor', 'metallicFactor', 'emissive', 'emissiveFactor']]}")
            
            # Load ALL PBR texture maps
            texture_types = {
                'baseColorTexture': 'base_color',
                'image': 'base_color',  # Fallback
                'metallicRoughnessTexture': 'metallic_roughness',
                'normalTexture': 'normal',
                'occlusionTexture': 'occlusion',
                'emissiveTexture': 'emissive',
                'emissive': 'emissive',  # Try without "Texture" suffix
                'opacityTexture': 'opacity'
            }
            
            textures_found = []
            for attr_name, tex_type in texture_types.items():
                if hasattr(material, attr_name):
                    texture_image = getattr(material, attr_name)
                    # Check if it's actually an image (not None, not a factor)
                    if texture_image is not None and hasattr(texture_image, 'size'):
                        # Store with descriptive key
                        mat_dict[f'image_{tex_type}'] = texture_image
                        textures_found.append(tex_type)
                        print(f"    [TEXTURE DEBUG] Found '{attr_name}' -> {tex_type}")
                    elif texture_image is not None:
                        print(f"    [TEXTURE DEBUG] '{attr_name}' exists but is not an image (type: {type(texture_image).__name__})")
            
            # Fallback: Search material.__dict__ for any PIL Images
            if not textures_found and hasattr(material, '__dict__'):
                for key, value in material.__dict__.items():
                    if value is not None and hasattr(value, 'size'):  # PIL Image check
                        mat_dict['image_base_color'] = value
                        textures_found.append('base_color (from __dict__)')
                        break
            
            # Last resort: check visual.image
            if not textures_found and hasattr(mesh.visual, 'image') and mesh.visual.image is not None:
                mat_dict['image_base_color'] = mesh.visual.image
                textures_found.append('base_color (from visual)')
            
            if textures_found:
                print(f"    ✓ Found textures: {', '.join(textures_found)}")
            
            if mat_dict:
                material_idx = len(self.materials)
                self.materials[material_idx] = mat_dict
        elif hasattr(mesh.visual, 'image') and mesh.visual.image is not None:
            # No material object, but visual has image directly
            mat_dict = {'image_base_color': mesh.visual.image}
            material_idx = len(self.materials)
            self.materials[material_idx] = mat_dict
            print(f"    ✓ Found textures: base_color (from visual)")
        
        # Get faces (convert to 1-indexed like OBJ)
        for face in mesh.faces:
            face_verts = [int(idx) + vertex_offset + 1 for idx in face]
            face_normals = [int(idx) + vertex_offset + 1 for idx in face]
            
            # UV coordinates - use the offset within texcoords list
            if has_uvs:
                face_uvs = [int(idx) + vertex_offset + 1 for idx in face]
            else:
                face_uvs = [0, 0, 0]
            
            self.faces.append((face_verts, face_normals, face_uvs, material_idx))
    
    def _load_texture_from_image(self, image):
        """Load texture from PIL Image object."""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Flip vertically for OpenGL (OpenGL expects bottom-left origin)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Get image data
            image_data = np.frombuffer(image.tobytes(), dtype=np.uint8)
            width, height = image.size
            
            # Create OpenGL texture
            texname = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texname)
            
            # Set texture parameters
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            
            # Upload texture data
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                        GL_RGB, GL_UNSIGNED_BYTE, image_data)
            
            return texname
        except Exception as e:
            print(f"    Error loading texture: {e}")
            return None
    
    def _analyze_model(self):
        """Analyze model bounds and calculate center/scale."""
        if not self.verts:
            print("⚠️  WARNING: No vertices found!")
            return
        
        min_x = min(v[0] for v in self.verts)
        max_x = max(v[0] for v in self.verts)
        min_y = min(v[1] for v in self.verts)
        max_y = max(v[1] for v in self.verts)
        min_z = min(v[2] for v in self.verts)
        max_z = max(v[2] for v in self.verts)
        
        self.center = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
        
        max_size = max(max_x - min_x, max_y - min_y, max_z - min_z)
        if max_size > 0:
            self.scale_factor = 5.0 / max_size
        
        print(f"Bounding box: {max_x - min_x:.2f} x {max_y - min_y:.2f} x {max_z - min_z:.2f}")
        print(f"Scale factor: {self.scale_factor:.4f}")
    
    def _calculate_tangents(self):
        """Calculate tangent vectors for normal mapping."""
        if not self.verts or not self.texcoords or not self.normals:
            # Need positions, UVs, and normals for tangent calculation
            print("  Skipping tangent calculation (missing UVs or normals)")
            return
        
        print("  Calculating tangents for normal mapping...")
        
        # Initialize tangents array (one per vertex)
        self.tangents = [np.array([0.0, 0.0, 0.0]) for _ in range(len(self.verts))]
        
        # Calculate tangents for each face
        for face in self.faces:
            vertices, normals, texture_coords, material_idx = face
            
            if len(vertices) < 3 or not any(texture_coords):
                continue  # Skip degenerate faces or faces without UVs
            
            # Get vertex indices (convert from 1-indexed to 0-indexed)
            i0 = vertices[0] - 1
            i1 = vertices[1] - 1
            i2 = vertices[2] - 1
            
            # Get positions
            v0 = np.array(self.verts[i0])
            v1 = np.array(self.verts[i1])
            v2 = np.array(self.verts[i2])
            
            # Get UV coordinates (convert from 1-indexed)
            if texture_coords[0] > 0 and texture_coords[1] > 0 and texture_coords[2] > 0:
                uv0 = np.array(self.texcoords[texture_coords[0] - 1])
                uv1 = np.array(self.texcoords[texture_coords[1] - 1])
                uv2 = np.array(self.texcoords[texture_coords[2] - 1])
            else:
                continue  # Skip if no valid UVs
            
            # Calculate edge vectors
            edge1 = v1 - v0
            edge2 = v2 - v0
            deltaUV1 = uv1 - uv0
            deltaUV2 = uv2 - uv0
            
            # Calculate tangent
            f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1] + 1e-6)
            tangent = np.array([
                f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]),
                f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]),
                f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])
            ])
            
            # Accumulate tangents for each vertex of the face
            self.tangents[i0] += tangent
            self.tangents[i1] += tangent
            self.tangents[i2] += tangent
        
        # Normalize and orthogonalize tangents
        for i in range(len(self.tangents)):
            if np.linalg.norm(self.tangents[i]) > 0:
                # Gram-Schmidt orthogonalization
                n = np.array(self.normals[i])
                t = self.tangents[i]
                
                # Make tangent orthogonal to normal
                t = t - n * np.dot(t, n)
                
                # Normalize
                if np.linalg.norm(t) > 0:
                    self.tangents[i] = tuple(t / np.linalg.norm(t))
                else:
                    # Fallback: create arbitrary perpendicular vector
                    if abs(n[0]) < 0.9:
                        self.tangents[i] = tuple(np.cross(n, np.array([1, 0, 0])))
                    else:
                        self.tangents[i] = tuple(np.cross(n, np.array([0, 1, 0])))
            else:
                # Fallback tangent
                self.tangents[i] = (1.0, 0.0, 0.0)
        
        print(f"  ✓ Calculated tangents for {len(self.tangents)} vertices")
    
    def compile_display_lists(self):
        """Compile display lists (call after GL context is created)."""
        print("Compiling GLB display lists...")
        
        # Now that we have an OpenGL context, create textures from stored images
        for mat_idx, mat in self.materials.items():
            # Find all image_ keys (all texture maps)
            image_keys = [k for k in mat.keys() if k.startswith('image_')]
            
            for img_key in image_keys:
                texture_id = self._load_texture_from_image(mat[img_key])
                if texture_id:
                    # Replace 'image_X' with 'texture_X'
                    tex_key = img_key.replace('image_', 'texture_')
                    mat[tex_key] = texture_id
                    print(f"  ✓ Created {tex_key.replace('texture_', '')} map")
                # Remove the PIL image to save memory
                del mat[img_key]
        
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
        
        print("✓ GLB display lists ready")
    
    def compile_vbos(self):
        """Compile vertex buffer objects for shader rendering."""
        print("Compiling VBOs for shader rendering...")
        
        # Build interleaved vertex data
        vertices = []
        indices = []
        current_index = 0
        
        for face in self.faces:
            face_verts, face_normals, face_uvs, material_idx = face
            
            # Process each vertex in the face
            for i in range(len(face_verts)):
                # Position
                v_idx = face_verts[i] - 1
                if v_idx < len(self.verts):
                    pos = self.verts[v_idx]
                    vertices.extend(pos)
                else:
                    vertices.extend([0.0, 0.0, 0.0])
                
                # Normal
                n_idx = face_normals[i] - 1
                if n_idx >= 0 and n_idx < len(self.normals):
                    normal = self.normals[n_idx]
                    vertices.extend(normal)
                else:
                    vertices.extend([0.0, 1.0, 0.0])
                
                # UV
                uv_idx = face_uvs[i] - 1
                if uv_idx >= 0 and uv_idx < len(self.texcoords):
                    uv = self.texcoords[uv_idx]
                    vertices.extend(uv)
                else:
                    vertices.extend([0.0, 0.0])
                
                # Tangent
                t_idx = v_idx
                if self.tangents and t_idx < len(self.tangents):
                    tangent = self.tangents[t_idx]
                    vertices.extend(tangent)
                else:
                    vertices.extend([1.0, 0.0, 0.0])
            
            # Create indices for the face (triangulate if needed)
            if len(face_verts) == 3:
                # Triangle
                indices.extend([current_index, current_index + 1, current_index + 2])
                current_index += 3
            elif len(face_verts) == 4:
                # Quad - split into two triangles
                indices.extend([current_index, current_index + 1, current_index + 2])
                indices.extend([current_index, current_index + 2, current_index + 3])
                current_index += 4
            else:
                # N-gon - fan triangulation
                for i in range(1, len(face_verts) - 1):
                    indices.extend([current_index, current_index + i, current_index + i + 1])
                current_index += len(face_verts)
        
        # Convert to numpy arrays
        vertex_data = np.array(vertices, dtype=np.float32)
        index_data = np.array(indices, dtype=np.uint32)
        self.vertex_count = len(indices)
        
        # Create VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Create VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        
        # Create EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, GL_STATIC_DRAW)
        
        # Vertex attributes
        stride = 11 * 4  # 11 floats per vertex (3 pos + 3 normal + 2 uv + 3 tangent)
        
        # Position (location = 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        
        # Normal (location = 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        
        # UV (location = 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        
        # Tangent (location = 3)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
        
        # Unbind
        glBindVertexArray(0)
        
        print(f"✓ VBOs ready: {self.vertex_count} indices, {len(vertices)//11} vertices")
    
    def _render_geometry(self, mode, enable_textures=True):
        """Render geometry for display list compilation."""
        glPushMatrix()
        glScalef(self.scale_factor, self.scale_factor, self.scale_factor)
        glTranslatef(-self.center[0], -self.center[1], -self.center[2])
        
        for face in self.faces:
            vertices, normals, texture_coords, material_idx = face
            
            # Apply material
            material_applied = False
            use_texture = False
            
            if material_idx is not None and material_idx in self.materials:
                mat = self.materials[material_idx]
                
                if 'texture_base_color' in mat and enable_textures:
                    # Bind base color texture
                    glBindTexture(GL_TEXTURE_2D, mat['texture_base_color'])
                    
                    # Set texture environment to modulate (multiply) with lighting
                    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                    
                    # Set color to white so texture shows its true colors
                    glColor3f(1.0, 1.0, 1.0)
                    material_applied = True
                    use_texture = True
                elif 'diffuse' in mat:
                    glColor(*mat['diffuse'])
                    material_applied = True
            
            if not material_applied:
                glColor(*self.default_color)
            
            # Compute face normal for flat shading
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
                # Set normal for smooth shading
                if mode == 'smooth' and self.normals and normals[i] > 0 and normals[i] <= len(self.normals):
                    glNormal3fv(self.normals[normals[i] - 1])
                
                # Set texture coordinate
                if self.texcoords and texture_coords[i] > 0 and texture_coords[i] <= len(self.texcoords):
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                
                # Vertex position
                if vertices[i] > 0 and vertices[i] <= len(self.verts):
                    glVertex3fv(self.verts[vertices[i] - 1])
            glEnd()
        
        glPopMatrix()
    
    def render(self, mode='smooth', enable_textures=True):
        """Render using pre-compiled display list."""
        # Enable/disable texturing before calling display list
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
        
        # Clean up
        glDisable(GL_TEXTURE_2D)
    
    def has_textures(self):
        """Check if model has any textures."""
        # Check for any texture_ keys (base_color, normal, metallic_roughness, etc.)
        for mat in self.materials.values():
            # Check for texture_ keys (after compilation)
            if any(k.startswith('texture_') for k in mat.keys()):
                return True
            # Check for image_ keys (before compilation)
            if any(k.startswith('image_') for k in mat.keys()):
                return True
        return False
    
    def get_available_maps(self):
        """Get list of available texture map types across all materials."""
        map_types = set()
        for mat in self.materials.values():
            for key in mat.keys():
                if key.startswith('texture_'):
                    map_type = key.replace('texture_', '')
                    map_types.add(map_type)
        return sorted(list(map_types))
    
    def has_map_type(self, map_type):
        """Check if model has a specific texture map type (e.g., 'normal', 'occlusion')."""
        texture_key = f'texture_{map_type}'
        return any(texture_key in mat for mat in self.materials.values())


def is_glb_file(filename: str) -> bool:
    """Check if file is GLB or glTF format."""
    supported_extensions = {'.glb', '.gltf'}
    return Path(filename).suffix.lower() in supported_extensions
