"""
GL Widget Module
OpenGL widget for rendering 3D models with PBR shader support.
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from pathlib import Path
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

from trackball import Trackball
from shader_manager import ShaderManager
from vbo_renderer import VBORenderer


class GLWidget(QGLWidget):
    """OpenGL widget for model display."""
    
    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.model = model
        self.trackball = Trackball(theta=-25)
        
        self.display_mode = 'smooth'
        self.lighting_enabled = True
        self.texture_enabled = True  # Textures enabled by default
        self.pbr_enabled = False  # PBR rendering disabled by default
        self.brightness = 0.3  # Default brightness (matches slider default)
        self.ambient_strength = 0.10  # Low default for maximum detail
        self.light_color = (1.0, 1.0, 1.0)  # Default white light (RGB)
        self.ambient_color = (1.0, 1.0, 1.0)  # Default white ambient (RGB)
        
        # Material property multipliers (for real-time adjustment)
        self.metallic_multiplier = 1.0  # Multiply metallic values
        self.roughness_multiplier = 1.0  # Multiply roughness values
        self.normal_strength = 1.0  # Normal map intensity
        self.ao_strength = 1.0  # Ambient occlusion intensity
        
        # HDR Environment Map (IBL)
        self.env_map_texture = None  # OpenGL cubemap texture ID
        self.env_intensity = 1.0  # HDR environment brightness
        self.env_rotation = 0.0  # HDR environment rotation in degrees
        
        # Emissive intensity
        self.emissive_intensity = 1.0
        
        # Debug visualization
        self.show_uv_checker = False
        self.show_bounding_box = False
        self.show_vertex_normals = False
        self._uv_checker_texture = None
        self._uv_checker_display_list = None
        self._normals_display_list = None
        self._bbox_display_list = None
        
        self.last_pos = None
        self.zoom = -20
        self.pan_x = 0
        self.pan_y = 0
        
        # Shader system
        self.shader_manager = None
        self.vbo_renderer = None
        self._texture_debug_printed = False  # Print texture debug only once
        
        self.setMinimumSize(800, 600)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Cache model stats
        self._cache_model_stats()
        
        # WASD movement
        self.move_speed = 0.15
        self.keys_pressed = set()
        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self._process_movement)
        self.move_timer.setInterval(16)  # ~60fps
    
    def initializeGL(self):
        glShadeModel(GL_SMOOTH)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(1.5)
        
        glClearColor(0.05, 0.05, 0.06, 1)
        
        # Compile display lists now that GL context exists
        if self.model is not None:
            self.model.compile_display_lists()
        
        # Initialize shader system for PBR rendering
        self._init_shaders()
        
        self.setup_lighting()
    
    def setup_lighting(self):
        """Setup three-point lighting."""
        if self.lighting_enabled:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_LIGHT1)
            glEnable(GL_LIGHT2)
        else:
            glDisable(GL_LIGHTING)
            return
        
        br = self.brightness
        r, g, b = self.light_color  # Use selected color
        
        # Key light
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(10, 10, 10, 0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(0.2*br*r, 0.2*br*g, 0.2*br*b, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(0.8*br*r, 0.8*br*g, 0.8*br*b, 1.0))
        
        # Fill light
        glLightfv(GL_LIGHT1, GL_POSITION, (GLfloat * 4)(-10, 5, 5, 0))
        glLightfv(GL_LIGHT1, GL_AMBIENT, (GLfloat * 4)(0.1*br*r, 0.1*br*g, 0.1*br*b, 1.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (GLfloat * 4)(0.4*br*r, 0.4*br*g, 0.4*br*b, 1.0))
        
        # Rim light
        glLightfv(GL_LIGHT2, GL_POSITION, (GLfloat * 4)(0, -5, -10, 0))
        glLightfv(GL_LIGHT2, GL_DIFFUSE, (GLfloat * 4)(0.3*br*r, 0.3*br*g, 0.3*br*b, 1.0))
    
    def _init_shaders(self):
        """Initialize PBR shader system."""
        try:
            # Find shader files
            shader_dir = Path(__file__).parent / 'shaders'
            if not shader_dir.exists():
                print("⚠️ Shader directory not found, PBR rendering disabled")
                return
            
            vert_path = shader_dir / 'pbr.vert'
            frag_path = shader_dir / 'pbr.frag'
            
            if not vert_path.exists() or not frag_path.exists():
                print("⚠️ PBR shader files not found, PBR rendering disabled")
                return
            
            # Initialize shader manager
            self.shader_manager = ShaderManager()
            
            # Load PBR shader
            shader = self.shader_manager.load_shader('pbr', str(vert_path), str(frag_path))
            
            if shader:
                print("✓ PBR shaders loaded successfully")
                
                # Create VBO renderer if model supports it
                if self.model is not None and hasattr(self.model, 'faces'):
                    self.vbo_renderer = VBORenderer(self.model)
                else:
                    print("⚠️ Model doesn't support VBO rendering")
            else:
                print("⚠️ Failed to load PBR shaders")
                
        except Exception as e:
            print(f"⚠️ Error initializing shaders: {e}")
            self.shader_manager = None
            self.vbo_renderer = None
    
    def render_with_shaders(self):
        """Render model using PBR shaders."""
        if not self.shader_manager or not self.vbo_renderer:
            return
        
        shader = self.shader_manager.get_shader('pbr')
        if not shader:
            return
        
        # Use shader
        shader.use()
        
        try:
            # Get matrices
            model_matrix = self._get_model_matrix()
            view_matrix = self._get_view_matrix()
            projection_matrix = self._get_projection_matrix()
            normal_matrix = self._get_normal_matrix(model_matrix)
            
            # Set matrix uniforms
            shader.set_mat4('uModel', model_matrix)
            shader.set_mat4('uView', view_matrix)
            shader.set_mat4('uProjection', projection_matrix)
            shader.set_mat3('uNormalMatrix', normal_matrix)
            
            # Set camera position
            camera_pos = self._get_camera_position()
            shader.set_vec3('uCameraPos', camera_pos)
            
            # Set light properties
            br = self.brightness
            r, g, b = self.light_color
            
            light_positions = [
                [8.0, 10.0, 12.0],   # Key light (upper right front, far)
                [-6.0, 5.0, 8.0],    # Fill light (left, medium)
                [3.0, -3.0, -10.0]   # Rim light (back, far)
            ]
            light_colors = [[r, g, b], [r, g, b], [r, g, b]]
            light_intensities = [10.0 * br, 4.0 * br, 3.0 * br]
            
            shader.set_vec3_array('uLightPositions', light_positions)
            shader.set_vec3_array('uLightColors', light_colors)
            shader.set_float_array('uLightIntensities', light_intensities)
            
            # Set ambient strength and color
            shader.set_float('uAmbientStrength', self.ambient_strength)
            shader.set_vec3('uAmbientColor', self.ambient_color)
            
            # Set material property multipliers
            shader.set_float('uMetallicMultiplier', self.metallic_multiplier)
            shader.set_float('uRoughnessMultiplier', self.roughness_multiplier)
            shader.set_float('uNormalStrength', self.normal_strength)
            shader.set_float('uAOStrength', self.ao_strength)
            shader.set_float('uEmissiveIntensity', self.emissive_intensity)
            
            # Set HDR environment map
            if self.env_map_texture is not None:
                import math
                glActiveTexture(GL_TEXTURE5)  # Use texture unit 5 for env map
                glBindTexture(GL_TEXTURE_CUBE_MAP, self.env_map_texture)
                shader.set_int('uEnvMap', 5)
                shader.set_bool('uHasEnvMap', True)
                shader.set_float('uEnvIntensity', self.env_intensity)
                shader.set_float('uEnvRotation', math.radians(self.env_rotation))
            else:
                shader.set_bool('uHasEnvMap', False)
            
            # Set material properties
            self._set_material_uniforms(shader)
            
            # Bind textures
            self._bind_textures(shader)
            
            # Render
            self.vbo_renderer.render()
            
        finally:
            # Cleanup
            shader.unuse()
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 0)
    
    def _get_model_matrix(self):
        """Get model transformation matrix."""
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(self.pan_x, self.pan_y, self.zoom)
        glMultMatrixf(self.trackball.matrix)
        glScalef(self.model.scale_factor, self.model.scale_factor, self.model.scale_factor)
        glTranslatef(-self.model.center[0], -self.model.center[1], -self.model.center[2])
        
        matrix = glGetFloatv(GL_MODELVIEW_MATRIX)
        glPopMatrix()
        return matrix
    
    def _get_view_matrix(self):
        """Get view matrix (identity for now, transform is in model matrix)."""
        return np.eye(4, dtype=np.float32)
    
    def _get_projection_matrix(self):
        """Get projection matrix."""
        return np.array(glGetFloatv(GL_PROJECTION_MATRIX), dtype=np.float32)
    
    def _get_normal_matrix(self, model_matrix):
        """Get normal matrix (transpose(inverse(mat3(model))))."""
        # Extract 3x3 from 4x4
        mat3 = model_matrix[:3, :3]
        # Inverse and transpose
        normal_mat = np.linalg.inv(mat3).T
        return normal_mat.astype(np.float32)
    
    def load_hdr_environment(self, filepath):
        """Load HDR environment map and create cubemap texture."""
        try:
            print(f"\nLoading HDR environment: {filepath}")
            
            # Try multiple methods to load HDR
            hdr_image = None
            file_ext = filepath.lower().rsplit('.', 1)[-1] if '.' in filepath else ''
            
            # Method 1: Try OpenCV (best for HDR/EXR files)
            try:
                import cv2
                # Use IMREAD_UNCHANGED for EXR to preserve float data
                flags = cv2.IMREAD_UNCHANGED
                hdr_image = cv2.imread(filepath, flags)
                if hdr_image is not None:
                    # Handle different channel counts
                    if len(hdr_image.shape) == 2:
                        # Grayscale - convert to RGB
                        hdr_image = np.stack([hdr_image] * 3, axis=-1)
                    elif hdr_image.shape[2] == 4:
                        # BGRA -> RGB
                        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGRA2RGB)
                    elif hdr_image.shape[2] == 3:
                        # BGR -> RGB
                        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
                    print(f"  ✓ Loaded with OpenCV: {hdr_image.shape}, dtype: {hdr_image.dtype}")
                else:
                    print(f"  OpenCV returned None (format may not be supported by this build)")
            except ImportError:
                print(f"  OpenCV not available (install: pip install opencv-python)")
            except Exception as e:
                print(f"  OpenCV failed: {e}")
            
            # Method 2: Try OpenEXR + Imath for EXR files
            if hdr_image is None and file_ext == 'exr':
                try:
                    import OpenEXR
                    import Imath
                    
                    exr_file = OpenEXR.InputFile(filepath)
                    header = exr_file.header()
                    dw = header['dataWindow']
                    width = dw.max.x - dw.min.x + 1
                    height = dw.max.y - dw.min.y + 1
                    
                    pt = Imath.PixelType(Imath.PixelType.FLOAT)
                    channels = exr_file.channels(['R', 'G', 'B'], pt)
                    
                    r = np.frombuffer(channels[0], dtype=np.float32).reshape(height, width)
                    g = np.frombuffer(channels[1], dtype=np.float32).reshape(height, width)
                    b = np.frombuffer(channels[2], dtype=np.float32).reshape(height, width)
                    hdr_image = np.stack([r, g, b], axis=-1)
                    
                    print(f"  ✓ Loaded with OpenEXR: {hdr_image.shape}, dtype: {hdr_image.dtype}")
                except ImportError:
                    print(f"  OpenEXR not available (install: pip install OpenEXR)")
                except Exception as e:
                    print(f"  OpenEXR failed: {e}")
            
            # Method 3: Try imageio
            if hdr_image is None:
                try:
                    import imageio.v2 as iio
                    hdr_image = iio.imread(filepath, format=file_ext.upper() if file_ext else None)
                    print(f"  ✓ Loaded with imageio: {hdr_image.shape}, dtype: {hdr_image.dtype}")
                except Exception as e1:
                    print(f"  imageio failed: {e1}")
                    try:
                        import imageio
                        hdr_image = imageio.imread(filepath)
                        print(f"  ✓ Loaded with imageio (legacy): {hdr_image.shape}, dtype: {hdr_image.dtype}")
                    except Exception as e2:
                        print(f"  imageio legacy failed: {e2}")
                    except ImportError:
                        pass
            
            if hdr_image is None:
                raise RuntimeError(
                    "Could not load HDR file. Please install one of:\n"
                    "  pip install opencv-contrib-python  (recommended, includes EXR support)\n"
                    "  pip install OpenEXR Imath          (for EXR files)\n"
                    "  pip install imageio[ffmpeg]"
                )
            
            # Convert to float32 if needed
            if hdr_image.dtype != np.float32:
                hdr_image = hdr_image.astype(np.float32)
            
            # Ensure RGB (some HDRs might be RGBA)
            if len(hdr_image.shape) == 3 and hdr_image.shape[2] == 4:
                hdr_image = hdr_image[:, :, :3]
                print(f"  Converted RGBA to RGB")
            
            # For now, create a simple cubemap by sampling the equirectangular map
            cubemap_size = 512
            
            # Delete old texture if exists
            if self.env_map_texture is not None:
                glDeleteTextures([self.env_map_texture])
            
            # Create cubemap texture
            self.env_map_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_CUBE_MAP, self.env_map_texture)
            
            # Generate mipmaps for roughness-based sampling
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
            
            # Convert equirectangular to cubemap faces
            print(f"  Converting to cubemap ({cubemap_size}x{cubemap_size})...")
            for face in range(6):
                face_data = self._sample_cubemap_face(hdr_image, face, cubemap_size)
                glTexImage2D(
                    GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                    0, GL_RGB16F, cubemap_size, cubemap_size,
                    0, GL_RGB, GL_FLOAT, face_data
                )
            
            # Generate mipmaps
            glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
            
            glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
            
            print(f"  ✓ HDR environment loaded successfully!")
            print(f"  Cubemap texture ID: {self.env_map_texture}")
            self.update()
            return True
            
        except Exception as e:
            print(f"  ✗ Error loading HDR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _sample_cubemap_face(self, equirect, face, size):
        """Sample cubemap face from equirectangular image."""
        h, w = equirect.shape[:2]
        face_data = np.zeros((size, size, 3), dtype=np.float32)
        
        # Define cube face directions
        # Face order: +X, -X, +Y, -Y, +Z, -Z
        for y in range(size):
            for x in range(size):
                # Normalize to [-1, 1]
                u = (x / (size - 1)) * 2.0 - 1.0
                v = (y / (size - 1)) * 2.0 - 1.0
                
                # Get direction vector for this face
                if face == 0:    # +X
                    direction = np.array([1.0, -v, -u])
                elif face == 1:  # -X
                    direction = np.array([-1.0, -v, u])
                elif face == 2:  # +Y
                    direction = np.array([u, 1.0, v])
                elif face == 3:  # -Y
                    direction = np.array([u, -1.0, -v])
                elif face == 4:  # +Z
                    direction = np.array([u, -v, 1.0])
                else:            # -Z
                    direction = np.array([-u, -v, -1.0])
                
                # Normalize direction
                direction = direction / np.linalg.norm(direction)
                
                # Convert to equirectangular coordinates
                theta = np.arctan2(direction[0], direction[2])  # Horizontal angle
                phi = np.arcsin(np.clip(direction[1], -1.0, 1.0))  # Vertical angle
                
                # Map to texture coordinates
                tex_u = (theta / (2.0 * np.pi)) + 0.5
                tex_v = 1.0 - ((phi / np.pi) + 0.5)  # Flip vertical (HDR convention)
                
                # Sample equirectangular image
                px = int(tex_u * (w - 1))
                py = int(tex_v * (h - 1))
                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)
                
                face_data[y, x] = equirect[py, px]
        
        return face_data
    
    def _get_normal_matrix(self, model_matrix):
        """Get normal matrix (transpose(inverse(mat3(model))))."""
        # Extract 3x3 from 4x4
        mat3 = model_matrix[:3, :3]
        # Inverse and transpose
        normal_mat = np.linalg.inv(mat3).T
        return normal_mat.astype(np.float32)
    
    def _get_camera_position(self):
        """Get camera position in world space."""
        # Camera is at (pan_x, pan_y, zoom) after trackball rotation
        # For simplicity, use approximate position
        return [self.pan_x, self.pan_y, abs(self.zoom)]
    
    def _set_material_uniforms(self, shader):
        """Set material property uniforms."""
        # Get first material (simplified for now)
        if self.model.materials:
            mat = list(self.model.materials.values())[0]
            
            # Base color factor
            base_color = mat.get('baseColorFactor')
            if base_color is not None and len(base_color) >= 3:
                shader.set_vec3('uBaseColorFactor', base_color[:3])
            elif 'diffuse' in mat and mat['diffuse'] is not None:
                shader.set_vec3('uBaseColorFactor', mat['diffuse'])
            else:
                shader.set_vec3('uBaseColorFactor', 0.7, 0.6, 0.5)
            
            # Metallic factor
            metallic = mat.get('metallicFactor')
            # If there's a metallic texture but factor is 0, default to 1.0 (let texture control it)
            if metallic is None or metallic == 0.0:
                if 'texture_metallic_roughness' in mat:
                    metallic_value = 1.0  # Use texture values
                else:
                    metallic_value = 0.0
            else:
                metallic_value = metallic
            shader.set_float('uMetallicFactor', metallic_value)
            
            # Roughness factor
            roughness = mat.get('roughnessFactor')
            roughness_value = roughness if roughness is not None else 1.0
            shader.set_float('uRoughnessFactor', roughness_value)
            
            # Emissive factor
            emissive = mat.get('emissiveFactor')
            if emissive is not None and len(emissive) >= 3:
                shader.set_vec3('uEmissiveFactor', emissive[:3])
            else:
                shader.set_vec3('uEmissiveFactor', 0.0, 0.0, 0.0)
        else:
            # Default material
            shader.set_vec3('uBaseColorFactor', 0.7, 0.6, 0.5)
            shader.set_float('uMetallicFactor', 0.0)
            shader.set_float('uRoughnessFactor', 0.5)
            shader.set_vec3('uEmissiveFactor', 0.0, 0.0, 0.0)
    
    def _bind_textures(self, shader):
        """Bind all PBR texture maps."""
        texture_unit = 0
        
        # Get first material
        if not self.model.materials:
            self._set_default_texture_flags(shader)
            return
        
        mat = list(self.model.materials.values())[0]
        
        # Debug: Show what texture keys exist (only once)
        debug = not self._texture_debug_printed
        if debug:
            texture_keys = [k for k in mat.keys() if k.startswith('texture_')]
            print(f"\n[TEXTURE BINDING DEBUG]")
            print(f"Available texture keys: {texture_keys}\n")
        
        # Base color map
        if 'texture_base_color' in mat:
            glActiveTexture(GL_TEXTURE0 + texture_unit)
            glBindTexture(GL_TEXTURE_2D, mat['texture_base_color'])
            shader.set_int('uBaseColorMap', texture_unit)
            shader.set_bool('uHasBaseColorMap', True)
            if debug:
                print(f"  ✓ Base color bound to unit {texture_unit}")
            texture_unit += 1
        else:
            shader.set_bool('uHasBaseColorMap', False)
        
        # Metallic/Roughness map
        if 'texture_metallic_roughness' in mat:
            glActiveTexture(GL_TEXTURE0 + texture_unit)
            glBindTexture(GL_TEXTURE_2D, mat['texture_metallic_roughness'])
            shader.set_int('uMetallicRoughnessMap', texture_unit)
            shader.set_bool('uHasMetallicRoughnessMap', True)
            if debug:
                print(f"  ✓ Metallic/Roughness bound to unit {texture_unit}")
            texture_unit += 1
        else:
            shader.set_bool('uHasMetallicRoughnessMap', False)
        
        # Normal map
        if 'texture_normal' in mat:
            glActiveTexture(GL_TEXTURE0 + texture_unit)
            glBindTexture(GL_TEXTURE_2D, mat['texture_normal'])
            shader.set_int('uNormalMap', texture_unit)
            shader.set_bool('uHasNormalMap', True)
            if debug:
                print(f"  ✓ Normal bound to unit {texture_unit}")
            texture_unit += 1
        else:
            shader.set_bool('uHasNormalMap', False)
        
        # Occlusion map
        if 'texture_occlusion' in mat:
            glActiveTexture(GL_TEXTURE0 + texture_unit)
            glBindTexture(GL_TEXTURE_2D, mat['texture_occlusion'])
            shader.set_int('uOcclusionMap', texture_unit)
            shader.set_bool('uHasOcclusionMap', True)
            if debug:
                print(f"  ✓ Occlusion (AO) bound to unit {texture_unit}")
            texture_unit += 1
        else:
            shader.set_bool('uHasOcclusionMap', False)
            if debug:
                print(f"  ✗ Occlusion (AO) NOT FOUND")
        
        # Emissive map
        if 'texture_emissive' in mat:
            glActiveTexture(GL_TEXTURE0 + texture_unit)
            glBindTexture(GL_TEXTURE_2D, mat['texture_emissive'])
            shader.set_int('uEmissiveMap', texture_unit)
            shader.set_bool('uHasEmissiveMap', True)
            if debug:
                print(f"  ✓ Emissive bound to unit {texture_unit}")
            texture_unit += 1
        else:
            shader.set_bool('uHasEmissiveMap', False)
            if debug:
                print(f"  ✗ Emissive NOT FOUND")
        
        # Mark as printed
        if debug:
            print()  # Empty line
            self._texture_debug_printed = True
            shader.set_int('uEmissiveMap', texture_unit)
            shader.set_bool('uHasEmissiveMap', True)
            texture_unit += 1
        else:
            shader.set_bool('uHasEmissiveMap', False)
    
    def _set_default_texture_flags(self, shader):
        """Set all texture flags to false."""
        shader.set_bool('uHasBaseColorMap', False)
        shader.set_bool('uHasMetallicRoughnessMap', False)
        shader.set_bool('uHasNormalMap', False)
        shader.set_bool('uHasOcclusionMap', False)
        shader.set_bool('uHasEmissiveMap', False)
    
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h > 0 else 1, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Nothing to render if no model loaded
        if self.model is None:
            self._draw_empty_viewport()
            return
        
        # Use PBR shader rendering if enabled and available
        if self.pbr_enabled and self.shader_manager and self.vbo_renderer:
            self.render_with_shaders()
            # Draw debug overlays in fixed-function after PBR
            self._draw_debug_overlays()
            self._draw_stats_overlay()
            return
        
        # Otherwise use fixed-function pipeline
        glLoadIdentity()
        
        glTranslatef(self.pan_x, self.pan_y, self.zoom)
        glMultMatrixf(self.trackball.matrix)
        
        if self.display_mode == 'wireframe':
            glDisable(GL_LIGHTING)
            glColor3f(0.8, 0.8, 0.8)
            self.model.render('wireframe', enable_textures=self.texture_enabled)
        elif self.display_mode == 'wireframe_culled':
            # Two-pass hidden-line wireframe:
            # Pass 1: Render solid to depth buffer only (no color output)
            glDisable(GL_LIGHTING)
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
            self.model.render('smooth', enable_textures=False)
            glDisable(GL_CULL_FACE)
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
            
            # Pass 2: Render wireframe with depth test (hidden lines blocked)
            glColor3f(0.8, 0.8, 0.8)
            glEnable(GL_POLYGON_OFFSET_LINE)
            glPolygonOffset(-1.0, -1.0)
            self.model.render('wireframe', enable_textures=False)
            glDisable(GL_POLYGON_OFFSET_LINE)
        elif self.display_mode == 'solid_wire':
            if self.lighting_enabled:
                glEnable(GL_LIGHTING)
            self.model.render('smooth', enable_textures=self.texture_enabled)
            
            # Force black wireframe: enable lighting with all-black material
            # and no lights, so display list glColor calls are ignored
            glEnable(GL_LIGHTING)
            glDisable(GL_LIGHT0)
            glDisable(GL_LIGHT1)
            glDisable(GL_LIGHT2)
            glDisable(GL_COLOR_MATERIAL)
            black = (GLfloat * 4)(0.0, 0.0, 0.0, 1.0)
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, black)
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, black)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black)
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black)
            
            glEnable(GL_POLYGON_OFFSET_LINE)
            glPolygonOffset(-1.0, -1.0)
            self.model.render('wireframe', enable_textures=False)
            glDisable(GL_POLYGON_OFFSET_LINE)
            
            # Restore state
            glEnable(GL_COLOR_MATERIAL)
            self.setup_lighting()
        else:
            if self.lighting_enabled:
                glEnable(GL_LIGHTING)
            self.model.render(self.display_mode, enable_textures=self.texture_enabled)
        
        # Draw debug overlays
        self._draw_debug_overlays()
        
        # Draw stats overlay
        self._draw_stats_overlay()
    
    def _cache_model_stats(self):
        """Cache vertex, face, and triangle counts from the loaded model."""
        self.stat_verts = 0
        self.stat_faces = 0
        self.stat_tris = 0
        
        if not self.model:
            return
        
        if hasattr(self.model, 'verts'):
            self.stat_verts = len(self.model.verts)
        
        if hasattr(self.model, 'faces'):
            self.stat_faces = len(self.model.faces)
            # Count triangles: faces with N vertices produce N-2 triangles
            for face in self.model.faces:
                vert_indices = face[0]  # First element is vertex index tuple
                n = len(vert_indices)
                self.stat_tris += max(0, n - 2)
    
    def _draw_stats_overlay(self):
        """Draw vertex/face/triangle count in the lower-left corner."""
        if self.model is None:
            return
        from PyQt5.QtGui import QFont, QColor
        
        font = QFont("Consolas", 9)
        self.setFont(font)
        glColor3f(0.7, 0.7, 0.7)
        
        margin = 40
        line_height = 16
        y = self.height() - margin - line_height * 2
        
        stats = [
            f"Verts: {self.stat_verts:,}",
            f"Faces: {self.stat_faces:,}",
            f"Tris:  {self.stat_tris:,}",
        ]
        
        # Disable lighting/depth so text is always visible
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        for text in stats:
            self.renderText(margin, y, text)
            y += line_height
        
        glEnable(GL_DEPTH_TEST)
    
    def _draw_debug_overlays(self):
        """Draw all enabled debug visualizations."""
        if self.model is None:
            return
        if self.show_uv_checker:
            self._draw_uv_checker()
        if self.show_bounding_box:
            self._draw_bounding_box()
        if self.show_vertex_normals:
            self._draw_vertex_normals()
    
    def _create_uv_checker_texture(self):
        """Generate a procedural checkerboard texture."""
        size = 256
        checks = 16
        cell = size // checks
        
        data = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            for x in range(size):
                cx = x // cell
                cy = y // cell
                if (cx + cy) % 2 == 0:
                    data[y, x] = [230, 230, 230]
                else:
                    data[y, x] = [40, 40, 40]
        
        # Add colored markers in corners for orientation
        # Top-left red, top-right green, bottom-left blue
        marker = cell
        data[0:marker, 0:marker] = [200, 60, 60]
        data[0:marker, size-marker:size] = [60, 200, 60]
        data[size-marker:size, 0:marker] = [60, 60, 200]
        
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        return tex_id
    
    def _draw_uv_checker(self):
        """Render the model with a UV checker texture overlay."""
        if self._uv_checker_texture is None:
            self._uv_checker_texture = self._create_uv_checker_texture()
            self._uv_checker_display_list = None  # Rebuild with new texture
        
        if self._uv_checker_display_list is None:
            self._uv_checker_display_list = self._build_uv_checker_display_list()
        
        # Set up fixed-function for checker overlay
        glUseProgram(0)
        
        glLoadIdentity()
        glTranslatef(self.pan_x, self.pan_y, self.zoom)
        glMultMatrixf(self.trackball.matrix)
        
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glBindTexture(GL_TEXTURE_2D, self._uv_checker_texture)
        glColor4f(1.0, 1.0, 1.0, 0.85)
        
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)
        
        glCallList(self._uv_checker_display_list)
        
        glDisable(GL_POLYGON_OFFSET_FILL)
        glDisable(GL_BLEND)
        glDisable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        if self.lighting_enabled:
            glEnable(GL_LIGHTING)
    
    def _build_uv_checker_display_list(self):
        """Build a display list for UV checker geometry."""
        model = self.model
        sf = model.scale_factor
        mcx, mcy, mcz = model.center
        
        dl = glGenLists(1)
        glNewList(dl, GL_COMPILE)
        
        glPushMatrix()
        glScalef(sf, sf, sf)
        glTranslatef(-mcx, -mcy, -mcz)
        
        for face in model.faces:
            vertices, normals, texture_coords, material = face
            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if texture_coords[i] > 0 and texture_coords[i] <= len(model.texcoords):
                    glTexCoord2fv(model.texcoords[texture_coords[i] - 1])
                else:
                    v = model.verts[vertices[i] - 1]
                    glTexCoord2f(v[0], v[1])
                if vertices[i] > 0 and vertices[i] <= len(model.verts):
                    glVertex3fv(model.verts[vertices[i] - 1])
            glEnd()
        
        glPopMatrix()
        glEndList()
        
        return dl
    
    def _draw_bounding_box(self):
        """Draw wireframe bounding box with dimension labels."""
        model = self.model
        if not hasattr(model, 'verts') or not model.verts:
            return
        
        glUseProgram(0)
        
        glLoadIdentity()
        glTranslatef(self.pan_x, self.pan_y, self.zoom)
        glMultMatrixf(self.trackball.matrix)
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        sf = model.scale_factor
        mcx, mcy, mcz = model.center
        
        # Calculate bounds
        min_x = min(v[0] for v in model.verts)
        max_x = max(v[0] for v in model.verts)
        min_y = min(v[1] for v in model.verts)
        max_y = max(v[1] for v in model.verts)
        min_z = min(v[2] for v in model.verts)
        max_z = max(v[2] for v in model.verts)
        
        # Original dimensions (before scaling)
        dim_x = max_x - min_x
        dim_y = max_y - min_y
        dim_z = max_z - min_z
        
        # Transform bounds to match model rendering transform
        x0 = (min_x - mcx) * sf
        x1 = (max_x - mcx) * sf
        y0 = (min_y - mcy) * sf
        y1 = (max_y - mcy) * sf
        z0 = (min_z - mcz) * sf
        z1 = (max_z - mcz) * sf
        
        # Draw box edges
        glColor3f(0.0, 1.0, 0.5)
        glLineWidth(1.5)
        
        glBegin(GL_LINES)
        # Bottom face
        for a, b in [((x0,y0,z0),(x1,y0,z0)), ((x1,y0,z0),(x1,y0,z1)),
                      ((x1,y0,z1),(x0,y0,z1)), ((x0,y0,z1),(x0,y0,z0))]:
            glVertex3f(*a); glVertex3f(*b)
        # Top face
        for a, b in [((x0,y1,z0),(x1,y1,z0)), ((x1,y1,z0),(x1,y1,z1)),
                      ((x1,y1,z1),(x0,y1,z1)), ((x0,y1,z1),(x0,y1,z0))]:
            glVertex3f(*a); glVertex3f(*b)
        # Vertical edges
        for x, z in [(x0,z0),(x1,z0),(x1,z1),(x0,z1)]:
            glVertex3f(x, y0, z); glVertex3f(x, y1, z)
        glEnd()
        
        glLineWidth(1.0)
        
        # Draw dimension labels
        glColor3f(0.0, 1.0, 0.5)
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        mid_z = (z0 + z1) / 2
        
        from PyQt5.QtGui import QFont
        font = QFont("Consolas", 8)
        self.setFont(font)
        
        self.renderText(mid_x, y0 - 0.3, z1, f"X: {dim_x:.2f}")
        self.renderText(x1 + 0.3, mid_y, z1, f"Y: {dim_y:.2f}")
        self.renderText(x1 + 0.3, y0 - 0.3, mid_z, f"Z: {dim_z:.2f}")
        
        glEnable(GL_DEPTH_TEST)
        if self.lighting_enabled:
            glEnable(GL_LIGHTING)
    
    def _build_normals_display_list(self):
        """Build a display list for vertex normals visualization."""
        model = self.model
        if not hasattr(model, 'verts') or not hasattr(model, 'faces'):
            return None
        
        sf = model.scale_factor
        mcx, mcy, mcz = model.center
        
        # Determine normal line length relative to model size
        min_x = min(v[0] for v in model.verts)
        max_x = max(v[0] for v in model.verts)
        min_y = min(v[1] for v in model.verts)
        max_y = max(v[1] for v in model.verts)
        min_z = min(v[2] for v in model.verts)
        max_z = max(v[2] for v in model.verts)
        extent = max(max_x - min_x, max_y - min_y, max_z - min_z)
        normal_length = extent * 0.02  # 2% of model size
        
        dl = glGenLists(1)
        glNewList(dl, GL_COMPILE)
        glBegin(GL_LINES)
        
        # Collect unique vertex-normal pairs from faces
        drawn = set()
        for face in model.faces:
            vertices, normals_idx, _, _ = face
            for i in range(len(vertices)):
                vi = vertices[i]
                ni = normals_idx[i] if i < len(normals_idx) else 0
                
                key = (vi, ni)
                if key in drawn or vi <= 0 or vi > len(model.verts):
                    continue
                drawn.add(key)
                
                v = model.verts[vi - 1]
                vx = (v[0] - mcx) * sf
                vy = (v[1] - mcy) * sf
                vz = (v[2] - mcz) * sf
                
                if ni > 0 and ni <= len(model.normals):
                    n = model.normals[ni - 1]
                else:
                    continue  # Skip vertices without normals
                
                ex = vx + n[0] * normal_length * sf
                ey = vy + n[1] * normal_length * sf
                ez = vz + n[2] * normal_length * sf
                
                # Color gradient: blue at base, cyan at tip
                glColor3f(0.2, 0.4, 1.0)
                glVertex3f(vx, vy, vz)
                glColor3f(0.2, 1.0, 1.0)
                glVertex3f(ex, ey, ez)
        
        glEnd()
        glEndList()
        
        return dl
    
    def _draw_vertex_normals(self):
        """Draw vertex normals as colored lines."""
        if self._normals_display_list is None:
            self._normals_display_list = self._build_normals_display_list()
            if self._normals_display_list is None:
                return
        
        glUseProgram(0)
        
        glLoadIdentity()
        glTranslatef(self.pan_x, self.pan_y, self.zoom)
        glMultMatrixf(self.trackball.matrix)
        
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        
        glCallList(self._normals_display_list)
        
        if self.lighting_enabled:
            glEnable(GL_LIGHTING)
    
    def load_new_model(self, model):
        """Replace the current model with a new one at runtime."""
        self.makeCurrent()
        
        # Clean up old VBO renderer
        if self.vbo_renderer:
            self.vbo_renderer.cleanup()
            self.vbo_renderer = None
        
        # Clean up old debug display lists
        if self._uv_checker_display_list is not None:
            glDeleteLists(self._uv_checker_display_list, 1)
            self._uv_checker_display_list = None
        if self._bbox_display_list is not None:
            glDeleteLists(self._bbox_display_list, 1)
            self._bbox_display_list = None
        if self._normals_display_list is not None:
            glDeleteLists(self._normals_display_list, 1)
            self._normals_display_list = None
        
        # Swap model
        self.model = model
        self._texture_debug_printed = False
        
        # Compile display lists for the new model
        if self.model is not None:
            self.model.compile_display_lists()
            
            # Rebuild VBO renderer if shaders are available
            if self.shader_manager and self.shader_manager.get_shader('pbr'):
                if hasattr(self.model, 'faces'):
                    self.vbo_renderer = VBORenderer(self.model)
        
        # Re-cache stats
        self._cache_model_stats()
        
        # Reset camera
        self.reset_camera()
        
        self.doneCurrent()
        self.update()
    
    def _draw_empty_viewport(self):
        """Draw a hint message when no model is loaded."""
        from PyQt5.QtGui import QFont
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glColor3f(0.4, 0.4, 0.45)
        font = QFont("Segoe UI", 14)
        self.setFont(font)
        self.renderText(self.width() // 2 - 120, self.height() // 2, "Open a model to get started")
        glEnable(GL_DEPTH_TEST)
    
    def mousePressEvent(self, event):
        self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        
        dx = event.x() - self.last_pos.x()
        dy = event.y() - self.last_pos.y()
        
        if event.buttons() & Qt.RightButton:
            self.trackball.drag(dx, dy, self.width(), self.height())
            self.update()
        elif event.buttons() & Qt.MiddleButton:
            self.pan_x += dx * 0.01
            self.pan_y -= dy * 0.01
            self.update()
        
        self.last_pos = event.pos()
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom += delta * 0.01
        self.update()
    
    def reset_camera(self):
        """Reset camera to default position."""
        self.trackball = Trackball(theta=-25)
        self.zoom = -20
        self.pan_x = 0
        self.pan_y = 0
        self.update()
    
    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E):
            self.keys_pressed.add(key)
            if not self.move_timer.isActive():
                self.move_timer.start()
            event.accept()
        elif key == Qt.Key_R:
            self.reset_camera()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        key = event.key()
        self.keys_pressed.discard(key)
        if not self.keys_pressed and self.move_timer.isActive():
            self.move_timer.stop()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E):
            event.accept()
        else:
            super().keyReleaseEvent(event)
    
    def _process_movement(self):
        """Process held WASD/QE keys for smooth camera movement."""
        moved = False
        if Qt.Key_W in self.keys_pressed:
            self.zoom += self.move_speed
            moved = True
        if Qt.Key_S in self.keys_pressed:
            self.zoom -= self.move_speed
            moved = True
        if Qt.Key_A in self.keys_pressed:
            self.pan_x -= self.move_speed * 0.5
            moved = True
        if Qt.Key_D in self.keys_pressed:
            self.pan_x += self.move_speed * 0.5
            moved = True
        if Qt.Key_Q in self.keys_pressed:
            self.pan_y += self.move_speed * 0.5
            moved = True
        if Qt.Key_E in self.keys_pressed:
            self.pan_y -= self.move_speed * 0.5
            moved = True
        if moved:
            self.update()
