"""
OBJ Model Loader Module
Handles loading and rendering of Wavefront OBJ files with MTL materials.
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from OpenGL.GL import *


def load_mtl(filename: str) -> Dict[str, Dict]:
    """Load MTL file - returns empty dict if not found."""
    contents = {}
    mtl = None
    
    filepath = Path(filename)
    if not filepath.exists():
        print(f"⚠️  Warning: MTL file not found: {filename}")
        return {}
    
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                values = line.split()
                if not values:
                    continue
                
                if values[0] == 'newmtl':
                    mtl = contents[values[1]] = {}
                elif mtl is None:
                    continue
                elif values[0] == 'map_Kd':
                    mtl[values[0]] = values[1]
                    texture_path = Path(filename).parent / mtl['map_Kd']
                    image = Image.open(texture_path).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    image_data = np.frombuffer(image.tobytes(), dtype=np.uint8)
                    width, height = image.size
                    
                    glEnable(GL_TEXTURE_2D)
                    texname = mtl['texture_Kd'] = glGenTextures(1)
                    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
                    glBindTexture(GL_TEXTURE_2D, texname)
                    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                                GL_RGB, GL_UNSIGNED_BYTE, image_data)
                else:
                    mtl[values[0]] = [float(x) for x in values[1:]]
    except Exception as e:
        print(f"⚠️  Warning: Error loading MTL: {e}")
        return {}
    
    return contents


class OBJModel:
    """Wavefront OBJ file loader with display lists."""
    
    def __init__(self, filename: str, flip_yz: bool = False, 
                 default_color: Tuple[float, float, float] = (0.7, 0.6, 0.5)):
        print(f"\nLoading: {filename}")
        
        self.verts = []
        self.normals = []
        self.texcoords = []
        self.tangents = []
        self.faces = []
        self.mtl = None
        self.materials = {}  # PBR material dict (empty for OBJ, used by shader pipeline)
        self.default_color = default_color
        self.center = [0, 0, 0]
        self.scale_factor = 1.0
        
        self.dl_smooth = None
        self.dl_flat = None
        self.dl_wireframe = None
        
        self._load_obj(filename, flip_yz)
        self._analyze_model()
    
    def _load_obj(self, filename: str, flip_yz: bool):
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"OBJ file not found: {filename}")
        
        material = None
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                values = line.split()
                if not values:
                    continue
                
                if values[0] == 'v':
                    v = [float(x) for x in values[1:4]]
                    if flip_yz:
                        v = [v[0], v[2], v[1]]
                    self.verts.append(tuple(v))
                elif values[0] == 'vn':
                    v = [float(x) for x in values[1:4]]
                    if flip_yz:
                        v = [v[0], v[2], v[1]]
                    self.normals.append(tuple(v))
                elif values[0] == 'vt':
                    self.texcoords.append(tuple([float(x) for x in values[1:3]]))
                elif values[0] in ('usemtl', 'usemat'):
                    material = values[1]
                elif values[0] == 'mtllib':
                    mtl_path = filepath.parent / values[1]
                    self.mtl = load_mtl(str(mtl_path))
                elif values[0] == 'f':
                    face = []
                    texcoords = []
                    norms = []
                    for v in values[1:]:
                        w = v.split('/')
                        face.append(int(w[0]))
                        texcoords.append(int(w[1]) if len(w) >= 2 and w[1] else 0)
                        norms.append(int(w[2]) if len(w) >= 3 and w[2] else 0)
                    self.faces.append((face, norms, texcoords, material))
    
    def _analyze_model(self):
        if not self.verts:
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
        
        print(f"Vertices: {len(self.verts)}, Faces: {len(self.faces)}")
    
    def compile_display_lists(self):
        """Compile display lists (call after GL context is created)."""
        print("Compiling display lists...")
        
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
        
        print("✓ Display lists ready")
    
    def _render_geometry(self, mode):
        glPushMatrix()
        glScalef(self.scale_factor, self.scale_factor, self.scale_factor)
        glTranslatef(-self.center[0], -self.center[1], -self.center[2])
        
        for face in self.faces:
            vertices, normals, texture_coords, material = face
            
            material_applied = False
            if self.mtl and material and material in self.mtl:
                mtl = self.mtl[material]
                if 'texture_Kd' in mtl:
                    glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
                    material_applied = True
                elif 'Kd' in mtl:
                    glColor(*mtl['Kd'])
                    material_applied = True
            
            if not material_applied:
                glColor(*self.default_color)
            
            if mode == 'flat' or not any(normals):
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
                if mode == 'smooth' and normals[i] > 0 and normals[i] <= len(self.normals):
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0 and texture_coords[i] <= len(self.texcoords):
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
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
        if self.mtl:
            return any('texture_Kd' in mat for mat in self.mtl.values())
        return False
    
    def get_available_maps(self):
        """Get list of available texture map types. OBJ only supports base color."""
        if self.has_textures():
            return ['base_color']
        return []
    
    def has_map_type(self, map_type):
        """Check if model has a specific texture map type."""
        return map_type == 'base_color' and self.has_textures()
