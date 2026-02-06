"""
VBO Renderer Module
Handles vertex buffer object rendering for PBR shaders.
"""

import numpy as np
import ctypes
from OpenGL.GL import *


class VBORenderer:
    """Manages VBO/VAO for shader-based rendering."""
    
    def __init__(self, model):
        """Create VBO/VAO from model data."""
        self.model = model
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.index_count = 0
        
        self._create_buffers()
    
    def _create_buffers(self):
        """Create vertex and element buffer objects."""
        print("Creating VBOs for PBR rendering...")
        
        # Prepare vertex data
        vertices, indices = self._prepare_vertex_data()
        
        if len(vertices) == 0:
            print("⚠️ No vertex data to create VBOs")
            return
        
        # Create VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Create VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Create EBO (Element Buffer Object)
        if len(indices) > 0:
            self.ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
            self.index_count = len(indices)
        
        # Set up vertex attributes
        stride = 11 * 4  # 11 floats per vertex (pos=3, norm=3, uv=2, tan=3) * 4 bytes
        
        # Position (location = 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        
        # Normal (location = 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        
        # TexCoord (location = 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        
        # Tangent (location = 3)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
        
        # Unbind
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        
        print(f"✓ Created VBO: {len(vertices)} vertices, {self.index_count} indices")
    
    def _prepare_vertex_data(self):
        """Prepare interleaved vertex data and indices."""
        vertices_list = []
        indices_list = []
        vertex_map = {}  # Map (pos, norm, uv, tan) -> index
        current_index = 0
        
        # Process each face
        for face in self.model.faces:
            face_verts, face_normals, face_uvs, material_idx = face
            
            face_indices = []
            
            for i in range(len(face_verts)):
                v_idx = face_verts[i] - 1
                n_idx = face_normals[i] - 1 if face_normals[i] > 0 else v_idx
                uv_idx = face_uvs[i] - 1 if face_uvs[i] > 0 else 0
                
                # Get vertex data
                pos = self.model.verts[v_idx] if v_idx < len(self.model.verts) else (0, 0, 0)
                norm = self.model.normals[n_idx] if n_idx < len(self.model.normals) else (0, 1, 0)
                uv = self.model.texcoords[uv_idx] if uv_idx < len(self.model.texcoords) else (0, 0)
                tan = self.model.tangents[v_idx] if v_idx < len(self.model.tangents) else (1, 0, 0)
                
                # Create vertex key
                vertex_key = (pos, norm, uv, tan)
                
                # Check if vertex already exists
                if vertex_key in vertex_map:
                    face_indices.append(vertex_map[vertex_key])
                else:
                    # Add new vertex
                    vertices_list.extend(pos)    # Position (3 floats)
                    vertices_list.extend(norm)   # Normal (3 floats)
                    vertices_list.extend(uv)     # TexCoord (2 floats)
                    vertices_list.extend(tan)    # Tangent (3 floats)
                    
                    vertex_map[vertex_key] = current_index
                    face_indices.append(current_index)
                    current_index += 1
            
            # Triangulate face (simple fan triangulation for n-gons)
            for i in range(1, len(face_indices) - 1):
                indices_list.append(face_indices[0])
                indices_list.append(face_indices[i])
                indices_list.append(face_indices[i + 1])
        
        # Convert to numpy arrays
        vertices = np.array(vertices_list, dtype=np.float32)
        indices = np.array(indices_list, dtype=np.uint32)
        
        return vertices, indices
    
    def render(self):
        """Render using VAO."""
        if self.vao and self.index_count > 0:
            glBindVertexArray(self.vao)
            glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
    
    def cleanup(self):
        """Delete buffers."""
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.ebo:
            glDeleteBuffers(1, [self.ebo])
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
