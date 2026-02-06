"""
Shader Manager Module
Handles GLSL shader compilation, linking, and uniform management.
"""

from pathlib import Path
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np


class ShaderProgram:
    """Manages a GLSL shader program."""
    
    def __init__(self, vertex_path, fragment_path):
        """Load and compile shaders from files."""
        self.program_id = None
        self.uniform_locations = {}
        
        # Load shader source code
        vertex_src = self._load_shader_source(vertex_path)
        fragment_src = self._load_shader_source(fragment_path)
        
        # Compile and link
        self._compile_and_link(vertex_src, fragment_src)
        
        print(f"✓ Shader program created from {Path(vertex_path).name} + {Path(fragment_path).name}")
    
    def _load_shader_source(self, filepath):
        """Load shader source code from file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Shader file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return f.read()
    
    def _compile_and_link(self, vertex_src, fragment_src):
        """Compile shaders and link program."""
        try:
            # Compile vertex shader
            vertex_shader = compileShader(vertex_src, GL_VERTEX_SHADER)
            
            # Compile fragment shader
            fragment_shader = compileShader(fragment_src, GL_FRAGMENT_SHADER)
            
            # Link program
            self.program_id = compileProgram(vertex_shader, fragment_shader)
            
            # Clean up individual shaders (they're now part of the program)
            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)
            
        except Exception as e:
            print(f"✗ Shader compilation error:")
            print(str(e))
            raise
    
    def use(self):
        """Activate this shader program."""
        if self.program_id:
            glUseProgram(self.program_id)
    
    def unuse(self):
        """Deactivate shader program."""
        glUseProgram(0)
    
    def get_uniform_location(self, name):
        """Get uniform location (cached)."""
        if name not in self.uniform_locations:
            self.uniform_locations[name] = glGetUniformLocation(self.program_id, name)
        return self.uniform_locations[name]
    
    # Uniform setters
    def set_bool(self, name, value):
        """Set boolean uniform."""
        loc = self.get_uniform_location(name)
        if loc != -1:
            glUniform1i(loc, int(value))
    
    def set_int(self, name, value):
        """Set integer uniform."""
        loc = self.get_uniform_location(name)
        if loc != -1:
            glUniform1i(loc, value)
    
    def set_float(self, name, value):
        """Set float uniform."""
        loc = self.get_uniform_location(name)
        if loc != -1:
            glUniform1f(loc, value)
    
    def set_vec3(self, name, x, y=None, z=None):
        """Set vec3 uniform."""
        loc = self.get_uniform_location(name)
        if loc != -1:
            if y is None:  # Assume x is a tuple/list
                glUniform3f(loc, x[0], x[1], x[2])
            else:
                glUniform3f(loc, x, y, z)
    
    def set_vec3_array(self, name, values):
        """Set array of vec3 uniforms."""
        loc = self.get_uniform_location(name)
        if loc != -1:
            # Flatten array and convert to float32
            flat = np.array(values, dtype=np.float32).flatten()
            glUniform3fv(loc, len(values), flat)
    
    def set_float_array(self, name, values):
        """Set array of float uniforms."""
        loc = self.get_uniform_location(name)
        if loc != -1:
            arr = np.array(values, dtype=np.float32)
            glUniform1fv(loc, len(values), arr)
    
    def set_mat3(self, name, matrix):
        """Set mat3 uniform."""
        loc = self.get_uniform_location(name)
        if loc != -1:
            mat = np.array(matrix, dtype=np.float32)
            glUniformMatrix3fv(loc, 1, GL_FALSE, mat)
    
    def set_mat4(self, name, matrix):
        """Set mat4 uniform."""
        loc = self.get_uniform_location(name)
        if loc != -1:
            mat = np.array(matrix, dtype=np.float32)
            glUniformMatrix4fv(loc, 1, GL_FALSE, mat)
    
    def cleanup(self):
        """Delete shader program."""
        if self.program_id:
            glDeleteProgram(self.program_id)
            self.program_id = None


class ShaderManager:
    """Manages multiple shader programs."""
    
    def __init__(self):
        self.shaders = {}
    
    def load_shader(self, name, vertex_path, fragment_path):
        """Load and compile a shader program."""
        try:
            shader = ShaderProgram(vertex_path, fragment_path)
            self.shaders[name] = shader
            return shader
        except Exception as e:
            print(f"Failed to load shader '{name}': {e}")
            return None
    
    def get_shader(self, name):
        """Get a shader program by name."""
        return self.shaders.get(name)
    
    def use_shader(self, name):
        """Activate a shader program by name."""
        shader = self.get_shader(name)
        if shader:
            shader.use()
            return shader
        return None
    
    def cleanup(self):
        """Delete all shader programs."""
        for shader in self.shaders.values():
            shader.cleanup()
        self.shaders.clear()
