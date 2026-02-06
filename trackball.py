"""
Trackball Module
Simple trackball camera control for 3D viewing.
"""

import math
from OpenGL.GL import GLfloat


class Trackball:
    """Simple trackball camera control using rotation matrix."""
    
    def __init__(self, theta=0, phi=0):
        self.rotation = [0, 0, 0, 1]
        self.theta = theta
        self.phi = phi
        self.matrix = None
        self._set_orientation(theta, phi)
    
    def drag(self, dx, dy, width, height):
        """Update rotation based on mouse drag."""
        sensitivity = 0.5
        self.theta += dy * sensitivity
        self.phi += dx * sensitivity
        self._set_orientation(self.theta, self.phi)
    
    def _set_orientation(self, theta, phi):
        """Calculate rotation matrix from angles."""
        theta_rad = math.radians(theta)
        phi_rad = math.radians(phi)
        
        ct, st = math.cos(theta_rad), math.sin(theta_rad)
        cp, sp = math.cos(phi_rad), math.sin(phi_rad)
        
        m = [0.0] * 16
        m[0] = cp
        m[2] = sp
        m[4] = st * sp
        m[5] = ct
        m[6] = -st * cp
        m[8] = -ct * sp
        m[9] = st
        m[10] = ct * cp
        m[15] = 1.0
        
        self.matrix = (GLfloat * 16)(*m)
