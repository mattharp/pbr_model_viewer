"""
Main Window Module
Main application window that contains the GL widget and control panel.
Supports OBJ and GLB file formats.
"""

from pathlib import Path
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import QTimer

from obj_loader import OBJModel
from glb_loader import GLBModel, is_glb_file
from gl_widget import GLWidget
from control_panel import CollapsiblePanel


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, model_path=None):
        super().__init__()
        self.setWindowTitle("PBR Model Viewer")
        self.resize(800, 600)
        self.showMaximized()
        
        # Load model (auto-detect format) or start empty
        self.model = self._load_model(model_path) if model_path else None
        
        # Create GL widget (handles None model)
        self.gl_widget = GLWidget(self.model)
        self.setCentralWidget(self.gl_widget)
        
        # Create control panel
        self.control_panel = CollapsiblePanel(self.gl_widget, self.gl_widget)
        
        # Position panel after widget is shown
        QTimer.singleShot(100, self.position_panel)
        
        # Update PBR checkbox availability after GL initialization
        QTimer.singleShot(200, self.control_panel.update_pbr_availability)
        
        self.print_controls()
    
    def _load_model(self, model_path):
        """Load model, auto-detecting format."""
        file_ext = Path(model_path).suffix.lower()
        
        if is_glb_file(model_path):
            print(f"Detected GLB/glTF file")
            return GLBModel(model_path, flip_yz=False)
        else:
            # Default to OBJ
            print(f"Detected OBJ file")
            return OBJModel(model_path, flip_yz=False)
    
    def open_model(self):
        """Open a file dialog to load a new model."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D Model",
            "",
            "3D Models (*.obj *.glb *.gltf);;OBJ Files (*.obj);;GLB Files (*.glb *.gltf);;All Files (*.*)"
        )
        
        if not filename:
            return
        
        try:
            new_model = self._load_model(filename)
        except Exception as e:
            print(f"\u2717 Error loading model: {e}")
            return
        
        self.model = new_model
        
        # Tell the GL widget to swap to the new model
        self.gl_widget.load_new_model(new_model)
        
        # Update control panel state for the new model
        self.control_panel.on_model_changed()
        
        # Re-check PBR availability
        QTimer.singleShot(100, self.control_panel.update_pbr_availability)
        
        self.setWindowTitle(f"PBR Model Viewer \u2014 {Path(filename).name}")
        print(f"\u2713 Model loaded: {filename}")
    
    def print_controls(self):
        """Print control information to console."""
        print("\n" + "="*60)
        print("CONTROLS:")
        print("="*60)
        print("Mouse:")
        print("  Right Click + Drag : Rotate model")
        print("  Middle Click + Drag: Pan camera")
        print("  Mouse Wheel        : Zoom in/out")
        print("\nKeyboard:")
        print("  W / S              : Zoom in/out")
        print("  A / D              : Pan left/right")
        print("  Q / E              : Pan up/down")
        print("  R                  : Reset camera")
        print("\nGUI Panel:")
        print("  Click and drag header to move panel")
        print("  Display Mode dropdown")
        print("  Lighting checkbox")
        print("  Light Color picker (click color swatch)")
        print("  Brightness slider")
        print("="*60 + "\n")
    
    def showEvent(self, event):
        """Position panel when window is shown."""
        super().showEvent(event)
        QTimer.singleShot(10, self.position_panel)
    
    def position_panel(self):
        """Position control panel."""
        if not hasattr(self, 'control_panel'):
            return
        
        self.control_panel._constrain_height()
        
        if self.control_panel.user_positioned and self.control_panel.relative_x is not None:
            self.control_panel.restore_relative_position()
        else:
            self.control_panel.set_default_position()
        
        self.control_panel.show()
        self.control_panel.raise_()
    
    def resizeEvent(self, event):
        """Maintain panel position on resize."""
        super().resizeEvent(event)
        
        if hasattr(self, 'control_panel'):
            self.control_panel._constrain_height()
            if self.control_panel.relative_x is not None:
                self.control_panel.restore_relative_position()
    
    def moveEvent(self, event):
        """Handle window moves between screens."""
        super().moveEvent(event)
        
        if hasattr(self, 'control_panel') and self.control_panel.relative_x is not None:
            QTimer.singleShot(10, self.control_panel.restore_relative_position)
    
    def changeEvent(self, event):
        """Handle window state changes (maximize, restore)."""
        super().changeEvent(event)
        
        if hasattr(self, 'control_panel') and self.control_panel.relative_x is not None:
            QTimer.singleShot(10, self.control_panel.restore_relative_position)
