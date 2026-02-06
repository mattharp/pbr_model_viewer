#!/usr/bin/env python3
"""
PyQt Model Viewer - Main Entry Point
Modern 3D model viewer with interactive GUI controls.

Supported Formats:
- OBJ (Wavefront)
- GLB (glTF Binary) - With embedded textures and materials
- glTF - GL Transmission Format
"""

import sys
from PyQt5.QtWidgets import QApplication

from main_window import MainWindow


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("PYQT MODEL VIEWER")
    print("Modern 3D viewer with interactive controls")
    print("="*60)

    if len(sys.argv) < 2:
        print("\nUsage: python viewer.py [model_file]")
        print("\nSupported Formats:")
        print("  OBJ  - Wavefront OBJ (native support)")
        print("  GLB  - glTF Binary (with textures/materials)")
        print("  glTF - GL Transmission Format")
        print("\nNo model specified \u2014 launching empty viewer.")
        print("Use the Open button in the control panel to load a model.\n")
    
    model_path = sys.argv[1] if len(sys.argv) >= 2 else None
    
    app = QApplication(sys.argv)
    window = MainWindow(model_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
