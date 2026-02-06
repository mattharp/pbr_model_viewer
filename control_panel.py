"""
Control Panel Module
Collapsible control panel with display mode, lighting, color picker, and brightness controls.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox,
    QSlider, QLabel, QPushButton, QFrame, QColorDialog, QSizePolicy, QLineEdit
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QFontMetrics, QFont, QDoubleValidator


class CollapsiblePanel(QWidget):
    """Draggable collapsible control panel."""
    
    def __init__(self, gl_widget, parent=None):
        super().__init__(parent)
        self.gl_widget = gl_widget
        self.is_collapsed = False
        
        self.setAutoFillBackground(True)
        
        # For dragging
        self.dragging = False
        self.drag_position = QPoint()
        
        # Track position
        self.user_positioned = False
        self.relative_x = None
        self.relative_y = None
        
        self._create_ui()
    
    def _create_ui(self):
        """Build the control panel UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self.header = QFrame()
        self.header.setCursor(Qt.OpenHandCursor)
        self.header.setStyleSheet("""
            QFrame {
                background-color: rgb(50, 50, 60);
                border: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
        """)
        self.header.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        
        # Calculate DPI-aware header height from font metrics
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        fm = QFontMetrics(header_font)
        text_height = fm.height()
        header_height = text_height + 8
        self.header.setFixedHeight(header_height)
        
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(0)
        
        # Collapse button - size relative to text height
        btn_size = text_height
        self.collapse_btn = QPushButton("▼")
        self.collapse_btn.setFixedSize(btn_size, btn_size)
        self.collapse_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: rgb(180, 180, 190);
                border: none;
                font-size: 10pt;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                color: white;
                background-color: rgba(255, 255, 255, 20);
                border-radius: 4px;
            }
        """)
        self.collapse_btn.clicked.connect(self.toggle_collapse)
        
        # Title
        title_label = QLabel("Controls")
        title_label.setFont(header_font)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: transparent;
                padding-left: 8px;
            }
        """)
        
        header_layout.addWidget(self.collapse_btn)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addWidget(self.header)
        
        # Content widget (goes inside scroll area)
        self.content = QFrame()
        self.content.setStyleSheet("""
            QFrame {
                background-color: rgb(40, 40, 45);
                border: none;
            }
        """)
        
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(12)
        
        # Open Model button
        self._add_open_button(content_layout)
        
        # Display Mode
        self._add_display_mode(content_layout)
        
        # Lighting
        self._add_lighting_controls(content_layout)
        
        # Directional Light (color and brightness)
        self._add_light_controls(content_layout)
        
        # Ambient Light
        self._add_ambient_control(content_layout)
        
        # Material Controls (PBR properties)
        self._add_material_controls(content_layout)
        
        # HDR Environment
        self._add_hdr_controls(content_layout)
        
        # Debug Visualization
        self._add_debug_controls(content_layout)
        
        # Scroll area wrapping content
        from PyQt5.QtWidgets import QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.content)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: rgb(40, 40, 45);
                border: none;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QScrollBar:vertical {
                background: rgb(40, 40, 45);
                width: 8px;
                border-radius: 4px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: rgb(80, 80, 90);
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgb(100, 100, 115);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)
        
        self.setMinimumWidth(250)
        self._constrain_height()
    
    def _create_value_input(self, initial_text, min_val, max_val, decimals=2):
        """Create a styled editable value input field."""
        value_input = QLineEdit(initial_text)
        value_input.setFixedWidth(50)
        value_input.setAlignment(Qt.AlignCenter)
        validator = QDoubleValidator(min_val, max_val, decimals)
        validator.setNotation(QDoubleValidator.StandardNotation)
        value_input.setValidator(validator)
        value_input.setStyleSheet("""
            QLineEdit {
                color: white;
                background-color: rgb(50, 50, 60);
                border: 1px solid rgb(70, 70, 80);
                border-radius: 3px;
                padding: 2px 4px;
                font-size: 9pt;
            }
            QLineEdit:focus {
                border: 1px solid rgb(80, 140, 200);
            }
            QLineEdit:disabled {
                color: rgb(90, 90, 95);
                background-color: rgb(35, 35, 40);
                border: 1px solid rgb(45, 45, 50);
            }
        """)
        return value_input
    
    def _add_open_button(self, layout):
        """Add Open Model button at the top of the panel."""
        self.open_btn = QPushButton("\U0001F4C2  Open Model...")
        self.open_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(70, 90, 130);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-size: 10pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(90, 110, 150);
            }
            QPushButton:pressed {
                background-color: rgb(55, 75, 115);
            }
        """)
        self.open_btn.clicked.connect(self._on_open_clicked)
        layout.addWidget(self.open_btn)
    
    def _on_open_clicked(self):
        """Forward open request to the main window."""
        # Walk up the parent chain to find MainWindow
        from main_window import MainWindow
        widget = self.parent()
        while widget is not None:
            if isinstance(widget, MainWindow):
                widget.open_model()
                return
            widget = widget.parent()
    
    def on_model_changed(self):
        """Update control panel state after a new model is loaded."""
        # Update texture checkbox
        if hasattr(self.gl_widget.model, 'has_textures'):
            has_tex = self.gl_widget.model.has_textures()
            self.texture_check.setEnabled(has_tex)
            self.texture_check.setChecked(has_tex)
        else:
            self.texture_check.setEnabled(False)
            self.texture_check.setChecked(False)
        
        # Reset PBR to off for new model
        self.pbr_check.setChecked(False)
        
        # Reset debug overlays
        self.uv_checker_check.setChecked(False)
        self.bbox_check.setChecked(False)
        self.normals_check.setChecked(False)
    
    def _add_display_mode(self, layout):
        """Add display mode dropdown."""
        mode_layout = QVBoxLayout()
        mode_label = QLabel("Display Mode:")
        mode_label.setStyleSheet("color: white; font-weight: bold; font-size: 10pt; background-color: rgb(50, 50, 60); border: none; border-radius: 4px; padding: 4px 8px;")
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Smooth Shaded",
            "Flat Shaded",
            "Wireframe",
            "Wireframe (Backface Culled)",
            "Solid + Wireframe"
        ])
        self.mode_combo.setStyleSheet("""
            QComboBox {
                background-color: rgb(50, 50, 60);
                color: white;
                border: 1px solid rgb(100, 100, 110);
                border-radius: 4px;
                padding: 6px;
                font-size: 10pt;
            }
            QComboBox:hover { border: 1px solid rgb(120, 120, 140); }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid white;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: rgb(50, 50, 60);
                color: white;
                selection-background-color: rgb(80, 120, 180);
                border: 1px solid rgb(100, 100, 110);
            }
        """)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)
    
    def _add_lighting_controls(self, layout):
        """Add lighting checkbox and color picker."""
        # Lighting Checkbox
        self.lighting_check = QCheckBox("Enable Lighting")
        self.lighting_check.setChecked(True)
        self.lighting_check.setStyleSheet("""
            QCheckBox {
                color: white;
                font-weight: bold;
                font-size: 10pt;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 3px;
                border: 2px solid rgb(100, 100, 110);
                background-color: rgb(50, 50, 60);
            }
            QCheckBox::indicator:checked {
                background-color: rgb(80, 140, 200);
                border: 2px solid rgb(100, 160, 220);
            }
        """)
        self.lighting_check.stateChanged.connect(self.on_lighting_changed)
        layout.addWidget(self.lighting_check)
        
        # Texture Checkbox
        self.texture_check = QCheckBox("Enable Textures")
        self.texture_check.setChecked(True)
        self.texture_check.setStyleSheet("""
            QCheckBox {
                color: white;
                font-weight: bold;
                font-size: 10pt;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 3px;
                border: 2px solid rgb(100, 100, 110);
                background-color: rgb(50, 50, 60);
            }
            QCheckBox::indicator:checked {
                background-color: rgb(80, 140, 200);
                border: 2px solid rgb(100, 160, 220);
            }
            QCheckBox:disabled {
                color: rgb(100, 100, 100);
            }
        """)
        self.texture_check.stateChanged.connect(self.on_texture_changed)
        
        # Check if model has textures and disable checkbox if not
        if self.gl_widget.model is not None and hasattr(self.gl_widget.model, 'has_textures'):
            if not self.gl_widget.model.has_textures():
                self.texture_check.setEnabled(False)
                self.texture_check.setChecked(False)
        
        layout.addWidget(self.texture_check)
        
        # PBR Rendering Checkbox
        self.pbr_check = QCheckBox("PBR Rendering")
        self.pbr_check.setChecked(False)
        self.pbr_check.setStyleSheet("""
            QCheckBox {
                color: white;
                font-weight: bold;
                font-size: 10pt;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 3px;
                border: 2px solid rgb(100, 100, 110);
                background-color: rgb(50, 50, 60);
            }
            QCheckBox::indicator:checked {
                background-color: rgb(100, 200, 100);
                border: 2px solid rgb(120, 220, 120);
            }
            QCheckBox:disabled {
                color: rgb(100, 100, 100);
            }
        """)
        self.pbr_check.stateChanged.connect(self.on_pbr_changed)
        
        # PBR checkbox will be enabled after GL initialization
        # (shaders load in initializeGL, which happens after panel creation)
        self.pbr_check.setEnabled(False)
        self.pbr_check.setToolTip("Checking for PBR shader support...")
        
        layout.addWidget(self.pbr_check)
    
    def _add_light_controls(self, layout):
        """Add directional light controls (color and brightness)."""
        light_layout = QVBoxLayout()
        light_label = QLabel("Directional Light:")
        light_label.setStyleSheet("color: white; font-weight: bold; font-size: 10pt; background-color: rgb(50, 50, 60); border: none; border-radius: 4px; padding: 4px 8px;")
        
        # Light color picker
        light_color_layout = QHBoxLayout()
        light_color_label = QLabel("Color:")
        light_color_label.setStyleSheet("color: rgb(200, 200, 200); font-size: 9pt;")
        
        self.color_button = QPushButton("")
        self.color_button.setFixedSize(60, 28)
        self.current_color = QColor(255, 255, 255)
        self.update_color_button()
        self.color_button.clicked.connect(self.choose_light_color)
        
        light_color_layout.addWidget(light_color_label)
        light_color_layout.addWidget(self.color_button)
        light_color_layout.addStretch()
        
        # Brightness slider
        slider_container = QHBoxLayout()
        self.brightness_value_input = self._create_value_input("0.3", 0.1, 2.0, 1)
        self.brightness_value_input.editingFinished.connect(self._on_brightness_input)
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(10)
        self.brightness_slider.setMaximum(200)
        self.brightness_slider.setValue(30)  # Default 0.3
        self.brightness_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: rgb(50, 50, 60);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: rgb(80, 140, 200);
                border: 2px solid rgb(100, 160, 220);
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: rgb(100, 160, 220);
            }
        """)
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        
        slider_container.addWidget(self.brightness_slider)
        slider_container.addWidget(self.brightness_value_input)
        
        light_layout.addWidget(light_label)
        light_layout.addLayout(light_color_layout)
        light_layout.addLayout(slider_container)
        layout.addLayout(light_layout)
    
    
    def _add_ambient_control(self, layout):
        """Add ambient lighting controls."""
        ambient_layout = QVBoxLayout()
        ambient_label = QLabel("Ambient Light:")
        ambient_label.setStyleSheet("color: white; font-weight: bold; font-size: 10pt; background-color: rgb(50, 50, 60); border: none; border-radius: 4px; padding: 4px 8px;")
        
        # Ambient color picker
        ambient_color_layout = QHBoxLayout()
        ambient_color_label = QLabel("Color:")
        ambient_color_label.setStyleSheet("color: rgb(200, 200, 200); font-size: 9pt;")
        
        self.ambient_color_button = QPushButton("")
        self.ambient_color_button.setFixedSize(60, 28)
        self.ambient_current_color = QColor(255, 255, 255)
        self.update_ambient_color_button()
        self.ambient_color_button.clicked.connect(self.choose_ambient_color)
        
        ambient_color_layout.addWidget(ambient_color_label)
        ambient_color_layout.addWidget(self.ambient_color_button)
        ambient_color_layout.addStretch()
        
        # Ambient strength slider
        slider_container = QHBoxLayout()
        self.ambient_value_input = self._create_value_input("0.03", 0.0, 1.0, 2)
        self.ambient_value_input.editingFinished.connect(self._on_ambient_input)
        
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setMinimum(0)
        self.ambient_slider.setMaximum(100)
        self.ambient_slider.setValue(3)  # Default 0.03
        self.ambient_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: rgb(50, 50, 60);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: rgb(140, 100, 200);
                border: 2px solid rgb(160, 120, 220);
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: rgb(160, 120, 220);
            }
        """)
        self.ambient_slider.valueChanged.connect(self.on_ambient_changed)
        
        slider_container.addWidget(self.ambient_slider)
        slider_container.addWidget(self.ambient_value_input)
        
        ambient_layout.addWidget(ambient_label)
        ambient_layout.addLayout(ambient_color_layout)
        ambient_layout.addLayout(slider_container)
        layout.addLayout(ambient_layout)
    
    def _add_material_controls(self, layout):
        """Add PBR material property sliders."""
        # Title
        material_title = QLabel("Material Properties:")
        material_title.setStyleSheet("""
            color: white;
            font-weight: bold;
            font-size: 10pt;
            background-color: rgb(50, 50, 60);
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
        """)
        layout.addWidget(material_title)
        
        # Metallic slider
        self._add_material_slider(
            layout, "Metallic:", "metallic",
            min_val=0, max_val=200, default=100,
            color="rgb(200, 140, 80)"
        )
        
        # Roughness slider
        self._add_material_slider(
            layout, "Roughness:", "roughness",
            min_val=0, max_val=200, default=100,
            color="rgb(140, 180, 140)"
        )
        
        # Normal Strength slider
        self._add_material_slider(
            layout, "Normal Strength:", "normal",
            min_val=0, max_val=200, default=100,
            color="rgb(100, 160, 200)"
        )
        
        # AO Strength slider
        self._add_material_slider(
            layout, "AO Strength:", "ao",
            min_val=0, max_val=100, default=100,
            color="rgb(180, 100, 160)"
        )
        
        # Emissive Intensity slider
        self._add_material_slider(
            layout, "Emissive Intensity:", "emissive",
            min_val=0, max_val=500, default=100,
            color="rgb(200, 180, 80)"
        )
    
    def _add_material_slider(self, layout, label_text, prop_name, min_val, max_val, default, color, disabled_by_default=True, direct_value=False):
        """Helper to add a single material property slider.
        If direct_value is True, the slider value is used directly (not divided by 100).
        """
        slider_layout = QVBoxLayout()
        slider_layout.setSpacing(2)
        
        # Label
        label = QLabel(label_text)
        label.setStyleSheet("color: rgb(200, 200, 200); font-size: 9pt;")
        setattr(self, f"{prop_name}_label", label)
        
        # Slider container
        slider_container = QHBoxLayout()
        
        # Editable value input
        if direct_value:
            value_input = self._create_value_input(f"{default}", float(min_val), float(max_val), 0)
        else:
            max_display = max_val / 100.0
            value_input = self._create_value_input(f"{default / 100.0:.2f}", min_val / 100.0, max_display, 2)
        value_input.editingFinished.connect(lambda: self._on_material_input(prop_name, min_val, max_val))
        setattr(self, f"{prop_name}_value_input", value_input)
        
        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 8px;
                background: rgb(50, 50, 60);
                border-radius: 4px;
            }}
            QSlider::groove:horizontal:disabled {{
                background: rgb(35, 35, 40);
            }}
            QSlider::handle:horizontal {{
                background: {color};
                border: 2px solid {color};
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:disabled {{
                background: rgb(60, 60, 65);
                border: 2px solid rgb(50, 50, 55);
            }}
            QSlider::handle:horizontal:hover {{
                background: {color};
                border: 2px solid white;
            }}
        """)
        
        # Connect to handler
        slider.valueChanged.connect(lambda v: self.on_material_changed(prop_name, v))
        setattr(self, f"{prop_name}_slider", slider)
        
        # Disable by default only for PBR material controls
        if disabled_by_default:
            slider.setEnabled(False)
            label.setStyleSheet("color: rgb(90, 90, 95); font-size: 9pt;")
            value_input.setEnabled(False)
        
        slider_container.addWidget(slider)
        slider_container.addWidget(value_input)
        
        slider_layout.addWidget(label)
        slider_layout.addLayout(slider_container)
        layout.addLayout(slider_layout)
    
    def _add_hdr_controls(self, layout):
        """Add HDR environment map controls."""
        from PyQt5.QtWidgets import QPushButton, QFileDialog
        
        # Title
        hdr_title = QLabel("HDR Environment:")
        hdr_title.setStyleSheet("""
            color: white;
            font-weight: bold;
            font-size: 10pt;
            background-color: rgb(50, 50, 60);
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
        """)
        layout.addWidget(hdr_title)
        
        # Load HDR button
        load_hdr_btn = QPushButton("Load HDR/EXR")
        load_hdr_btn.setStyleSheet("""
            QPushButton {
                background-color: rgb(80, 100, 140);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px;
                font-size: 9pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(100, 120, 160);
            }
            QPushButton:pressed {
                background-color: rgb(60, 80, 120);
            }
        """)
        load_hdr_btn.clicked.connect(self.load_hdr_file)
        layout.addWidget(load_hdr_btn)
        
        # HDR intensity slider
        self._add_material_slider(
            layout, "HDR Intensity:", "hdr_intensity",
            min_val=0, max_val=300, default=100,
            color="rgb(160, 140, 80)",
            disabled_by_default=True  # Disabled until HDR is loaded
        )
        
        # HDR rotation slider (0-360 degrees)
        self._add_material_slider(
            layout, "HDR Rotation:", "hdr_rotation",
            min_val=0, max_val=360, default=0,
            color="rgb(120, 160, 140)",
            disabled_by_default=True,
            direct_value=True
        )
    
    def _add_debug_controls(self, layout):
        """Add debug visualization toggles."""
        # Title
        debug_title = QLabel("Debug:")
        debug_title.setStyleSheet("""
            color: white;
            font-weight: bold;
            font-size: 10pt;
            background-color: rgb(50, 50, 60);
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
        """)
        layout.addWidget(debug_title)
        
        debug_checkbox_style = """
            QCheckBox {
                color: rgb(200, 200, 200);
                font-size: 9pt;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid rgb(100, 100, 110);
                background-color: rgb(50, 50, 60);
            }
            QCheckBox::indicator:checked {
                background-color: rgb(200, 160, 60);
                border: 2px solid rgb(220, 180, 80);
            }
        """
        
        # UV Checker toggle
        self.uv_checker_check = QCheckBox("UV Checker Map")
        self.uv_checker_check.setChecked(False)
        self.uv_checker_check.setStyleSheet(debug_checkbox_style)
        self.uv_checker_check.stateChanged.connect(self.on_uv_checker_changed)
        layout.addWidget(self.uv_checker_check)
        
        # Bounding Box toggle
        self.bbox_check = QCheckBox("Bounding Box")
        self.bbox_check.setChecked(False)
        self.bbox_check.setStyleSheet(debug_checkbox_style)
        self.bbox_check.stateChanged.connect(self.on_bbox_changed)
        layout.addWidget(self.bbox_check)
        
        # Vertex Normals toggle
        self.normals_check = QCheckBox("Vertex Normals")
        self.normals_check.setChecked(False)
        self.normals_check.setStyleSheet(debug_checkbox_style)
        self.normals_check.stateChanged.connect(self.on_normals_changed)
        layout.addWidget(self.normals_check)
    
    # Event handlers
    def on_mode_changed(self, index):
        modes = ['smooth', 'flat', 'wireframe', 'wireframe_culled', 'solid_wire']
        self.gl_widget.display_mode = modes[index]
        self.gl_widget.update()
    
    def on_lighting_changed(self, state):
        self.gl_widget.lighting_enabled = (state == Qt.Checked)
        self.gl_widget.setup_lighting()
        self.gl_widget.update()
    
    def on_texture_changed(self, state):
        self.gl_widget.texture_enabled = (state == Qt.Checked)
        self.gl_widget.update()
    
    def on_uv_checker_changed(self, state):
        self.gl_widget.show_uv_checker = (state == Qt.Checked)
        self.gl_widget.update()
    
    def on_bbox_changed(self, state):
        self.gl_widget.show_bounding_box = (state == Qt.Checked)
        self.gl_widget.update()
    
    def on_normals_changed(self, state):
        self.gl_widget.show_vertex_normals = (state == Qt.Checked)
        self.gl_widget.update()
    
    def _set_slider_enabled(self, prop_name, enabled):
        """Enable or disable a material slider and its label/input."""
        slider = getattr(self, f"{prop_name}_slider", None)
        label = getattr(self, f"{prop_name}_label", None)
        value_input = getattr(self, f"{prop_name}_value_input", None)
        
        if slider:
            slider.setEnabled(enabled)
        if label:
            if enabled:
                label.setStyleSheet("color: rgb(200, 200, 200); font-size: 9pt;")
            else:
                label.setStyleSheet("color: rgb(90, 90, 95); font-size: 9pt;")
        if value_input:
            value_input.setEnabled(enabled)
    
    def on_pbr_changed(self, state):
        """Toggle PBR rendering mode."""
        pbr_enabled = (state == Qt.Checked)
        self.gl_widget.pbr_enabled = pbr_enabled
        
        if pbr_enabled:
            # Check which maps are available to enable matching sliders
            available = []
            if hasattr(self.gl_widget.model, 'get_available_maps'):
                available = self.gl_widget.model.get_available_maps()
            
            # Always enable metallic/roughness (they work with factors even without maps)
            self._set_slider_enabled('metallic', True)
            self._set_slider_enabled('roughness', True)
            
            # Enable/disable based on map availability
            self._set_slider_enabled('normal', 'normal' in available)
            self._set_slider_enabled('ao', 'occlusion' in available)
            self._set_slider_enabled('emissive', 'emissive' in available)
        else:
            for prop in ('metallic', 'roughness', 'normal', 'ao', 'emissive'):
                self._set_slider_enabled(prop, False)
        
        self.gl_widget.update()
    
    def update_pbr_availability(self):
        """Update PBR checkbox based on shader availability and model materials."""
        has_shaders = hasattr(self.gl_widget, 'shader_manager') and self.gl_widget.shader_manager
        model = self.gl_widget.model
        has_pbr_materials = (
            model is not None
            and hasattr(model, 'materials')
            and bool(model.materials)
        )
        
        if has_shaders and has_pbr_materials:
            self.pbr_check.setEnabled(True)
            self.pbr_check.setToolTip("Toggle physically-based rendering (Cook-Torrance BRDF)")
            print("\u2713 PBR checkbox enabled")
        else:
            self.pbr_check.setEnabled(False)
            self.pbr_check.setChecked(False)
            if not has_shaders:
                self.pbr_check.setToolTip("PBR shaders not available")
            elif not has_pbr_materials:
                self.pbr_check.setToolTip("Model has no PBR materials")
    
    def on_brightness_changed(self, value):
        self.gl_widget.brightness = value / 100.0
        self.brightness_value_input.setText(f"{self.gl_widget.brightness:.1f}")
        self.gl_widget.setup_lighting()
        self.gl_widget.update()
    
    def _on_brightness_input(self):
        """Handle manual brightness value entry."""
        try:
            val = float(self.brightness_value_input.text())
            slider_val = int(val * 100)
            slider_val = max(self.brightness_slider.minimum(), min(slider_val, self.brightness_slider.maximum()))
            self.brightness_slider.setValue(slider_val)
        except ValueError:
            pass
    
    def on_ambient_changed(self, value):
        """Handle ambient light slider change."""
        self.gl_widget.ambient_strength = value / 100.0
        self.ambient_value_input.setText(f"{self.gl_widget.ambient_strength:.2f}")
        self.gl_widget.update()
    
    def _on_ambient_input(self):
        """Handle manual ambient value entry."""
        try:
            val = float(self.ambient_value_input.text())
            slider_val = int(val * 100)
            slider_val = max(self.ambient_slider.minimum(), min(slider_val, self.ambient_slider.maximum()))
            self.ambient_slider.setValue(slider_val)
        except ValueError:
            pass
    
    def on_material_changed(self, prop_name, value):
        """Handle material property slider changes."""
        # Convert slider value to multiplier (0-200 -> 0.0-2.0, except AO which is 0-100 -> 0.0-1.0)
        if prop_name == "ao":
            multiplier = value / 100.0
        else:
            multiplier = value / 100.0
        
        # Update the appropriate property
        if prop_name == "metallic":
            self.gl_widget.metallic_multiplier = multiplier
        elif prop_name == "roughness":
            self.gl_widget.roughness_multiplier = multiplier
        elif prop_name == "normal":
            self.gl_widget.normal_strength = multiplier
        elif prop_name == "ao":
            self.gl_widget.ao_strength = multiplier
        elif prop_name == "emissive":
            self.gl_widget.emissive_intensity = multiplier
        elif prop_name == "hdr_intensity":
            self.gl_widget.env_intensity = multiplier
        elif prop_name == "hdr_rotation":
            self.gl_widget.env_rotation = value  # Direct degrees, not /100
        
        # Update input field
        value_input = getattr(self, f"{prop_name}_value_input")
        if prop_name == "hdr_rotation":
            value_input.setText(f"{value}")
        else:
            value_input.setText(f"{multiplier:.2f}")
        
        # Update display
        self.gl_widget.update()
    
    def _on_material_input(self, prop_name, min_val, max_val):
        """Handle manual material value entry."""
        try:
            value_input = getattr(self, f"{prop_name}_value_input")
            val = float(value_input.text())
            if prop_name == "hdr_rotation":
                slider_val = int(val)
            else:
                slider_val = int(val * 100)
            slider = getattr(self, f"{prop_name}_slider")
            slider_val = max(min_val, min(slider_val, max_val))
            slider.setValue(slider_val)
        except (ValueError, AttributeError):
            pass
    
    def load_hdr_file(self):
        """Open file dialog to load HDR/EXR environment map."""
        from PyQt5.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load HDR Environment",
            "",
            "HDR Files (*.hdr *.exr);;All Files (*.*)"
        )
        
        if filename:
            success = self.gl_widget.load_hdr_environment(filename)
            if success:
                # Enable HDR sliders now that HDR is loaded
                self._set_slider_enabled('hdr_intensity', True)
                self._set_slider_enabled('hdr_rotation', True)
                print(f"✓ Loaded HDR: {filename}")
            else:
                print(f"✗ Failed to load HDR: {filename}")
    
    def choose_light_color(self):
        """Open color picker with live preview."""
        original_color = QColor(self.current_color)
        
        color_dialog = QColorDialog(self.current_color, self)
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)
        color_dialog.setWindowTitle("Choose Light Color")
        color_dialog.currentColorChanged.connect(self.on_color_preview)
        
        if color_dialog.exec_() == QColorDialog.Accepted:
            self.current_color = color_dialog.currentColor()
            self.update_color_button()
            self.gl_widget.light_color = (
                self.current_color.redF(),
                self.current_color.greenF(),
                self.current_color.blueF()
            )
            self.gl_widget.setup_lighting()
            self.gl_widget.update()
        else:
            self.current_color = original_color
            self.update_color_button()
            self.gl_widget.light_color = (
                original_color.redF(),
                original_color.greenF(),
                original_color.blueF()
            )
            self.gl_widget.setup_lighting()
            self.gl_widget.update()
    
    def on_color_preview(self, color):
        """Real-time color preview."""
        if color.isValid():
            self.gl_widget.light_color = (color.redF(), color.greenF(), color.blueF())
            self.gl_widget.setup_lighting()
            self.gl_widget.update()
            
            r, g, b = color.red(), color.green(), color.blue()
            self.color_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({r}, {g}, {b});
                    border: 2px solid rgb(100, 100, 110);
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    border: 2px solid rgb(120, 120, 140);
                }}
            """)
    
    def update_color_button(self):
        """Update color button appearance."""
        r, g, b = self.current_color.red(), self.current_color.green(), self.current_color.blue()
        self.color_button.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                border: 2px solid rgb(100, 100, 110);
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border: 2px solid rgb(120, 120, 140);
            }}
        """)
    
    def choose_ambient_color(self):
        """Open color picker for ambient light with live preview."""
        original_color = QColor(self.ambient_current_color)
        
        color_dialog = QColorDialog(self.ambient_current_color, self)
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)
        color_dialog.setWindowTitle("Choose Ambient Light Color")
        color_dialog.currentColorChanged.connect(self.on_ambient_color_preview)
        
        if color_dialog.exec_() == QColorDialog.Accepted:
            self.ambient_current_color = color_dialog.currentColor()
            self.update_ambient_color_button()
            self.gl_widget.ambient_color = (
                self.ambient_current_color.redF(),
                self.ambient_current_color.greenF(),
                self.ambient_current_color.blueF()
            )
            self.gl_widget.update()
        else:
            self.ambient_current_color = original_color
            self.update_ambient_color_button()
            self.gl_widget.ambient_color = (
                original_color.redF(),
                original_color.greenF(),
                original_color.blueF()
            )
            self.gl_widget.update()
    
    def on_ambient_color_preview(self, color):
        """Real-time ambient color preview."""
        if color.isValid():
            self.gl_widget.ambient_color = (color.redF(), color.greenF(), color.blueF())
            self.gl_widget.update()
            
            r, g, b = color.red(), color.green(), color.blue()
            self.ambient_color_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgb({r}, {g}, {b});
                    border: 2px solid rgb(100, 100, 110);
                    border-radius: 4px;
                }}
                QPushButton:hover {{
                    border: 2px solid rgb(120, 120, 140);
                }}
            """)
    
    def update_ambient_color_button(self):
        """Update ambient color button appearance."""
        r, g, b = self.ambient_current_color.red(), self.ambient_current_color.green(), self.ambient_current_color.blue()
        self.ambient_color_button.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                border: 2px solid rgb(100, 100, 110);
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border: 2px solid rgb(120, 120, 140);
            }}
        """)
    
    def toggle_collapse(self):
        """Toggle content visibility."""
        self.is_collapsed = not self.is_collapsed
        self.scroll_area.setVisible(not self.is_collapsed)
        self.collapse_btn.setText("▶" if self.is_collapsed else "▼")
        
        if self.is_collapsed:
            self.header.setStyleSheet("""
                QFrame {
                    background-color: rgb(50, 50, 60);
                    border: none;
                    border-radius: 8px;
                }
            """)
        else:
            self.header.setStyleSheet("""
                QFrame {
                    background-color: rgb(50, 50, 60);
                    border: none;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                }
            """)
        
        self._constrain_height()
    
    def _constrain_height(self):
        """Constrain panel height to not exceed parent window."""
        if self.is_collapsed:
            self.setFixedHeight(self.header.sizeHint().height())
            return
        
        # Calculate max available height (parent height minus margin)
        parent = self.parent()
        if parent:
            max_height = parent.height() - 20
        else:
            max_height = 9999
        
        # Let content determine its preferred size
        preferred = self.header.sizeHint().height() + self.content.sizeHint().height() + 4
        
        if preferred <= max_height:
            # Fits without scrolling
            self.setFixedHeight(preferred)
        else:
            self.setFixedHeight(max_height)
    
    # Mouse events for dragging
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.header.geometry().contains(event.pos()):
            self.dragging = True
            self.drag_position = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            event.ignore()
    
    def mouseMoveEvent(self, event):
        if self.dragging and event.buttons() & Qt.LeftButton:
            delta = event.pos() - self.drag_position
            new_pos = self.pos() + delta
            
            if self.parent():
                parent_rect = self.parent().rect()
                x = max(0, min(new_pos.x(), parent_rect.width() - self.width()))
                y = max(0, min(new_pos.y(), parent_rect.height() - self.height()))
                self.move(x, y)
            else:
                self.move(new_pos)
            
            event.accept()
        else:
            event.ignore()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.setCursor(Qt.OpenHandCursor if self.header.geometry().contains(event.pos()) else Qt.ArrowCursor)
            
            self.user_positioned = True
            if self.parent():
                parent_width = self.parent().width()
                parent_height = self.parent().height()
                if parent_width > 0 and parent_height > 0:
                    self.relative_x = self.x() / parent_width
                    self.relative_y = self.y() / parent_height
            
            event.accept()
        else:
            event.ignore()
    
    def leaveEvent(self, event):
        if not self.dragging:
            self.setCursor(Qt.ArrowCursor)
    
    # Position management
    def restore_relative_position(self):
        """Restore position from relative coordinates."""
        if not self.parent() or self.relative_x is None or self.relative_y is None:
            return
        
        parent_width = self.parent().width()
        parent_height = self.parent().height()
        
        x = int(self.relative_x * parent_width)
        y = int(self.relative_y * parent_height)
        
        max_x = parent_width - self.width()
        max_y = parent_height - self.height()
        
        x = max(0, min(x, max_x))
        y = max(0, min(y, max_y))
        
        self.move(x, y)
    
    def set_default_position(self):
        """Set default position (bottom right)."""
        if not self.parent():
            return
        
        parent_width = self.parent().width()
        parent_height = self.parent().height()
        
        x = parent_width - self.width() - 20
        y = parent_height - self.height() - 20
        
        self.move(x, y)
        
        if parent_width > 0 and parent_height > 0:
            self.relative_x = x / parent_width
            self.relative_y = y / parent_height
