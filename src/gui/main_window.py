import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QLabel, QPushButton, QFileDialog, QSlider,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QProgressBar,
                             QMessageBox, QTabWidget, QListWidget, QListWidgetItem,
                             QRadioButton, QButtonGroup, QFrame, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QFont
from gui.workers import StackingWorker, PostProcessingWorker, DualObjectWorker
from core.processing import FrameAnalyzer
from core.planet_configs import get_config, TARGET_LABEL_TO_KEY
from core.derotation import needs_derotation

# ─── Dark Space Theme ────────────────────────────────────────────────────────
_DARK_THEME = """
QMainWindow, QDialog { background-color: #0d1117; }

QWidget {
    background-color: #0d1117;
    color: #cdd9e5;
    font-family: "Segoe UI", "Inter", Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 6px 6px 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #58a6ff;
    font-weight: bold;
    font-size: 11px;
}

/* ── Default buttons ── */
QPushButton {
    background-color: #21262d;
    color: #cdd9e5;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 5px 12px;
    min-height: 26px;
}
QPushButton:hover:enabled  { background-color: #30363d; border-color: #8b949e; color: #e6edf3; }
QPushButton:pressed        { background-color: #0d1117; }
QPushButton:disabled       { background-color: #161b22; color: #484f58; border-color: #21262d; }

/* ── Stack Frames (green) ── */
QPushButton#stackBtn {
    background-color: #0f3d20;
    border-color: #2ea043;
    color: #aff5b4;
    font-weight: bold;
    font-size: 14px;
    min-height: 38px;
}
QPushButton#stackBtn:hover:enabled  { background-color: #196127; border-color: #3fb950; }
QPushButton#stackBtn:disabled       { background-color: #0d1117; border-color: #21262d; color: #484f58; }

/* ── Apply Post-Processing (blue) ── */
QPushButton#applyBtn {
    background-color: #0c2d6b;
    border-color: #388bfd;
    color: #a5d3fb;
    font-weight: bold;
    font-size: 14px;
    min-height: 38px;
}
QPushButton#applyBtn:hover:enabled  { background-color: #1158c7; border-color: #58a6ff; }
QPushButton#applyBtn:disabled       { background-color: #0d1117; border-color: #21262d; color: #484f58; }

/* ── Save Image (amber) ── */
QPushButton#saveBtn {
    background-color: #3d1f00;
    border-color: #d29922;
    color: #f0c27f;
    font-weight: bold;
    min-height: 32px;
}
QPushButton#saveBtn:hover:enabled   { background-color: #6b3a00; border-color: #e3b341; }
QPushButton#saveBtn:disabled        { background-color: #0d1117; border-color: #21262d; color: #484f58; }

/* ── Dual-Object Blend (purple) ── */
QPushButton#dualBlendBtn {
    background-color: #2e1065;
    border-color: #8957e5;
    color: #d2b0ff;
    min-height: 32px;
}
QPushButton#dualBlendBtn:hover:enabled  { background-color: #3d1a8a; border-color: #a371f7; }
QPushButton#dualBlendBtn:disabled       { background-color: #0d1117; border-color: #21262d; color: #484f58; }

/* ── Load Preset (cyan) ── */
QPushButton#presetBtn {
    background-color: #0a2d3d;
    border-color: #0c8599;
    color: #96d5e5;
    min-height: 28px;
}
QPushButton#presetBtn:hover:enabled { background-color: #0d4b61; border-color: #22d3ee; }
QPushButton#presetBtn:disabled      { background-color: #0d1117; border-color: #21262d; color: #484f58; }

/* ── Sliders ── */
QSlider::groove:horizontal {
    height: 4px;
    background: #30363d;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #388bfd;
    border: 1px solid #1f6feb;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover  { background: #58a6ff; }
QSlider::sub-page:horizontal      { background: #1f6feb; border-radius: 2px; }

/* ── Inputs ── */
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 3px 8px;
    color: #cdd9e5;
    min-height: 26px;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #8b949e; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    selection-background-color: #30363d;
    color: #cdd9e5;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #30363d;
    border: none;
    width: 16px;
}

/* ── Checkboxes & Radios ── */
QCheckBox, QRadioButton { color: #cdd9e5; spacing: 6px; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #484f58;
    background-color: #21262d;
}
QCheckBox::indicator  { border-radius: 4px; }
QRadioButton::indicator { border-radius: 8px; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #1f6feb;
    border-color: #388bfd;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #8b949e; }

/* ── Progress bar ── */
QProgressBar {
    border: 1px solid #30363d;
    border-radius: 4px;
    background-color: #21262d;
    text-align: center;
    color: #8b949e;
    min-height: 14px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #1f6feb, stop:1 #8957e5);
    border-radius: 3px;
}

/* ── Tabs ── */
QTabWidget::pane  { border: 1px solid #30363d; border-radius: 6px; background: #0d1117; }
QTabBar::tab {
    background-color: #161b22;
    color: #8b949e;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 6px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected          { background-color: #0d1117; color: #58a6ff; border-bottom: 1px solid #0d1117; }
QTabBar::tab:hover:!selected   { background-color: #21262d; color: #cdd9e5; }

/* ── List ── */
QListWidget {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #cdd9e5;
}
QListWidget::item:selected { background-color: #30363d; color: #58a6ff; }

/* ── Labels ── */
QLabel { color: #8b949e; background: transparent; }
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AstroPhotoAutomator — Planetary Imaging Suite")
        self.resize(1200, 900)
        
        self.video_paths = []  # Support multiple videos
        self.stacked_image = None
        self.processed_image = None
        self.stacking_worker = None
        self.postproc_worker = None
        self.dual_worker = None
        self.detected_object = None   # Set after stacking completes
        self.planet_config = None     # Set by target selector or auto-detect

        self.init_ui()
        self.setStyleSheet(_DARK_THEME)
        font = QFont("Segoe UI", 11)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        self.setFont(font)

    def init_ui(self):
        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QHBoxLayout(container)
        
        # --- Left Control Panel ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(390)
        control_layout.setSpacing(8)
        
        # File Loading
        file_group = QGroupBox("Video Files")
        file_layout = QVBoxLayout(file_group)
        
        btn_layout = QHBoxLayout()
        self.add_video_btn = QPushButton("Add Video(s)")
        self.add_video_btn.clicked.connect(self.add_videos)
        btn_layout.addWidget(self.add_video_btn)
        
        self.remove_video_btn = QPushButton("Remove")
        self.remove_video_btn.clicked.connect(self.remove_selected_video)
        self.remove_video_btn.setEnabled(False)
        btn_layout.addWidget(self.remove_video_btn)
        
        self.clear_videos_btn = QPushButton("Clear All")
        self.clear_videos_btn.clicked.connect(self.clear_videos)
        self.clear_videos_btn.setEnabled(False)
        btn_layout.addWidget(self.clear_videos_btn)
        
        file_layout.addLayout(btn_layout)
        
        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(100)
        self.video_list.itemSelectionChanged.connect(self.on_video_selection_changed)
        file_layout.addWidget(self.video_list)
        
        self.video_count_label = QLabel("No videos loaded")
        file_layout.addWidget(self.video_count_label)
        
        control_layout.addWidget(file_group)
        
        # === STAGE 1: STACKING ===
        stacking_group = QGroupBox("① Stacking")
        stacking_layout = QVBoxLayout(stacking_group)
        stacking_layout.setSpacing(6)

        pct_row = QHBoxLayout()
        pct_row.addWidget(QLabel("Stack:"))
        self.stack_percent = QSlider(Qt.Orientation.Horizontal)
        self.stack_percent.setRange(1, 100)
        self.stack_percent.setValue(20)
        self.stack_percent.valueChanged.connect(self.on_stack_slider_change)
        self.stack_percent_label = QLabel("20%")
        self.stack_percent_label.setFixedWidth(42)
        self.stack_percent_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pct_row.addWidget(self.stack_percent)
        pct_row.addWidget(self.stack_percent_label)
        stacking_layout.addLayout(pct_row)
        
        self.stack_btn = QPushButton("Stack Frames")
        self.stack_btn.setObjectName("stackBtn")
        self.stack_btn.clicked.connect(self.start_stacking)
        self.stack_btn.setEnabled(False)
        stacking_layout.addWidget(self.stack_btn)

        # --- Advanced Stacking Options ---
        adv_group = QGroupBox("Advanced")
        adv_layout = QVBoxLayout(adv_group)

        # Max Frames
        frame_limit_layout = QHBoxLayout()
        frame_limit_layout.addWidget(QLabel("Max Frames to Load (0 = All):"))
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 100000)
        self.max_frames_spin.setValue(2000) 
        self.max_frames_spin.setSingleStep(500)
        frame_limit_layout.addWidget(self.max_frames_spin)
        adv_layout.addLayout(frame_limit_layout)
        
        # Stacking Mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Stacking Mode:"))
        self.stack_mode_combo = QComboBox() # requires import
        self.stack_mode_combo.addItems(["Percentage (%)", "Frame Count (#)", "Auto (Best Quality)"])
        self.stack_mode_combo.currentIndexChanged.connect(self.update_stack_slider_mode)
        mode_layout.addWidget(self.stack_mode_combo)
        adv_layout.addLayout(mode_layout)

        # Alignment Mode
        align_layout = QHBoxLayout()
        align_layout.addWidget(QLabel("Alignment:"))
        self.align_mode_combo = QComboBox() 
        self.align_mode_combo.addItems(["Translation (Fast)", "Affine (Rotation)", "Optical Flow (Distortion)"])
        align_layout.addWidget(self.align_mode_combo)
        adv_layout.addLayout(align_layout)
        
        # Target object selector — drives recommended stacking % and alignment mode
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Object:"))
        self.target_combo = QComboBox()
        self.target_combo.addItems(list(TARGET_LABEL_TO_KEY.keys()))
        self.target_combo.setToolTip(
            "Select your imaging target to auto-apply recommended stacking and post-processing settings."
        )
        self.target_combo.currentTextChanged.connect(self.on_target_changed)
        target_layout.addWidget(self.target_combo)
        adv_layout.addLayout(target_layout)

        # Derotation (automatic for fast-rotating planets like Jupiter)
        self.derotate_check = QCheckBox("Enable Planetary Derotation")
        self.derotate_check.setToolTip(
            "Counter-rotate frames to compensate for planetary rotation during the capture session.\n"
            "Auto-enabled for Jupiter (9.9 h period) and Saturn (10.7 h period)."
        )
        adv_layout.addWidget(self.derotate_check)

        # Pano Mode
        self.pano_mode_check = QCheckBox("Panorama Mode (Stitch videos)")
        self.pano_mode_check.setToolTip("Stack each video separately and stitch them into a panorama.")
        adv_layout.addWidget(self.pano_mode_check)
        
        stacking_layout.addWidget(adv_group)
        
        control_layout.addWidget(stacking_group)
        
        # === STAGE 2: POST-PROCESSING ===
        postproc_group = QGroupBox("② Post-Processing")
        postproc_layout = QVBoxLayout(postproc_group)
        
        # Mode Selection
        mode_layout = QHBoxLayout()
        self.auto_mode_btn = QRadioButton("Automatic")
        self.manual_mode_btn = QRadioButton("Manual")
        self.manual_mode_btn.setChecked(True)
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.auto_mode_btn)
        self.mode_group.addButton(self.manual_mode_btn)
        
        # Connect signal
        self.mode_group.buttonToggled.connect(self.toggle_postproc_mode)
        
        mode_layout.addWidget(self.auto_mode_btn)
        mode_layout.addWidget(self.manual_mode_btn)
        postproc_layout.addLayout(mode_layout)
        
        # Manual Controls Container
        self.manual_controls_widget = QWidget()
        manual_layout = QVBoxLayout(self.manual_controls_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        
        # Wavelet Options — compact grid: scale label | slider | live value
        wavelet_header = QLabel("Wavelet Sharpening")
        wavelet_header.setStyleSheet("color: #58a6ff; font-weight: bold; font-size: 11px;")
        manual_layout.addWidget(wavelet_header)

        wavelet_grid = QGridLayout()
        wavelet_grid.setSpacing(4)
        wavelet_grid.setColumnStretch(1, 1)  # slider column stretches

        self.layer_sliders = []
        self.layer_value_labels = []
        scales = [1, 2, 4, 8, 16, 32]
        for i, scale in enumerate(scales):
            scale_lbl = QLabel(f"L{i+1} {scale}px")
            scale_lbl.setFixedWidth(52)
            scale_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 50)
            slider.setValue(0)

            val_lbl = QLabel("0.0")
            val_lbl.setFixedWidth(28)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet("color: #cdd9e5; font-size: 11px; font-family: monospace;")

            def _make_updater(v_lbl):
                def _update(v):
                    v_lbl.setText(f"{v / 10:.1f}")
                return _update

            slider.valueChanged.connect(_make_updater(val_lbl))
            slider.valueChanged.connect(self.on_postproc_param_changed)

            wavelet_grid.addWidget(scale_lbl, i, 0)
            wavelet_grid.addWidget(slider,    i, 1)
            wavelet_grid.addWidget(val_lbl,   i, 2)

            self.layer_sliders.append(slider)
            self.layer_value_labels.append(val_lbl)

        manual_layout.addLayout(wavelet_grid)
            
        # Denoise Option
        denoise_container = QHBoxLayout()
        denoise_container.addWidget(QLabel("Denoise Strength:"))
        self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_slider.setRange(0, 20)
        self.denoise_slider.setValue(0)
        self.denoise_slider.valueChanged.connect(self.on_postproc_param_changed)
        denoise_container.addWidget(self.denoise_slider)
        manual_layout.addLayout(denoise_container)
        
        # Color Options
        self.auto_color_check = QCheckBox("Auto Color Balance & Align")
        self.auto_color_check.stateChanged.connect(self.on_postproc_param_changed)
        manual_layout.addWidget(self.auto_color_check)
        
        postproc_layout.addWidget(self.manual_controls_widget)
        
        self.load_preset_btn = QPushButton("Load Planet Preset")
        self.load_preset_btn.setObjectName("presetBtn")
        self.load_preset_btn.setEnabled(False)
        self.load_preset_btn.setToolTip("Populate wavelet sliders with recommended settings for the detected target.")
        self.load_preset_btn.clicked.connect(self.apply_planet_preset)
        postproc_layout.addWidget(self.load_preset_btn)

        self.apply_postproc_btn = QPushButton("Apply Post-Processing")
        self.apply_postproc_btn.setObjectName("applyBtn")
        self.apply_postproc_btn.clicked.connect(self.start_post_processing)
        self.apply_postproc_btn.setEnabled(False)
        postproc_layout.addWidget(self.apply_postproc_btn)

        # Moon exposure boost for dual-object mode
        boost_row = QHBoxLayout()
        boost_row.addWidget(QLabel("Moon Boost:"))
        self.moon_boost_spin = QDoubleSpinBox()
        self.moon_boost_spin.setRange(0.0, 10.0)
        self.moon_boost_spin.setSingleStep(0.5)
        self.moon_boost_spin.setValue(0.0)
        self.moon_boost_spin.setDecimals(1)
        self.moon_boost_spin.setToolTip(
            "Exposure boost applied to frames before Moon stacking.\n"
            "0.0 = auto-detect from first frame (recommended).\n"
            "Increase if the Moon is still invisible after auto-detect."
        )
        boost_row.addWidget(self.moon_boost_spin)
        boost_row.addWidget(QLabel("× (0=auto)"))
        postproc_layout.addLayout(boost_row)

        self.dual_blend_btn = QPushButton("Dual-Object Blend  Moon + Planet")
        self.dual_blend_btn.setObjectName("dualBlendBtn")
        self.dual_blend_btn.setEnabled(False)
        self.dual_blend_btn.setToolTip(
            "Stack once with Moon-optimal settings (boosted exposure) and once with\n"
            "planet-optimal settings, then composite the planet region over the Moon."
        )
        self.dual_blend_btn.clicked.connect(self.start_dual_object_blend)
        postproc_layout.addWidget(self.dual_blend_btn)

        self.save_btn = QPushButton("Save Image")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        postproc_layout.addWidget(self.save_btn)
        
        control_layout.addWidget(postproc_group)
        
        # Progress + status
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        control_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready — load video(s) to begin.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "color: #58a6ff; font-size: 11px; padding: 2px 0;"
        )
        control_layout.addWidget(self.status_label)

        control_layout.addStretch()
        
        # --- Right Preview ---
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        
        # Tab widget for stacked vs processed
        self.preview_tabs = QTabWidget()
        
        _preview_style = (
            "background-color: #010409;"
            "border: 1px solid #30363d;"
            "border-radius: 6px;"
            "color: #484f58;"
        )
        self.stacked_view = QLabel("No image loaded")
        self.stacked_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stacked_view.setStyleSheet(_preview_style)
        self.stacked_view.setMinimumSize(600, 600)

        self.processed_view = QLabel("Apply post-processing to see result")
        self.processed_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_view.setStyleSheet(_preview_style)
        self.processed_view.setMinimumSize(600, 600)
        
        self.preview_tabs.addTab(self.stacked_view, "Stacked Image")
        self.preview_tabs.addTab(self.processed_view, "Processed Image")
        
        preview_layout.addWidget(self.preview_tabs)
        
        main_layout.addWidget(control_panel)
        main_layout.addWidget(preview_panel)
        
    def add_videos(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Video(s)", "", "Video Files (*.avi *.mp4 *.ser)")
        if file_paths:
            for file_path in file_paths:
                if file_path not in self.video_paths:
                    self.video_paths.append(file_path)
                    self.video_list.addItem(os.path.basename(file_path))
            
            self.update_video_controls()
    
    def remove_selected_video(self):
        current_row = self.video_list.currentRow()
        if current_row >= 0:
            self.video_paths.pop(current_row)
            self.video_list.takeItem(current_row)
            self.update_video_controls()
    
    def clear_videos(self):
        self.video_paths.clear()
        self.video_list.clear()
        self.update_video_controls()
    
    def on_video_selection_changed(self):
        has_selection = self.video_list.currentRow() >= 0
        self.remove_video_btn.setEnabled(has_selection)
    
    def update_video_controls(self):
        count = len(self.video_paths)
        if count == 0:
            self.video_count_label.setText("No videos loaded")
            self.stack_btn.setEnabled(False)
            self.clear_videos_btn.setEnabled(False)
        elif count == 1:
            self.video_count_label.setText("1 video loaded")
            self.stack_btn.setEnabled(True)
            self.clear_videos_btn.setEnabled(True)
        else:
            self.video_count_label.setText(f"{count} videos loaded (will be combined)")
            self.stack_btn.setEnabled(True)
            self.clear_videos_btn.setEnabled(True)
        self.status_label.setText("Ready to stack." if count > 0 else "Load video(s) to begin.")
    
    def on_target_changed(self, label):
        key = TARGET_LABEL_TO_KEY.get(label)
        if key is None:
            self.planet_config = None
            self.derotate_check.setChecked(False)
            return
        config = get_config(key)
        self.planet_config = config
        # Apply recommended stacking settings immediately
        if self.stack_mode_combo.currentIndex() != 0:
            self.stack_mode_combo.setCurrentIndex(0)
        self.stack_percent.setValue(config["stack_percent"])
        align_map = {"translate": 0, "affine": 1, "optical_flow": 2}
        self.align_mode_combo.setCurrentIndex(align_map.get(config["align_mode"], 0))
        # Auto-enable derotation only for targets with fast enough rotation to matter
        self.derotate_check.setChecked(needs_derotation(key))
        self.status_label.setText(f"Preset for {label} applied to stacking settings.")

    def apply_planet_preset(self):
        if self.planet_config is None:
            return
        self.manual_mode_btn.setChecked(True)
        scales = [1, 2, 4, 8, 16, 32]
        wavelet_dict = {float(sigma): w for sigma, w in self.planet_config["wavelet_layers"]}
        for slider, scale in zip(self.layer_sliders, scales):
            weight = wavelet_dict.get(float(scale), 0.0)
            slider.setValue(int(weight * 10))
        self.denoise_slider.setValue(self.planet_config["denoise"])
        obj = self.detected_object or self.target_combo.currentText()
        self.status_label.setText(f"Loaded {obj} wavelet preset — adjust if needed, then Apply.")

    def update_stack_slider_mode(self, index):
        if index == 0: # Percentage
            self.stack_percent.setRange(1, 100)
            self.stack_percent.setValue(20)
            self.stack_percent.setEnabled(True)
            self.stack_percent_label.setText("20%")
        elif index == 1: # Frame Count
            self.stack_percent.setRange(1, 10000)
            self.stack_percent.setValue(500)
            self.stack_percent.setEnabled(True)
            self.stack_percent_label.setText("500 frames")
        else: # Auto
            self.stack_percent.setEnabled(False)
            self.stack_percent_label.setText("Auto-Detect")
            
    def start_stacking(self):
        if not self.video_paths:
            return
            
        self.stack_btn.setEnabled(False)
        self.add_video_btn.setEnabled(False)
        self.remove_video_btn.setEnabled(False)
        self.clear_videos_btn.setEnabled(False)
        self.apply_postproc_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0) # Indeterminate
        
        # Gather stacking settings
        stack_val = self.stack_percent.value()
        max_load = self.max_frames_spin.value()
        if max_load == 0: max_load = None
        
        idx = self.stack_mode_combo.currentIndex()
        if idx == 0:
            stack_mode = "percent"
        elif idx == 1:
            stack_mode = "count"
        else:
            stack_mode = "auto"
            
        align_mode = "translate"
        if self.align_mode_combo.currentIndex() == 1: align_mode = "affine"
        elif self.align_mode_combo.currentIndex() == 2: align_mode = "optical_flow"
        
        pano_mode = self.pano_mode_check.isChecked()
        
        derotate = self.derotate_check.isChecked()
        planet_name = TARGET_LABEL_TO_KEY.get(self.target_combo.currentText())

        self.stacking_worker = StackingWorker(
            self.video_paths, stack_val, stack_mode, max_load, align_mode, pano_mode,
            planet_config=self.planet_config, derotate=derotate, planet_name=planet_name
        )
        self.stacking_worker.progress.connect(self.update_status)
        self.stacking_worker.finished.connect(self.stacking_finished)
        self.stacking_worker.error.connect(self.processing_error)
        self.stacking_worker.start()
        
    def stacking_finished(self, stacked_image):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)

        recognized_obj = FrameAnalyzer.recognize_object(stacked_image)
        self.detected_object = recognized_obj

        # If user chose Auto-Detect, resolve planet config from recognition result now
        if self.target_combo.currentText() == "Auto-Detect":
            self.planet_config = get_config(recognized_obj)

        self.status_label.setText(
            f"Stacking complete — detected: {recognized_obj}. Load preset or set sliders, then Apply."
        )
        self.load_preset_btn.setText(f"Load {recognized_obj} Preset")
        self.load_preset_btn.setEnabled(True)

        self.stacked_image = stacked_image
        self.display_image(stacked_image, self.stacked_view)

        # Enable post-processing controls
        self.apply_postproc_btn.setEnabled(True)
        self.dual_blend_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.stack_btn.setEnabled(True)
        self.add_video_btn.setEnabled(True)
        self.clear_videos_btn.setEnabled(True)

        self.preview_tabs.setCurrentIndex(0)
        
    def on_stack_slider_change(self, v):
        if self.stack_mode_combo.currentIndex() == 0:
            self.stack_percent_label.setText(f"{v}%")
        else:
            self.stack_percent_label.setText(f"{v} frames")
    
    def on_postproc_param_changed(self):
        # Auto-apply is disabled; user must click apply button
        pass
    
    def toggle_postproc_mode(self):
        is_manual = self.manual_mode_btn.isChecked()
        self.manual_controls_widget.setVisible(is_manual)

    def start_post_processing(self):
        if self.stacked_image is None:
            return
        
        self.apply_postproc_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        
        # Gather post-processing settings
        layers = []
        auto_color = False
        denoise = 0
        
        if self.auto_mode_btn.isChecked():
            # Automatic settings - handled by worker's smart optimizer
            auto_mode = True
        else:
            auto_mode = False
            # Manual settings
            scales = [1, 2, 4, 8, 16, 32]
            for i, slider in enumerate(self.layer_sliders):
                weight = slider.value() / 10.0
                sigma = float(scales[i])
                if weight > 0:
                    layers.append((sigma, weight))
            
            auto_color = self.auto_color_check.isChecked()
            denoise = self.denoise_slider.value()
        
        self.postproc_worker = PostProcessingWorker(
            self.stacked_image, layers, auto_color, denoise, auto_mode=auto_mode,
            planet_config=self.planet_config
        )
        self.postproc_worker.progress.connect(self.update_status)
        self.postproc_worker.finished.connect(self.postproc_finished)
        self.postproc_worker.error.connect(self.processing_error)
        self.postproc_worker.start()
    
    def postproc_finished(self, processed_image):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.status_label.setText("Post-processing Complete!")
        
        self.processed_image = processed_image
        self.display_image(processed_image, self.processed_view)
        
        self.apply_postproc_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        # Switch to processed tab
        self.preview_tabs.setCurrentIndex(1)

    def start_dual_object_blend(self):
        """
        Run independent Moon and planet stacks from the source video, then composite.
        The Moon pass applies an exposure boost to recover the dim Moon from
        planet-exposed frames (Jupiter/Saturn conjunctions).
        """
        if not self.video_paths:
            QMessageBox.warning(self, "No Video", "Load video file(s) before running Dual-Object Blend.")
            return

        # Determine which planet to composite
        target_label = self.target_combo.currentText()
        planet_key = TARGET_LABEL_TO_KEY.get(target_label)
        if planet_key in (None, "Moon (Surface)", "Planet (Jupiter/Mars/Venus)", "Unknown Celestial Body"):
            planet_key = "Jupiter"  # Default to Jupiter for Moon/generic targets

        self.dual_blend_btn.setEnabled(False)
        self.apply_postproc_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setRange(0, 0)

        max_load = self.max_frames_spin.value() or None
        boost = self.moon_boost_spin.value()  # 0.0 = auto-detect

        self.dual_worker = DualObjectWorker(
            self.video_paths,
            max_frames_load=max_load,
            planet_name=planet_key,
            moon_boost_factor=boost,
        )
        self.dual_worker.progress.connect(self.update_status)
        self.dual_worker.finished.connect(self.dual_blend_finished)
        self.dual_worker.error.connect(self.processing_error)
        self.dual_worker.start()

    def dual_blend_finished(self, blended_image):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.status_label.setText("Dual-Object Blend complete — planet and Moon composited.")

        self.processed_image = blended_image
        self.display_image(blended_image, self.processed_view)

        self.dual_blend_btn.setEnabled(True)
        self.apply_postproc_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.preview_tabs.setCurrentIndex(1)

    def save_image(self):
        # Determine which image to save
        if self.processed_image is not None:
            image_to_save = self.processed_image
            default_name = "processed_image.png"
        elif self.stacked_image is not None:
            image_to_save = self.stacked_image
            default_name = "stacked_image.png"
        else:
            QMessageBox.warning(self, "No Image", "No image available to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Image", 
            default_name, 
            "PNG Files (*.png);;TIFF Files (*.tiff *.tif);;JPEG Files (*.jpg *.jpeg)"
        )
        
        if file_path:
            cv2.imwrite(file_path, image_to_save)
            self.status_label.setText(f"Image saved to {os.path.basename(file_path)}")
            QMessageBox.information(self, "Success", f"Image saved successfully to:\n{file_path}")
        
    def update_status(self, message):
        self.status_label.setText(message)
        
    def processing_error(self, error_msg):
        self.progress_bar.setRange(0, 100)
        self.status_label.setText("Error occurred.")
        QMessageBox.critical(self, "Processing Error", error_msg)
        self.stack_btn.setEnabled(True)
        self.add_video_btn.setEnabled(True)
        self.clear_videos_btn.setEnabled(True)
        if self.stacked_image is not None:
            self.apply_postproc_btn.setEnabled(True)
            self.dual_blend_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
        
    def display_image(self, arr, label_widget):
        if arr is None:
            return
            
        # Convert to RGB for Qt
        if len(arr.shape) == 3:
            h, w, ch = arr.shape
            # OpenCV is BGR, Qt needs RGB
            rgb_image = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            h, w = arr.shape
            qt_img = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
            
        pixmap = QPixmap.fromImage(qt_img)
        
        # Scale to fit view
        scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label_widget.setPixmap(scaled_pixmap)
