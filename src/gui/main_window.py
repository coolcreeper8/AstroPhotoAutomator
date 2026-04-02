import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QSlider, 
                             QSpinBox, QCheckBox, QGroupBox, QProgressBar, QMessageBox,
                             QTabWidget, QListWidget, QListWidgetItem, QRadioButton,
                             QButtonGroup, QFrame, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from gui.workers import StackingWorker, PostProcessingWorker
from core.processing import FrameAnalyzer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AstroPhotoAutomator")
        self.resize(1200, 900)
        
        self.video_paths = []  # Support multiple videos
        self.stacked_image = None
        self.processed_image = None
        self.stacking_worker = None
        self.postproc_worker = None
        
        self.init_ui()
        
    def init_ui(self):
        container = QWidget()
        self.setCentralWidget(container)
        main_layout = QHBoxLayout(container)
        
        # --- Left Control Panel ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(350)
        
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
        stacking_group = QGroupBox("Stage 1: Stacking")
        stacking_layout = QVBoxLayout(stacking_group)
        
        stacking_layout.addWidget(QLabel("Percentage to Stack (%):"))
        self.stack_percent = QSlider(Qt.Orientation.Horizontal)
        self.stack_percent.setRange(1, 100)
        self.stack_percent.setValue(20)
        self.stack_percent_label = QLabel("20%")
        self.stack_percent.valueChanged.connect(self.on_stack_slider_change)
        
        stacking_layout.addWidget(self.stack_percent)
        stacking_layout.addWidget(self.stack_percent_label)
        
        self.stack_btn = QPushButton("Stack Frames")
        self.stack_btn.clicked.connect(self.start_stacking)
        self.stack_btn.setEnabled(False)
        self.stack_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        self.stack_btn.stylesheet = "background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;"
        stacking_layout.addWidget(self.stack_btn)

        # --- Advanced Stacking Options ---
        adv_group = QGroupBox("Advanced Settings")
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
        
        # Pano Mode
        self.pano_mode_check = QCheckBox("Panorama Mode (Stitch videos)")
        self.pano_mode_check.setToolTip("Stack each video separately and stitch them into a panorama.")
        adv_layout.addWidget(self.pano_mode_check)
        
        stacking_layout.addWidget(adv_group)
        
        control_layout.addWidget(stacking_group)
        
        # === STAGE 2: POST-PROCESSING ===
        postproc_group = QGroupBox("Stage 2: Post-Processing")
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
        
        # Wavelet Options
        manual_layout.addWidget(QLabel("Wavelet Sharpening (Dyadic):"))
        self.layer_sliders = []
        # Dyadic Scales: 1, 2, 4, 8, 16, 32
        scales = [1, 2, 4, 8, 16, 32]
        for i, scale in enumerate(scales):
            layer_container = QWidget()
            layer_layout = QVBoxLayout(layer_container)
            layer_layout.setContentsMargins(0, 0, 0, 0)
            
            layer_layout.addWidget(QLabel(f"  Layer {i+1} (Denoise/Detail {scale}px)"))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 50) # Divide by 10 for actual weight
            slider.setValue(0)
            slider.valueChanged.connect(self.on_postproc_param_changed)
            layer_layout.addWidget(slider)
            
            manual_layout.addWidget(layer_container)
            self.layer_sliders.append(slider)
            
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
        
        self.apply_postproc_btn = QPushButton("Apply Post-Processing")
        self.apply_postproc_btn.clicked.connect(self.start_post_processing)
        self.apply_postproc_btn.setEnabled(False)
        self.apply_postproc_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; font-weight: bold;")
        postproc_layout.addWidget(self.apply_postproc_btn)
        
        self.save_btn = QPushButton("Save Image")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 10px; font-weight: bold;")
        postproc_layout.addWidget(self.save_btn)
        
        control_layout.addWidget(postproc_group)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        
        # --- Right Preview ---
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        
        # Tab widget for stacked vs processed
        self.preview_tabs = QTabWidget()
        
        self.stacked_view = QLabel()
        self.stacked_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stacked_view.setStyleSheet("background-color: #222; border: 1px solid #444;")
        self.stacked_view.setMinimumSize(600, 600)
        
        self.processed_view = QLabel()
        self.processed_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_view.setStyleSheet("background-color: #222; border: 1px solid #444;")
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
        
        self.stacking_worker = StackingWorker(self.video_paths, stack_val, stack_mode, max_load, align_mode, pano_mode)
        self.stacking_worker.progress.connect(self.update_status)
        self.stacking_worker.finished.connect(self.stacking_finished)
        self.stacking_worker.error.connect(self.processing_error)
        self.stacking_worker.start()
        
    def stacking_finished(self, stacked_image):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        recognized_obj = FrameAnalyzer.recognize_object(stacked_image)
        self.status_label.setText(f"Stacking Complete! Recognized Object: {recognized_obj}. Ready for post-processing.")
        
        self.stacked_image = stacked_image
        self.display_image(stacked_image, self.stacked_view)
        
        # Enable post-processing controls
        self.apply_postproc_btn.setEnabled(True)
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
        
        self.postproc_worker = PostProcessingWorker(self.stacked_image, layers, auto_color, denoise, auto_mode=auto_mode)
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
            # Convert BGR to RGB if needed for saving
            if len(image_to_save.shape) == 3:
                save_image = cv2.cvtColor(image_to_save, cv2.COLOR_BGR2RGB)
            else:
                save_image = image_to_save
                
            cv2.imwrite(file_path, save_image)
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
