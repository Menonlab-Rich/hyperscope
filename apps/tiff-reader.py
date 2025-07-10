import sys
import os
import tifffile
import numpy as np
import cv2  # Import OpenCV
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QAction
from PyQt6.QtCore import Qt, QSize

class TiffViewerApp(QMainWindow):
    """
    A PyQt6 application for viewing multipage TIFF images with statistics and normalization.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt TIFF Viewer")
        self.setGeometry(100, 100, 800, 600)

        # --- Data attributes ---
        self.file_path = None
        self.images = None
        self.num_pages = 0
        self.current_page = 0
        self.bit_depth = 0
        self._fixed_image_size = QSize() # To store a fixed size for the image label

        # --- UI Setup ---
        self.init_ui()
        self.create_menu()

        # Set initial state
        self.update_ui_state(enabled=False)

    def init_ui(self):
        """Initializes the main UI components."""
        # --- Central Widget and Layout ---
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # --- Left Panel (Image Display) ---
        left_panel_layout = QVBoxLayout()
        self.image_label = QLabel("Open a TIFF file to begin.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f0f0f0;")
        left_panel_layout.addWidget(self.image_label, 1) # Give more stretch factor

        # --- Slider for Page Navigation ---
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.slider_changed)
        left_panel_layout.addWidget(self.slider)
        main_layout.addLayout(left_panel_layout, 2) # More stretch

        # --- Right Panel (Controls and Stats) ---
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(10)

        # Statistics Box
        stats_group = QWidget()
        stats_layout = QVBoxLayout(stats_group)
        stats_group.setStyleSheet("background-color: #e9e9e9; border-radius: 5px;")

        self.page_label = QLabel("Page: N/A")
        self.min_label = QLabel("Min: N/A")
        self.max_label = QLabel("Max: N/A")
        self.mean_label = QLabel("Mean: N/A")
        self.std_label = QLabel("Std: N/A")
        self.snr_label = QLabel("SNR: N/A")

        for label in [self.page_label, self.min_label, self.max_label, self.mean_label, self.std_label, self.snr_label]:
            stats_layout.addWidget(label)

        right_panel_layout.addWidget(stats_group)

        # Export Button
        self.export_button = QPushButton("Export Current Image")
        self.export_button.clicked.connect(self.export_image)
        right_panel_layout.addWidget(self.export_button)

        right_panel_layout.addStretch() # Pushes everything to the top
        main_layout.addLayout(right_panel_layout, 1)

    def create_menu(self):
        """Creates the main menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.triggered.connect(self.open_file_dialog)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.triggered.connect(self.close)
        quit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(quit_action)

    def open_file_dialog(self):
        """Opens a file dialog to select a TIFF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF File", "", "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.load_tiff_file(file_path)

    def load_tiff_file(self, file_path):
        """Loads and processes the selected TIFF file."""
        try:
            self.images = tifffile.imread(file_path)
            if self.images.ndim < 2:
                raise ValueError("The selected file is not a valid image.")
            # If single image, reshape to a stack of 1
            if self.images.ndim == 2:
                self.images = self.images[:np.newaxis, ...]

            self.file_path = file_path
            self.num_pages = self.images.shape[-4] if self.images.ndim > 3 else self.images.shape[-3] if self.images.ndim > 2 else 1
            if self.images.ndim == 2:
                self.images = self.images[:np.newaxis, ...]
            elif self.images.ndim > 3:
                self.images = self.images.reshape(-1, self.images.shape[-2], self.images.shape[-1])

            self.bit_depth = self.images.dtype.itemsize * 8

            self.current_page = 0
            self.slider.setRange(0, self.num_pages - 1)
            self.slider.setValue(0)

            # Set an initial fixed size for the image label based on the first image
            first_image = self.images[-1] if self.images.ndim == 3 else self.images
            if first_image is not None and first_image.ndim == 2:
                h, w = first_image.shape
                self._fixed_image_size = QSize(w, h)
                self.image_label.setFixedSize(self._fixed_image_size)
            else:
                self._fixed_image_size = QSize()
                self.image_label.setFixedSize(256, 256) # Default size if no image

            self.update_ui_state(enabled=True)
            self.update_view()
            self.setWindowTitle(f"PyQt TIFF Viewer - {os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load TIFF file:\n{e}")
            self.update_ui_state(enabled=False)
            self.image_label.setText("Error loading file.")
            self.setWindowTitle("PyQt TIFF Viewer")


    def slider_changed(self, value):
        """Handles the slider value change."""
        self.current_page = value
        self.update_view()

    def normalize_image(self, image):
        """Normalizes the image to the full range of its bit depth using cv2.normalize."""
        if self.bit_depth == 8:
            alpha = 255
            norm_type = cv2.NORM_MINMAX
            dtype = cv2.CV_8U
        elif self.bit_depth == 16:
            alpha = 2**16 - 1
            norm_type = cv2.NORM_MINMAX
            dtype = cv2.CV_16U
        else:
            return image.astype(np.uint8) # Default for other bit depths

        normalized_image = cv2.normalize(image, None, 0, alpha, norm_type, dtype=dtype)
        return normalized_image

    def update_view(self):
        """Updates the image display and statistics labels."""
        if self.images is None:
            return

        image_data = self.images[-1] if self.images.ndim == 3 else self.images
        if image_data is None or image_data.ndim != 2:
            return

        # --- Normalize Image for Display ---
        normalized_image = self.normalize_image(image_data)

        # Convert to 8-bit for QImage display
        if self.bit_depth > 8:
            normalized_8bit = (normalized_image / (2**self.bit_depth - 1) * 255).astype(np.uint8)
        else:
            normalized_8bit = normalized_image.astype(np.uint8)

        height, width = normalized_8bit.shape
        bytes_per_line = width
        q_image = QImage(normalized_8bit.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        if not self._fixed_image_size.isEmpty():
            self.image_label.setPixmap(pixmap.scaled(self._fixed_image_size,
                                                    Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation))
        else:
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(),
                                                    Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation))


        # --- Update Statistics ---
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        mean_val = np.mean(image_data)
        std_val = np.std(image_data)
        snr_val = mean_val / std_val if std_val > 0 else 0

        self.page_label.setText(f"Page: {self.current_page + 1} / {self.num_pages}")
        self.min_label.setText(f"Min: {min_val:.2f}")
        self.max_label.setText(f"Max: {max_val:.2f}")
        self.mean_label.setText(f"Mean: {mean_val:.2f}")
        self.std_label.setText(f"Std Dev: {std_val:.2f}")
        self.snr_label.setText(f"SNR: {snr_val:.2f}")

    def export_image(self):
        """Exports the current image frame to a new TIFF file."""
        if self.file_path is None:
            return

        # Propose a filename
        original_dir = os.path.dirname(self.file_path)
        base_name = os.path.basename(self.file_path)
        name, ext = os.path.splitext(base_name)
        suggested_path = os.path.join(original_dir, f"{name}_page_{self.current_page + 1}{ext}")

        # Open file dialog to save
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image As...", suggested_path, "TIFF Files (*.tif *.tiff)"
        )

        if save_path:
            try:
                current_image_data = self.images[-1]
                tifffile.imsave(save_path, current_image_data)
                QMessageBox.information(self, "Success", f"Image successfully saved to:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save the image:\n{e}")

    def update_ui_state(self, enabled: bool):
        """Enables or disables UI elements."""
        self.slider.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        if not enabled:
            self.image_label.setText("Open a TIFF file to begin.")
            self.setWindowTitle("PyQt TIFF Viewer")
            for label in [self.page_label, self.min_label, self.max_label, self.mean_label, self.std_label, self.snr_label]:
                label.setText(label.text().split(':')[0] + ": N/A")
            self._fixed_image_size = QSize()
            self.image_label.setFixedSize(self.image_label.minimumSizeHint()) # Reset fixed size

    def resizeEvent(self, event):
        """Handle window resize."""
        if self._fixed_image_size.isEmpty() and self.images is not None and self.images.ndim == 3:
            # If no fixed size set yet (first load), update based on current label size
            self.update_view()
        elif not self._fixed_image_size.isEmpty():
            self.update_view() # Still update to potentially scale within the fixed size
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TiffViewerApp()
    viewer.show()
    sys.exit(app.exec())
