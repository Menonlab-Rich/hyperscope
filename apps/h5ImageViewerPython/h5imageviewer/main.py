import sys
import cv2
import os
import h5py
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout,
    QHBoxLayout, QWidget, QLabel, QPushButton, QTreeWidget,
    QTreeWidgetItem, QSplitter, QMessageBox, QScrollArea,
    QComboBox, QSlider, QMenuBar, QMenu, QDialog,
    QFormLayout, QLineEdit, QDialogButtonBox, QSizePolicy,
    QCheckBox
)
from PySide6.QtGui import QPixmap, QImage, QAction, QKeySequence, QPainter, QPen, QColor
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QRect
from datetime import datetime, timedelta
import re
import torch
from transformers import SamModel, SamProcessor
from . import utils


class ImageExportDialog(QDialog):
    """Dialog for exporting images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Image")
        self.setMinimumWidth(300)

        layout = QFormLayout(self)

        self.filename_field = QLineEdit()
        layout.addRow("Filename:", self.filename_field)

        self.format_selector = QComboBox()
        self.format_selector.addItems(["TIFF (.tif)", "JPEG (.jpg)", "PNG (.png)"])
        layout.addRow("Format:", self.format_selector)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    def get_export_info(self):
        """Return the file name and format."""
        filename = self.filename_field.text()
        format_idx = self.format_selector.currentIndex()
        if format_idx == 0:
            format_ext = ".tif"
            save_format = "TIFF"
        elif format_idx == 1:
            format_ext = ".jpg"
            save_format = "JPEG"
        else:
            format_ext = ".png"
            save_format = "PNG"

        if not filename.lower().endswith(format_ext):
            filename += format_ext

        return filename, save_format


class H5TreeWidget(QTreeWidget):
    """Tree widget to display HDF5 file structure."""
    item_selected = Signal(str, object)
    prompts_detected = Signal(bool, str)
    timestamps_detected = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["HDF5 Structure"])
        self.file = None
        self.has_prompts = False
        self.prompts_path = None
        self.has_timestamps = False
        self.timestamps_path = None
        self.itemClicked.connect(self._item_clicked)
        self.itemDoubleClicked.connect(self._item_double_clicked)

    def load_file(self, file_path):
        """Load an HDF5 file and populate the tree."""
        self.clear()
        self.has_prompts = False
        self.prompts_path = None
        self.has_timestamps = False
        self.timestamps_path = None

        try:
            if self.file is not None:
                self.file.close()

            self.file = h5py.File(file_path, 'r')
            self._add_group(self.file, None)
            self._check_for_prompts(self.file)
            self._check_for_timestamps(self.file)

            self.prompts_detected.emit(self.has_prompts, self.prompts_path)
            self.timestamps_detected.emit(self.has_timestamps, self.timestamps_path)

            self.expandItem(self.topLevelItem(0))
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")
            return False

    def _check_for_timestamps(self, group):
        """Recursively check if a timestamps dataset exists."""
        for key, item in group.items():
            if key.lower() == "timestamps" and isinstance(item, h5py.Dataset):
                self.has_timestamps = True
                self.timestamps_path = item.name
                return
            if isinstance(item, h5py.Group):
                self._check_for_timestamps(item)
                if self.has_timestamps:
                    return

    def _check_for_prompts(self, group):
        """Recursively check if a prompts dataset exists."""
        for key, item in group.items():
            if key.lower() == "prompts" and isinstance(item, h5py.Dataset):
                self.has_prompts = True
                self.prompts_path = item.name
                return
            if isinstance(item, h5py.Group):
                self._check_for_prompts(item)
                if self.has_prompts:
                    return

    def _add_group(self, group, parent_item):
        """Recursively add groups and datasets to the tree."""
        if parent_item is None:
            item = QTreeWidgetItem(self, [group.name or "/"])
        else:
            name = group.name.split("/")[-1] or "/"
            item = QTreeWidgetItem(parent_item, [name])

        item.setData(0, Qt.UserRole, group.name)

        for key, obj in group.items():
            if isinstance(obj, h5py.Group):
                self._add_group(obj, item)
            elif isinstance(obj, h5py.Dataset):
                dataset_name = key
                child = QTreeWidgetItem(item, [dataset_name])
                child.setData(0, Qt.UserRole, obj.name)
                shape_str = f"Shape: {obj.shape}"
                dtype_str = f"Type: {obj.dtype}"
                QTreeWidgetItem(child, [shape_str])
                QTreeWidgetItem(child, [dtype_str])

    def _item_clicked(self, item, column):
        """Handle item click event."""
        path = item.data(0, Qt.UserRole)
        if path and self.file:
            try:
                if path in self.file:
                    obj = self.file[path]
                    if isinstance(obj, h5py.Dataset):
                        shape = obj.shape
                        is_valid_format = False

                        if len(shape) == 2:
                            is_valid_format = True
                        elif len(shape) == 3:
                            if shape[-1] <= 4:
                                is_valid_format = True
                            else:
                                is_valid_format = True
                        elif len(shape) == 4 and shape[-1] <= 4:
                            is_valid_format = True

                        if not is_valid_format:
                            QMessageBox.information(
                                self, "Not an Image",
                                f"Dataset dimensions {shape} are not image-like."
                            )
                            return

                        data_size_mb = np.prod(shape) * obj.dtype.itemsize / (1024 * 1024)
                        if data_size_mb > 100:
                            response = QMessageBox.question(
                                self, "Large Dataset",
                                f"Dataset is {data_size_mb:.1f} MB. Load? (May be slow)",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                            )
                            if response == QMessageBox.No:
                                return

                        self.item_selected.emit(path, obj)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to access dataset: {str(e)}")

    def _item_double_clicked(self, item, column):
        """Handle item double click event."""
        path = item.data(0, Qt.UserRole)
        if path and self.file:
            try:
                obj = self.file[path]
                if isinstance(obj, h5py.Dataset):
                    attrs = dict(obj.attrs)
                    attr_text = "Dataset attributes:\n\n"
                    attr_text += "\n".join([f"{k}: {v}" for k, v in attrs.items()]) if attrs else "No attributes."
                    attr_text += f"\n\nInfo:\nShape: {obj.shape}\nType: {obj.dtype}"
                    QMessageBox.information(self, f"Dataset Details: {path}", attr_text)
                elif isinstance(obj, h5py.Group):
                    item.setExpanded(not item.isExpanded())
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to access item: {str(e)}")


class ImageDisplayWidget(QWidget):
    """Widget to display image data and handle segmentation."""
    image_shown = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_dataset = None
        self.current_path = None
        self.current_index = 0
        self.current_image = None
        self.prompts_dataset = None
        self.prompts_path = None
        self.timestamps_dataset = None
        self.timestamps_path = None
        self.has_timestamps = False
        self.current_min = 0
        self.current_max = 0
        self.timestamp_offset = timedelta(0)
        self.is_displaying = False
        self.scroll_timer = QTimer(self)
        self.scroll_direction = None
        self.scroll_timer.timeout.connect(self._handle_scroll_timer)
        self.image_shown.connect(self._on_image_shown)

        # For bounding box selection
        self.rubber_band = None
        self.selection_start = None

        self.initUI()
        self.init_sam()

    def init_sam(self):
        """Initialize the Segment Anything Model."""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            self.model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
            QMessageBox.information(self, "SAM Loaded", "Segment Anything Model loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "SAM Error", f"Failed to load SAM model: {str(e)}")
            self.model = None
            self.processor = None

    def initUI(self):
        """Initialize the UI components."""
        self.layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setScaledContents(True)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setMouseTracking(True)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event

        self.nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("< Previous")
        self.prev_button.setEnabled(False)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.image_counter_label = QLabel("0/0")
        self.image_counter_label.setAlignment(Qt.AlignCenter)
        self.next_button = QPushButton("Next >")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.show_next_image)
        self.nav_layout.addWidget(self.prev_button)
        self.nav_layout.addWidget(self.image_counter_label)
        self.nav_layout.addWidget(self.next_button)

        self.jump_layout = QHBoxLayout()
        self.jump_label = QLabel("Jump to index:")
        self.jump_input = QLineEdit()
        self.jump_input.setFixedWidth(50)
        self.jump_button = QPushButton("Go")
        self.jump_button.clicked.connect(self.jump_to_image)
        self.jump_input.returnPressed.connect(self.jump_to_image)
        self.jump_layout.addWidget(self.jump_label)
        self.jump_layout.addWidget(self.jump_input)
        self.jump_layout.addWidget(self.jump_button)
        self.jump_layout.addStretch()

        self.prompts_layout = QHBoxLayout()
        self.show_prompts_checkbox = QCheckBox("Show Prompts")
        self.show_prompts_checkbox.setVisible(False)
        self.show_prompts_checkbox.stateChanged.connect(self.on_show_prompts_changed)
        self.prompts_label = QLabel()
        self.prompts_label.setVisible(False)
        self.prompts_layout.addWidget(self.show_prompts_checkbox)
        self.prompts_layout.addWidget(self.prompts_label, 1)

        self.timestamp_label = QLabel()
        self.timestamp_label.setVisible(False)
        self.info_label = QLabel()

        self.segment_button = QPushButton("Generate Segmentation")
        self.segment_button.setEnabled(False)
        self.segment_button.clicked.connect(self.generate_segmentation)

        self.save_mask_button = QPushButton("Save Mask")
        self.save_mask_button.setEnabled(False)
        self.save_mask_button.clicked.connect(self.save_mask)

        self.layout.addWidget(self.scroll_area, 1)
        self.layout.addLayout(self.nav_layout)
        self.layout.addLayout(self.jump_layout)
        self.layout.addLayout(self.prompts_layout)
        self.layout.addWidget(self.timestamp_label)
        self.layout.addWidget(self.info_label)
        self.layout.addWidget(self.segment_button)
        self.layout.addWidget(self.save_mask_button)

        self.jump_label.setVisible(False)
        self.jump_input.setVisible(False)
        self.jump_button.setVisible(False)

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.selection_start = event.pos()
            if self.rubber_band is None:
                self.rubber_band = QLabel(self.image_label)
                self.rubber_band.setStyleSheet("background-color: rgba(0, 255, 0, 100); border: 1px solid green;")
            self.rubber_band.setGeometry(QRect(self.selection_start, event.pos()).normalized())
            self.rubber_band.show()

    def mouse_move_event(self, event):
        if self.selection_start:
            self.rubber_band.setGeometry(QRect(self.selection_start, event.pos()).normalized())

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton and self.selection_start:
            self.selection_end = event.pos()
            self.rubber_band.hide()
            self.segment_button.setEnabled(True)

    def generate_segmentation(self):
        if self.current_image is None or self.selection_start is None or self.model is None:
            return

        pixmap = self.image_label.pixmap()
        img_w = pixmap.width()
        img_h = pixmap.height()
        label_w = self.image_label.width()
        label_h = self.image_label.height()

        x_scale = img_w / label_w
        y_scale = img_h / label_h

        x1 = int(self.selection_start.x() * x_scale)
        y1 = int(self.selection_start.y() * y_scale)
        x2 = int(self.selection_end.x() * x_scale)
        y2 = int(self.selection_end.y() * y_scale)

        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

        try:
            inputs = self.processor(self.current_image, input_boxes=[[bbox]], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]

            self.display_mask(masks[0][0])
            self.save_mask_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Segmentation Error", f"Failed to generate segmentation: {str(e)}")

    def display_mask(self, mask_tensor):
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_np, 'L').convert('RGBA')

        overlay = Image.new('RGBA', self.current_image.size, (0, 0, 0, 0))
        drawing = Image.new('RGBA', self.current_image.size, (0, 255, 0, 128))
        overlay.paste(drawing, (0, 0), mask_img)

        combined = Image.alpha_composite(self.current_image.convert('RGBA'), overlay)
        qimage = QImage(combined.tobytes(), combined.width, combined.height, QImage.Format_RGBA8888)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))

    def save_mask(self):
        if self.current_image is None:
            QMessageBox.warning(self, "No Image", "No image to save mask for.")
            return

        dialog = ImageExportDialog(self)
        if dialog.exec():
            filename, save_format = dialog.get_export_info()
            try:
                pixmap = self.image_label.pixmap()
                pixmap.save(filename, save_format)
                QMessageBox.information(self, "Mask Saved", f"Mask saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save mask: {str(e)}")
    
    def start_scrolling(self, direction):
        if self.scroll_direction != direction:
            self.scroll_direction = direction
            self.scroll_timer.stop()
            self._handle_scroll_timer()
            self.scroll_timer.start(50)

    def stop_scrolling(self):
        self.scroll_timer.stop()
        self.scroll_direction = None

    def _handle_scroll_timer(self):
        if not self.is_displaying:
            if self.scroll_direction == 'next':
                self.show_next_image()
            elif self.scroll_direction == 'prev':
                self.show_previous_image()

    @Slot()
    def _on_image_shown(self):
        self.is_displaying = False
        if self.scroll_timer.isActive():
            QTimer.singleShot(0, self._handle_scroll_timer)

    def save_timestamps_data(self, has_timestamps, path):
        self.has_timestamps = has_timestamps
        if has_timestamps:
            self.timestamps_path = path
        else:
            self.timestamps_path = None
            self.timestamps_dataset = None
            self.timestamp_label.setVisible(False)
            self.timestamp_offset = timedelta(0)

    def _extract_timestamp_from_filename(self, filename):
        if not filename: return None
        match = re.search(r'_(\d{8}_\d{6})\.h5$', os.path.basename(filename), re.IGNORECASE)
        if match:
            try:
                return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
            except ValueError as e:
                print(f"Error parsing filename timestamp: {e}")
                return None
        print(f"Filename {os.path.basename(filename)} does not match expected pattern.")
        return None

    @Slot(str, object)
    def set_timestamps_dataset(self, path, dataset):
        self.timestamp_offset = timedelta(0)
        self.timestamps_dataset = None
        self.timestamp_label.setVisible(False)

        if self.has_timestamps and self.timestamps_path and dataset:
            try:
                h5_file = dataset.file
                self.timestamps_dataset = h5_file[self.timestamps_path]
                self.timestamp_label.setVisible(True)
                filename_timestamp_obj = self._extract_timestamp_from_filename(h5_file.filename)

                if filename_timestamp_obj and len(self.timestamps_dataset) > 0:
                    last_dataset_timestamp_raw = self.timestamps_dataset[-1]
                    try:
                        last_dataset_datetime = datetime.fromtimestamp(last_dataset_timestamp_raw)
                        self.timestamp_offset = filename_timestamp_obj - last_dataset_datetime
                        print(f"Calculated timestamp offset: {self.timestamp_offset}")
                    except (TypeError, ValueError) as e:
                        QMessageBox.warning(self, "Timestamp Error",
                                            f"Could not convert dataset timestamp ({last_dataset_timestamp_raw}) "
                                            f"to datetime: {e}. Offset set to zero.")
                        self.timestamp_offset = timedelta(0)
                        self.timestamps_dataset = None
                        self.timestamp_label.setVisible(False)
                elif not filename_timestamp_obj and h5_file.filename:
                    QMessageBox.warning(self, "Filename Error",
                                        f"Could not extract timestamp from filename: {os.path.basename(h5_file.filename)}. "
                                        f"Offset set to zero.")
                self.show_current_timestamp()

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load timestamps or calculate offset: {str(e)}")
                self.timestamps_dataset = None
                self.timestamp_label.setVisible(False)
                self.timestamp_offset = timedelta(0)
        else:
            self.timestamps_dataset = None
            self.timestamp_label.setVisible(False)
            self.timestamp_offset = timedelta(0)

    @Slot(bool, str)
    def set_prompts_dataset(self, has_prompts, path):
        if has_prompts and path:
            self.prompts_path = path
            self.show_prompts_checkbox.setVisible(True)
        else:
            self.prompts_path = None
            self.prompts_dataset = None
            self.show_prompts_checkbox.setVisible(False)
            self.prompts_label.setVisible(False)
            self.show_prompts_checkbox.setChecked(False)

    def on_show_prompts_changed(self, state):
        if state == Qt.Checked and self.prompts_path and self.current_dataset:
            if self.prompts_dataset is None:
                try:
                    h5_file = self.current_dataset.file
                    self.prompts_dataset = h5_file[self.prompts_path]
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load prompts: {str(e)}")
                    self.show_prompts_checkbox.setChecked(False)
                    return
            self.show_current_prompt()
            self.prompts_label.setVisible(True)
        else:
            self.prompts_label.setVisible(False)

    def show_current_prompt(self):
        if self.prompts_dataset is None or not self.show_prompts_checkbox.isChecked():
            return
        try:
            if self.current_index < len(self.prompts_dataset):
                prompt = self.prompts_dataset[self.current_index]
                if isinstance(prompt, (np.string_, np.bytes_)):
                    prompt = prompt.decode('utf-8')
                else:
                    prompt = str(prompt)
                self.prompts_label.setText(f"Prompt: {prompt}")
            else:
                self.prompts_label.setText("No prompt available.")
        except Exception as e:
            self.prompts_label.setText(f"Error loading prompt: {str(e)}")

    def show_current_timestamp(self):
        if self.timestamps_dataset is None:
            self.timestamp_label.setVisible(False)
            return
        self.timestamp_label.setVisible(True)
        try:
            if self.current_index < len(self.timestamps_dataset):
                timestamp_raw = self.timestamps_dataset[self.current_index]
                if timestamp_raw > 7.34E15:
                    timestamp_raw -= 7.34E15
                try:
                    dt_object_raw = datetime.fromtimestamp(timestamp_raw)
                    dt_object_corrected = dt_object_raw + self.timestamp_offset
                    timestamp_str = dt_object_corrected.strftime("%y-%m-%d %H:%M:%S.%f")[:-3]
                except (TypeError, ValueError):
                    timestamp_str = f"{timestamp_raw} (raw - offset: {self.timestamp_offset})"
                self.timestamp_label.setText(f"Timestamp: {timestamp_str}")
            else:
                self.timestamp_label.setText("No timestamp available.")
        except Exception as e:
            self.timestamp_label.setText(f"Error loading timestamp: {str(e)}")

    @Slot(str, object)
    def display_dataset(self, path, dataset):
        try:
            self.current_dataset = dataset
            self.current_path = path
            shape = dataset.shape
            is_multi = (len(shape) == 3 and shape[-1] > 4) or len(shape) == 4
            self.current_index = 0
            self._display_image_slice(0 if is_multi else None)
            self.jump_label.setVisible(is_multi)
            self.jump_input.setVisible(is_multi)
            self.jump_button.setVisible(is_multi)
            self.show_current_timestamp()
        except Exception as e:
            self.display_error(f"Failed to display dataset: {str(e)}")

    def _display_image_slice(self, index):
        if self.is_displaying:
            return
        self.is_displaying = True
        try:
            image_data = self._get_image_slice(index)
            if image_data is None:
                self.display_error("Failed to retrieve image data.")
                return
            qimage = self._normalize_and_convert_to_qimage(image_data)
            if qimage is None:
                self.display_error("Failed to convert data to image format.")
                return
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap)
            self.current_image = self._get_pil_image(image_data)
            self._update_navigation_controls(index)
            self.show_current_prompt()
            self.show_current_timestamp()
            QApplication.processEvents()
        except Exception as e:
            self.display_error(f"Failed to display image slice: {str(e)}")
        finally:
            self.is_displaying = False
            self.image_shown.emit()

    def _get_image_slice(self, index):
        shape = self.current_dataset.shape
        if len(shape) == 2:
            return self.current_dataset[:]
        elif len(shape) == 3:
            if shape[-1] <= 4 and index is None:
                return self.current_dataset[:]
            elif index is not None:
                return self.current_dataset[index]
        elif len(shape) == 4 and index is not None:
            return self.current_dataset[index]
        elif index is None:
            return self.current_dataset[0] if shape[0] == 1 else self.current_dataset[:]
        return None

    def _normalize_and_convert_to_qimage(self, data):
        try:
            data = np.ascontiguousarray(data)
            data = data.astype(np.float32)
            data = utils.compute_windowed_std_dev_vectorized(data, (16,16))
            if data.dtype != np.uint8:
                data_norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            else:
                data_norm = data
            self.current_min = data.min()
            self.current_max = data.max()
            height, width = data_norm.shape[:2]
            if len(data_norm.shape) == 2:
                return QImage(data_norm.data, width, height, width, QImage.Format_Grayscale8)
            elif len(data_norm.shape) == 3:
                channels = data_norm.shape[2]
                if channels == 1:
                    return QImage(data_norm.data, width, height, width, QImage.Format_Grayscale8)
                elif channels == 3:
                    data_rgb = cv2.cvtColor(data_norm, cv2.COLOR_BGR2RGB)
                    return QImage(data_rgb.data, width, height, width * 3, QImage.Format_RGB888)
                elif channels == 4:
                    data_rgba = cv2.cvtColor(data_norm, cv2.COLOR_BGRA2RGBA)
                    return QImage(data_rgba.data, width, height, width * 4, QImage.Format_RGBA8888)
            return None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"QImage conversion failed: {str(e)}")
            return None

    def _get_pil_image(self, data):
        try:
            if data.dtype != np.uint8:
                data_norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            else:
                data_norm = data
            if len(data_norm.shape) == 2: return Image.fromarray(data_norm, 'L')
            if len(data_norm.shape) == 3:
                ch = data_norm.shape[2]
                if ch == 1: return Image.fromarray(data_norm[:, :, 0], 'L')
                if ch == 3: return Image.fromarray(cv2.cvtColor(data_norm, cv2.COLOR_BGR2RGB), 'RGB')
                if ch == 4: return Image.fromarray(cv2.cvtColor(data_norm, cv2.COLOR_BGRA2RGBA), 'RGBA')
            return None
        except Exception as e:
            QMessageBox.critical(self, "Error", f"PIL conversion failed: {str(e)}")
            return None

    def show_previous_image(self):
        shape = self.current_dataset.shape
        is_multi = (len(shape) == 3 and shape[-1] > 4) or len(shape) == 4
        if self.current_dataset is not None and self.current_index > 0 and is_multi and not self.is_displaying:
            self.current_index -= 1
            self._display_image_slice(self.current_index)

    def show_next_image(self):
        shape = self.current_dataset.shape
        is_multi = (len(shape) == 3 and shape[-1] > 4) or len(shape) == 4
        if self.current_dataset is not None and self.current_index < self.current_dataset.shape[0] - 1 and is_multi and not self.is_displaying:
            self.current_index += 1
            self._display_image_slice(self.current_index)

    def jump_to_image(self):
        if self.current_dataset is None or self.is_displaying: return
        try:
            index = int(self.jump_input.text()) - 1
            if 0 <= index < self.current_dataset.shape[0]:
                self.current_index = index
                self._display_image_slice(self.current_index)
            else:
                QMessageBox.warning(self, "Invalid Index", "Index out of range.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a number.")

    def _update_navigation_controls(self, index):
        if self.current_dataset is None: return
        shape = self.current_dataset.shape
        is_multi = (len(shape) == 3 and shape[-1] > 4) or len(shape) == 4
        if is_multi:
            total = shape[0]
            self.prev_button.setEnabled(self.current_index > 0)
            self.next_button.setEnabled(self.current_index < total - 1)
            self.image_counter_label.setText(f"{self.current_index + 1}/{total}")
            self.jump_input.setText(str(self.current_index + 1))
            current_idx_str = str(self.current_index + 1)
        else:
            total = 1
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.image_counter_label.setText("1/1")
            self.jump_input.clear()
            current_idx_str = "1"
        self.info_label.setText(f"Path: {self.current_path}\n"
                                f"Shape: {shape}\nType: {self.current_dataset.dtype}\n"
                                f"Index: {current_idx_str}\n"
                                f"Min/Max: {self.current_min}/{self.current_max}")

    def display_error(self, message):
        self.image_label.setText(message)
        self.current_image = None
        self.info_label.setText("")
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.image_counter_label.setText("0/0")
        self.jump_label.setVisible(False)
        self.jump_input.setVisible(False)
        self.jump_button.setVisible(False)
        self.timestamp_label.setVisible(False)
        self.segment_button.setEnabled(False)
        self.save_mask_button.setEnabled(False)


class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDF5 Image Viewer with SAM")
        self.setGeometry(100, 100, 1200, 800)
        self.h5_tree = H5TreeWidget(self)
        self.image_display = ImageDisplayWidget(self)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.h5_tree)
        splitter.addWidget(self.image_display)
        splitter.setSizes([300, 900])
        self.setCentralWidget(splitter)
        self.h5_tree.item_selected.connect(self.image_display.display_dataset)
        self.h5_tree.item_selected.connect(self.image_display.set_timestamps_dataset)
        self.h5_tree.prompts_detected.connect(self.image_display.set_prompts_dataset)
        self.h5_tree.timestamps_detected.connect(self.image_display.save_timestamps_data)
        self.image_display.image_shown.connect(self.update_export_action_state)
        self._create_actions()
        self._create_menus()
        self.setFocusPolicy(Qt.StrongFocus)
        self.image_display.setFocus()

    def _create_actions(self):
        self.open_action = QAction("&Open HDF5 File...", self)
        self.open_action.triggered.connect(self.open_file)
        self.open_action.setShortcut(QKeySequence.Open)
        self.export_action = QAction("&Export Image...", self)
        self.export_action.triggered.connect(self.export_image)
        self.export_action.setEnabled(False)
        self.export_action.setShortcut(QKeySequence.Save)
        self.exit_action = QAction("E&xit", self)
        self.exit_action.triggered.connect(self.close)
        self.exit_action.setShortcut(QKeySequence.Quit)

    def _create_menus(self):
        self.file_menu = self.menuBar().addMenu("&File")
        self.file_menu.addAction(self.open_action)
        self.file_menu.addAction(self.export_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open HDF5 File", "", "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if file_path:
            if self.h5_tree.load_file(file_path):
                self.update_export_action_state()

    def export_image(self):
        if self.image_display.current_image:
            dialog = ImageExportDialog(self)
            if dialog.exec():
                filename, save_format = dialog.get_export_info()
                try:
                    self.image_display.current_image.save(filename, format=save_format)
                    QMessageBox.information(self, "Export Successful", f"Image exported to {filename}")
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", f"Failed to export image: {str(e)}")
        else:
            QMessageBox.warning(self, "No Image", "No image is currently displayed to export.")

    def update_export_action_state(self):
        self.export_action.setEnabled(self.image_display.current_image is not None)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            if not event.isAutoRepeat():
                self.image_display.start_scrolling('prev')
            event.accept()
        elif key == Qt.Key_Right:
            if not event.isAutoRepeat():
                self.image_display.start_scrolling('next')
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        key = event.key()
        if not event.isAutoRepeat() and (key == Qt.Key_Left or key == Qt.Key_Right):
            self.image_display.stop_scrolling()
            event.accept()
        else:
            super().keyReleaseEvent(event)


def main():
    """Main function to run the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
