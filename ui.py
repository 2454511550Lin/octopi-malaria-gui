import sys
import os
import cv2
import numpy as np
import imageio
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QLineEdit, QPushButton, QScrollArea,QGridLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch.multiprocessing as mp

def numpy2png(img, resize_factor=5):
    try:
        # Ensure the image is in the correct shape (H, W, C)
        if img.shape[0] == 4:  # If the first dimension is 4, it's likely (C, H, W)
            img = img.transpose(1, 2, 0)
        
        # Separate fluorescence and DPC channels
        img_fluorescence = img[:, :, [2,1,0]]  # First 3 channels, but in reverse order
        img_dpc = img[:, :, 3]  # Last channel

        # Normalize the fluorescence image
        img_fluorescence = (img_fluorescence - img_fluorescence.min()) / (img_fluorescence.max() - img_fluorescence.min())
        img_fluorescence = (img_fluorescence * 255).astype(np.uint8)

        # Normalize the DPC image
        img_dpc = (img_dpc - img_dpc.min()) / (img_dpc.max() - img_dpc.min())
        img_dpc = (img_dpc * 255).astype(np.uint8)
        img_dpc = np.dstack([img_dpc, img_dpc, img_dpc])  # Make it 3 channels

        # Combine fluorescence and DPC
        img_overlay = cv2.addWeighted(img_fluorescence, 0.64, img_dpc, 0.36, 0)

        # Resize
        img_overlay = cv2.resize(img_overlay, (img_overlay.shape[1]*resize_factor, img_overlay.shape[0]*resize_factor), interpolation=cv2.INTER_NEAREST)

        return img_overlay
    except Exception as e:
        print(f"Error in numpy2png: {e}")
        return None

class ImageAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Image Analysis")
        self.setGeometry(100, 100, 1200, 800)

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # Create FOV tab
        fov_tab = QWidget()
        fov_layout = QVBoxLayout()
        fov_tab.setLayout(fov_layout)
        self.fov_list = QListWidget()
        fov_layout.addWidget(self.fov_list)
        tab_widget.addTab(fov_tab, "FOV List")

        # Create Cropped Images tab
        cropped_tab = QWidget()
        cropped_layout = QVBoxLayout()
        cropped_tab.setLayout(cropped_layout)

        # Add filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Min Score:"))
        self.score_filter = QLineEdit()
        self.score_filter.setPlaceholderText("Enter minimum score")
        filter_layout.addWidget(self.score_filter)
        filter_button = QPushButton("Apply Filter")
        filter_button.clicked.connect(self.apply_filter)
        filter_layout.addWidget(filter_button)
        cropped_layout.addLayout(filter_layout)

        # Add stats display
        self.stats_label = QLabel("Total Spots: 0 | Total RBC Count: 0")
        cropped_layout.addWidget(self.stats_label)

        # Add scrollable area for cropped images
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.cropped_layout = QGridLayout()
        self.scroll_widget.setLayout(self.cropped_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        cropped_layout.addWidget(self.scroll_area)

        tab_widget.addTab(cropped_tab, "Cropped Images")

        # Initialize data structures
        self.fov_data = {}
        self.cropped_images = []
        self.cropped_scores = []

    def update_fov_list(self, fov_id):
        self.fov_list.addItem(fov_id)
        self.fov_data[fov_id] = {'spots': 0, 'rbc_count': 0}

    def update_rbc_count(self, fov_id, count):
        self.fov_data[fov_id]['rbc_count'] = count
        self.update_stats()

    def update_stats(self):
        total_spots = sum(data['spots'] for data in self.fov_data.values())
        total_rbc = sum(data['rbc_count'] for data in self.fov_data.values())
        self.stats_label.setText(f"Total Spots: {total_spots} | Total RBC Count: {total_rbc}")

    def apply_filter(self):
        try:
            min_score = float(self.score_filter.text())
            self.display_cropped_images(min_score)
        except ValueError:
            print("Invalid score filter")

    def update_cropped_images(self, fov_id, images, scores):
        print(f"Received images for FOV {fov_id}. Shape: {images.shape}, Type: {images.dtype}")
        print(f"Received scores. Shape: {scores.shape}, Type: {scores.dtype}")
        
        self.cropped_images.extend(images)
        self.cropped_scores.extend(scores)
        self.fov_data[fov_id]['spots'] = images.shape[0]
        self.update_stats()
        self.display_cropped_images()

    def display_cropped_images(self, min_score=0):
        # Clear existing images
        for i in reversed(range(self.cropped_layout.count())): 
            self.cropped_layout.itemAt(i).widget().setParent(None)

        # Calculate the number of columns based on the window width
        window_width = self.width()
        image_width = 200  # Adjust this value as needed
        num_columns = max(1, window_width // (image_width + 10))  # +10 for some padding

        row, col = 0, 0
        for img, score in zip(self.cropped_images, self.cropped_scores):
            if score >= min_score:
                overlay_img = numpy2png(img)
                if overlay_img is not None:
                    height, width, channel = overlay_img.shape
                    bytes_per_line = 3 * width
                    qimg = QImage(overlay_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    scaled_pixmap = pixmap.scaled(image_width, image_width, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    
                    label = QLabel()
                    label.setPixmap(scaled_pixmap)
                    label.setAlignment(Qt.AlignCenter)
                    
                    # Create a widget to hold the image and score
                    widget = QWidget()
                    layout = QVBoxLayout()
                    layout.addWidget(label)
                    layout.addWidget(QLabel(f"Score: {score:.2f}"))
                    widget.setLayout(layout)
                    
                    self.cropped_layout.addWidget(widget, row, col)
                    
                    col += 1
                    if col >= num_columns:
                        col = 0
                        row += 1
                else:
                    print(f"Failed to process image with score {score}")

        # Force update
        self.cropped_layout.update()

        # Force update
        self.cropped_layout.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.display_cropped_images(float(self.score_filter.text() or 0))

class UIThread(QThread):
    update_fov = pyqtSignal(str)
    update_images = pyqtSignal(str, np.ndarray, np.ndarray)
    update_rbc = pyqtSignal(str, int)

    def __init__(self, shared_memory_final, shared_memory_classification, shared_memory_segmentation, final_lock):
        super().__init__()
        self.shared_memory_final = shared_memory_final
        self.shared_memory_classification = shared_memory_classification
        self.shared_memory_segmentation = shared_memory_segmentation
        self.final_lock = final_lock
        self.processed_fovs = set()

    def run(self):
        while True:
            with self.final_lock:
                fovs_to_process = [fov_id for fov_id, data in self.shared_memory_final.items() 
                                   if not data['displayed'] and fov_id not in self.processed_fovs]

            for fov_id in fovs_to_process:
                self.process_fov(fov_id)

            self.msleep(100)  # Sleep for 100ms to prevent high CPU usage

    def process_fov(self, fov_id):
        self.update_fov.emit(fov_id)
        
        classification_data = self.shared_memory_classification.get(fov_id, {})
        images = classification_data.get('cropped_images', np.array([]))
        scores = classification_data.get('scores', np.array([]))
        
        if len(images) > 0 and len(scores) > 0:
            print(f"Processing FOV {fov_id}")
            self.update_images.emit(fov_id, images, scores)
        else:
            print(f"No images or scores for FOV {fov_id}")
        
        segmentation_data = self.shared_memory_segmentation.get(fov_id, {})
        rbc_count = segmentation_data.get('n_cells', 0)
        self.update_rbc.emit(fov_id, rbc_count)
        
        with self.final_lock:
            self.shared_memory_final[fov_id]['displayed'] = True
            self.processed_fovs.add(fov_id)
        
        print(f"Finished processing FOV {fov_id}")

def ui_process(input_queue: mp.Queue, output: mp.Queue, shared_memory_final, shared_memory_classification, shared_memory_segmentation, final_lock):
    start_ui(shared_memory_final, shared_memory_classification, shared_memory_segmentation, final_lock)

def start_ui(shared_memory_final, shared_memory_classification, shared_memory_segmentation, final_lock):
    app = QApplication(sys.argv)
    window = ImageAnalysisUI()
    
    ui_thread = UIThread(shared_memory_final, shared_memory_classification, shared_memory_segmentation, final_lock)
    ui_thread.update_fov.connect(window.update_fov_list)
    ui_thread.update_images.connect(window.update_cropped_images)
    ui_thread.update_rbc.connect(window.update_rbc_count)
    ui_thread.start()
    
    window.show()
    sys.exit(app.exec_())