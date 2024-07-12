import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QListWidget, QLineEdit, QPushButton, QScrollArea, QGridLayout, 
                             QSplitter, QSizePolicy, QTableWidget, QTableWidgetItem, QHeaderView,QAbstractItemView)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import pyqtgraph as pg
import numpy as np
import torch.multiprocessing as mp
import threading
from queue import Empty
import cv2


MINIMUM_SCORE_THRESHOLD = 0.31  # Adjust this value as needed

def numpy2png(img, resize_factor=5):
    try:
        # Ensure the image is in the correct shape (H, W, C)
        if img.shape[0] == 4:  # If the first dimension is 4, it's likely (C, H, W)
            img = img.transpose(1, 2, 0)
        
        # Separate fluorescence and DPC channels
        img_fluorescence = img[:, :, [2,1,0]]  # First 3 channels, but in reverse order
        img_dpc = img[:, :, 3]  # Last channel

        # Normalize the fluorescence image
        epsilon = 1e-7
        img_fluorescence = (img_fluorescence - img_fluorescence.min()) / (img_fluorescence.max() - img_fluorescence.min() + epsilon)
        img_fluorescence = (img_fluorescence * 255).astype(np.uint8)

        # Normalize the DPC image
        img_dpc = (img_dpc - img_dpc.min()) / (img_dpc.max() - img_dpc.min())
        img_dpc = (img_dpc * 255).astype(np.uint8)
        img_dpc = np.dstack([img_dpc, img_dpc, img_dpc])  # Make it 3 channels

        # Combine fluorescence and DPC
        img_overlay = cv2.addWeighted(img_fluorescence, 0.64, img_dpc, 0.36, 0)

        # Resize
        if resize_factor is not None:
            if resize_factor >=1:
                img_overlay = cv2.resize(img_overlay, (img_overlay.shape[1]*resize_factor, img_overlay.shape[0]*resize_factor), interpolation=cv2.INTER_NEAREST)
            if resize_factor < 1:
                img_overlay = cv2.resize(img_overlay, (int(img_overlay.shape[1]*resize_factor), int(img_overlay.shape[0]*resize_factor)), interpolation=cv2.INTER_NEAREST)

        return img_overlay
    except Exception as e:
        print(f"Error in numpy2png: {e}")
        return None

class ImageAnalysisUI(QMainWindow):
    shutdown_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Image Analysis")
        self.setGeometry(100, 100, 1920, 1080)
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QTabWidget::pane { border: 1px solid #d0d0d0; background-color: white; }
            QTabBar::tab { background-color: #e0e0e0; padding: 8px 16px; margin-right: 2px; }
            QTabBar::tab:selected { background-color: white; border-bottom: 2px solid #007bff; }
            QPushButton { background-color: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 4px; }
            QPushButton:hover { background-color: #0056b3; }
            QLineEdit { padding: 6px; border: 1px solid #d0d0d0; border-radius: 4px; }
            QLabel { color: #333333; }
        """)
        self.image_lock = threading.Lock()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.setup_ui()

        self.image_cache = {}
        self.fov_image_cache = {}
        self.fov_data = {}
        self.max_cache_size = 50
        self.current_fov_index = -1
        self.newest_fov_id = None

        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.update_cropped_images_display)

    def setup_ui(self):
        # Top layout with shutdown button
        top_layout = QHBoxLayout()
        top_layout.addStretch()
        self.shutdown_button = QPushButton("Shutdown")
        self.shutdown_button.clicked.connect(self.shutdown)
        self.shutdown_button.setFixedWidth(120)
        self.shutdown_button.setStyleSheet("""
            QPushButton { 
                background-color: #dc3545; 
                font-weight: bold;
                padding: 5px 10px;
            }
            QPushButton:hover { background-color: #c82333; }
        """)
        top_layout.addWidget(self.shutdown_button)
        self.main_layout.addLayout(top_layout)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        # FOV Tab
        fov_tab = QWidget()
        fov_layout = QVBoxLayout(fov_tab)
        splitter = QSplitter(Qt.Horizontal)
        fov_layout.addWidget(splitter)

        # Left side: FOV image
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.fov_image_view = pg.ImageView()
        self.setup_fov_image_view()
        left_layout.addWidget(self.fov_image_view)

        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(self.show_previous_fov)
        self.next_button.clicked.connect(self.show_next_fov)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        left_layout.addLayout(nav_layout)

        splitter.addWidget(left_widget)

        # Right side: FOV list
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.fov_table = QTableWidget()
        self.fov_table.setColumnCount(3)
        self.fov_table.setHorizontalHeaderLabels(["FOV id", "RBCs", "Positives"])
        self.fov_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.fov_table.verticalHeader().setVisible(False)
        self.fov_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.fov_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.fov_table.itemClicked.connect(self.fov_table_item_clicked)
        self.fov_table.setStyleSheet("""
            QTableWidget { border: 1px solid #d0d0d0; border-radius: 4px; }
            QTableWidget::item { padding: 6px; }
            QTableWidget::item:selected { background-color: #007bff; color: white; }
        """)
        right_layout.addWidget(self.fov_table)

        splitter.addWidget(right_widget)
        splitter.setSizes([1300, 300])

        self.tab_widget.addTab(fov_tab, "FOVs List")

        # Cropped Images Tab
        cropped_tab = QWidget()
        cropped_layout = QVBoxLayout(cropped_tab)

        self.stats_label = QLabel("Total RBC Count: 0 | Total Malaria Positives: 0")
        self.stats_label.setStyleSheet("""
            font-weight: bold; 
            margin-top: 10px; 
            margin-bottom: 10px;
            font-size: 50px;  /* Adjust the size as needed */
            font-family: Arial, sans-serif;  /* Change the font family as desired */
            color: black;  /* Change the color as needed */
        """)
        cropped_layout.addWidget(self.stats_label)

        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.cropped_layout = QGridLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        cropped_layout.addWidget(self.scroll_area)

        self.tab_widget.addTab(cropped_tab, "Malaria Detection Report")

    def setup_fov_image_view(self):
        self.fov_image_view.ui.roiBtn.hide()
        self.fov_image_view.ui.menuBtn.hide()
        self.fov_image_view.ui.histogram.hide()
        self.fov_image_view.getView().setAspectLocked(True)
        self.fov_image_view.getView().invertY(True)
        self.fov_image_view.view.setMouseEnabled(x=False, y=False)
        #self.fov_image_view.view.hideAxis('left')
        #self.fov_image_view.view.hideAxis('bottom')
        self.fov_image_view.view.setBackgroundColor((255, 255, 255))


    def shutdown(self):
        self.shutdown_signal.emit()
        self.close()

    def update_cropped_images(self, fov_id, images, scores):
        with self.image_lock:
            malaria_positives = 0
            for img, score in zip(images, scores):
                img_hash = hash(img.tobytes())
                if img_hash not in self.image_cache:
                    overlay_img = numpy2png(img, resize_factor=None)
                    if overlay_img is not None:
                        qimg = self.create_qimage(overlay_img)
                        self.image_cache[img_hash] = (qimg, score)
                        if score >= MINIMUM_SCORE_THRESHOLD:
                            malaria_positives += 1
            self.update_malaria_positives(fov_id, malaria_positives)
        self.update_cropped_images_display()

    def create_qimage(self, overlay_img):
        height, width, channel = overlay_img.shape
        bytes_per_line = 3 * width
        return QImage(overlay_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def update_display(self):
        self.display_cropped_images(float(self.score_filter.text() or 0))

    def update_cropped_images_display(self):
        for i in reversed(range(self.cropped_layout.count())): 
            self.cropped_layout.itemAt(i).widget().setParent(None)

        window_width = self.scroll_area.width()
        image_width = 100
        num_columns = max(1, window_width // (image_width + 5))

        row, col = 0, 0
        self.total_malaria_positives = 0
        with self.image_lock:
            for img_hash, (qimg, score) in self.image_cache.items():
                if score >= MINIMUM_SCORE_THRESHOLD:
                    self.total_malaria_positives += 1
                    # ... (rest of the function remains the same)
                    pixmap = QPixmap.fromImage(qimg)
                    scaled_pixmap = pixmap.scaled(image_width, image_width, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    
                    label = QLabel()
                    label.setPixmap(scaled_pixmap)
                    label.setAlignment(Qt.AlignCenter)
                    
                    widget = QWidget()
                    layout = QVBoxLayout(widget)
                    layout.addWidget(label)
                    layout.addWidget(QLabel(f"Score: {score:.2f}"))
                    
                    self.cropped_layout.addWidget(widget, row, col)
                    
                    col += 1
                    if col >= num_columns:
                        col = 0
                        row += 1

        self.scroll_widget.setLayout(self.cropped_layout)
        self.update_stats()

        self.scroll_widget.setLayout(self.cropped_layout)
        self.update_stats()

    def update_fov_list(self, fov_id):
        row_position = self.fov_table.rowCount()
        self.fov_table.insertRow(row_position)
        self.fov_table.setItem(row_position, 0, QTableWidgetItem(fov_id))
        self.fov_table.setItem(row_position, 1, QTableWidgetItem("0"))  # Initial RBC Count
        self.fov_table.setItem(row_position, 2, QTableWidgetItem("0"))  # Initial Malaria Positives
        self.fov_data[fov_id] = {'rbc_count': 0, 'malaria_positives': 0}

    def update_fov_image(self, fov_id, dpc_image, fluorescent_image):
        # Combine DPC and fluorescent images
        overlay_img = self.create_overlay(dpc_image, fluorescent_image)
        
        # Cache the numpy array
        self.fov_image_cache[fov_id] = overlay_img
        
        # Limit cache size
        if len(self.fov_image_cache) > self.max_cache_size:
            oldest_fov = next(iter(self.fov_image_cache))
            del self.fov_image_cache[oldest_fov]
        
        self.newest_fov_id = fov_id
        self.current_fov_index = list(self.fov_image_cache.keys()).index(fov_id)
        self.display_current_fov()

    def display_current_fov(self):
        if self.newest_fov_id and self.newest_fov_id in self.fov_image_cache:
            overlay_img = self.fov_image_cache[self.newest_fov_id]

            # Update the PyQtGraph ImageView
            self.fov_image_view.setImage(overlay_img, autoLevels=False, levels=(0, 255))

            # Highlight the selected row in the table
            row = self.find_fov_row(self.newest_fov_id)
            if row is not None:
                self.fov_table.selectRow(row)
        else:
            print(f"No FOV image available to display")

    def show_previous_fov(self):
        if self.current_fov_index > 0:
            self.current_fov_index -= 1
            self.newest_fov_id = list(self.fov_image_cache.keys())[self.current_fov_index]
            self.display_current_fov()

    def show_next_fov(self):
        if self.current_fov_index < len(self.fov_image_cache) - 1:
            self.current_fov_index += 1
            self.newest_fov_id = list(self.fov_image_cache.keys())[self.current_fov_index]
            self.display_current_fov()

    def fov_table_item_clicked(self, item):
        fov_id = self.fov_table.item(item.row(), 0).text()
        if fov_id in self.fov_image_cache:
            self.current_fov_index = list(self.fov_image_cache.keys()).index(fov_id)
            self.newest_fov_id = fov_id
            self.display_current_fov()
        else:
            print(f"FOV {fov_id} not in cache. It may need to be loaded.")


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_timer.start(200)

    def create_overlay(self, dpc_image, fluorescent_image):
        # stack fluorescent and DPC images to 4xHxW
        # then direcly call numpy2png
        img = np.stack([fluorescent_image[:,:,0], fluorescent_image[:,:,1], fluorescent_image[:,:,2], dpc_image], axis=0)
        img =  numpy2png(img,resize_factor=0.5)
        
        return img

    def update_rbc_count(self, fov_id, count):
        self.fov_data[fov_id]['rbc_count'] = count
        row = self.find_fov_row(fov_id)
        if row is not None:
            self.fov_table.setItem(row, 1, QTableWidgetItem(str(count)))
        self.update_stats()

    def find_fov_row(self, fov_id):
        for row in range(self.fov_table.rowCount()):
            if self.fov_table.item(row, 0).text() == fov_id:
                return row
        return None
    def update_malaria_positives(self, fov_id, count):
        self.fov_data[fov_id]['malaria_positives'] = count
        row = self.find_fov_row(fov_id)
        if row is not None:
            self.fov_table.setItem(row, 2, QTableWidgetItem(str(count)))
        self.update_stats()
    def update_stats(self):
        total_rbc = sum(data['rbc_count'] for data in self.fov_data.values())
        self.stats_label.setText(f"Total RBC Count: {total_rbc} | Total Malaria Positives: {self.total_malaria_positives}")

class UIThread(QThread):
    update_fov = pyqtSignal(str)
    update_images = pyqtSignal(str, np.ndarray, np.ndarray)
    update_rbc = pyqtSignal(str, int)
    update_fov_image = pyqtSignal(str, np.ndarray, np.ndarray)  # New signal for full FOV images

    def __init__(self, input_queue,output,shared_memory_final, shared_memory_classification, shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc,shared_memory_timing,final_lock,timing_lock,window):
        super().__init__()
        self.input_queue = input_queue
        self.output = output    
        self.shared_memory_final = shared_memory_final
        self.shared_memory_classification = shared_memory_classification
        self.shared_memory_segmentation = shared_memory_segmentation
        self.shared_memory_acquisition = shared_memory_acquisition  # New shared memory for full images
        self.shared_memory_dpc = shared_memory_dpc
        self.shared_memory_timing = shared_memory_timing
        self.final_lock = final_lock
        self.timing_lock = timing_lock
        self.processed_fovs = set()

        self.window = window

    def process_fov(self, fov_id):
        self.update_fov.emit(fov_id)
        
        # Emit full FOV images
        acquisition_data = self.shared_memory_acquisition.get(fov_id, {})
        dpc_data = self.shared_memory_dpc.get(fov_id, {})
        dpc_image = dpc_data.get('dpc_image', np.array([]))
        fluorescent_image = acquisition_data.get('fluorescent', np.array([]))
        
        #print(f"Processing FOV {fov_id} to display")
        #print(f"DPC image shape: {dpc_image.shape}, Fluorescent image shape: {fluorescent_image.shape}")

        if dpc_image.size > 0 and fluorescent_image.size > 0:
            #print(f"Emitting update_fov_image signal for FOV {fov_id}")
            self.update_fov_image.emit(fov_id, dpc_image, fluorescent_image)
        else:
            print(f"Missing DPC or fluorescent image for FOV {fov_id}")

        classification_data = self.shared_memory_classification.get(fov_id, {})
        images = classification_data.get('cropped_images', np.array([]))
        scores = classification_data.get('scores', np.array([]))
        
        if len(images) > 0 and len(scores) > 0:
            self.update_images.emit(fov_id, images, scores)
        else:
            print(f"No images or scores for FOV {fov_id}")
        
        segmentation_data = self.shared_memory_segmentation.get(fov_id, {})
        rbc_count = segmentation_data.get('n_cells', 0)
        self.update_rbc.emit(fov_id, rbc_count)
            


    def run(self):

        while True:
            try:
                fov_id = self.input_queue.get(timeout=0.1)
                self.log_time(fov_id, "UI Process", "start")

                with self.final_lock:
                    if fov_id in self.shared_memory_final and not self.shared_memory_final[fov_id]['displayed']:
                        self.process_fov(fov_id)
                        self.log_time(fov_id, "UI Process", "end")

                        temp_dict = self.shared_memory_final[fov_id]
                        temp_dict['displayed'] = True
                        self.shared_memory_final[fov_id] = temp_dict    
                        self.processed_fovs.add(fov_id)
                        if self.shared_memory_final[fov_id]['saved']:
                            self.output.put(fov_id)

            except Empty:
                #print("UI Process: No FOV to process")
                pass

    
    def log_time(self,fov_id: str, process_name: str, event: str):
        import time
        with self.timing_lock:
            if fov_id not in self.shared_memory_timing:
                self.shared_memory_timing[fov_id] = {}
            if process_name not in self.shared_memory_timing[fov_id]:
                self.shared_memory_timing[fov_id][process_name] = {}

            temp_dict = self.shared_memory_timing[fov_id]
            temp_process_dict = temp_dict.get(process_name, {})
            temp_process_dict[event] = time.time()
            temp_dict[process_name] = temp_process_dict
            self.shared_memory_timing[fov_id] = temp_dict

    def connect_shutdown(self, shutdown_func):
        self.window.shutdown_signal.connect(shutdown_func)

def ui_process(input_queue: mp.Queue, output: mp.Queue, shared_memory_final, shared_memory_classification, shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, shared_memory_timing,final_lock,timing_lock):
    start_ui(input_queue,output,shared_memory_final, shared_memory_classification, shared_memory_segmentation,shared_memory_acquisition,shared_memory_dpc,shared_memory_timing, final_lock,timing_lock)

def start_ui(input_queue, output, shared_memory_final, shared_memory_classification, shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, shared_memory_timing, final_lock, timing_lock):
    app = QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    window = ImageAnalysisUI()
    
    ui_thread = UIThread(input_queue, output, shared_memory_final, shared_memory_classification, shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, shared_memory_timing, final_lock, timing_lock, window)
    ui_thread.update_fov.connect(window.update_fov_list)
    ui_thread.update_images.connect(window.update_cropped_images)
    ui_thread.update_rbc.connect(window.update_rbc_count)
    ui_thread.update_fov_image.connect(window.update_fov_image)
    
    def shutdown():
        app.quit()
    
    ui_thread.connect_shutdown(shutdown)
    ui_thread.start()
    
    window.show()
    sys.exit(app.exec_())