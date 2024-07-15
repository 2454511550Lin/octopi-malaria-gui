import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton,  
                             QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,QAbstractItemView, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import pyqtgraph as pg
import numpy as np
import torch.multiprocessing as mp
import threading
from queue import Empty

from utils import numpy2png_ui as numpy2png
from virtual_list import VirtualImageListWidget

MINIMUM_SCORE_THRESHOLD = 0.31  # Adjust this value as needed



class ImageAnalysisUI(QMainWindow):
    shutdown_signal = pyqtSignal()
    

    def __init__(self, start_event):
        super().__init__()
        self.start_event = start_event
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

        self.patient_id = ""
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

        self.fov_image_data = {} 

    def setup_ui(self):
        # Patient ID Label (initially empty)
        self.patient_id_label = QLabel("")
        self.patient_id_label.setAlignment(Qt.AlignCenter)
        self.patient_id_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #007bff;
            padding: 10px;
            background-color: #e6f2ff;
            border-radius: 5px;
        """)
        self.main_layout.addWidget(self.patient_id_label)

        # Top layout with shutdown button
        top_layout = QHBoxLayout()
        top_layout.addStretch()

        self.new_patient_button = QPushButton("New Patient")
        self.new_patient_button.clicked.connect(self.new_patient)
        self.new_patient_button.setFixedWidth(120)
        self.new_patient_button.setStyleSheet("""
            QPushButton { 
                background-color: #28a745; 
                font-weight: bold;
                padding: 5px 10px;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        top_layout.addWidget(self.new_patient_button)

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

        # Start Tab
        start_tab = QWidget()
        start_layout = QVBoxLayout(start_tab)
        
        welcome_label = QLabel("Welcome to Octopi")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 50px; margin-bottom: 20px;")
        start_layout.addWidget(welcome_label)

        # Patient ID input (centered and compact)
        patient_id_container = QWidget()
        patient_id_layout = QHBoxLayout(patient_id_container)
        patient_id_layout.setContentsMargins(0, 0, 0, 0)
        patient_id_label = QLabel("Patient ID:")

        font = patient_id_label.font()
        font.setPointSize(20)  # Set the font size to 20
        patient_id_label.setFont(font)

        self.patient_id_input = QLineEdit()

        # Set the font size of the QLineEdit
        font = self.patient_id_input.font()
        font.setPointSize(20)  # Set the font size to 20
        self.patient_id_input.setFont(font)

        self.patient_id_input = QLineEdit()
        self.patient_id_input.setFixedWidth(200)  # Set a fixed width for the input field
        self.patient_id_input.setPlaceholderText("Enter Patient ID")
        patient_id_layout.addWidget(patient_id_label)
        patient_id_layout.addWidget(self.patient_id_input)
        patient_id_layout.addStretch()
        start_layout.addWidget(patient_id_container, alignment=Qt.AlignCenter|Qt.AlignBottom)
        
        self.start_button = QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.start_analysis)
        self.start_button.setFixedWidth(300)
        self.start_button.setStyleSheet("""
            QPushButton { 
                background-color: #28a745; 
                font-weight: bold;
                padding: 30px 30px;
                font-size: 30px;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        start_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        
        self.tab_widget.addTab(start_tab, "Start")

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
        self.cropped_tab = QWidget()
        self.cropped_layout = QVBoxLayout(self.cropped_tab)

        self.stats_label = QLabel("Total RBC Count: 0 | Total Malaria Positives: 0")
        self.stats_label.setStyleSheet("""
            font-weight: bold; 
            margin-top: 10px; 
            margin-bottom: 10px;
            font-size: 50px;
            font-family: Arial, sans-serif;
            color: black;
        """)
        self.cropped_layout.addWidget(self.stats_label)

        self.virtual_image_list = VirtualImageListWidget()
        self.cropped_layout.addWidget(self.virtual_image_list)

        self.tab_widget.addTab(self.cropped_tab, "Malaria Detection Report")

    def setup_fov_image_view(self):
        self.fov_image_view.ui.roiBtn.hide()
        self.fov_image_view.ui.menuBtn.hide()
        self.fov_image_view.ui.histogram.hide()
        self.fov_image_view.view.setMouseEnabled(x=False, y=False)
        self.fov_image_view.view.setBackgroundColor((255, 255, 255))


    def shutdown(self):
        self.shutdown_signal.emit()
        self.close()

    def new_patient(self):
        # Clear all caches and data
        self.image_cache.clear()
        self.fov_image_cache.clear()
        self.fov_data.clear()
        self.fov_image_data.clear()
        
        # Reset UI elements
        self.fov_table.setRowCount(0)
        self.virtual_image_list.clear()
        self.fov_image_view.clear()
        self.patient_id_label.setText("")
        self.stats_label.setText("Total RBC Count: 0 | Total Malaria Positives: 0")
        
        # Reset other variables
        self.current_fov_index = -1
        self.newest_fov_id = None
        self.patient_id = ""
        
        # Clear the patient ID input and re-enable the start button
        self.patient_id_input.clear()
        self.start_button.setEnabled(True)
        self.start_button.setText("Start Analysis")
        
        # Switch back to the start tab
        self.tab_widget.setCurrentIndex(0)

        # Signal the main process to stop
        self.start_event.clear()

    def update_cropped_images(self, fov_id, images, scores):
        with self.image_lock:
            malaria_positives = 0
            updated_images = []
            for img, score in zip(images, scores):
                img_hash = hash(img.tobytes())
                if img_hash not in self.image_cache:
                    overlay_img = numpy2png(img, resize_factor=None)
                    if overlay_img is not None:
                        qimg = self.create_qimage(overlay_img)
                        self.image_cache[img_hash] = (qimg, score)
                        if score >= MINIMUM_SCORE_THRESHOLD:
                            malaria_positives += 1
                            updated_images.append((qimg, score))
                else:
                    qimg, cached_score = self.image_cache[img_hash]
                    if cached_score >= MINIMUM_SCORE_THRESHOLD:
                        malaria_positives += 1
                        updated_images.append((qimg, cached_score))
            
            self.update_malaria_positives(fov_id, malaria_positives)
            self.fov_image_data[fov_id] = updated_images

        self.update_all_fov_images()

    def update_all_fov_images(self):
        self.virtual_image_list.clear()
        for fov_id, images in self.fov_image_data.items():
            self.virtual_image_list.update_images(images, fov_id)
        self.update_stats()
    
    def create_qimage(self, overlay_img):
        height, width, channel = overlay_img.shape
        bytes_per_line = 3 * width
        return QImage(overlay_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def update_display(self):
        self.display_cropped_images(float(self.score_filter.text() or 0))

    def update_cropped_images_display(self):
        pass

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
        total_positives = self.virtual_image_list.model.rowCount()
        self.stats_label.setText(f"Total RBC Count: {total_rbc} | Total Malaria Positives: {total_positives}")


    def start_analysis(self):
        self.patient_id = self.patient_id_input.text().strip()
        if not self.patient_id:
            QMessageBox.warning(self, "Input Error", "Please enter a Patient ID before starting the analysis.")
            return
        
        self.patient_id_label.setText(f"Patient ID: {self.patient_id}")
        self.start_event.set()  # Signal the main process to start
        self.tab_widget.setCurrentIndex(1)  # Switch to FOVs List tab
        self.start_button.setEnabled(False)
        self.start_button.setText("In Progress")



class UIThread(QThread):
    update_fov = pyqtSignal(str)
    update_images = pyqtSignal(str, np.ndarray, np.ndarray)
    update_rbc = pyqtSignal(str, int)
    update_fov_image = pyqtSignal(str, np.ndarray, np.ndarray)

    def __init__(self, input_queue, output, shared_memory_final, shared_memory_classification, 
                 shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, 
                 shared_memory_timing, final_lock, timing_lock, window):
        super().__init__()
        self.input_queue = input_queue
        self.output = output    
        self.shared_memory_final = shared_memory_final
        self.shared_memory_classification = shared_memory_classification
        self.shared_memory_segmentation = shared_memory_segmentation
        self.shared_memory_acquisition = shared_memory_acquisition
        self.shared_memory_dpc = shared_memory_dpc
        self.shared_memory_timing = shared_memory_timing
        self.final_lock = final_lock
        self.timing_lock = timing_lock
        self.processed_fovs = set()
        self.window = window

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
                pass

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

    def shutdown(self):
        self.shutdown_signal.emit()
        QApplication.quit()  # This will close all windows

    def closeEvent(self, event):
        # This method is called when the window is about to be closed
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.shutdown()
            event.accept()
        else:
            event.ignore()


def ui_process(input_queue, output, shared_memory_final, shared_memory_classification, 
               shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, 
               shared_memory_timing, final_lock, timing_lock, start_event, shutdown_event):
    app = QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    window = ImageAnalysisUI(start_event)
    
    ui_thread = UIThread(input_queue, output, shared_memory_final, shared_memory_classification, 
                         shared_memory_segmentation, shared_memory_acquisition, shared_memory_dpc, 
                         shared_memory_timing, final_lock, timing_lock, window)
    ui_thread.update_fov.connect(window.update_fov_list)
    ui_thread.update_images.connect(window.update_cropped_images)
    ui_thread.update_rbc.connect(window.update_rbc_count)
    ui_thread.update_fov_image.connect(window.update_fov_image)
    
    def handle_shutdown():
        shutdown_event.set()
        app.quit()
    
    window.shutdown_signal.connect(handle_shutdown)
    
    ui_thread.start()
    
    window.show()
    app.exec_()
    shutdown_event.set()  # Ensure shutdown_event is set when app closes

def start_ui(input_queue, output, shared_memory_final, shared_memory_classification, shared_memory_segmentation, 
             shared_memory_acquisition, shared_memory_dpc, shared_memory_timing, final_lock, timing_lock,start_event):
    app = QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    window = ImageAnalysisUI(start_event)
    
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