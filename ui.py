import sys
import numpy as np
import threading
from queue import Empty
from utils import numpy2png_ui as numpy2png
import numpy as np

import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox, QStyleFactory, QFileDialog,
    QComboBox, QCheckBox, QGroupBox, QGridLayout,QSpinBox, QFrame
)
from PyQt5.QtGui import QImage, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

import pyqtgraph as pg
from widgets import VirtualImageListWidget, ExpandableImageWidget

import time, os

from utils import SharedConfig

MINIMUM_SCORE_THRESHOLD = 0.31  # Adjust this value as needed
SILENT_MODE = False

class ImageAnalysisUI(QMainWindow):
    shutdown_signal = pyqtSignal()
    

    def __init__(self, start_event,shared_config:SharedConfig):
        super().__init__()
        self.start_event = start_event
        self.shared_config = shared_config
        self.setWindowTitle("Octopi")
        self.setGeometry(100, 100, 1920, 1080)
        
        # Set the application style to Fusion for a more modern look
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        
        # Set a custom color palette
        palette = self.palette()
        palette.setColor(palette.Window, QColor("#ECF0F1"))
        palette.setColor(palette.WindowText, QColor("#2C3E50"))
        palette.setColor(palette.Button, QColor("#2C3E50"))
        palette.setColor(palette.ButtonText, QColor("#FFFFFF"))
        palette.setColor(palette.Highlight, QColor("#3498DB"))
        self.setPalette(palette)
        
        self.setStyleSheet("""
        * { font-size: 20px;  }
            QMainWindow { background-color: #ECF0F1; }
            QTabWidget::pane { border: 1px solid #BDC3C7; background-color: white; }
            QTabBar::tab { 
                background-color: #ECF0F1; 
                padding: 8px 16px; 
                margin-right: 2px; 
                border-top-left-radius: 4px; 
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { 
                background-color: white; 
                border-bottom: 2px solid #3498DB; 
            }
            QPushButton { 
                background-color: #2C3E50; 
                color: white; 
                border: none; 
                padding: 8px 16px; 
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #357eab; }
            QLineEdit { 
                padding: 6px; 
                border: 1px solid #BDC3C7; 
                border-radius: 4px; 
                background-color: white;
            }
            QLabel { color: #2C3E50; }
            QTableWidget { 
                gridline-color: #BDC3C7;
                selection-background-color: #3498DB;
            }
            QTableWidget::item:hover { background-color: #3ea8c2; }
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
        self.max_cache_size = 10
        self.current_fov_index = -1
        self.selected_fov_id = None

        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)

        self.fov_image_data = {} 

        self.silent_mode = SILENT_MODE

        self.first_fov_time = None
        self.latest_fov_time = None

       

    def setup_ui(self):

        # Patient ID Label (initially empty)
        self.patient_id_label = QLabel("")
        self.patient_id_label.setAlignment(Qt.AlignCenter)
        self.patient_id_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #2C3E50;
            padding: 10px;
            background-color: #e6f2ff;
            border-radius: 5px;
        """)
        self.main_layout.addWidget(self.patient_id_label)

        # Top layout with shutdown button
        top_layout = QHBoxLayout()
        top_layout.addStretch()

        self.silent_mode_toggle = QPushButton("Silent Mode: Off")
        self.silent_mode_toggle.setCheckable(True)
        self.silent_mode_toggle.clicked.connect(self.toggle_silent_mode)


        top_layout.addWidget(self.silent_mode_toggle)

        self.new_patient_button = QPushButton("New Patient")
        self.new_patient_button.clicked.connect(self.new_patient)
        self.new_patient_button.setStyleSheet("""
            QPushButton { 
                background-color: #48abe8; 
                font-weight: bold;
            }
            QPushButton:hover { background-color: #357eab; }
        """)
        top_layout.addWidget(self.new_patient_button)

        self.shutdown_button = QPushButton("Shutdown")
        self.shutdown_button.clicked.connect(self.shutdown)
        self.shutdown_button.setStyleSheet("""
            QPushButton { 
                background-color: #dc3545; 
                font-weight: bold;
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

        # Card frame
        card = QFrame(self)
        card.setObjectName("card")
        card.setFixedSize(400, 500)
        card.setStyleSheet("""
            #card {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 25px;
            }
        """)
        card_layout = QVBoxLayout(card)

        # Welcome label
        welcome_label = QLabel("Welcome to Octopi")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("""
            font-size: 30px;
            font-weight: bold;
            color: #2C3E50;
            padding: 10px;
        """)

        card_layout.addWidget(welcome_label)

        # Patient ID input
        patient_id_layout = QVBoxLayout()
        patient_id_label = QLabel("Patient ID:")
        patient_id_label.setStyleSheet("""
            margin-bottom: 0px; 
        """)
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setPlaceholderText("Enter Patient ID")
        self.patient_id_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #ccc;
                border-radius: 4px;
                margin-bottom: 65px; 
                margin-top: 1px;
            }
        """)
        patient_id_layout.addWidget(patient_id_label)
        patient_id_layout.addWidget(self.patient_id_input)
        card_layout.addLayout(patient_id_layout)

        # To Loading Position button
        self.loading_position_button = QPushButton("To Loading Position")
        card_layout.addWidget(self.loading_position_button)
        self.loading_position_button.clicked.connect(self.move_to_loading_position)

        # Start Scanning button
        self.start_button = QPushButton("Start Scanning")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        card_layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_analysis)

        start_layout = QHBoxLayout(start_tab)
        start_layout.addWidget(card, alignment=Qt.AlignCenter)

        self.tab_widget.addTab(start_tab, "Start")

        # Modify the FOV Tab
        fov_tab = QWidget()
        fov_layout = QHBoxLayout(fov_tab)
        splitter = QSplitter(Qt.Horizontal)
        fov_layout.addWidget(splitter)

        # Left side: FOV image
        fov_widget = QWidget()
        left_layout = QVBoxLayout(fov_widget)
        
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(self.show_previous_fov)
        self.next_button.clicked.connect(self.show_next_fov)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        left_layout.addLayout(nav_layout)

        self.fov_image_view = pg.ImageView()
        self.setup_fov_image_view()
        left_layout.addWidget(self.fov_image_view)

        splitter.addWidget(fov_widget)

        # Middle: FOV list
        list_widget = QWidget()
        middle_layout = QVBoxLayout(list_widget)

         # Add a new label for average processing time
        self.avg_processing_time_label = QLabel("Avg Processing Time: N/A")
        self.avg_processing_time_label.setStyleSheet("""
            font-size: 14px;
            color: #2C3E50;
            padding: 5px;
            background-color: #ECF0F1;
            border-radius: 3px;
        """)
        middle_layout.addWidget(self.avg_processing_time_label)

        # Timer to update average processing time
        self.update_avg_timer = QTimer(self)
        self.update_avg_timer.timeout.connect(self.update_avg_processing_time)
        self.update_avg_timer.start(1000) 

        self.stats_label_small = QLabel("FoVs: 0 | Total RBCs: 0 | Total Positives: 0")
        self.stats_label_small.setStyleSheet("""
            font-size: 14px;
        """)     
        # add on top of the fov table
        middle_layout.addWidget(self.stats_label_small)

        self.fov_table = QTableWidget()
        self.fov_table.setColumnCount(3)
        self.fov_table.setHorizontalHeaderLabels(["FOV id", "RBCs", "Positives"])
        self.fov_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.fov_table.verticalHeader().setVisible(False)
        self.fov_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.fov_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.fov_table.itemClicked.connect(self.fov_table_item_clicked)
        middle_layout.addWidget(self.fov_table)
        
        # Right side: Positive Images

        positive_widget = QWidget()
        right_layout = QVBoxLayout(positive_widget)

        self.positive_images_widget = ExpandableImageWidget()
        right_layout.addWidget(self.positive_images_widget)

        splitter.addWidget(positive_widget)
        splitter.addWidget(list_widget)

        total_width = self.width()
        unit = total_width / 10  
        splitter.setSizes([int(4*unit), int(4*unit), int(2*unit)])

        self.tab_widget.addTab(fov_tab, "FOVs List")

         # Cropped Images Tab
        self.cropped_tab = QWidget()
        self.cropped_layout = QVBoxLayout(self.cropped_tab)

        self.stats_label = QLabel("FoVs: 0 | Total RBC Count: 0 | Total Malaria Positives: 0")
        self.stats_label.setStyleSheet("""
            margin-top: 10px; 
            margin-bottom: 10px;
            font-size: 50px;
            color: black;
        """)
        self.cropped_layout.addWidget(self.stats_label)

        self.virtual_image_list = VirtualImageListWidget()
        self.cropped_layout.addWidget(self.virtual_image_list)

        self.tab_widget.addTab(self.cropped_tab, "Malaria Detection Report")

        # A tab for live view
        live_view_tab = QWidget()
        live_view_layout = QVBoxLayout(live_view_tab)


        # Channel selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QHBoxLayout(channel_group)
        self.channel_combo = QComboBox()
        self.load_channels()
        channel_layout.addWidget(self.channel_combo, alignment=Qt.AlignTop | Qt.AlignLeft)
        live_view_layout.addWidget(channel_group)

        # Microscope controls
       
        self.live_button = QPushButton("LIVE")
        self.live_button.clicked.connect(self.start_live_view)

        live_view_layout.addWidget(self.live_button, alignment=Qt.AlignTop | Qt.AlignLeft)

        live_view_layout.addStretch(1)


        self.tab_widget.addTab(live_view_tab, "Live View")
        

        # a tab for settings
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)

        # Directory selection
        directory_group = QGroupBox("Save Directory")
        directory_layout = QHBoxLayout(directory_group)
        self.directory_input = QLineEdit()
        # set a fixed width for the input field
        self.directory_input.setFixedWidth(500)
        # set a default input
        self.directory_input.setText(os.path.join(os.getcwd(), "saved_data"))
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_directory)
        directory_layout.addWidget(self.directory_input)
        directory_layout.addWidget(self.browse_button)
        settings_layout.addWidget(directory_group, alignment=Qt.AlignTop | Qt.AlignLeft)

        # Image options
        options_group = QGroupBox("Image Saving Options")
        options_layout = QVBoxLayout(options_group)
        self.raw_images_check = QCheckBox("Raw Images")
        self.overlay_images_check = QCheckBox("Overlay Images")
        self.positives_images_check = QCheckBox("Spots Images")
        self.raw_images_check.setChecked(True)
        self.overlay_images_check.setChecked(True)
        self.positives_images_check.setChecked(True)
        options_layout.addWidget(self.raw_images_check)
        options_layout.addWidget(self.overlay_images_check)
        options_layout.addWidget(self.positives_images_check)
        settings_layout.addWidget(options_group, alignment=Qt.AlignTop | Qt.AlignLeft)

        # Position selection
        position_group = QGroupBox("Field of View Selection")
        position_layout = QGridLayout(position_group)
        position_layout.addWidget(QLabel("X:"), 0, 0)
        self.x_input = QSpinBox()
        self.x_input.setRange(1, 100)  # Adjust the range as needed
        self.x_input.setValue(1)  # Set default value to 1
        self.x_input.setStyleSheet("QSpinBox { width: 1px; height: 25px; }")
        position_layout.addWidget(self.x_input, 0, 1)
        position_layout.addWidget(QLabel("Y:"), 1, 0)
        self.y_input = QSpinBox()
        self.y_input.setRange(1, 100)  # Adjust the range as needed
        self.y_input.setValue(1)  # Set default value to 1
        self.y_input.setStyleSheet("QSpinBox { width: 1px; height: 25px; }")
        position_layout.addWidget(self.y_input, 1, 1)
        # shrink the first column to a certain ratio
        position_layout.setColumnStretch(0, 1)
        position_layout.setColumnStretch(1, 5)
        settings_layout.addWidget(position_group, alignment=Qt.AlignTop | Qt.AlignLeft)

        settings_layout.addStretch(1)

        settings_tab.setLayout(settings_layout)
        self.tab_widget.addTab(settings_tab, "Settings")

    def setup_fov_image_view(self):
        self.fov_image_view.ui.roiBtn.hide()
        self.fov_image_view.ui.menuBtn.hide()
        self.fov_image_view.ui.histogram.hide()
        self.fov_image_view.view.setMouseEnabled(x=True, y=True)
        self.fov_image_view.view.setBackgroundColor((255, 255, 255))

    def toggle_silent_mode(self):
        self.silent_mode = not self.silent_mode
        self.silent_mode_toggle.setText(f"Silent Mode: {'On' if self.silent_mode else 'Off'}")
        if self.silent_mode:
            self.fov_image_view.clear()
            self.positive_images_widget.set_invisaible()
        else:
            self.positive_images_widget.set_visible()
    
    def load_channels(self):
        try:
            tree = ET.parse('channel_configurations.xml')
            root = tree.getroot()
            channels = [mode.get('Name') for mode in root.findall('mode')]
            self.channel_combo.addItems(channels)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
        except FileNotFoundError:
            print("channel_configurations.xml file not found")

    def start_live_view(self):
        
        # check what is the text of the button
        if self.live_button.text() == "LIVE (Running)":
            self.live_button.setText("LIVE")
            self.live_button.setStyleSheet("""
                QPushButton {
                    background-color: #2C3E50;
                }
                QPushButton:hover { background-color: #357eab; }
            """)

        else:
            self.live_button.setText("LIVE (Running)")
        # change the color 
            self.live_button.setStyleSheet("""
                QPushButton { 
                    background-color: #48abe8; 
                }
                QPushButton:hover { background-color: #357eab; }
            """)


    def move_to_loading_position(self):
        if self.loading_position_button.text() == "To Loading Position":
            self.loading_position_button.setText("To Scanning Position")
            self.loading_position_button.setStyleSheet("""
                QPushButton { 
                    background-color: #48abe8; 
                }
                QPushButton:hover { background-color: #357eab; }
            """)
        else:
            self.loading_position_button.setText("To Loading Position")
            self.loading_position_button.setStyleSheet("""
                QPushButton {
                    background-color: #2C3E50;
                }
                QPushButton:hover { background-color: #357eab; }
            """)

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
        self.positive_images_widget.image_list.clear()
        self.stats_label.setText("FoVs: 0 | Total RBC Count: 0 | Total Malaria Positives: 0")
        
        # Reset other variables
        self.current_fov_index = -1
        self.selected_fov_id = None
        self.patient_id = ""
        
        # Clear the patient ID input and re-enable the start button
        self.patient_id_input.clear()
        self.start_button.setEnabled(True)
        self.start_button.setText("Start Scanning")
        
        # Switch back to the start tab
        self.tab_widget.setCurrentIndex(0)

        # Signal the main process to stop
        self.start_event.clear()

    def update_avg_processing_time(self):
        if self.first_fov_time is not None and self.latest_fov_time:
            total_time = self.latest_fov_time - self.first_fov_time
            avg_time = total_time / len(self.fov_data) 
            self.avg_processing_time_label.setText(f"Avg Processing Time: {avg_time:.3f} s")
        else:
            self.avg_processing_time_label.setText("Avg Processing Time: N/A")

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.directory_input.setText(directory)
    
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

        if fov_id == self.selected_fov_id and self.positive_images_widget.image_list.isVisible():
            self.update_positive_images(fov_id)

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

    def update_fov_list(self, fov_id):
        row_position = self.fov_table.rowCount()
        self.fov_table.insertRow(row_position)
        self.fov_table.setItem(row_position, 0, QTableWidgetItem(fov_id))
        self.fov_table.setItem(row_position, 1, QTableWidgetItem("0"))  # Initial RBC Count
        self.fov_table.setItem(row_position, 2, QTableWidgetItem("0"))  # Initial Malaria Positives
        self.fov_data[fov_id] = {'rbc_count': 0, 'malaria_positives': 0}

        current_time = time.time()
        if self.first_fov_time is None:
            self.first_fov_time = current_time

    def update_fov_image(self, fov_id, dpc_image, fluorescent_image):
        # Combine DPC and fluorescent images
        overlay_img = self.create_overlay(dpc_image, fluorescent_image)
        
        # Cache the numpy array
        self.fov_image_cache[fov_id] = overlay_img
        
        # Limit cache size
        if len(self.fov_image_cache) > self.max_cache_size:
            oldest_fov = next(iter(self.fov_image_cache))
            del self.fov_image_cache[oldest_fov]
        
        self.selected_fov_id = fov_id
        self.current_fov_index = list(self.fov_image_cache.keys()).index(fov_id)
        if not self.silent_mode:
            self.display_current_fov()

    def display_current_fov(self):
        if self.selected_fov_id and self.selected_fov_id in self.fov_image_cache:
            overlay_img = self.fov_image_cache[self.selected_fov_id]

            # Update the PyQtGraph ImageView
            self.fov_image_view.setImage(overlay_img, autoLevels=False, levels=(0, 255))

            # Highlight the selected row in the table
            row = self.find_fov_row(self.selected_fov_id)
            if row is not None:
                self.fov_table.selectRow(row)
        else:
            print(f"No FOV image available to display")

    def show_previous_fov(self):
        current_row = self.fov_table.currentRow()
        if current_row > 0:
            previous_row = current_row - 1
            fov_id = self.fov_table.item(previous_row, 0).text()
            self.load_fov_cache(fov_id)
            self.fov_table.selectRow(previous_row)
            self.update_positive_images(fov_id)

    def show_next_fov(self):
        current_row = self.fov_table.currentRow()
        if current_row < self.fov_table.rowCount() - 1:
            next_row = current_row + 1
            fov_id = self.fov_table.item(next_row, 0).text()
            self.load_fov_cache(fov_id)
            self.fov_table.selectRow(next_row)
            self.update_positive_images(fov_id)

    def fov_table_item_clicked(self, item):
        fov_id = self.fov_table.item(item.row(), 0).text()
        self.load_fov_cache(fov_id)

    def load_fov_cache(self,fov_id):
        if fov_id in self.fov_image_cache:
            self.current_fov_index = list(self.fov_image_cache.keys()).index(fov_id)
        else:
            print(f"FOV {fov_id} not in cache. It may need to be loaded.")
            # load from the local disk
            filename = f"{self.shared_config.get_path()}/{fov_id}_overlay.npy"
            img_array = np.load(filename)
            self.fov_image_cache[fov_id] = numpy2png(img_array, resize_factor=0.5)
            # delete the oldest image
            if len(self.fov_image_cache) > self.max_cache_size:
                oldest_fov = next(iter(self.fov_image_cache))
                del self.fov_image_cache[oldest_fov]

        self.selected_fov_id = fov_id
        self.display_current_fov()
        self.update_positive_images(fov_id)
            

    def update_positive_images(self, fov_id):
        if fov_id in self.fov_image_data:
            self.positive_images_widget.update_images(self.fov_image_data[fov_id], fov_id)

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

        self.latest_fov_time = time.time()

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
        self.stats_label.setText(f"FoVs: {len(self.fov_data)} | Total RBC Count: {total_rbc} | Total Malaria Positives: {total_positives}")
        self.stats_label_small.setText(f"FoVs: {len(self.fov_data)} | Total RBCs: {total_rbc} | Total Positives: {total_positives}")

    def start_analysis(self):
        self.patient_id = self.patient_id_input.text().strip()
        directory = self.directory_input.text().strip()

        if not self.patient_id:
            QMessageBox.warning(self, "Input Error", "Please enter a Patient ID before starting the analysis.")
            return

        if not directory:
            QMessageBox.warning(self, "Input Error", "Please enter or select a directory for saving data.")
            return

        # Create patient directory
        patient_directory = os.path.join(directory, self.patient_id)
        try:
            os.makedirs(patient_directory, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Failed to create patient directory: {e}")
            return
        
        self.shared_config.set_path(patient_directory)
        
        self.shared_config.save_raw_images = self.raw_images_check.isChecked()
        self.shared_config.save_overlay_images = self.overlay_images_check.isChecked()
        self.shared_config.save_positives_images = self.positives_images_check.isChecked()

        self.patient_id_label.setText(f"Patient ID: {self.patient_id}")
        self.start_event.set()  # Signal the main process to start
        self.tab_widget.setCurrentIndex(1)  # Switch to FOVs List tab
        self.start_button.setEnabled(False)
        self.start_button.setText("Scanning in progress")

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
               shared_memory_timing, final_lock, timing_lock, start_event, shutdown_event, shared_config):

    app = QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder='row-major')
    window = ImageAnalysisUI(start_event, shared_config)
    
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