from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListView
from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QRect, QSize
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QStyledItemDelegate

class ImageItem:
    def __init__(self, image, score, fov_id):
        self.image = image
        self.score = score
        self.fov_id = fov_id

class ImageListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def data(self, index, role):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return f"Score: {self.items[index.row()].score:.2f}"
        elif role == Qt.DecorationRole:
            return self.items[index.row()].image

    def addItem(self, image, score, fov_id):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self.items.append(ImageItem(image, score, fov_id))
        self.endInsertRows()

    def clear(self):
        self.beginResetModel()
        self.items.clear()
        self.endResetModel()

class ImageDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.item_size = QSize(150, 190)  # Adjusted size to accommodate FOV ID

    def paint(self, painter, option, index):
        image = index.data(Qt.DecorationRole)
        text = index.data(Qt.DisplayRole)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Draw image
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_rect = QRect(option.rect.x() + 5, option.rect.y() + 5, 140, 140)
        painter.drawPixmap(image_rect, scaled_pixmap)

        # Draw text (centered)
        text_rect = QRect(option.rect.x(), option.rect.y() + 150, 150, 40)
        painter.drawText(text_rect, Qt.AlignCenter, text)

        painter.restore()

    def sizeHint(self, option, index):
        return self.item_size

class VirtualImageListWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.list_view = QListView()
        self.model = ImageListModel()
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(ImageDelegate())
        self.list_view.setViewMode(QListView.IconMode)
        self.list_view.setResizeMode(QListView.Adjust)
        self.list_view.setSpacing(10)
        self.layout.addWidget(self.list_view)

    def add_image(self, image, score, fov_id):
        self.model.addItem(image, score, fov_id)

    def clear(self):
        self.model.clear()

    def update_images(self, images, fov_id):
        for image, score in images:
            self.add_image(image, score, fov_id)
