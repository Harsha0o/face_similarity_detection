import sys
import os
import cv2
import math
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt

# Function to calculate Euclidean distance between two images
def euclidean_distance(img1, img2):
    distance = 0
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])

    for i in range(min_height):
        for j in range(min_width):
            distance += (img1[i][j] / 255.0 - img2[i][j] / 255.0) ** 2
    return math.sqrt(distance)

class ImageSimilarityApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Similarity App')
        self.setWindowIcon(QIcon("face-recognition.png"))  # Set the window icon
        self.setGeometry(100, 100, 800, 600)

        self.setStyleSheet("""
            QWidget {
                background-color: #F0F0F0;
            }
            QLabel {
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton {
                font-size: 14px;
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        main_layout = QVBoxLayout()

        self.label = QLabel('Select the input image and folder to compare.', self)
        self.label.setFont(QFont("Arial", 16))
        self.label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label)

        button_layout = QHBoxLayout()

        self.select_image_button = QPushButton('Select Image', self)
        self.select_image_button.clicked.connect(self.select_image)
        button_layout.addWidget(self.select_image_button)

        self.select_folder_button = QPushButton('Select Folder', self)
        self.select_folder_button.clicked.connect(self.select_folder)
        button_layout.addWidget(self.select_folder_button)

        main_layout.addLayout(button_layout)

        self.compare_button = QPushButton('Compare Images', self)
        self.compare_button.clicked.connect(self.compare_images)
        main_layout.addWidget(self.compare_button)

        self.result_label = QLabel('', self)
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        image_display_layout = QHBoxLayout()

        self.input_image_label = QLabel('', self)
        self.input_image_label.setAlignment(Qt.AlignCenter)
        image_display_layout.addWidget(self.input_image_label)

        self.match_label = QLabel('', self)
        self.match_label.setAlignment(Qt.AlignCenter)
        image_display_layout.addWidget(self.match_label)

        main_layout.addLayout(image_display_layout)

        self.setLayout(main_layout)

        self.input_img_path = ""
        self.folder_path = ""

    def select_image(self):
        options = QFileDialog.Options()
        self.input_img_path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if self.input_img_path:
            self.label.setText(f"Selected Image: {self.input_img_path}")
            pixmap = QPixmap(self.input_img_path)
            self.input_image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))

    def select_folder(self):
        options = QFileDialog.Options()
        self.folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if self.folder_path:
            self.label.setText(f"Selected Folder: {self.folder_path}")

    def compare_images(self):
        if not self.input_img_path or not self.folder_path:
            QMessageBox.critical(self, "Error", "Please select both image and folder.")
            return

        input_img = cv2.imread(self.input_img_path, cv2.IMREAD_GRAYSCALE)
        if input_img is None:
            QMessageBox.critical(self, "Error", "Unable to read the input image.")
            return

        images = []
        image_filenames = []
        for filename in os.listdir(self.folder_path):
            img_path = os.path.join(self.folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                image_filenames.append(img_path)
            else:
                print(f"Warning: Unable to read image '{img_path}'.")

        if not images:
            QMessageBox.critical(self, "Error", "No images found in the folder.")
            return

        input_img = input_img / 255.0
        distances = []
        for img in images:
            img = img / 255.0
            distance = euclidean_distance(input_img, img)
            distances.append(distance)

        closest_match_index = distances.index(min(distances))
        closest_match_path = image_filenames[closest_match_index]
        self.result_label.setText(f"Closest match: {os.path.basename(closest_match_path)} with distance {min(distances):.2f}")

        # Load and display the closest match image
        pixmap = QPixmap(closest_match_path)
        self.match_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageSimilarityApp()
    ex.show()
    sys.exit(app.exec_())
