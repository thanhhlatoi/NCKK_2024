import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QTextEdit
from PyQt5.QtGui import QPixmap
from ultralytics import YOLO
from test4 import clear_crop_directory, crop_and_save, process_license_plate

# Khởi tạo mô hình YOLOv8
model = YOLO("C:/XuLyAnh/YoloV8/runs/detect/train/weights/best.pt")

class ImageUploader(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.imagePath = None

    def initUI(self):
        self.setWindowTitle('Image Uploader')
        self.setGeometry(100, 100, 400, 400)

        self.layout = QVBoxLayout()

        self.label = QLabel('No image selected', self)
        self.layout.addWidget(self.label)

        self.imageLabel = QLabel(self)
        self.layout.addWidget(self.imageLabel)

        self.uploadButton = QPushButton('Upload Image', self)
        self.uploadButton.clicked.connect(self.uploadImage)
        self.layout.addWidget(self.uploadButton)

        self.confirmButton = QPushButton('Confirm', self)
        self.confirmButton.clicked.connect(self.confirmImage)
        self.layout.addWidget(self.confirmButton)

        self.resultText = QTextEdit(self)
        self.resultText.setReadOnly(True)
        self.layout.addWidget(self.resultText)

        self.setLayout(self.layout)

    def uploadImage(self):
        options = QFileDialog.Options()
        self.imagePath, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if self.imagePath:
            self.label.setText(f'Selected: {self.imagePath}')
            self.displayImage()

    def displayImage(self):
        pixmap = QPixmap(self.imagePath)
        self.imageLabel.setPixmap(pixmap.scaled(300, 300, aspectRatioMode=True))

    def confirmImage(self):
        if self.imagePath:
            license_numbers = self.processImage(self.imagePath)
            if license_numbers:
                self.resultText.append("Detected License Plates:")
                for number in license_numbers:
                    self.resultText.append(f"Detected License Plate: {number}")
            else:
                self.resultText.append("No license plates detected.")
        else:
            self.resultText.setPlainText('No image selected!')

    def processImage(self, imagepath, conf=0.1):
        try:
            clear_crop_directory()  # Xóa tất cả các ảnh trong thư mục results_crop
            img = cv2.imread(imagepath)
            results = model(imagepath, conf=conf, save=True)

            if results:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # Lấy các bounding box

                detected_boxes = []
                for box in boxes:
                    if len(box) >= 4:
                        x_min, y_min, x_max, y_max = box[:4]  # Chỉ lấy 4 giá trị
                        detected_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))

                if detected_boxes:
                    cropped_images = crop_and_save(imagepath, detected_boxes)
                    license_numbers = process_license_plate(cropped_images)  # Đảm bảo hàm này trả về danh sách biển số
                    return license_numbers if license_numbers else []
                else:
                    return []
            else:
                return []
        except Exception as e:
            self.resultText.append(f"Error: {e}")
            return []

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageUploader()
    ex.show()
    sys.exit(app.exec_())
