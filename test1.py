import cv2
import pytesseract
import numpy as np
import os

# Đường dẫn đến Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Đường dẫn thư mục đầu ra
output_dir = 'C:/XuLyAnh/YoloV8/output'
os.makedirs(output_dir, exist_ok=True)

# Đọc hình ảnh biển số
license_plate_image = cv2.imread('C:/XuLyAnh/YoloV8/results_crop/cropped_0.jpg')

# 1. Tiền xử lý
img_gray = cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_thresh = cv2.adaptiveThreshold(img_blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

# 2. Tìm contours
contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 3. Phân đoạn ký tự
recognized_text = ""
characters = []
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    if w > 5 and h > 10:
        character = img_thresh[y:y + h, x:x + w]
        characters.append(character)

        # Resize ký tự và nhận diện
        char_resized = cv2.resize(character, (20, 40))
        text = pytesseract.image_to_string(char_resized, config="--oem 3 --psm 10").strip()

        recognized_text += text  # Thêm ký tự vào chuỗi kết quả
        print(f"Ký tự {i + 1}: {text}")

        # Lưu từng ký tự vào file
        cv2.imwrite(os.path.join(output_dir, f'ky_tu_{i + 1}.jpg'), char_resized)

# In ra kết quả nhận diện
print("Kết quả nhận diện biển số:", recognized_text)

# Lưu hình ảnh kết quả
cv2.imwrite(os.path.join(output_dir, 'ket_qua_nhan_dien.jpg'), license_plate_image)
cv2.imwrite(os.path.join(output_dir, 'anh_xam.jpg'), img_gray)
cv2.imwrite(os.path.join(output_dir, 'anh_nguong.jpg'), img_thresh)

# Hiển thị hình ảnh
cv2.imshow("Biển số gốc", license_plate_image)
cv2.imshow("Ảnh xám", img_gray)
cv2.imshow("Ảnh ngưỡng", img_thresh)

# Hiển thị từng ký tự
for i, char in enumerate(characters):
    cv2.imshow(f"Ký tự {i + 1}", char)

cv2.waitKey(0)
cv2.destroyAllWindows()
