import easyocr
import cv2
import re

# Khởi tạo EasyOCR
reader = easyocr.Reader(['vi'])

# Đọc ảnh
image_path = 'C:/XuLyAnh/YoloV8/results_crop/cropped_0.jpg'
image = cv2.imread(image_path)

# Tiền xử lý
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Nhận diện văn bản
results = reader.readtext(binary_image)

# Danh sách để lưu các ký tự
all_chars = []

# Hiển thị kết quả
for (bbox, text, prob) in results:
    # Vẽ bounding box
    cv2.rectangle(image, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 255, 0), 2)
    # Lưu văn bản vào danh sách
    all_chars.append(text)

# Kết hợp các ký tự nhận diện
combined_text = ' '.join(all_chars)

# Chuẩn hóa định dạng
def format_license_plate(text):
    # Sử dụng regex để tìm và định dạng lại
    pattern = r'(\d{2}[A-Z])\s?(\d{3})\.(\d{2})'
    match = re.search(pattern, text)
    if match:
        return f"{match.group(1)} {match.group(2)}.{match.group(3)}"
    return text

# Định dạng lại kết quả
formatted_result = format_license_plate(combined_text)

# In ra kết quả đã định dạng
print("Kết quả nhận diện:", formatted_result)

# Hiển thị ảnh kết quả
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
