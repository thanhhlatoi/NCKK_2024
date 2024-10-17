import easyocr
import cv2

# Khởi tạo EasyOCR
reader = easyocr.Reader(['vi'])

# Đọc ảnh
image_path = 'C:/XuLyAnh/YoloV8/results_crop/cropped_0.jpg'
image = cv2.imread(image_path)

# Nhận diện văn bản
results = reader.readtext(image)

# Danh sách để lưu các ký tự
all_chars = []

# Hiển thị kết quả
for (bbox, text, prob) in results:
    # Tính toán kích thước của từ
    width = int(bbox[2][0] - bbox[0][0])
    height = int(bbox[2][1] - bbox[0][1])

    # Lặp qua từng ký tự trong từ
    for i, char in enumerate(text):
        # Tính toán tọa độ cho từng ký tự
        char_width = width / len(text)
        char_bbox = [
            (bbox[0][0] + i * char_width, bbox[0][1]),
            (bbox[0][0] + (i + 1) * char_width, bbox[2][1])
        ]

        top_left = (int(char_bbox[0][0]), int(char_bbox[0][1]))
        bottom_right = (int(char_bbox[1][0]), int(char_bbox[1][1]))

        # Vẽ bounding box cho từng ký tự
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Hiển thị ký tự
        cv2.putText(image, char, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Lưu ký tự vào danh sách
        all_chars.append(char)

# In ra các ký tự đã nhận diện
print("Các ký tự nhận diện được:", ''.join(all_chars))

# Hiển thị ảnh kết quả
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
