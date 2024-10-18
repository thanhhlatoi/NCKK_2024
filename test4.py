import os
import cv2
import pytesseract

# Dinh nghia cac ky tu tren bien so
char_list = '0123456789ABCDEFGHKLMNPRSTUVXYZ'

def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

def clear_crop_directory(directory='results_crop'):
    """Xóa tất cả các tệp trong thư mục."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Xóa tệp
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def crop_and_save(imagepath, boxes):
    image = cv2.imread(imagepath)

    if not os.path.exists('results_crop'):
        os.makedirs('results_crop')

    cropped_images = []
    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        crop_img = image[y_min:y_max, x_min:x_max]
        crop_path = f"results_crop/cropped_{i}.jpg"
        cv2.imwrite(crop_path, crop_img)
        print(f"Cropped image saved at: {crop_path}")
        cropped_images.append(crop_img)  # Lưu cropped images để xử lý sau này

    return cropped_images

def process_license_plate(cropped_images):
    for lp_img in cropped_images:
        # Chuyển đổi ảnh biển số về gray
        gray = cv2.cvtColor(lp_img, cv2.COLOR_BGR2GRAY)

        # Áp dụng threshold để phân tách số và nền
        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Nhận diện biển số
        text = pytesseract.image_to_string(binary, lang="eng", config="--psm 7")

        # Fine-tune biển số
        fine_tuned_text = fine_tune(text)
        print(f"Detected License Plate: {fine_tuned_text}")

        # Hiển thị ảnh biển số và kết quả
        cv2.imshow("Cropped License Plate", lp_img)
        cv2.imshow("Binary License Plate", binary)
        cv2.waitKey(0)  # Giữ cửa sổ mở cho đến khi nhấn phím
        cv2.destroyAllWindows()  # Đóng cửa sổ sau khi nhấn phím
