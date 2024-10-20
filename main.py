import os
import cv2
from ultralytics import YOLO
from test4 import clear_crop_directory, crop_and_save, process_license_plate

# Khởi tạo mô hình YOLOv11
model = YOLO("C:/XuLyAnh/YoloV8/runs/detect/train/weights/best.pt")

def load_model(imagepath="e1.jpg", conf=0.1):
    img = cv2.imread(imagepath)
    results = model(imagepath, conf=conf, save=True)

    if results:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Lấy các bounding box

        detected_boxes = []
        for box in boxes:
            if len(box) >= 4:
                x_min, y_min, x_max, y_max = box[:4]  # Chỉ lấy 4 giá trị
                detected_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))

        return imagepath, detected_boxes

    print("No results returned from model.")
    return imagepath, []

def main():
    try:
        clear_crop_directory()  # Xóa tất cả các ảnh trong thư mục results_crop
        path, boxes = load_model("C:/XuLyAnh/YoloV8/img/img_2.png")  # Thay đổi đường dẫn đến hình ảnh của bạn

        if boxes:
            cropped_images = crop_and_save(path, boxes)
            process_license_plate(cropped_images)  # Xử lý biển số sau khi cắt
        else:
            print("No detected plates to crop.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
