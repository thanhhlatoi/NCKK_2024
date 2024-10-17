from ultralytics import YOLO
import cv2
import os

# Khởi tạo mô hình YOLOv11
model = YOLO("C:/XuLyAnh/YoloV8/runs/detect/train/weights/best.pt")

def load_model(imagepath="e1.jpg", conf=0.1):
    img = cv2.imread(imagepath)
    results = model(imagepath, conf=conf,save=True)

    if results:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Lấy các bounding box

        detected_boxes = []
        for box in boxes:
            # Giảm giá trị unpack xuống còn 4
            if len(box) >= 4:
                x_min, y_min, x_max, y_max = box[:4]  # Chỉ lấy 4 giá trị

                detected_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))

        return imagepath, detected_boxes

    print("No results returned from model.")
    return imagepath, []

def crop_and_save(imagepath, boxes):
    image = cv2.imread(imagepath)

    if not os.path.exists('results_crop'):
        os.makedirs('results_crop')

    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        # Cắt hình ảnh
        crop_img = image[y_min:y_max, x_min:x_max]

        # Lưu ảnh đã cắt
        crop_path = f"results_crop/cropped_{i}.jpg"
        cv2.imwrite(crop_path, crop_img)
        print(f"Cropped image saved at: {crop_path}")

def main():
    try:
        path, boxes = load_model("C:/XuLyAnh/YoloV8/img_1.png")  # Thay đổi đường dẫn đến hình ảnh của bạn

        if boxes:
            crop_and_save(path, boxes)
        else:
            print("No detected plates to crop.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
