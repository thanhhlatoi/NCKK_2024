import cv2
import numpy as np


def is_colored_license_plate(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)

    # Chuyển đổi sang không gian màu HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Đặt ngưỡng cho màu vàng (biển số xe màu vàng)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Đặt ngưỡng cho màu xanh (biển số xe màu xanh)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Đặt ngưỡng cho màu đỏ (biển số xe màu đỏ)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Đặt ngưỡng cho màu trắng (biển số xe màu trắng)
    lower_white = np.array([0, 0, 200])  # Ngưỡng cho màu sáng
    upper_white = np.array([180, 25, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Tính toán tỷ lệ vùng màu
    yellow_area = cv2.countNonZero(yellow_mask)
    blue_area = cv2.countNonZero(blue_mask)
    red_area = cv2.countNonZero(red_mask)
    white_area = cv2.countNonZero(white_mask)

    total_area = image.shape[0] * image.shape[1]

    # Kiểm tra tỷ lệ vùng màu
    if yellow_area / total_area > 0.1:  # nếu diện tích vàng > 10%
        return "Biển số màu vàng"
    elif blue_area / total_area > 0.1:  # nếu diện tích xanh > 10%
        return "Biển số màu xanh"
    elif red_area / total_area > 0.1:  # nếu diện tích đỏ > 10%
        return "Biển số màu đỏ"
    elif white_area / total_area > 0.1:  # nếu diện tích trắng > 10%
        return "Biển số màu trắng"
    else:
        return "Biển số không có màu hoặc không xác định"


# Sử dụng hàm
image_path = 'C:/XuLyAnh/YoloV8/results_crop/cropped_0.jpg'
result = is_colored_license_plate(image_path)
print(result)
