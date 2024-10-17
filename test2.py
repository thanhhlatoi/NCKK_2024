import cv2 as cv
import numpy as np

# Đọc hình ảnh
image_path = 'C:/XuLyAnh/YoloV8/results_crop/cropped_0.jpg'  # Thay thế bằng đường dẫn hình ảnh của bạn
image = cv.imread(image_path)

# Chuyển đổi sang ảnh xám
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Làm mượt hình ảnh
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# Nhị phân hóa với ngưỡng Otsu
_, binary_inv = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Tìm contours
contours, hierarchy = cv.findContours(binary_inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# Danh sách để lưu các contours hợp lệ
valid_contours = []

# Duyệt qua từng contour
for contour in contours:
    # Lấy kích thước bounding box
    x, y, w, h = cv.boundingRect(contour)

    # Kiểm tra các quy luật
    if w < h:  # width < height
        if h >= 10:  # thử chiều cao tối thiểu là 10 pixels
            aspect_ratio = w / h  # tỉ lệ aspect
            if 0.2 < aspect_ratio < 0.5:  # tỉ lệ aspect phải trong khoảng
                # Tính số pixel màu trắng trong contour
                mask = np.zeros(binary_inv.shape, dtype=np.uint8)
                cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)
                white_pixels = cv.countNonZero(cv.bitwise_and(mask, binary_inv))
                total_pixels = cv.contourArea(contour)

                if total_pixels > 0:  # tránh chia cho 0
                    white_ratio = (white_pixels / total_pixels) * 100  # tỉ lệ phần trăm
                    if white_ratio < 80:  # tỉ lệ pixel màu trắng nhỏ hơn 80%
                        valid_contours.append(contour)

# Vẽ tất cả contours để kiểm tra
output_image = cv.cvtColor(binary_inv, cv.COLOR_GRAY2BGR)  # Chuyển đổi sang hình ảnh màu để vẽ
cv.drawContours(output_image, contours, -1, (255, 0, 0), 1)  # Vẽ tất cả contours bằng màu đỏ
cv.drawContours(output_image, valid_contours, -1, (0, 255, 0), 2)  # Vẽ contours hợp lệ bằng màu xanh

# Hiển thị hình ảnh chứa tất cả contours và contours hợp lệ
cv.imshow('Contours', output_image)

# Chờ cho đến khi nhấn phím bất kỳ và sau đó đóng tất cả cửa sổ
cv.waitKey(0)
cv.destroyAllWindows()

# In số lượng contours hợp lệ
print(f'Total valid contours found: {len(valid_contours)}')
