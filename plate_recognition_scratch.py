import cv2
import imutils
import math
import numpy as np

# Đọc ảnh
image = cv2.imread('C:/XuLyAnh/YoloV8/results_crop/cropped_4.jpg')

# Thay đổi kích thước ảnh để hiển thị
image = imutils.resize(image, width=300)
cv2.imshow("Original Image", image)

# Chuyển đổi ảnh sang grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray_image)

# Làm mịn ảnh
gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
cv2.imshow("Smoothed Image", gray_image)

# Phát hiện cạnh
edged = cv2.Canny(gray_image, 30, 200)
cv2.imshow("Edged Image", edged)

# Tìm các contour
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Contours", image1)

# Lấy 5 contour lớn nhất
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

screenCnt = []

for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)

    if len(approx) == 4:
        screenCnt.append(approx)
        # Vẽ contour trên ảnh gốc
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)

        # Cắt biển số
        x, y, w, h = cv2.boundingRect(c)
        new_img = image[y:y + h, x:x + w]
        cv2.imshow("License Plate", new_img)
        break

if not screenCnt:
    print("No plate detected")
else:
    # Tính toán góc nghiêng của biển số
    (x1, y1) = screenCnt[0][0]
    (x2, y2) = screenCnt[1][0]
    (x3, y3) = screenCnt[2][0]
    (x4, y4) = screenCnt[3][0]

    # Tính toán góc
    doi = abs(y1 - y2)
    ke = abs(x1 - x2)
    angle = math.atan2(doi, ke) * (180.0 / math.pi)

    # Cắt và xoay biển số
    mask = np.zeros(gray_image.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt[0]], 0, 255, -1)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))

    roi = image[topx:bottomx, topy:bottomy]
    ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
    roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))

    roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
    cv2.imshow("Aligned License Plate", roi)

# Đợi người dùng nhấn phím
cv2.waitKey(0)
cv2.destroyAllWindows()
