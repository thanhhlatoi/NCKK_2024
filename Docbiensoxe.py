# import cv2
# import pytesseract
# import numpy as np
# import re
#
# # Thiết lập đường dẫn của tesseract OCR
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#
# # Đường dẫn đến ảnh
# image_path = 'C:/XuLyAnh/YoloV8/results_crop/cropped_0.jpg'
# # image_path ="C:/XuLyAnh/YoloV8/img/aa1.jpg"
# # Đọc ảnh từ đường dẫn
# img = cv2.imread(image_path)
# #cv2.imshow("Anh ban dau",img)
# #img = cv2.resize(img, (640, 480))
# # Tiền xử lý ảnh
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #cv2.imshow("Anh Gray",gray)
# # Chuyển đổi sang ảnh nhị phân
# _, binary_img = cv2.threshold(gray, 100, 100, cv2.THRESH_BINARY)
# cv2.imshow("anh den trang",binary_img)
# # Sử dụng hàm Canny để phát hiện các cạnh trong ảnh
# # edges = cv2.Canny(binary_img, 50, 150)
# # cv2.imshow("edge",edges)
# # Tìm contours trong ảnh
# contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print("so luong contour tim dc", len(contours))
#
# for contour in contours:
#     # Lọc các contours có diện tích lớn hơn ngưỡng
#     area = cv2.contourArea(contour)
#     print(area)
#     if cv2.contourArea(contour) > 1000:
#         approx = cv2.approxPolyDP(contour,0.02*cv2.arcLength(contour,True), True)
#         if len(approx) == 4:
#             x, y, w, h = cv2.boundingRect(contour)
#
#             # Cắt vùng chứa biển số xe từ frame gốc
#             roi = img[y:y + h, x:x + w]
#             cv2.imshow("ROI",roi)
#
#             # Sử dụng Tesseract OCR để đọc ký tự từ vùng chứa biển số xe
#             text = pytesseract.image_to_string(roi, config='--psm 8')
#             # Lọc chỉ giữ lại các ký tự thuộc bảng chữ cái, số và dấu "-"
#             text = re.sub(r'[^a-zA-Z0-9-]', '', text)
#
#             # Hiển thị kết quả lên màn hình
#             print("Biển số xe: ", text)
#             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
# # Hiển thị ảnh
# cv2.imshow('Car Plate Recognition', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


