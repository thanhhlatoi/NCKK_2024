import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import warnings

# Tắt thông báo từ TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def plate_recognizie(image_path):
    np.set_printoptions(suppress=True)

    # Tải mô hình
    model = tf.keras.models.load_model('image classifier.h5', compile=False)

    # Kiểm tra cấu trúc của mô hình
    model.summary()

    # Tạo mảng cho dữ liệu
    data = np.ndarray(shape=(1, 28, 28, 1), dtype=np.float32)

    # Mở ảnh từ đường dẫn
    image = Image.open(image_path)
    image = image.convert("L")  # Chuyển sang grayscale
    image = ImageOps.fit(image, (28, 28), Image.ANTIALIAS)  # Đặt kích thước về 28x28
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    data[0] = normalized_image_array[..., np.newaxis]  # Thêm chiều cho kênh màu

    # Kiểm tra kích thước ảnh
    print("Image shape:", image_array.shape)
    print("Data shape:", data.shape)  # In ra kích thước dữ liệu đầu vào

    # Dự đoán
    prediction = model.predict(data)

    # Xác định loại đối tượng
    if prediction[0][0] >= 0.5:
        object = 'Domestic_Motor'
        probability = prediction[0][0]
    else:
        object = 'Others'
        probability = 1 - prediction[0][0]

    print(f"Prediction: {object}, Probability: {probability:.2f}")

def main():
    # Thay đổi đường dẫn đến ảnh bạn muốn nhập
    image_path = 'C:/XuLyAnh/YoloV8/results_crop/cropped_4.jpg'
    plate_recognizie(image_path)

if __name__ == "__main__":
    main()
