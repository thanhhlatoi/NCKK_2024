from PIL import Image
import os

source_folder = r"C:\Users\tient\Desktop\New folder\coco128\images\hihi\BIEN_SO_XE\\"
destination_folder = r"C:\Users\tient\Desktop\New folder\coco128\images\train2017\\"

# Đảm bảo thư mục đích tồn tại
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

directory = os.listdir(source_folder)

if __name__ == '__main__':
    print(directory)

for item in directory:
    if item.endswith(".jpg") or item.endswith(".png"):  # Chỉ xử lý file ảnh
        img = Image.open(os.path.join(source_folder, item))

        # Đặt new_width và new_height đều là 640
        new_width = 640
        new_height = 640

        # Thực hiện resize về kích thước 640x640
        imgResize = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        imgResize.save(os.path.join(destination_folder, item[:-4] + ".jpg"), quality=100)
