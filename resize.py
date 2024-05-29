from PIL import Image
import os

# 定义要处理的文件夹路径
folder_path = "E:\DACHUANG\sgs\images"

# 获取文件夹下所有文件名
file_names = os.listdir(folder_path)

# 遍历文件夹下的每个文件
for file_name in file_names:
    # 拼接文件的完整路径
    file_path = os.path.join(folder_path, file_name)

    # 检查文件是否为图片
    if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
        # 打开图片
        image = Image.open(file_path)

        # 调整大小为416x416
        resized_image = image.resize((416, 416))

        # 保存调整大小后的图片
        resized_image.save(file_path)
        print(f"{file_name} resized successfully.")
    else:
        print(f"{file_name} is not an image file.")
