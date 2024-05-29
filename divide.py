import os
import shutil
from sklearn.model_selection import train_test_split

# 定义图像文件夹和标签文件夹路径
image_folder = "E:\DACHUANG\sgs\images"
label_folder = "E:\DACHUANG\sgs\labels"

# 定义保存划分后数据集的文件夹路径
output_folder = "E:\DACHUANG\sgs"
train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")
valid_folder = os.path.join(output_folder, "valid")

# 创建输出文件夹及其子文件夹
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)

# 获取所有图像文件和对应的标签文件路径
image_files = os.listdir(image_folder)
label_files = [os.path.splitext(file)[0] + ".txt" for file in image_files]

# 划分训练集、测试集和验证集
train_images, test_images, train_labels, test_labels = train_test_split(image_files, label_files, test_size=0.2,
                                                                        random_state=42)
test_images, valid_images, test_labels, valid_labels = train_test_split(test_images, test_labels, test_size=0.5,
                                                                        random_state=42)


# 将图像和标签文件移动到相应的文件夹中
# 将图像和标签文件移动到相应的文件夹中
def move_files_to_folders(image_files, label_files, folder_name):
    # 创建子文件夹
    os.makedirs(os.path.join(folder_name, "images"), exist_ok=True)
    os.makedirs(os.path.join(folder_name, "labels"), exist_ok=True)

    for image_file, label_file in zip(image_files, label_files):
        # 移动图像文件
        image_src = os.path.join(image_folder, image_file)
        image_dest = os.path.join(folder_name, "images", image_file)
        shutil.copy(image_src, image_dest)

        # 移动标签文件
        label_src = os.path.join(label_folder, label_file)
        label_dest = os.path.join(folder_name, "labels", label_file)
        shutil.copy(label_src, label_dest)


# 将数据移动到训练集文件夹中
move_files_to_folders(train_images, train_labels, train_folder)

# 将数据移动到测试集文件夹中
move_files_to_folders(test_images, test_labels, test_folder)

# 将数据移动到验证集文件夹中
move_files_to_folders(valid_images, valid_labels, valid_folder)
