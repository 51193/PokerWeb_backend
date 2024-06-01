from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


def create_folder_structure(base_folder):
    """
    创建输出文件夹及其子文件夹。
    """
    for folder in ['train', 'test', 'valid']:
        (base_folder / folder / 'images').mkdir(parents=True, exist_ok=True)
        (base_folder / folder / 'labels').mkdir(parents=True, exist_ok=True)


def move_files(image_files, label_files, dest_folder):
    """
    将图像和标签文件移动到相应的文件夹中。
    """
    for image_file, label_file in zip(image_files, label_files):
        try:
            shutil.copy(image_file, dest_folder / 'images' / image_file.name)
            shutil.copy(label_file, dest_folder / 'labels' / label_file.name)
        except Exception as e:
            print(f"Error copying files {image_file} and {label_file}: {e}")


def main():
    # 定义路径
    image_folder = Path("datasets/sgs/images")
    label_folder = Path("datasets/sgs/labels")
    output_folder = Path("datasets/sgs")

    # 创建文件夹结构
    create_folder_structure(output_folder)

    # 获取所有图像文件和对应的标签文件路径
    image_files = list(image_folder.glob('*.jpg'))
    label_files = [label_folder / (file.stem + ".txt") for file in image_files]

    # 划分训练集、测试集和验证集
    train_images, test_images, train_labels, test_labels = train_test_split(image_files, label_files, test_size=0.2,
                                                                            random_state=42)
    test_images, valid_images, test_labels, valid_labels = train_test_split(test_images, test_labels, test_size=0.5,
                                                                            random_state=42)

    # 移动文件
    move_files(train_images, train_labels, output_folder / 'train')
    move_files(test_images, test_labels, output_folder / 'test')
    move_files(valid_images, valid_labels, output_folder / 'valid')


if __name__ == "__main__":
    main()
