import pathlib
import torch
import yaml
from ultralytics import YOLO  # 导入YOLO模型
from QtFusion.path import abs_path
device = "0" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':  # 确保该模块被直接运行时才执行以下代码

    data_name = "SGS"
    data_path = pathlib.Path(f'datasets/{data_name}/{data_name}.yaml').resolve()
    unix_style_path = data_path.as_posix()  # 获取 UNIX 风格路径

    # 获取目录路径
    directory_path = data_path.parent.as_posix()

    try:
        # 读取 YAML 文件，保持原有顺序
        with data_path.open('r') as file:
            data = yaml.safe_load(file)

        # 修改 path 项
        if 'path' in data:
            data['path'] = directory_path

            # 将修改后的数据写回 YAML 文件
            with data_path.open('w') as file:
                yaml.safe_dump(data, file, sort_keys=False)
    except FileNotFoundError:
        print(f"文件未找到：{data_path}")
    except yaml.YAMLError as e:
        print(f"YAML 错误：{e}")
    except Exception as e:
        print(f"发生错误：{e}")

    model = YOLO(abs_path('./weights/yolov8n.pt'), task='detect')  # 加载预训练的YOLOv8模型
    results = model.train(  # 开始训练模型
        data=data_path,  # 指定训练数据的配置文件路径
        device=device,  # 自动选择进行训练
        workers=2,  # 指定使用2个工作进程加载数据
        imgsz=640,  # 指定输入图像的大小为640x640
        epochs=10,  # 指定训练100个epoch
        batch=8,  # 指定每个批次的大小为8
        name='train_v8_' + data_name  # 指定训练任务的名称
    )

