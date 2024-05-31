# -*- coding: utf-8 -*-
import cv2
import torch
from QtFusion.models import Detector  # 从QtFusion库中导入Detector抽象基类
from datasets.PokerCards.label_name import Chinese_name_poker  # 导入扑克牌的中文名称
from datasets.SGS.label_name import Chinese_name_sgs  # 导入SGS牌的中文名称
from ultralytics import YOLO  # 导入YOLO类，用于加载YOLO模型
from ultralytics.utils.torch_utils import select_device  # 用于选择设备（CPU或GPU）

# 选择设备，优先使用GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 初始参数配置
ini_params = {
    'device': device,
    'conf': 0.6,  # 物体置信度阈值
    'iou': 0.5,  # 用于非极大值抑制的IOU阈值
    'classes': None,  # 类别过滤器，这里设置为None表示不过滤任何类别
    'verbose': False
}

def count_classes(det_info, class_names):
    """
    统计检测信息中每个类别的数量。

    :param det_info: 检测信息列表，每个元素是一个字典，包含class_name, bbox, conf, class_id
    :param class_names: 所有可能类别的名称列表
    :return: 包含每个类别数量的列表，顺序与class_names相同
    """
    count_dict = {name: 0 for name in class_names}
    for info in det_info:
        class_name = info['class_name']
        if class_name in count_dict:
            count_dict[class_name] += 1
    return [count_dict[name] for name in class_names]

class YOLOv8Detector(Detector):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = None
        self.img = None  # 初始化图像为None
        self.names = list(Chinese_name_poker.values())  # 获取所有类别的中文名称
        self.params = params if params else ini_params  # 如果提供了参数则使用提供的参数，否则使用默认参数

    def load_model(self, model_path):
        """
        加载YOLO模型并进行初始化。

        :param model_path: 模型路径
        """
        self.device = select_device(self.params['device'])
        self.model = YOLO(model_path)
        self._update_names()
        self._warm_up_model()

    def _update_names(self):
        """
        更新类别名称为中文名称。
        """
        names_dict = self.model.names
        self.names = [Chinese_name_poker.get(v, v) for v in names_dict.values()]

    def _warm_up_model(self):
        """
        预热模型以加快后续推理速度。
        """
        dummy_input = torch.zeros(1, 3, *[self.imgsz] * 2).to(self.device)
        self.model(dummy_input.type_as(next(self.model.model.parameters())))

    def change_name(self, name):
        """
        根据输入的名称更改类别名称。

        :param name: 名称，可以是"SGS"或"poker"
        """
        if name == "SGS":
            self.names = list(Chinese_name_sgs.values())
            self.names = [Chinese_name_sgs.get(v, v) for v in self.model.names.values()]
        elif name == "poker":
            self.names = list(Chinese_name_poker.values())
            self.names = [Chinese_name_poker.get(v, v) for v in self.model.names.values()]

    def preprocess(self, img):
        """
        预处理图像。

        :param img: 输入图像
        :return: 预处理后的图像
        """
        self.img = img
        return img

    def predict(self, img):
        """
        使用模型进行预测。

        :param img: 输入图像
        :return: 预测结果
        """
        results = self.model(img, **self.params)
        return results

    def postprocess(self, pred):
        """
        对预测结果进行后处理。

        :param pred: 预测结果
        :return: 处理后的结果列表
        """
        results = []
        for res in pred[0].boxes:
            for box in res:
                class_id = int(box.cls.cpu())
                bbox = [int(coord) for coord in box.xyxy.cpu().squeeze().tolist()]
                result = {
                    "class_name": self.names[class_id],
                    "bbox": bbox,
                    "score": box.conf.cpu().squeeze().item(),
                    "class_id": class_id,
                }
                results.append(result)
        return results

    def set_param(self, params):
        """
        设置参数。

        :param params: 参数字典
        """
        self.params.update(params)
