import sys  # 导入sys模块，用于访问与Python解释器相关的变量和函数
import time  # 导入time模块，用于处理时间
from PySide6 import QtWidgets  # 导入PySide6库中的QtWidgets模块，用于创建GUI
from QtFusion.widgets import QMainWindow  # 从QtFusion库中导入QMainWindow类，用于创建窗口
from QtFusion.utils import cv_imread, drawRectBox, get_cls_color  # 从QtFusion库中导入cv_imread和drawRectBox函数，用于读取图像和绘制矩形框
from QtFusion.path import abs_path
from YOLOv8Model import YOLOv8Detector  # 从YOLOv8Model模块中导入YOLOv8Detector类，用于加载YOLOv8模型并进行目标检测


class MainWindow(QMainWindow):
    def __init__(self, width, height):
        super().__init__()
        self.resize(width, height)
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(0, 0, width, height)

def frame_process(image, model, colors, window):
    pre_img = model.preprocess(image)
    pred = model.predict(pre_img)

    det = pred[0]
    if det is not None and len(det):
        det_info = model.postprocess(pred)
        for info in det_info:
            name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info['class_id']
            label = f'{name} {conf * 100:.0f}%'
            image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=colors[cls_id])

    window.dispImage(window.label, image)

def main():
    # 初始化模型
    model = YOLOv8Detector()
    model.load_model(abs_path("weights/sgs.pt", path_type="current"))
    model.change_name("SGS")

    colors = get_cls_color(model.names)

    # 读取图像
    img_path = abs_path("test_media/34.jpg")
    image = cv_imread(img_path)
    height, width, _ = image.shape

    # 创建应用程序和窗口
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(width, height)

    # 处理图像并显示
    t1 = time.time()
    frame_process(image, model, colors, window)
    t2 = time.time()
    use_time = t2 - t1

    print("推理时间: %.2f 秒" % use_time)

    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
