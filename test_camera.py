import sys
import cv2
from PySide6 import QtWidgets
from QtFusion.widgets import QMainWindow
from QtFusion.handlers import MediaHandler
from QtFusion.utils import drawRectBox, get_cls_color
from QtFusion.path import abs_path
from YOLOv8Model import YOLOv8Detector


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


def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        sys.exit()

    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        sys.exit()

    height, width, _ = frame.shape
    cap.release()
    return width, height


def main():
    model = YOLOv8Detector()
    model.load_model(abs_path("weights/sgs.pt", path_type="current"))
    model.change_name("SGS")
    colors = get_cls_color(model.names)

    width, height = initialize_camera()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(width, height)

    video_handler = MediaHandler(fps=30)
    video_handler.frameReady.connect(lambda img: frame_process(img, model, colors, window))
    video_handler.setDevice(device=0)
    video_handler.startMedia()

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
