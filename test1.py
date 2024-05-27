import random  # 导入random模块，用于生成随机数
import cv2  # 导入OpenCV库，用于处理图像
from QtFusion.utils import cv_imread, drawRectBox  # 从QtFusion库中导入cv_imread和drawRectBox函数，用于读取图像和绘制矩形框
from QtFusion.path import abs_path
from YOLOv8v5Model import YOLOv8v5Detector  # 从YOLOv8Model模块中导入YOLOv8Detector类，用于加载YOLOv8模型并进行目标检测
from datasets.PokerCards.label_name import Label_list

cls_name = Label_list  # 定义类名列表
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(cls_name))]  # 为每个目标类别生成一个随机颜色

model = YOLOv8v5Detector()  # 创建YOLOv8Detector对象
model.load_model(abs_path("weights/best-yolov8n.pt", path_type="current"))  # 加载预训练的YOLOv8模型
def frame_process(image):  # 定义帧处理函数，用于处理每一帧图像
    image = cv2.resize(image, (850, 500))  # 将图像的大小调整为850x500
    pre_img = model.preprocess(image)  # 对图像进行预处理
    pred, superimposed_img = model.predict(pre_img)  # 使用模型进行预测
    det = pred[0]  # 获取预测结果
    # 如果有检测信息则进入
    if det is not None and len(det):
        det_info = model.postprocess(pred)  # 对预测结果进行后处理
        for info in det_info:  # 遍历检测信息
            name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info[
                'class_id']  # 获取类别名称、边界框、置信度和类别ID
            label = '%s %.0f%%' % (name, conf * 100)  # 创建标签，包含类别名称和置信度
            # 画出检测到的目标物
            image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=colors[cls_id])  # 在图像上绘制边界框和标签
    return image

# 视频文件路径
video_path = 'test_media/123.mp4'
# 输出视频文件路径
output_path = 'output_video.mp4'

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# 使用mp4v编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (850,500), isColor=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = frame_process(frame)
    out.write(processed_frame)

cap.release()
out.release()

print('视频处理完成，已保存至:', output_path)
