import subprocess
import random
import time

import cv2
from QtFusion.utils import cv_imread, drawRectBox
from QtFusion.path import abs_path
from YOLOv8v5Model import YOLOv8v5Detector
from datasets.PokerCards.label_name import Label_list
import os  # 导入os模块，用于操作文件路径

# 定义处理函数a
cls_name = Label_list
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(cls_name))]

model = YOLOv8v5Detector()
model.load_model(abs_path("weights/best-yolov8n.pt", path_type="current"))

def frame_process(image):
    image = cv2.resize(image, (850, 500))
    pre_img = model.preprocess(image)
    t1 = time.time()  # 获取当前时间
    pred, superimposed_img = model.predict(pre_img)  # 使用模型进行预测
    t2 = time.time()  # 获取当前时间
    use_time = t2 - t1  # 计算预测所花费的时间

    print("推理时间: %.2f" % use_time)  # 打印预测所花费的时间
    det = pred[0]
    if det is not None and len(det):
        det_info = model.postprocess(pred)
        for info in det_info:
            name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info['class_id']
            label = '%s %.0f%%' % (name, conf * 100)
            image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=colors[cls_id])
    return image

# 输入视频文件名
input_video = 'test_media/123.mp4'

# 创建桌面路径
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

# 打开输入视频文件
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: 无法打开视频文件")
    exit()

# 获取输入视频的基本信息
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 创建用于写入视频的VideoWriter对象
output_video = os.path.join(desktop_path, 'output_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# 逐帧处理视频
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 调用处理函数a处理每一帧
    processed_frame = frame_process(frame)

    # 将处理后的帧写入输出视频
    out.write(processed_frame)

    # 显示处理后的帧
    cv2.imshow('Processed Frame', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

# 将输出视频移动到桌面
final_output_video = os.path.join(desktop_path, 'final_output_video.mp4')
os.rename(output_video, final_output_video)

# 使用FFmpeg将处理后的帧重新组合成视频（可选）
# subprocess.call(['ffmpeg', '-i', final_output_video, '-c:v', 'copy', 'final_output_video.mp4'])
