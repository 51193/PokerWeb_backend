# 所需加载的模型目录
import cv2
from ultralytics import YOLO

path = 'weights/poker.pt'
# 需要检测的图片地址
img_path = "test_media/65cbee4c15cd05d27ad22832fe50d382.jpeg"

# 加载预训练模型
model = YOLO(path, task='detect')
# 检测图片
results = model(img_path, conf=0.7)
detected_names = []
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])  # 获取类别索引
        class_name = model.names.get(class_id, 'Unknown')  # 获取标签名称
        detected_names.append(class_name)
print("检测到的目标名称：", detected_names)

res = results[0].plot()
cv2.imshow("YOLOv8 Detection", res)
cv2.waitKey(0)
