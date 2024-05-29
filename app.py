import base64
import io
import tempfile
import time

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import random  # 导入random模块，用于生成随机数
from QtFusion.config import QF_Config
import cv2  # 导入OpenCV库，用于处理图像
from QtFusion.utils import drawRectBox  # 从QtFusion库中导入cv_imread和drawRectBox函数，用于读取图像和绘制矩形框
from QtFusion.path import abs_path
from flask_socketio import SocketIO

from YOLOv8Model import YOLOv8Detector  # 从YOLOv8Model模块中导入YOLOv8Detector类，用于加载YOLOv8模型并进行目标检测
from datasets.PokerCards.label_name import Label_list_poker
from datasets.SGS.label_name import Label_list_sgs
import numpy as np
from PIL import Image
import pymysql
import jwt
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False  # 禁止中文转义
QF_Config.set_verbose(False)
socketio = SocketIO(app, cors_allowed_origins="*")

colors = [[random.randint(100, 255) for _ in range(3)] for _ in range(len(Label_list_poker))]  # 为每个目标类别生成一个随机颜色

model = YOLOv8Detector()  # 创建YOLOv8Detector对象
model.load_model(abs_path("weights/best-yolov8n.pt", path_type="current"))  # 加载预训练的YOLOv8模型

# MySQL 连接配置
MYSQL_HOST = '127.0.0.1'
MYSQL_USER = 'root'  # 替换为你的 MySQL 用户名
MYSQL_PASSWORD = 'JASONyyx2002'  # 替换为你的 MySQL 密码
MYSQL_DB = 'DC'  # 替换为你的数据库名称

SECRET_KEY = "key"

Game_List = ["sgs", "poker"]


def get_mysql_connection():
    return pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DB)


# 检查数据库连接状态
try:
    connection1 = get_mysql_connection()
    print("数据库连接成功！")
except pymysql.Error as e:
    print("数据库连接失败:", e)


def frame_process(image):  # 定义帧处理函数，用于处理每一帧图像
    # image = cv2.resize(image, (850, 500))  # 将图像的大小调整为850x500
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
    else:
        return image
    return image


@app.route('/')
def hello_world():
    return 'Hello, World!'


def create_token(username):
    payload = {
        'exp': datetime.utcnow() + timedelta(days=1),  # 令牌有效期为1d
        'iat': datetime.utcnow(),
        'sub': username  # 通常 'sub' 表示 subject，即用户的唯一标识
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    return token


def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload['sub']
    except jwt.ExpiredSignatureError:
        return None  # Token has expired
    except jwt.InvalidTokenError:
        return None  # Token is invalid


# 封装的令牌验证函数
def get_username_from_token():
    token = request.headers.get('Authorization')
    if not token:
        return None, jsonify({'error': '未提供令牌'}), 401
    token = token.split(' ')[1]  # 移除 "Bearer " 前缀
    username = verify_token(token)
    if not username:
        return None, jsonify({'error': '令牌无效或已过期'}), 401
    return username, None, None


@app.route('/change-model', methods=['POST'])
def change_model():
    name = request.json.get('name')
    if name in Game_List:
        global colors
        if name == "poker":
            colors = [[random.randint(0, 155) for _ in range(3)] for _ in range(len(Label_list_poker))]
        elif name == "sgs":
            colors = [[random.randint(0, 100) for _ in range(3)] for _ in range(len(Label_list_sgs))]
        model.load_model(abs_path(f"weights/{name}.pt", path_type="current"))
        model.change_name(name)
        return jsonify({'message': '模型切换成功'})
    else:
        return jsonify({'error': '目标不存在'})


@app.route('/user/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        if username is None or password is None:
            return jsonify({'error': '用户名和密码不能为空'}), 400
        # 获取数据库连接
        connection = get_mysql_connection()
        cursor = connection.cursor()
        # 执行 SQL 查询
        cursor.execute("SELECT * FROM user WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()

        # 关闭游标和数据库连接
        cursor.close()
        connection.close()

        # 判断是否查询到用户
        if user:
            token = create_token(username)
            return jsonify({'message': '登录成功', 'token': token}), 200
        else:
            return jsonify({'error': '用户名或密码错误'}), 401
    except Exception as e:
        app.logger.error(f"数据库操作失败: {e}")
        return jsonify({'error': '数据库操作失败'}), 500


@app.route('/user/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        phone = data.get('phone')
        email = data.get('email')
        if username is None or password is None:
            return jsonify({'error': '用户名和密码不能为空'}), 400

        # 获取数据库连接
        connection = get_mysql_connection()
        cursor = connection.cursor()

        # 检查用户名是否已经存在
        cursor.execute("SELECT * FROM user WHERE username=%s", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            cursor.close()
            connection.close()
            return jsonify({'error': '用户名已存在'}), 409

        # 执行 SQL 插入语句
        cursor.execute("INSERT INTO user (username, password,phone,email) VALUES (%s, %s,%s,%s)",
                       (username, password, phone, email))
        connection.commit()

        # 关闭游标和数据库连接
        cursor.close()
        connection.close()

        return jsonify({'message': '注册成功'}), 201
    except Exception as e:
        app.logger.error(f"数据库操作失败: {e}")
        return jsonify({'error': '数据库操作失败'}), 500


@app.route('/user/info', methods=['GET'])
def get_user_info():
    try:
        username, error_response, status_code = get_username_from_token()
        if error_response:
            return error_response, status_code
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT phone,email,asset FROM user WHERE username = %s", (username,))
        result = cursor.fetchone()
        # 关闭游标和数据库连接
        cursor.close()
        connection.close()
        if result:
            phone, email, asset = result
            # asset = result[2]
            return jsonify({'username': username, 'phone': phone, 'email': email, 'asset': asset}), 200
        else:
            return jsonify({'error': '未找到用户信息'}), 404
    except Exception as e:
        app.logger.error(f"获取用户信息失败: {e}")
        return jsonify({'error': '获取用户信息失败'}), 500


# 修改密码接口
@app.route('/user/change-password', methods=['POST'])
def change_password():
    username, error_response, status_code = get_username_from_token()
    if error_response:
        return error_response, status_code
    data = request.get_json()
    old_password = data.get('oldPassword')
    new_password = data.get('newPassword')

    if not old_password or not new_password:
        return jsonify({'error': '旧密码和新密码不能为空'}), 400

    connection = get_mysql_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT * FROM user WHERE username=%s AND password=%s", (username, old_password))
        user = cursor.fetchone()

        if not user:
            return jsonify({'error': '旧密码错误'}), 400

        cursor.execute("UPDATE user SET password=%s WHERE username=%s", (new_password, username))
        connection.commit()
        return jsonify({'message': '修改成功'}), 200

    except Exception as e:
        app.logger.error(f"修改密码失败: {e}")
        return jsonify({'error': '数据库操作失败'}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/detect/video', methods=['POST'])
def detect_video():
    # 检查是否有文件在请求中
    if 'video' not in request.files:
        return 'No file part', 400
    file = request.files['video']
    # 如果用户没有选择文件，浏览器可能会提交一个没有文件名的空部分
    if file.filename == '':
        return 'No selected file', 400

    temp_dir = 'temp'
    # 创建临时文件来保存上传的视频和输出视频
    input_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=temp_dir)
    output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=temp_dir)
    # 保存上传的视频到临时文件
    file.save(input_temp_file.name)
    # 处理视频
    cap = cv2.VideoCapture(input_temp_file.name)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
    out = cv2.VideoWriter(output_temp_file.name, fourcc, fps, (width, height), isColor=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = frame_process(frame)
        out.write(processed_frame)
    cap.release()
    out.release()
    input_temp_file.close()
    output_temp_file.close()
    if output_temp_file.closed:
        print(1)
    return send_file(output_temp_file.name, mimetype='video/mp4', as_attachment=True)


@app.route('/detect/image', methods=['POST'])
def detect_cam():
    # 从请求体中获取图像的Base64编码
    data = request.get_json()
    if data is None or 'image' not in data:
        return jsonify({"error": "Invalid request, no image provided"}), 400

    image_data = data['image']
    # 解码Base64图像数据
    header, encoded = image_data.split(",", 1)
    image_decoded = base64.b64decode(encoded)
    image = cv2.imdecode(np.frombuffer(image_decoded, np.uint8), cv2.IMREAD_COLOR)
    image = frame_process(image)

    # 将处理后的图像转换回Base64以发送回客户端
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 返回处理后的图像数据
    return jsonify({"processedImage": f"data:image/jpeg;base64,{img_base64}"})


@app.route('/detectCam1', methods=['POST'])
def detect_cam1():
    # 从请求体中获取图像的Base64编码
    data = request.get_json()
    if data is None or 'image' not in data:
        return jsonify({"error": "Invalid request, no image provided"}), 400

    image_data = data['image']
    # 解码Base64图像数据
    header, encoded = image_data.split(",", 1)
    image_decoded = base64.b64decode(encoded)
    image = cv2.imdecode(np.frombuffer(image_decoded, np.uint8), cv2.IMREAD_COLOR)
    pre_img = model.preprocess(image)  # 对图像进行预处理
    pred, superimposed_img = model.predict(pre_img)  # 使用模型进行预测
    det = pred[0]  # 获取预测结果
    card = []
    # 如果有检测信息则进入
    if det is not None and len(det):
        det_info = model.postprocess(pred)  # 对预测结果进行后处理
        for info in det_info:  # 遍历检测信息
            name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info[
                'class_id']  # 获取类别名称、边界框、置信度和类别ID
            if conf > 0.8:
                card.append(name)
    # 返回处理后的图像数据
    return card


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


wait = dict()
receive = dict()
receive_threshold = 0.2
cooldown = dict()
cool = 1
count = 1
discard = 1
frame = 5
timer = time.time()


# 返回字符串
@socketio.on('image1')
def handle_image(message):
    global count
    global discard
    global frame
    global timer

    count += 1

    sid = request.sid
    if sid not in wait:
        wait[sid] = 0

    curSize = wait.get(sid)
    if curSize > frame:
        discard += 1
        return

    wait[sid] = wait[sid] + 1
    if isinstance(message, bytes):  # 检查消息是否为二进制
        # 将二进制数据转换为图像
        arr = np.frombuffer(message, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        pre_img = model.preprocess(image)  # 对图像进行预处理
        pred, superimposed_img = model.predict(pre_img)  # 使用模型进行预测
        det = pred[0]  # 获取预测结果
        card = []
        # 如果有检测信息则进入
        if det is not None and len(det):
            det_info = model.postprocess(pred)  # 对预测结果进行后处理
            for info in det_info:  # 遍历检测信息
                name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info[
                    'class_id']  # 获取类别名称、边界框、置信度和类别ID
                if conf > 0.8:
                    card.append(name)
        socketio.emit('processed', card)
    wait[sid] = wait[sid] - 1


# 返回实时图片
@socketio.on('image')
# def handle_image(message):
#     global cool
#     global count
#     global discard
#     global frame
#     global timer
#
#     sid = request.sid
#
#     start_time = time.time()
#     if sid not in receive:
#         receive[sid] = start_time
#     elif receive[sid] + receive_threshold > start_time:
#         receive[sid] = start_time
#         return
#     else:
#         receive[sid] = start_time
#
#     count += 1
#
#     if sid not in wait:
#         wait[sid] = 0
#
#     if sid not in cooldown:
#         cooldown[sid] = 0
#
#     cur_size = wait.get(sid)
#     cur_cooldown = cooldown.get(sid)
#     if cur_size > frame or cur_cooldown > start_time:
#         discard += 1
#         return
#
#     wait[sid] = wait[sid] + 1
#     if isinstance(message, bytes):  # 检查消息是否为二进制
#         # 将二进制数据转换为图像
#         arr = np.frombuffer(message, np.uint8)
#         image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#         image = frame_process(image)
#         _, buffer = cv2.imencode('.jpg', image)
#         socketio.emit('processed', buffer.tobytes())
#
#     end_time = time.time()
#     consume_time = end_time - start_time
#     if consume_time > 0.4 and end_time - timer > 5:
#         if frame > 1:
#             frame -= 1
#             timer = time.time()
#         else:
#             cooldown[sid] = start_time + cool
#             cool += 1
#         print(str(frame) + '-' + str(cool))
#     elif consume_time < 0.4 and end_time - timer > 5:
#         if frame < 2 <= cool:
#             cool -= 1
#         elif frame < 5:
#             frame += 1
#             timer = time.time()
#         print(str(frame) + '-' + str(cool))
#     wait[sid] = wait[sid] - 1
def handle_image(message):
    if isinstance(message, bytes):  # 检查消息是否为二进制
        # 将二进制数据转换为图像
        arr = np.frombuffer(message, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        image = frame_process(image)
        _, buffer = cv2.imencode('.jpg', image)
        socketio.emit('processed', buffer.tobytes())


if __name__ == '__main__':
    app.run(debug=True)
