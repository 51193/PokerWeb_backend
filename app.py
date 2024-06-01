import base64
import os
import tempfile
import threading
import time
import queue

from queue import Empty

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

Game_List = ["SGS", "poker"]


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
    pred = model.predict(pre_img)  # 使用模型进行预测
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
        elif name == "SGS":
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


@app.route('/getGames', methods=['GET'])
def get_games():
    try:
        # 获取分页参数
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('pageSize', 8))
        offset = (page - 1) * page_size

        connection = get_mysql_connection()
        cursor = connection.cursor()
        # 查询游戏列表
        cursor.execute("SELECT name,url FROM games LIMIT %s OFFSET %s", (page_size, offset))
        games = cursor.fetchall()
        # 查询总游戏数量
        cursor.execute("SELECT COUNT(*) FROM games")
        total = cursor.fetchone()[0]
        # 关闭游标和数据库连接
        cursor.close()
        connection.close()
        # 假设你的图片存储在服务器的某个目录下，例如 '/path/to/images/'
        image_directory = 'image/'
        # 格式化游戏列表，并将图片文件读取为Base64编码
        games_list = []
        for game in games:
            # 构建图片的完整路径
            image_path = os.path.join(image_directory, f"{game[0]}.jpg")  # 假设图片文件扩展名为.jpg
            # 读取图片文件并转换为Base64编码
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            # 添加到游戏列表
            games_list.append({
                'name': game[0],
                'image': f"data:image/jpeg;base64,{encoded_image}",  # 假设MIME类型为image/jpeg
                'url': game[1]
            })

        return jsonify({'games': games_list, 'total': total}), 200

    except Exception as e:
        app.logger.error(f"获取游戏列表失败: {e}")
        return jsonify({'error': '获取游戏列表失败'}), 500


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
    pred = model.predict(pre_img)  # 使用模型进行预测
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


# 返回字符串
@socketio.on('image1')
def handle_image(message):
    if isinstance(message, bytes):  # 检查消息是否为二进制
        # 将二进制数据转换为图像
        arr = np.frombuffer(message, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        pre_img = model.preprocess(image)  # 对图像进行预处理
        pred = model.predict(pre_img)  # 使用模型进行预测
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


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in realtimeQueues:
        realtimeQueues[sid].put(None)
        processingThreads[sid].join()

        with locks[sid]:
            del realtimeQueues[sid]
            del processingThreads[sid]
            del conditions[sid]
            if sid in sample_rates:  # 清理该sid的sample_rates记录
                del sample_rates[sid]
    print('Client disconnected')


# 使用Queue来存储每个sid的消息队列
realtimeQueues = {}
# 每个sid的处理线程
processingThreads = {}
# 条件变量和锁
conditions = {}
locks = {}

# 初始化每个sid的处理时间跟踪和上次调整时间
processTimes = {}
lastAdjustmentTimes = {}
sample_rates = {}  # 线程局部存储的批处理数量
target_process_time = 0.5  # 目标处理时间为0.5秒


def adjust_sampling_rate(sid, process_time):
    min_samples, max_samples = 1, 10
    cooldown = 10

    current_time = time.time()

    if sid not in lastAdjustmentTimes or current_time - lastAdjustmentTimes[sid] > cooldown:
        if process_time > target_process_time and sample_rates[sid] > min_samples:  # 如果处理时间大于目标处理时间
            sample_rates[sid] -= 1
        elif process_time < target_process_time and sample_rates[sid] < max_samples:  # 如果处理时间小于目标处理时间
            sample_rates[sid] += 1

        lastAdjustmentTimes[sid] = current_time
        processTimes[sid] = process_time
        print(f'Adjusted number of samples to {sample_rates[sid]} for sid {sid}')


def process_images(sid):
    while True:
        start_time = time.time()
        selected_messages = []
        queue_size = 0

        # 只在需要访问共享资源时持有锁
        with conditions[sid]:
            conditions[sid].wait_for(lambda: not realtimeQueues[sid].empty())
            queue_size = realtimeQueues[sid].qsize()

            if queue_size == 0:
                continue

            num_samples = sample_rates.get(sid, 4)

            if num_samples != 1:
                if queue_size >= num_samples:
                    indexes = [int(queue_size * i / num_samples) for i in range(1, num_samples)]
                else:
                    indexes = list(range(queue_size))
            else:
                indexes = [queue_size - 1]

            for i in range(queue_size):
                message = realtimeQueues[sid].get_nowait()
                if message is None:
                    break
                if i in indexes:
                    selected_messages.append(message)

        # 图像处理不需要锁保护
        for message in selected_messages:
            if message is None:
                break
            arr = np.frombuffer(message, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is not None:
                image = frame_process(image)
                _, buffer = cv2.imencode('.jpg', image)
                socketio.emit('processed', buffer.tobytes(), room=sid)
            else:
                print("Failed to decode image")

        end_time = time.time()
        process_time = end_time - start_time
        adjust_sampling_rate(sid, process_time)


@socketio.on('image')
def handle_image(message):
    sid = request.sid
    if sid not in realtimeQueues:
        locks[sid] = threading.Lock()
        conditions[sid] = threading.Condition(lock=locks[sid])
        realtimeQueues[sid] = queue.Queue()
        sample_rates[sid] = 4  # 每个sid初始批处理数量
        processingThreads[sid] = threading.Thread(target=process_images, args=(sid,))
        processingThreads[sid].start()
        processTimes[sid] = 0  # 初始化处理时间
        lastAdjustmentTimes[sid] = time.time()  # 初始化调整时间

    with conditions[sid]:
        realtimeQueues[sid].put(message)
        conditions[sid].notify()


if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # app.run(debug=True)
