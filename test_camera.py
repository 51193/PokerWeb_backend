# -*- coding: utf-8 -*-
import sys  # 导入sys模块，用于访问与Python解释器相关的变量和函数
import time  # 导入time模块，用于获取当前时间
import cv2  # 导入OpenCV库，用于图像处理
from PySide6.QtWidgets import QTableWidgetItem, QTableWidget, QVBoxLayout, QWidget, QSizePolicy, QLabel, QHBoxLayout, \
    QPushButton
from QtFusion.widgets import QMainWindow  # 从QtFusion库导入FBaseWindow类，用于创建主窗口
from QtFusion.handlers import MediaHandler  # 从QtFusion库导入MediaHandler类，用于处理媒体流
from QtFusion.utils import drawRectBox, get_cls_color  # 从QtFusion库导入drawRectBox函数，用于在图像上绘制矩形框
from PySide6 import QtWidgets, QtCore  # 导入PySide6库的QtWidgets和QtCore模块，用于创建GUI
from QtFusion.path import abs_path
from QtFusion.config import QF_Config
from YOLOv8Model import YOLOv8Detector  # 从YOLOv8Model模块导入YOLOv8Detector类，用于物体检测
from datasets.PokerCards.label_name import Label_list
from PySide6.QtCore import QTimer
from collections import Counter
from collections import defaultdict

QF_Config.set_verbose(False)

# 创建一个空字典用于存储扑克牌
poker_deck = {}

# 定义花色和点数
suits = ['黑桃', '红桃', '方块', '梅花']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

# 遍历花色和点数，构建扑克牌字典
for suit in suits:
    for rank in ranks:
        # 构建扑克牌的名称
        card_name = f"{suit}{rank}"
        # 默认次数为1
        count = 0
        # 将扑克牌名称和次数添加到字典中
        poker_deck[card_name] = count


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1200, 500)  # 增大整体窗口
        self.setWindowTitle('Video and Table')

        # 创建主布局
        main_layout = QHBoxLayout()

        # 创建视频窗口和表格的布局
        video_layout = QVBoxLayout()
        table_layout = QVBoxLayout()

        # 创建视频窗口
        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("border: 1px solid black;")  # 添加边框以便观察
        video_layout.addWidget(self.video_label)

        # 创建表格
        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(1)  # 设置表格列数
        self.tableWidget.setHorizontalHeaderLabels(["识别结果"])  # 设置表格列标签
        self.tableWidget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)  # 设置表格的大小策略
        table_layout.addWidget(self.tableWidget)

        # 创建清除记录按钮
        self.clear_button = QPushButton('清除记录', self)
        self.clear_button.clicked.connect(self.clear_records)  # 连接按钮点击事件处理函数
        video_layout.addWidget(self.clear_button)

        # 将视频窗口和表格的布局添加到主布局中
        main_layout.addLayout(video_layout, 3)  # 视频窗口占三份
        main_layout.addLayout(table_layout, 1)  # 表格占一份

        # 创建一个widget，并设置布局
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 固定视频框的大小
        # self.video_label.setFixedSize(900, 500)
        self.video_label.setFixedSize(640, 480)

        # 创建一个标签用于显示牌型
        self.poker_type_label = QLabel('牌型：', self)
        main_layout.addWidget(self.poker_type_label)

        self.card_timer = QTimer(self)  # 创建计时器
        self.card_timer.setInterval(1500)  # 设置计时器间隔为2秒
        self.card_timer.timeout.connect(self.check_card_type)  # 连接计时器的timeout信号到处理函数
        # 统计计时器启动期间识别到的牌
        self.cards_during_timer = []

        self.pair_count = 2  # 对子需要的最小牌数
        self.sequence_length = 5  # 顺子需要的最小连续牌数

    # 更新表格内容的槽函数
    def update_table(self, strings):
        # 清空表格内容
        self.tableWidget.setRowCount(0)

        # 遍历字符串数组，并将每个字符串添加到表格中
        for i, string in enumerate(strings):
            self.tableWidget.insertRow(i)
            self.tableWidget.setItem(i, 0, QTableWidgetItem(string))

    def clear_records(self):
        # 清空检测到的结果记录
        Card_record.clear()
        self.poker_type_label = QLabel('牌型：', self)
        # 更新表格显示
        self.update_table(Card_record)
        # 清除记牌器中的记录
        window2.clear_cards()

    def card_added(self):
        # 当记录添加时启动计时器
        if not self.card_timer.isActive():
            self.card_timer.start()
        else:
            # 如果计时器已经在运行，则延长计时器时间
            self.card_timer.stop()
            self.card_timer.start()

    def check_card_type(self):
        if len(self.cards_during_timer) == 1:
            self.poker_type_label.setText("牌型：" + self.cards_during_timer[0])
            self.cards_during_timer = []
            # 停止计时器
            self.card_timer.stop()
            return 0

        # 提取每张牌的点数
        card_numbers = [card[2:] for card in self.cards_during_timer]
        # 统计每个点数出现的次数
        number_counts = Counter(card_numbers)

        if len(self.cards_during_timer) == 2:
            # 判断对子
            if any(count >= 2 for number, count in number_counts.items()):
                self.poker_type_label.setText(f"牌型：对{card_numbers[0]}")
            else:
                self.poker_type_label.setText(f"牌型：错误")
            self.cards_during_timer = []
            # 停止计时器
            self.card_timer.stop()
            return 0

        if len(self.cards_during_timer) == 4:
            # 判断炸弹
            if any(count == 4 for number, count in number_counts.items()):
                for number, count in number_counts.items():
                    if count == 4:
                        self.poker_type_label.setText(f"牌型：炸弹 {number}")
                        self.cards_during_timer = []
                        # 停止计时器
                        self.card_timer.stop()
                        return 0
            # 判断三带一
            if any(count >= 3 for number, count in number_counts.items()):
                for number, count in number_counts.items():
                    if count >= 3:
                        # 寻找额外的一张牌
                        extra_card = next(card for card, count in number_counts.items() if card != number)
                        self.poker_type_label.setText(f"牌型：三{number}带一{extra_card}")
                        self.cards_during_timer = []
                        # 停止计时器
                        self.card_timer.stop()
                        return 0

            else:
                self.poker_type_label.setText("牌型：错误")
                self.cards_during_timer = []
                # 停止计时器
                self.card_timer.stop()
                return 0

        if len(self.cards_during_timer) > 4:
            # 判断三带一对
            if any(count >= 3 and len(self.cards_during_timer) == 5 for number, count in number_counts.items()):
                if any(count == 1 for number, count in number_counts.items()):
                    self.poker_type_label.setText("牌型：错误")
                    self.cards_during_timer = []
                    # 停止计时器
                    self.card_timer.stop()
                    return 0
                for number, count in number_counts.items():
                    if count >= 3:
                        # 寻找额外的一张牌
                        extra_card = next(card for card, count in number_counts.items() if card != number)
                        self.poker_type_label.setText(f"牌型：三{number}带一对{extra_card}")
                        self.cards_during_timer = []
                        # 停止计时器
                        self.card_timer.stop()
                        return 0

            card_seq = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
            # 判断顺子
            if all(count == 1 for count in number_counts.values()):
                sequence_length = 0
                sequence_numbers = []  # 用于存储顺子中的点数
                has_sequence = False
                for number in card_seq:
                    if number in card_numbers:
                        sequence_length += 1
                        sequence_numbers.append(number)
                        if sequence_length == len(self.cards_during_timer):
                            has_sequence = True
                            break
                    else:
                        sequence_length = 0
                        sequence_numbers = []
                if has_sequence:
                    sequence_str = ' '.join(sequence_numbers)  # 将点数列表转换为字符串
                    self.poker_type_label.setText(f"牌型：顺子 {sequence_str}")
                    self.cards_during_timer = []
                    # 停止计时器
                    self.card_timer.stop()
                    return 0
                else:
                    self.poker_type_label.setText("牌型：错误")
                    self.cards_during_timer = []
                    # 停止计时器
                    self.card_timer.stop()
                    return 0
            # 判断连对
            if all(count == 2 for count in number_counts.values()):
                pair_sequence_length = 0
                has_pair_sequence = False
                pair_sequence_numbers = []  # 用于存储连对中的点数
                for number in card_seq:
                    if number in card_numbers:
                        pair_sequence_length += 1
                        pair_sequence_numbers.append(number)
                        pair_sequence_numbers.append(number)
                        if pair_sequence_length == len(self.cards_during_timer) / 2:
                            has_pair_sequence = True
                            break
                    else:
                        pair_sequence_length = 0
                        pair_sequence_numbers = []
                if has_pair_sequence:
                    pair_sequence_str = ''.join(pair_sequence_numbers)  # 将点数列表转换为字符串
                    self.poker_type_label.setText(f"牌型：连对 {pair_sequence_str}")
                    self.cards_during_timer = []
                    # 停止计时器
                    self.card_timer.stop()
                    return 0
                else:
                    self.poker_type_label.setText("牌型：错误")
                    self.cards_during_timer = []
                    # 停止计时器
                    self.card_timer.stop()
                    return 0
            # 判断飞机
            if sum(1 for count in number_counts.values() if count >= 3) >= 2:
                three_sequence_length = 0
                has_three_sequence = False
                three_numbers = []  # 用于存储连对中的点数
                for number in card_seq:
                    if number in number_counts and number_counts[number] == 3:
                        three_sequence_length += 1
                        three_numbers.append(number)
                        if three_sequence_length == 2:
                            has_three_sequence = True
                            break
                    else:
                        three_sequence_length = 0
                        three_numbers = []
                if has_three_sequence:
                    str = ''.join(three_numbers)  # 将点数列表转换为字符串
                    self.poker_type_label.setText(f"牌型：飞机 {str}")
                    self.cards_during_timer = []
                    # 停止计时器
                    self.card_timer.stop()
                    return 0
                else:
                    self.poker_type_label.setText("牌型：错误")
                    self.cards_during_timer = []
                    # 停止计时器
                    self.card_timer.stop()
                    return 0
            self.cards_during_timer = []
            # 停止计时器
            self.card_timer.stop()
            return 0

    def update_cards_during_timer(self, card):
        # 更新计时器启动期间识别到的牌列表
        self.cards_during_timer.append(card)

    def is_sequence(self, card):
        # 判断牌是否为顺子
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        rank_index = ranks.index(card[2:])  # 获取牌的点数在ranks列表中的索引
        return all(card[:2] + ranks[rank_index + i] in Card_record for i in range(5))  # 判断连续的五张牌是否都在记录中

    def update_card_type_label(self, card_type):
        # 更新界面显示牌的类型
        self.poker_type_label.setText(f'牌型：{card_type}')


class CardCounter(QMainWindow):
    def __init__(self, cards):
        super().__init__()
        self.resize(1400, 200)
        self.setWindowTitle("记牌器")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()
        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(4)  # 四种花色
        self.table_widget.setColumnCount(13)  # 13 个点数
        self.table_widget.setHorizontalHeaderLabels(["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"])
        self.table_widget.setVerticalHeaderLabels(["黑桃", "红桃", "方块", "梅花"])

        layout.addWidget(self.table_widget)
        self.central_widget.setLayout(layout)

        self.update_cards(cards)

    def update_table(self, strings):
        for i in range(4):
            for j in range(13):
                item = QTableWidgetItem("1")  # 初始全部为1
                self.table_widget.setItem(i, j, item)

        for card in strings:
            suit, rank = self.parse_card(card)
            if suit in ["黑桃", "红桃", "方块", "梅花"] and rank in range(1, 14):
                item = self.table_widget.item(["黑桃", "红桃", "方块", "梅花"].index(suit), rank - 1)
                print(suit)
                if item:
                    item.setText("0")

    def update_cards(self, cards):
        self.update_table(cards)

    def parse_card(self, card):
        suit = card[:2]
        rank_str = card[2:]
        rank_mapping = {"A": 1, "J": 11, "Q": 12, "K": 13}
        rank = rank_mapping.get(rank_str, -1)  # 如果没有匹配到则返回-1
        if rank == -1:
            try:
                rank = int(rank_str)
            except ValueError:
                pass  # 错误的点数表示为-1
        return suit, rank
        # 清除扑克牌记录的方法

    def clear_cards(self):
        self.update_table([])  # 清空表格中的内容


def frame_process(image):  # 定义帧处理函数，用于处理每一帧图像
    image = cv2.resize(image, (850, 500))  # 将图像的大小调整为850x500
    pre_img = model.preprocess(image)  # 对图像进行预处理

    t1 = time.time()  # 获取当前时间
    pred, superimposed_img = model.predict(pre_img)  # 使用模型进行预测
    t2 = time.time()  # 获取当前时间
    use_time = t2 - t1  # 计算预测所花费的时间

    # print("推理时间: %.2f" % use_time)  # 打印预测所花费的时间
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
            if conf * 100 > 84:
                if poker_deck[name] < 10:
                    poker_deck[name] += 1
                else:
                    name_already_exists = False
                    for string in Card_record:
                        if name in string:
                            name_already_exists = True
                            break
                    if not name_already_exists:
                        Card_record.append(name)
                        print(name)
                        window.update_table(Card_record)
                        window.card_added()  # 当新的牌被添加时启动计时器
                        window.update_cards_during_timer(name)  # 更新计时器启动期间识别到的牌列表
                        window2.update_table(Card_record)
            # else:
            #     poker_deck[name] = 0

    window.dispImage(window.video_label, image)  # 在窗口的label上显示图像


cls_name = Label_list  # 定义类名列表

model = YOLOv8Detector()  # 创建YOLOv8Detector对象
model.load_model(abs_path("weights/best-yolov8n.pt", path_type="current"))  # 加载预训练的YOLOv8模型
colors = get_cls_color(model.names)  # 获取类别颜色
Card_record = []
app = QtWidgets.QApplication(sys.argv)  # 创建QApplication对象
window = MainWindow()  # 创建MainWindow对象
window2 = CardCounter(Card_record)

videoHandler = MediaHandler(fps=10)  # 创建MediaHandler对象，设置帧率为30
videoHandler.frameReady.connect(frame_process)  # 当有新的帧准备好时，调用frame_process函数
videoHandler.setDevice(device=0)  # 设置设备为0，即默认的摄像头
videoHandler.startMedia()  # 开始处理媒体流

# 显示窗口
window.show()
window2.show()
# 进入 Qt 应用程序的主循环
sys.exit(app.exec())
