# -*- encoding: utf-8 -*-
'''
@File    :   MyMainWindow.py

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/18           WuJunQi      1.0     Internet+project
'''
import argparse
import asyncio
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer

from PyQt5.QtGui import QPixmap, QTextCursor, QPalette, QBrush
from PyQt5.QtWidgets import QMainWindow, QApplication, QFrame
from PyQt5.uic.uiparser import QtCore
from torch import square
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh,
                           xywh2xyxy, clip_coords)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from cffi import model
from numpy import random

from pyqt5_plugins.examplebutton import QtWidgets
from torch.backends import cudnn

from MainWindow import Ui_MainWindow
from State import StateEnum
from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, print_args, ROOT, check_img_size, check_requirements, \
    LOGGER, strip_optimizer
from utils.plots import save_one_box
from utils.torch_utils import select_device


class MyMainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 窗口图标
        self.setWindowIcon(QtGui.QIcon(r"../resourcedir/1.png"))
        # 背景初始化
        # palette = QPalette()
        # # 设置画刷
        # palette.setBrush(QPalette.Background, QBrush(QPixmap(r"E:\desktop\back1.png")))
        # # palette.setColor(QPalette.Background,Qt.red)
        # self.setPalette(palette)
        self.start = False
        # 设置视频初始边框
        i = QPixmap(r"../resourcedir/2.png")
        self.label.setPixmap(i)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        # 设置阴影 据说只有加了这步才能设置边框颜色。///可选样式有Raised、Sunken、Plain（这个无法设置颜色）等
        self.label.setFrameShadow(QtWidgets.QFrame.Raised)
        # 设置背景颜色，包括边框颜色
        # self.label.setStyleSheet()
        self.label.setFrameShape(QFrame.Box)
        # 设置边框样式
        # 设置背景填充颜色'background-color: rgb(0, 0, 0)'
        # 设置边框颜色border-color: rgb(255, 170, 0);
        self.label.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 170, 0);background-color: rgb(100, 149, 237);')




        imagegreen = QPixmap()
        self.flag = False
        # 初始化图片
        imagegreen.load(r"../resourcedir/red.png")
        self.pictureLabel.setPixmap(imagegreen)
        self.State = StateEnum.RED
        self.video_name = r"../video/text1.avi"
        # 定时监控 同时初始化槽函数
        self.timer_video = QTimer()
        self.timer_video.timeout.connect(self.show_video_frame)
        self.startflag = False
        # 初始化时间
        self.offset = 0
        self.greentime = 5
        self.redtime = 30
        self.yellowtime = 5
        self.showPlainText.setReadOnly(True)
        self.facttime = self.greentime
        # 用opencv来显示视频
        self.cap = cv2.VideoCapture()
        self.out = None
        # self.video_path = r"../video/text1.avi"
        # self.video_path = r"E:\OBS录屏\2022-05-17 17-52-50.mp4"
        self.video_path = r"../video/demo.mp4"
        # yolvo5初始化
        self.opt = self.parse_opt()
        print(self.opt)
        check_requirements(exclude=('tensorboard', 'thop'))

    def judgeifadd(self, minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2):
        minx = max(minx1, minx2)
        miny = max(miny1, miny2)
        maxx = min(maxx1, maxx2)
        maxy = min(maxy1, maxy2)
        if minx < maxx and miny < maxy:
            return True
        else:
            return False

    def show_video_frame(self):
        print(torch.cuda.is_available())
        self.opt.source = self.video_path
        self.opt.save_crop = True
        # self.opt.classes = 0
        self.opt.line_thickness = 2
        self.opt.dnn = True
        self.opt.weights = r"../weight/best.pt"
        # 斑马线
        # self.opt.weights = ROOT / 'yolov5s.pt'
        self.opt.conf_thres = 0.25
        self.opt.hide_conf = True
        self.opt.hide_labels = True
        self.opt.device = ''
        self.opt.nosave = True
        name_list = []
        # 输入的路径变为字符串
        source = str(self.opt.source)
        # 是否保存图片和txt文件
        save_img = not self.opt.nosave and not source.endswith('.txt')  # save inference images
        # 判断文件是否是视频流
        # Path()提取文件名 例如：Path("./data/test_images/bus.jpg") Path.name->bus.jpg Path.parent->./data/test_images Path.suffix->.jpg
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # 提取文件后缀名是否符合要求的文件，例如：是否格式是jpg, png, asf, avi等
        # .lower()转化成小写 .upper()转化成大写 .title()首字符转化成大写，其余为小写, .startswith('http://')返回True or Flase
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        # .isnumeric()是否是由数字组成，返回True or False
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            # 返回文件
            source = check_file(source)  # download
        # Directories
        # 预测路径是否存在，不存在新建，按照实验文件以此递增新建
        save_dir = increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok)  # increment run
        (save_dir / 'labels' if self.opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # 获取设备 CPU/CUDA
        self.opt.device = select_device(self.opt.device)
        # 检测编译框架PYTORCH/TENSORFLOW/TENSORRT
        model = DetectMultiBackend(self.opt.weights, device=self.opt.device, dnn=self.opt.dnn, data=self.opt.data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回
        self.opt.imgsz = check_img_size(self.opt.imgsz, s=stride)  # check image size

        # # Half
        # # 如果不是CPU，使用半进度(图片半精度/模型半精度)
        # half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        # if pt or jit:
        #     model.model.half() if half else model.model.float()
        # # TENSORRT加速
        # elif engine and model.trt_fp16_input != half:
        #     LOGGER.info('model ' + (
        #         'requires' if model.trt_fp16_input else 'incompatible with') + ' --half. Adjusting automatically.')
        #     half = model.trt_fp16_input

        ################################################# 2. 加载数据 #####################################################
        # Dataloader 加载数据
        # 使用视频流或者页面
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=self.opt.imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            # 直接从source文件下读取图片
            dataset = LoadImages(source, img_size=self.opt.imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        # 保存的路径
        vid_path, vid_writer = [None] * bs, [None] * bs

        ################################################# 3. 网络预测 #####################################################
        # Run inference
        # warmup 热身
        model.warmup(imgsz=(1 if pt else bs, 3, *self.opt.imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            # 转化到GPU上
            im = torch.from_numpy(im).to(self.opt.device)
            # 是否使用半精度
            im = im.half() if self.opt.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                # 增加一个维度
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            # 可视化文件路径
            self.opt.visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.opt.visualize else False
            """
            pred.shape=(1, num_boxes, 5+num_class)
            h,w为传入网络图片的长和宽,注意dataset在检测时使用了矩形推理,所以这里h不一定等于w
            num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
            pred[..., 0:4]为预测框坐标=预测框坐标为xywh(中心点+宽长)格式
            pred[..., 4]为objectness置信度
            pred[..., 5:-1]为分类结果
            """
            pred = model(im, augment=self.opt.augment, visualize=self.opt.visualize)
            t3 = time_sync()
            # 预测的时间
            dt[1] += t3 - t2

            # NMS
            # 非极大值抑制
            """
            pred: 网络的输出结果
            conf_thres:置信度阈值
            ou_thres:iou阈值
            classes: 是否只保留特定的类别
            agnostic_nms: 进行nms是否也去除不同类别之间的框
            max-det: 保留的最大检测框数量
            ---NMS, 预测框格式: xywh(中心点+长宽)-->xyxy(左上角右下角)
            pred是一个列表list[torch.tensor], 长度为batch_size
            每一个torch.tensor的shape为(num_boxes, 6), 内容为box + conf + cls
            """
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes,
                                       self.opt.agnostic_nms, max_det=self.opt.max_det)
            # 预测+NMS的时间
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            # 对每张图片做处理
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    # 如果输入源是webcam则batch_size>=1 取出dataset中的一张图片
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    # 但是大部分我们一般都是从LoadImages流读取本都文件中的照片或者视频 所以batch_size=1
                    # p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
                    # s: 输出信息 初始为 ''
                    # im0: 原始图片 letterbox + pad 之前的图片
                    # frame: 视频流
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                # 当前路径yolov5/data/images/
                p = Path(p)  # to Path
                # 图片/视频的保存路径save_path 如 runs\\detect\\exp8\\bus.jpg
                save_path = str(save_dir / p.name)  # im.jpg
                # 设置保存框坐标的txt文件路径，每张图片对应一个框坐标信息
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                # 设置打印图片的信息
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # 保存截图
                imc = im0.copy() if self.opt.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.opt.line_thickness, example=str(names))

                self.people = []
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    # 将预测信息映射到原图
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # 打印检测到的类别数量
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        if names[int(c)] == 'crossing':
                            temp = "检测到的人的数量：" + str(int(n)) + "\n"
                            self.peoplenum.setText(temp)
                            QApplication.processEvents()
                            continue
                        if names[int(c)] == 'person':
                            temp = "斑马线数量:" + str(int(n)) + "\n"
                            self.crossingsnum.setText(temp)
                            QApplication.processEvents()
                            continue
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    # 保存结果： txt/图片画框/crop-image
                    people = []
                    for *xyxy, conf, cls in reversed(det):
                        # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id + score + xywh
                        if self.opt.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                            print(xywh)
                            line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # # 在原图上画框 + 将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下 在原图像画图或者保存结果
                        if save_img or self.opt.save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.opt.hide_labels else (
                                names[c] if self.opt.hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if self.opt.save_crop:
                                # 在原图上画框 + 将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下
                                # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                                im = imc
                                square = False
                                BGR = False
                                gain = 1.02
                                pad = 10

                                xyxy = torch.tensor(xyxy).view(-1, 4)
                                b = xyxy2xywh(xyxy)  # boxes
                                if square:
                                    b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
                                b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
                                xyxy = xywh2xyxy(b).long()
                                clip_coords(xyxy, im.shape)
                                crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]),
                                       ::(1 if BGR else -1)]


                                # temp = "检测到的人的数量：" + str(int(n)) + "\n"
                                # self.peoplenum.setText(temp)
                                # temp = "斑马线数量:" + str(int(n)) + "\n"
                                # self.crossingsnum.setText(temp)
                                # print(names[int(c)], int(n))

                                if names[c] == 'crossing':
                                    # 左上角右下角
                                    print("person坐标", int(xyxy[0, 0]), int(xyxy[0, 1]), int(xyxy[0, 2]), int(xyxy[0, 3]))
                                    temp = [int(xyxy[0, 0]), int(xyxy[0, 1]), int(xyxy[0, 2]), int(xyxy[0, 3])]
                                    people.append(temp)
                                # 速度计算
                                if names[c] == 'person':
                                    print("crossing坐标", int(xyxy[0, 0]), int(xyxy[0, 1]), int(xyxy[0, 2]), int(xyxy[0, 3]))
                                    tem = [int(xyxy[0, 0]), int(xyxy[0, 1]), int(xyxy[0, 2]), int(xyxy[0, 3])]
                                    self.flag = False
                                    for te in people:
                                        if self.judgeifadd(tem[0], tem[1], tem[2], tem[3], te[0], te[1], te[2], te[3]):
                                            self.factpeoplenum.setText("有行人在斑马线上")
                                            self.flag = True
                                    if not self.flag:
                                        self.factpeoplenum.setText("无行人在斑马线上")




                # Stream results
                im0 = annotator.result()

                # 显示图片
                if self.opt.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                # Save results (image with detections)
                # 保存图片
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
                self.start = True
                # im0 = self.anthor()
                # 显示到屏幕上
                # self.out.write(im0)
                show = cv2.resize(im0, (640, 480))
                self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                         QtGui.QImage.Format_RGB888)
                self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Print results
        # 打印每张图片的速度
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.opt.imgsz)}' % t)
        # 保存图片或者txt
        if self.opt.save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.opt.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if self.opt.update:
            strip_optimizer(self.opt.weights)  # update model (to fix SourceChangeWarning)

    def button_video_open(self):
        self.video_path = self.video_name
        # self.video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
        #     self, "打开视频", "", "All Files(*);;*.mp4;;*.avi;;")
        #
        # if not self.video_name:
        #     return
        #
        # flag = self.cap.open(self.video_name)
        #
        # print(self.video_name)
        # if flag == False:
        #     QtWidgets.QMessageBox.warning(
        #         self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        # else:
        #     pass
        #     # self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
        #     #     *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
        #     # self.timer_video.start(30)

    def offsettime(self):
        print("按下")
        # lock.acquire()
        self.offset += 1
        # lock.release()

    def stopsimulation(self):
        print("暂停了")
        self.startflag = False

    def setState(self, State):
        self.State = State

    def settime(self, redtime, greentime, yellowtime):
        self.redtime = redtime
        self.greentime = greentime
        self.yellowtime = yellowtime

    def startsimulation(self):
        self.settime(int(self.redEdit.text()),int(self.greenEdit.text()),int(self.yellowEdit.text()))
        self.startflag = True
        # video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
        #     self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        # if not video_name:
        #     return
        # flag = self.cap.open(video_name)
        # self.video_path = video_name
        # if flag == False:
        #     QtWidgets.QMessageBox.warning(
        #         self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        # else:
        #     # self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(
        #     #     *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
        t1 = threading.Thread(target=self.show_video_frame)
        print("开始了")
        t1.start()
        while self.start == False:
            print("等待启动")
            time.sleep(1)
        t = threading.Thread(target=self.thread_body)
        print("开始了")
        t.start()

        if self.State == StateEnum.GREEN:
            temp = "绿灯还剩：" + str(self.greentime) + "s\n"
            # self.showPlainText.insertPlainText(temp)
            self.greentimelabel.setText(temp)
            self.facttime = self.greentime
        elif self.State == StateEnum.RED:
            temp = "红灯还剩：" + str(self.redtime) + "s\n"
            # self.showPlainText.insertPlainText(temp)
            self.redtimelabel.setText(temp)
            self.facttime = self.redtime
        elif self.State == StateEnum.YELLOW:
            temp = "黄灯还剩：" + str(self.yellowtime) + "s\n"
            # self.showPlainText.insertPlainText(temp)
            self.yellowtimelabel.setText(temp)
            self.facttime = self.yellowtime
        QApplication.processEvents()


    def addText(self, text):
        pass

    def moveCursor(self):
        self.showPlainText.moveCursor(QTextCursor.End)
        self.plainTextEdit.moveCursor(QTextCursor.End)

    def thread_body(self):
        print("开始模拟")
        while self.startflag:
            print("当前状态是", self.State)
            if self.State == StateEnum.GREEN:
                # 重新绘画图像
                # imagegreen = QPixmap("E:\\desktop\\green.png")
                # self.pictureLabel.setPixmap(imagegreen)
                # self.pictureLabel.repaint()
                time.sleep(1)
                self.facttime -= 1
                print("绿灯更新前剩余时间：", self.facttime)
                # 每秒之后更新偏差时间
                self.facttime += self.offset
                self.offset = 0
                print("绿灯更新后剩余时间：", self.facttime)
                temp = "绿灯还剩：" + str(self.facttime) + "s\n"
                self.showPlainText.insertPlainText(temp)
                self.greentimelabel.setText(temp)
                if self.facttime == 0:
                    print("进入黄灯阶段")
                    self.State = StateEnum.YELLOW
                    self.facttime = self.yellowtime
                    image = QPixmap(r"../resourcedir/yellow.png")
                    self.pictureLabel.setPixmap(image)
                    temp = "黄灯还剩：" + str(self.yellowtime) + "s\n"
                    self.yellowtimelabel.setText(temp)
                    self.showPlainText.insertPlainText(temp)
                    QApplication.processEvents()  # 刷新界面
                QApplication.processEvents()  # 刷新界面
                # 再次判断是否终止
                if not self.startflag:
                    break
                # 状态结束进入到下一个状态

            elif self.State == StateEnum.YELLOW:
                # 重新绘画图像
                # imagegreen = QPixmap("E:\\desktop\\yellow.png")
                # self.pictureLabel.setPixmap(imagegreen)
                # self.pictureLabel.repaint()
                time.sleep(1)
                self.facttime -= 1
                print("黄灯更新前剩余时间：", self.facttime)
                # 每秒之后更新偏差时间
                self.facttime += self.offset
                self.offset = 0
                print("黄灯更新后剩余时间：", self.facttime)
                temp = "黄灯还剩：" + str(self.facttime) + "s\n"
                self.showPlainText.insertPlainText(temp)
                self.yellowtimelabel.setText(temp)
                if self.facttime == 0:
                    print("进入红灯阶段")
                    self.State = StateEnum.RED
                    self.facttime = self.redtime
                    image = QPixmap(r"../resourcedir/red.png")
                    self.pictureLabel.setPixmap(image)
                    temp = "红灯还剩：" + str(self.redtime) + "s\n"
                    self.showPlainText.insertPlainText(temp)
                    self.redtimelabel.setText(temp)
                    QApplication.processEvents()  # 刷新界面
                QApplication.processEvents()  # 刷新界面
                # 再次判断是否终止
                if not self.startflag:
                    break
                # 状态结束进入到下一个状态

            elif self.State == StateEnum.RED:
                # 重新绘画图像
                # imagegreen = QPixmap("E:\\desktop\\red.png")
                # self.pictureLabel.setPixmap(imagegreen)
                # self.pictureLabel.repaint()
                time.sleep(1)
                self.facttime -= 1
                print("红灯更新前剩余时间：", self.facttime)
                # 每秒之后更新偏差时间
                self.facttime += self.offset
                self.offset = 0
                print("红灯更新后剩余时间：", self.facttime)

                temp = "红灯还剩：" + str(self.facttime) + "s\n"
                self.showPlainText.insertPlainText(temp)
                self.redtimelabel.setText(temp)
                if self.facttime == 0:
                    if self.flag:
                        self.plainTextEdit.insertPlainText("延长红灯时间，预测斑马线上会有人无法通过\n")
                        self.peoplenum.setText("有行人在斑马线上")
                        self.offsettime()
                        self.offsettime()
                        self.offsettime()
                        time.sleep(10)
                        continue
                    print("进入绿灯阶段")
                    self.State = StateEnum.GREEN
                    self.facttime = self.greentime
                    image = QPixmap(r"../resourcedir/green.png")
                    self.pictureLabel.setPixmap(image)
                    temp = "绿灯还剩：" + str(self.redtime) + "s\n"
                    self.showPlainText.insertPlainText(temp)
                    self.greentimelabel.setText(temp)
                    QApplication.processEvents()  # 刷新界面
                QApplication.processEvents()  # 刷新界面
                # 再次判断是否终止
                if not self.startflag:
                    break
                # 状态结束进入到下一个状态

    def __del__(self):
        print("程序停止了")
        self.startflag = False

    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weight/best.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default=self.video_path, help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', default=False, action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', default=False, action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', default=False, action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', default=False, action='store_true', help='use OpenCV DNN for ONNX inference')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        print(opt)
        return opt

    def anthor(self):
        # 内层嵌套的一层
        # weights = 'F:\\Internet+\\456\\weights\\best.pt'  # model.pt path(s)
        weights = '../weight/best.pt'
        source = 'F:\\Internet+\\stand-yolov5\\yolov5\\runs\\detect\\exp52\\123456.jpg'  # file/dir/URL/glob, 0 for webcam
        data = ROOT / 'data/coco128.yaml'  # dataset.yaml path
        imgsz = (640, 640)  # inference size (height, width)
        conf_thres = 0.1  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        device = 'cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = True  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # augmented inference
        visualize = False  # visualize features
        update = False  # update all models
        project = 'F:\\Internet+\\stand-yolov5\\yolov5\\main\\runs\\detect',  # save results to project/name
        name = 'exp'  # save results to project/name
        exist_ok = False  # existing project/name ok, do not increment
        line_thickness = 1  # bounding box thickness (pixels)
        hide_labels = True  # hide labels
        hide_conf = True  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = True  # use OpenCV DNN for ONNX inference
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # # Directories
        # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = model(im, augment=augment, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            im = imc
                            square = False
                            gain = 1.02
                            pad = 10
                            xyxy = torch.tensor(xyxy).view(-1, 4)
                            b = xyxy2xywh(xyxy)  # boxes
                            if square:
                                b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
                            b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
                            xyxy = xywh2xyxy(b).long()
                            clip_coords(xyxy, im.shape)

                # Stream results
                im0 = annotator.result()
                return im0
