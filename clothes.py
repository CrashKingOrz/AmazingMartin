# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'clothes.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImageReader, QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QWidget, QMainWindow, QApplication, QGraphicsPixmapItem, QGraphicsScene
import cv2

from clothes_transfer.dress_in_order import parse_clothes, clothes_transfer

__appname__ = 'Versatile-Style-House'
defaultFilename = '.'


class Ui_ClothesWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("ClothesWindow")
        MainWindow.resize(1317, 810)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("res/icons/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolTipDuration(-1)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imgViewRaw = QtWidgets.QGraphicsView(self.centralwidget)
        self.imgViewRaw.setGeometry(QtCore.QRect(70, 10, 350, 350))
        self.imgViewRaw.setObjectName("imgViewRaw")
        self.chooseRawPic = QtWidgets.QToolButton(self.centralwidget)
        self.chooseRawPic.setGeometry(QtCore.QRect(150, 370, 171, 41))
        self.chooseRawPic.setObjectName("chooseRawPic")
        self.chooseModelPic = QtWidgets.QToolButton(self.centralwidget)
        self.chooseModelPic.setGeometry(QtCore.QRect(570, 370, 171, 41))
        self.chooseModelPic.setObjectName("chooseModelPic")
        self.transferPic = QtWidgets.QToolButton(self.centralwidget)
        self.transferPic.setGeometry(QtCore.QRect(990, 370, 171, 41))
        self.transferPic.setObjectName("transferPic")
        self.imgViewModel = QtWidgets.QGraphicsView(self.centralwidget)
        self.imgViewModel.setGeometry(QtCore.QRect(480, 10, 350, 350))
        self.imgViewModel.setObjectName("imgViewModel")
        self.imgViewOutput = QtWidgets.QGraphicsView(self.centralwidget)
        self.imgViewOutput.setGeometry(QtCore.QRect(890, 10, 350, 350))
        self.imgViewOutput.setObjectName("imgViewOutput")
        self.imgViewClothes = QtWidgets.QGraphicsView(self.centralwidget)
        self.imgViewClothes.setGeometry(QtCore.QRect(310, 440, 704, 256))
        self.imgViewClothes.setObjectName("imgViewClothes")
        self.clothes1 = QtWidgets.QCheckBox(self.centralwidget)
        self.clothes1.setGeometry(QtCore.QRect(370, 710, 71, 41))
        self.clothes1.setObjectName("clothes1")
        self.clothes2 = QtWidgets.QCheckBox(self.centralwidget)
        self.clothes2.setGeometry(QtCore.QRect(610, 710, 71, 41))
        self.clothes2.setObjectName("clothes2")
        self.clothes3 = QtWidgets.QCheckBox(self.centralwidget)
        self.clothes3.setGeometry(QtCore.QRect(860, 710, 71, 41))
        self.clothes3.setObjectName("clothes3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(180, 550, 101, 31))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1317, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("res/icons/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen.setIcon(icon1)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("res/icons/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave.setIcon(icon2)
        self.actionSave.setObjectName("actionSave")
        self.actionExit = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("res/icons/exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionExit.setIcon(icon3)
        self.actionExit.setObjectName("actionExit")
        self.actionFace = QtWidgets.QAction(MainWindow)
        self.actionFace.setObjectName("actionFace")
        self.actionHair = QtWidgets.QAction(MainWindow)
        self.actionHair.setObjectName("actionHair")
        self.actionClothes = QtWidgets.QAction(MainWindow)
        self.actionClothes.setObjectName("actionClothes")
        self.faceDetect = QtWidgets.QAction(MainWindow)
        self.faceDetect.setObjectName("faceDetect")
        self.aboutUs = QtWidgets.QAction(MainWindow)
        self.aboutUs.setObjectName("aboutUs")
        self.menu.addAction(self.actionOpen)
        self.menu.addSeparator()
        self.menu.addAction(self.actionSave)
        self.menu.addSeparator()
        self.menu.addAction(self.actionExit)
        self.menu_2.addAction(self.actionFace)
        self.menu_2.addSeparator()
        self.menu_2.addAction(self.actionHair)
        self.menu_2.addSeparator()
        self.menu_2.addAction(self.actionClothes)
        self.menu_3.addAction(self.faceDetect)
        self.menu_4.addAction(self.aboutUs)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())

        self.retranslateUi(MainWindow)
        self.actionExit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "百变造型屋"))
        self.chooseRawPic.setText(_translate("MainWindow", "选择待处理图片"))
        self.chooseModelPic.setText(_translate("MainWindow", "选择模特图片"))
        self.transferPic.setText(_translate("MainWindow", "开始合成"))
        self.clothes1.setText(_translate("MainWindow", "服装1"))
        self.clothes2.setText(_translate("MainWindow", "服装2"))
        self.clothes3.setText(_translate("MainWindow", "服装3"))
        self.label.setText(_translate("MainWindow", "服装提取结果："))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "模式"))
        self.menu_3.setTitle(_translate("MainWindow", "设置"))
        self.menu_4.setTitle(_translate("MainWindow", "关于"))
        self.actionOpen.setText(_translate("MainWindow", "打开"))
        self.actionSave.setText(_translate("MainWindow", "保存(Ctrl+S)"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionExit.setText(_translate("MainWindow", "退出(Ctrl+Q)"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionFace.setText(_translate("MainWindow", "妆容"))
        self.actionHair.setText(_translate("MainWindow", "发型"))
        self.actionClothes.setText(_translate("MainWindow", "服装"))
        self.faceDetect.setText(_translate("MainWindow", "人脸识别"))
        self.aboutUs.setText(_translate("MainWindow", "关于我们"))


class ClothesWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ClothesWindow()
        self.ui.setupUi(self)
        self.filePath = None

        self.rawImgPath = None
        self.modelImgPath = None
        self.clothesImgPath = "./clothes.png"

        self.rawImg = None
        self.modelImg = None
        self.clothesImg = None

        self.rawImg_backup = None

        self.outputImg = None
        self.outputImgPath = 'output.png'

        self.clothes_type_list = []
        self.clothes_type_list_choose = []

        self.ui.chooseRawPic.clicked.connect(self.openRawImg)
        self.ui.chooseModelPic.clicked.connect(self.openClothesImg)

        self.ui.actionOpen.triggered.connect(self.openRawImg)
        self.ui.actionSave.triggered.connect(self.saveImg)

        self.ui.clothes1.stateChanged.connect(self.check_clothes1)
        self.ui.clothes2.stateChanged.connect(self.check_clothes2)
        self.ui.clothes3.stateChanged.connect(self.check_clothes3)

        self.ui.transferPic.clicked.connect(self.process)
        self.init()

    def init(self):
        self.ui.clothes1.setEnabled(0)
        self.ui.clothes2.setEnabled(0)
        self.ui.clothes3.setEnabled(0)

    def process(self):
        # Todo:调用合成的API
        if self.rawImgPath and self.modelImgPath and self.clothes_type_list_choose:
            self.result_image = clothes_transfer(self.rawImgPath, self.modelImgPath, self.clothes_type_list_choose)
            self.outputImg = self.result_image

        if self.outputImg is not None:
            width, height = 340, 340

            resize_size = [width, height]
            img = self.resize_image(self.outputImg, resize_size)
            width, height = img.shape[1], img.shape[0]

            img_rgb_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = QImage(img_rgb_data.data, width, height, width * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            item = QGraphicsPixmapItem(pix)
            scene = QGraphicsScene()  # 创建场景
            scene.addItem(item)
            self.ui.imgViewOutput.setScene(scene)

    def check_clothes1(self):
        if len(self.clothes_type_list) >= 1:
            if self.ui.clothes1.isChecked():
                self.clothes_type_list_choose.append(self.clothes_type_list[0])
            else:
                if self.clothes_type_list[0] in self.clothes_type_list_choose:
                    self.clothes_type_list_choose.remove(self.clothes_type_list[0])
        print("服装选择列表", self.clothes_type_list_choose)

    def check_clothes2(self):
        if len(self.clothes_type_list) >= 2:
            if self.ui.clothes2.isChecked():
                self.clothes_type_list_choose.append(self.clothes_type_list[1])
            else:
                if self.clothes_type_list[1] in self.clothes_type_list_choose:
                    self.clothes_type_list_choose.remove(self.clothes_type_list[1])
        print("服装选择列表", self.clothes_type_list_choose)

    def check_clothes3(self):
        if len(self.clothes_type_list) >= 3:
            if self.ui.clothes3.isChecked():
                self.clothes_type_list_choose.append(self.clothes_type_list[2])
            else:
                if self.clothes_type_list[2] in self.clothes_type_list_choose:
                    self.clothes_type_list_choose.remove(self.clothes_type_list[2])
        print("服装选择列表", self.clothes_type_list_choose)

    # 打开文件预览选择图片
    def openRawImg(self):
        path = os.path.dirname(self.filePath) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image (%s)" % ' '.join(formats)
        filename = QFileDialog.getOpenFileName(self, '%s - 打开图片' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
        self.rawImgPath = filename
        print(self.rawImgPath)
        # jpg = QtGui.QPixmap(filename).scaled(self.ui.imgViewRaw.width(), self.ui.imgViewRaw.height())
        # self.ui.imgViewRaw.setPixmap(jpg)

        # 读取图片显示
        img = cv2.imread(self.rawImgPath)
        if img is None:
            return
        self.rawImg = img
        self.rawImg_backup = img
        # width, height = img.shape[1], img.shape[0]
        width, height = 340, 340

        resize_size = [width, height]
        img = self.resize_image(img, resize_size)
        width, height = img.shape[1], img.shape[0]

        img_rgb_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        frame = QImage(img_rgb_data.data, width, height, width * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.ui.imgViewRaw.setScene(scene)

    # 打开文件预览选择图片
    def openClothesImg(self):
        path = os.path.dirname(self.filePath) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image (%s)" % ' '.join(formats)
        filename = QFileDialog.getOpenFileName(self, '%s - 打开图片' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
        self.modelImgPath = filename

        # 读取图片显示
        img = cv2.imread(self.modelImgPath)
        if img is None:
            return
        self.modelImg = img
        # width, height = img.shape[1], img.shape[0]
        width, height = 340, 340

        resize_size = [width, height]
        img = self.resize_image(img, resize_size)
        width, height = img.shape[1], img.shape[0]

        img_rgb_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = QImage(img_rgb_data.data, width, height, width * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.ui.imgViewModel.setScene(scene)

        self.get_clothes_type()

    def get_clothes_type(self):
        # Todo:调用提取服装的API
        self.clothesImg, self.clothes_type_list = parse_clothes(self.modelImgPath)
        # self.clothes_type_list = [5, 1, 3]
        # self.clothes_type_list = [5]
        # self.clothesImg = self.modelImg
        width, height = 700, 252

        resize_size = [width, height]
        img = self.resize_image(self.clothesImg, resize_size)
        width, height = img.shape[1], img.shape[0]

        img_rgb_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = QImage(img_rgb_data.data, width, height, width * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.ui.imgViewClothes.setScene(scene)

        # 将处理出的服装图片保存
        if self.clothesImg is not None:
            cv2.imwrite(self.clothesImgPath, self.clothesImg)

        # 处理多选框
        if len(self.clothes_type_list) >= 3:
            self.ui.clothes1.setEnabled(1)
            self.ui.clothes2.setEnabled(1)
            self.ui.clothes3.setEnabled(1)
        elif len(self.clothes_type_list) == 2:
            self.ui.clothes1.setEnabled(1)
            self.ui.clothes2.setEnabled(1)
            self.ui.clothes3.setEnabled(0)
        elif len(self.clothes_type_list) == 1:
            self.ui.clothes1.setEnabled(1)
            self.ui.clothes2.setEnabled(0)
            self.ui.clothes3.setEnabled(0)
        else:
            self.ui.clothes1.setEnabled(0)
            self.ui.clothes2.setEnabled(0)
            self.ui.clothes3.setEnabled(0)

    def saveImg(self):
        path = os.path.dirname(self.filePath) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image (%s)" % ' '.join(formats)
        self.outputImgPath = QFileDialog.getSaveFileName(self, "保存图片位置", path, filters)
        print(self.outputImgPath, type(self.outputImgPath))
        if self.outputImg is not None:
            cv2.imwrite(self.outputImgPath[0], self.outputImg)

    @staticmethod
    def resize_image(img, resize_size):
        w, h = img.shape[1], img.shape[0]

        if w <= resize_size[0] and h <= resize_size[1]:
            return img

        h1 = resize_size[1]
        w1 = int(h1 / h * w)

        if w1 > resize_size[0]:
            w1 = resize_size[0]
            h1 = int(w1 / w * h)

        resize_img = cv2.resize(img, (w1, h1))

        return resize_img


if __name__ == "__main__":
    import sys

    # QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    ui = ClothesWindow()
    ui.show()
    sys.exit(app.exec_())
