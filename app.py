# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\app.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImageReader, QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QGraphicsPixmapItem, QGraphicsScene
import cv2

from clothes import ClothesWindow

from makeup_transfer.makeup_transfer import makeup_transfer
from hairstyle_transfer.align_face import cut_face
from hairstyle_transfer.change_hair import hairstyle_transfer

__appname__ = 'Versatile-Style-House'
defaultFilename = '.'


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 800)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("res/icons/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolTipDuration(-1)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 添加图片预览
        self.imgViewRaw = QtWidgets.QGraphicsView(self.centralwidget)
        self.imgViewRaw.setGeometry(QtCore.QRect(30, 40, 431, 581))
        self.imgViewRaw.setObjectName("imgViewRaw")
        self.imgViewFace = QtWidgets.QGraphicsView(self.centralwidget)
        self.imgViewFace.setGeometry(QtCore.QRect(650, 40, 211, 201))
        self.imgViewFace.setObjectName("imgViewFace")
        self.imgVIewOutput = QtWidgets.QGraphicsView(self.centralwidget)
        # self.imgVIewOutput.setGeometry(QtCore.QRect(930, 40, 541, 701))
        self.imgVIewOutput.setGeometry(QtCore.QRect(950, 150, 300, 400))
        self.imgVIewOutput.setObjectName("imgVIewOutput")
        self.imgViewHair = QtWidgets.QGraphicsView(self.centralwidget)
        self.imgViewHair.setGeometry(QtCore.QRect(650, 420, 211, 201))
        self.imgViewHair.setObjectName("imgViewHair")
        # self.imgViewClothes = QtWidgets.QGraphicsView(self.centralwidget)
        # self.imgViewClothes.setGeometry(QtCore.QRect(650, 540, 211, 201))
        # self.imgViewClothes.setObjectName("imgViewClothes")

        # 添加多选框
        self.checkBoxFace = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxFace.setGeometry(QtCore.QRect(510, 80, 121, 41))
        self.checkBoxFace.setObjectName("checkBoxFace")
        self.checkBoxHair = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxHair.setGeometry(QtCore.QRect(510, 460, 121, 41))
        self.checkBoxHair.setObjectName("checkBoxHair")
        # self.checkBoxClothes = QtWidgets.QCheckBox(self.centralwidget)
        # self.checkBoxClothes.setGeometry(QtCore.QRect(510, 580, 121, 41))
        # self.checkBoxClothes.setObjectName("checkBoxClothes")

        # 添加照片选择
        self.chooseRawPic = QtWidgets.QPushButton(self.centralwidget)
        self.chooseRawPic.setGeometry(QtCore.QRect(150, 660, 171, 41))
        self.chooseRawPic.setObjectName("chooseRawPic")
        self.chooseFacePic = QtWidgets.QToolButton(self.centralwidget)
        self.chooseFacePic.setGeometry(QtCore.QRect(510, 130, 91, 31))
        self.chooseFacePic.setObjectName("chooseFacePic")
        self.chooseHairPic = QtWidgets.QToolButton(self.centralwidget)
        self.chooseHairPic.setGeometry(QtCore.QRect(510, 510, 91, 31))
        self.chooseHairPic.setObjectName("chooseHairPic")
        # self.chooseClothesPic = QtWidgets.QToolButton(self.centralwidget)
        # self.chooseClothesPic.setGeometry(QtCore.QRect(510, 630, 91, 31))
        # self.chooseClothesPic.setObjectName("chooseClothesPic")

        # 提交按钮
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1050, 580, 121, 41))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)

        # 添加菜单
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1500, 26))
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
        MainWindow.setWindowTitle(_translate("MainWindow", "百变造型"))
        self.chooseRawPic.setText(_translate("MainWindow", "选择待处理图"))
        self.checkBoxFace.setText(_translate("MainWindow", "妆容样式"))
        self.checkBoxHair.setText(_translate("MainWindow", "发型样式"))
        # self.checkBoxClothes.setText(_translate("MainWindow", "服装样式"))
        self.chooseFacePic.setText(_translate("MainWindow", "选择图片"))
        self.chooseHairPic.setText(_translate("MainWindow", "选择图片"))
        # self.chooseClothesPic.setText(_translate("MainWindow", "选择图片"))
        self.pushButton.setText(_translate("MainWindow", "开始合成"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "模式"))
        self.menu_3.setTitle(_translate("MainWindow", "设置"))
        self.menu_4.setTitle(_translate("MainWindow", "关于"))
        self.actionOpen.setText(_translate("MainWindow", "打开"))
        self.actionSave.setText(_translate("MainWindow", "保存(Ctrl+S)"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionExit.setText(_translate("MainWindow", "退出Ctrl+Q)"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionFace.setText(_translate("MainWindow", "妆容"))
        self.actionHair.setText(_translate("MainWindow", "发型"))
        self.actionClothes.setText(_translate("MainWindow", "服装"))
        self.faceDetect.setText(_translate("MainWindow", "人脸识别"))
        self.aboutUs.setText(_translate("MainWindow", "关于我们"))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.filePath = None

        self.rawImgPath = None
        self.faceImgPath = None
        self.hairImgPath = None
        # self.clothesImgPath = None

        self.rawImg = None
        self.faceImg = None
        self.hairImg = None
        # self.clothesImg = None

        self.rawImg_backup = None

        self.outputImg = None
        self.outputImgPath = 'output.png'

        self.using_face = False
        self.using_hair = False
        # self.using_clothes = False

        self.ui.chooseRawPic.clicked.connect(self.openRawImg)
        self.ui.chooseFacePic.clicked.connect(self.openFaceImg)
        self.ui.chooseHairPic.clicked.connect(self.openHairImg)
        # self.ui.chooseClothesPic.clicked.connect(self.openClothesImg)
        self.ui.pushButton.clicked.connect(self.process)

        self.ui.checkBoxFace.stateChanged.connect(self.checkFace)
        self.ui.checkBoxHair.stateChanged.connect(self.checkHair)
        # self.ui.checkBoxClothes.stateChanged.connect(self.checkClothes)

        self.ui.actionOpen.triggered.connect(self.openRawImg)
        self.ui.actionSave.triggered.connect(self.saveImg)
        self.ui.actionClothes.triggered.connect(self.open_clothes_page)
        self.init()

    def init(self):
        self.checkFace()
        self.checkHair()
        # self.checkClothes()
        self.check_ok()

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

    def process(self):
        if self.rawImg is None:
            return

        if self.using_hair and self.hairImg is not None:
            # 调用发型API
            face_path, hair_path = cut_face(self.rawImgPath, self.hairImgPath)
            self.rawImg = hairstyle_transfer(face_path, hair_path)

        if self.using_face and self.faceImg is not None:
            # 调用妆容API
            self.rawImg = makeup_transfer(self.rawImg, self.faceImg)

        if self.rawImg is None:
            return
        self.outputImg = self.rawImg

        img = self.outputImg

        width, height = 280, 400

        resize_size = [width, height]
        img = self.resize_image(img, resize_size)
        width, height = img.shape[1], img.shape[0]

        img_rgb_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = QImage(img_rgb_data.data, width, height, width * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.ui.imgVIewOutput.setScene(scene)

    def check_ok(self):
        if self.rawImg is not None and ((self.using_face and self.faceImg is not None) or (
                self.using_hair and self.hairImg is not None)):
            self.ui.pushButton.setEnabled(1)
        else:
            self.ui.pushButton.setEnabled(0)

    def checkFace(self):
        if self.ui.checkBoxFace.isChecked():
            self.ui.chooseFacePic.setEnabled(1)
            self.using_face = True
        else:
            self.ui.chooseFacePic.setEnabled(0)
            self.using_face = False

    def checkHair(self):
        if self.ui.checkBoxHair.isChecked():
            self.ui.chooseHairPic.setEnabled(1)
            self.using_hair = True
        else:
            self.ui.chooseHairPic.setEnabled(0)
            self.using_hair = False

    # def checkClothes(self):
    #     if self.ui.checkBoxClothes.isChecked():
    #         self.ui.chooseClothesPic.setEnabled(1)
    #         self.using_clothes = True
    #     else:
    #         self.ui.chooseClothesPic.setEnabled(0)
    #         self.using_clothes = False

    def saveImg(self):
        path = os.path.dirname(self.filePath) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image (%s)" % ' '.join(formats)
        self.outputImgPath = QFileDialog.getSaveFileName(self, "保存图片位置", path, filters)
        print(self.outputImgPath, type(self.outputImgPath))
        if self.outputImg is not None:
            cv2.imwrite(self.outputImgPath[0], self.outputImg)

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

        width, height = 425, 575

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
        self.check_ok()

    # 打开文件预览选择图片
    def openFaceImg(self):
        path = os.path.dirname(self.filePath) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image (%s)" % ' '.join(formats)
        filename = QFileDialog.getOpenFileName(self, '%s - 打开图片' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
        self.faceImgPath = filename

        # 读取图片显示
        img = cv2.imread(self.faceImgPath)
        if img is None:
            return
        self.faceImg = img

        width, height = 205, 190

        resize_size = [width, height]
        img = self.resize_image(img, resize_size)
        width, height = img.shape[1], img.shape[0]

        img_rgb_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        frame = QImage(img_rgb_data.data, width, height, width * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.ui.imgViewFace.setScene(scene)
        self.check_ok()

    # 打开文件预览选择图片
    def openHairImg(self):
        path = os.path.dirname(self.filePath) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image (%s)" % ' '.join(formats)
        filename = QFileDialog.getOpenFileName(self, '%s - 打开图片' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
        self.hairImgPath = filename

        # 读取图片显示
        img = cv2.imread(self.hairImgPath)
        if img is None:
            return
        self.hairImg = img

        width, height = 205, 190

        resize_size = [width, height]
        img = self.resize_image(img, resize_size)
        width, height = img.shape[1], img.shape[0]

        img_rgb_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = QImage(img_rgb_data.data, width, height, width * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.ui.imgViewHair.setScene(scene)
        self.check_ok()

    # 打开文件预览选择图片
    # def openClothesImg(self):
    #     path = os.path.dirname(self.filePath) if self.filePath else '.'
    #     formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
    #     filters = "Image (%s)" % ' '.join(formats)
    #     filename = QFileDialog.getOpenFileName(self, '%s - 打开图片' % __appname__, path, filters)
    #     if filename:
    #         if isinstance(filename, (tuple, list)):
    #             filename = filename[0]
    #     self.clothesImgPath = filename
    #
    #     # 读取图片显示
    #     img = cv2.imread(self.clothesImgPath)
    #     if img is None:
    #         return
    #     self.clothesImg = img
    #     # width, height = img.shape[1], img.shape[0]
    #     width, height = 205, 190
    #
    #     resize_size = [width, height]
    #     img = self.resize_image(img, resize_size)
    #     width, height = img.shape[1], img.shape[0]
    #
    #     img_rgb_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     frame = QImage(img_rgb_data.data, width, height, width * 3, QImage.Format_RGB888)
    #     pix = QPixmap.fromImage(frame)
    #     item = QGraphicsPixmapItem(pix)
    #     scene = QGraphicsScene()  # 创建场景
    #     scene.addItem(item)
    #     self.ui.imgViewClothes.setScene(scene)
    #     self.check_ok()

    def open_clothes_page(self):
        self.clothes_ui = ClothesWindow()
        self.clothes_ui.show()
        # self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
