from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox
import Create_sets
import Train_data
import predictor
import time


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        self.directory = ''
        self.accept_flag = 0
        self.create_flag = 1
        self.submit_flag = 0
        self.batch = 200
        self.pixel = 28
        self.step = 100
        self.reset_flag = 0
        self.test_ratio = 0.4
        self.prob = 0.5
        self.rgb = 3
        self.kernel_size = 3
        self.file_name = 'test'
        self.seperate = 0
        self.show_img = False
        super(MyWindow, self).__init__()
        MyWindow.resize(self, 800, 600)
        MyWindow.setWindowTitle(self, "车型识别工具")
        self.browse = QtWidgets.QPushButton(self)
        self.browse.setGeometry(QtCore.QRect(600, 20, 75, 23))
        self.browse.setObjectName("browse")
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setGeometry(QtCore.QRect(120, 20, 461, 20))
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setEnabled(False)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(250, 50, 301, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.ok = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.ok.setObjectName("ok")
        self.horizontalLayout.addWidget(self.ok)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.cancel = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.cancel.setObjectName("cancel")
        self.horizontalLayout.addWidget(self.cancel)
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(110, 110, 561, 161))
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(20, 20, 54, 21))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(200, 20, 54, 21))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(380, 20, 54, 21))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_2.setGeometry(QtCore.QRect(80, 20, 113, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_3.setGeometry(QtCore.QRect(260, 20, 113, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_4.setGeometry(QtCore.QRect(440, 20, 113, 20))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_5.setGeometry(QtCore.QRect(80, 60, 231, 20))  # 5   file_name
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(20, 60, 54, 21))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_6.setGeometry(QtCore.QRect(440, 60, 113, 20))  # 6   test ratio
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(330, 60, 101, 21))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_7.setGeometry(QtCore.QRect(80, 100, 113, 20))  # 7   prob
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(20, 100, 54, 21))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.lineEdit_8 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_8.setGeometry(QtCore.QRect(260, 100, 41, 20))  # 8   rgb
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(200, 100, 54, 21))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.lineEdit_9 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_9.setGeometry(QtCore.QRect(440, 100, 113, 20))  # 9   kernel
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(313, 100, 121, 21))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_10.setGeometry(QtCore.QRect(440, 130, 41, 20))  # 9   kernel
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.label_10 = QtWidgets.QLabel(self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(90, 130, 301, 21))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(250, 305, 301, 31))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.submit = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.submit.setObjectName("submit")
        self.horizontalLayout_3.addWidget(self.submit)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.reset = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.reset.setObjectName("reset")
        self.horizontalLayout_3.addWidget(self.reset)
        self.create = QtWidgets.QPushButton(self)
        self.create.setGeometry(QtCore.QRect(100, 365, 75, 23))
        self.create.setObjectName("create")
        self.create.setEnabled(False)
        self.train = QtWidgets.QPushButton(self)
        self.train.setGeometry(QtCore.QRect(350, 365, 75, 23))
        self.train.setObjectName("train")
        self.train.setEnabled(False)
        self.run = QtWidgets.QPushButton(self)
        self.run.setGeometry(QtCore.QRect(600, 365, 75, 23))
        self.run.setObjectName("run")
        self.run.setEnabled(False)

        self.browse.setText("浏览")
        self.ok.setText("确认")
        self.cancel.setText("取消")
        self.groupBox.setTitle("变量设置")
        self.label.setText("batch")
        self.label_2.setText("pixel")
        self.label_3.setText("step")
        self.label_5.setText("文件名")
        self.label_6.setText("test_ratio")
        self.label_7.setText("prob")
        self.label_8.setText("rgb")
        self.label_9.setText("kernel_size")
        self.label_10.setText("是否输出重定义图片（True/False）")
        self.create.setText("创建")
        self.train.setText("训练")
        self.run.setText("预测")
        self.lineEdit_2.setText(str(self.batch))
        self.lineEdit_3.setText(str(self.pixel))
        self.lineEdit_4.setText(str(self.step))
        self.lineEdit_5.setText(self.file_name)
        self.lineEdit_6.setText(str(self.test_ratio))
        self.lineEdit_7.setText(str(self.prob))
        self.lineEdit_8.setText(str(self.rgb))
        self.lineEdit_9.setText(str(self.kernel_size))
        self.lineEdit_10.setText(str(self.show_img))
        self.submit.setText("提交")
        self.reset.setText("重置")

        self.browse.clicked.connect(self.msg)
        self.ok.clicked.connect(self.func_accept)
        self.cancel.clicked.connect(self.func_cancel)
        self.submit.clicked.connect(self.func_submit)
        self.reset.clicked.connect(self.func_reset)
        self.train.clicked.connect(self.func_train)
        self.run.clicked.connect(self.func_run)
        self.create.clicked.connect(self.func_create)

    def msg(self):
        self.directory = QFileDialog.getExistingDirectory(self,
                                                          "选取文件夹",
                                                          "D:/")  # 起始路径
        self.lineEdit.setText(self.directory)
        # return self.directory

    def func_accept(self):
        if self.directory:
            self.browse.setEnabled(False)
            self.accept_flag = 1
            if self.submit_flag ==1:
                self.run.setEnabled(True)
        else:
            QMessageBox.warning(self, "Warning", "地址未输入！")

    def func_cancel(self):
        if not self.accept_flag:
            self.lineEdit.setText('')

    def func_submit(self):
        self.submit_flag =1
        self.lineEdit_2.setEnabled(False)
        self.lineEdit_3.setEnabled(False)
        self.lineEdit_4.setEnabled(False)
        self.lineEdit_5.setEnabled(False)
        self.lineEdit_6.setEnabled(False)
        self.lineEdit_7.setEnabled(False)
        self.lineEdit_8.setEnabled(False)
        self.lineEdit_9.setEnabled(False)
        self.lineEdit_10.setEnabled(False)
        self.create.setEnabled(True)
        self.train.setEnabled(True)
        if self.accept_flag == 1:
            self.run.setEnabled(True)
        batch = self.lineEdit_2.text().strip()
        pixels = self.lineEdit_3.text().strip()
        step = self.lineEdit_4.text().strip()
        file_name = self.lineEdit_5.text().strip()
        test_ratio = self.lineEdit_6.text().strip()
        prob = self.lineEdit_7.text().strip()
        rgb = self.lineEdit_8.text().strip()
        kernel_size = self.lineEdit_9.text().strip()
        show_img = self.lineEdit_10.text().strip()
        # if (int(pixels) == 28):
        #     if (float(test_ratio) == 0.4):
        #         self.create_flag = 0
        #         self.seperate = 0
        #     else:
        #         self.seperate = 1

        self.batch = int(batch)
        self.pixels = int(pixels)
        print(self.pixels)
        self.step = int(step)
        self.test_ratio = float(test_ratio)
        self.prob = float(prob)
        self.rgb = int(rgb)
        self.kernel_size = int(kernel_size)
        self.file_name = file_name
        if show_img == 'True':
            self.show_img = True
        elif show_img == 'False':
            self.show_img = False
        else:
            QMessageBox.warning(self, "Warning", "输入不合法！默认为False")
            self.lineEdit_10.setText(str(False))
        # print(self.show_img)
        self.reset_flag = 1

    def func_reset(self):
        if not self.reset_flag:
            self.lineEdit_2.setText(str(self.batch))
            self.lineEdit_3.setText(str(self.pixel))
            self.lineEdit_4.setText(str(self.step))
            self.lineEdit_5.setText(self.file_name)
            self.lineEdit_6.setText(str(self.test_ratio))
            self.lineEdit_7.setText(str(self.prob))
            self.lineEdit_8.setText(str(self.rgb))
            self.lineEdit_9.setText(str(self.kernel_size))
            self.lineEdit_10.setText(str(self.show_img))

    def func_create(self):
        try:
            if self.show_img:
                Create_sets.Cs(self.pixels, self.create_flag, True, self.test_ratio, self.seperate, self.rgb)
            else:
                Create_sets.Cs(self.pixels, self.create_flag, False, self.test_ratio, self.seperate, self.rgb)
            self.create.setEnabled(False)
            QMessageBox.information(self, "Success", "创建成功！")
        except:
            QMessageBox.information(self, "Failed", "创建失败！")

    def func_train(self):
        try:
            Train_data.Td(self.batch, self.pixels, self.step, self.file_name, self.prob, self.rgb, self.kernel_size)
            self.train.setEnabled(False)
            self.create.setEnabled(False)
            QMessageBox.information(self, "Success", "训练成功！")
        except:
            QMessageBox.information(self, "Failed", "训练失败！")

    def func_run(self):
        try:
            predictor.pred(self.pixels, self.directory, self.file_name, self.rgb, self.kernel_size, self.batch, self.step,self.prob)
            self.run.setEnabled(False)
            self.train.setEnabled(False)
            self.train.setEnabled(False)
        except:
            QMessageBox.information(self, "Failed", "分类出错！")


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
