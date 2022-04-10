# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'separation.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(871, 583)
        MainWindow.setStyleSheet("background-color: rgb(220, 220, 220);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.tab1)
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 90))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("background-color: rgb(157, 157, 157);\n"
"")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 10, -1, 0)
        self.horizontalLayout.setSpacing(20)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tab1_show = PlotWidget(self.tab1)
        self.tab1_show.setObjectName("tab1_show")
        self.horizontalLayout.addWidget(self.tab1_show)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout.setContentsMargins(-1, -1, 30, 30)
        self.verticalLayout.setSpacing(50)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tab1_openFile = QtWidgets.QPushButton(self.tab1)
        self.tab1_openFile.setMaximumSize(QtCore.QSize(100, 16777215))
        self.tab1_openFile.setObjectName("tab1_openFile")
        self.verticalLayout.addWidget(self.tab1_openFile)
        self.tab1_Song = QtWidgets.QPushButton(self.tab1)
        self.tab1_Song.setMaximumSize(QtCore.QSize(100, 16777215))
        self.tab1_Song.setObjectName("tab1_Song")
        self.verticalLayout.addWidget(self.tab1_Song)
        self.tab1_vocals = QtWidgets.QPushButton(self.tab1)
        self.tab1_vocals.setMaximumSize(QtCore.QSize(100, 16777215))
        self.tab1_vocals.setObjectName("tab1_vocals")
        self.verticalLayout.addWidget(self.tab1_vocals)
        self.tab1_music = QtWidgets.QPushButton(self.tab1)
        self.tab1_music.setMaximumSize(QtCore.QSize(100, 16777215))
        self.tab1_music.setObjectName("tab1_music")
        self.verticalLayout.addWidget(self.tab1_music)
        self.verticalLayout.setStretch(0, 10)
        self.verticalLayout.setStretch(2, 10)
        self.verticalLayout.setStretch(3, 10)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.tabWidget.addTab(self.tab1, "")
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setObjectName("tab2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tab2_signal1 = QtWidgets.QPushButton(self.tab2)
        self.tab2_signal1.setMaximumSize(QtCore.QSize(200, 16777215))
        self.tab2_signal1.setObjectName("tab2_signal1")
        self.gridLayout_2.addWidget(self.tab2_signal1, 4, 1, 1, 1)
        self.tab2_signal2 = QtWidgets.QPushButton(self.tab2)
        self.tab2_signal2.setMaximumSize(QtCore.QSize(200, 16777215))
        self.tab2_signal2.setObjectName("tab2_signal2")
        self.gridLayout_2.addWidget(self.tab2_signal2, 5, 1, 1, 1)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.tab2_openFile1 = QtWidgets.QPushButton(self.tab2)
        self.tab2_openFile1.setMaximumSize(QtCore.QSize(200, 16777215))
        self.tab2_openFile1.setObjectName("tab2_openFile1")
        self.verticalLayout_9.addWidget(self.tab2_openFile1)
        self.file1Play = QtWidgets.QPushButton(self.tab2)
        self.file1Play.setMaximumSize(QtCore.QSize(200, 16777215))
        self.file1Play.setObjectName("file1Play")
        self.verticalLayout_9.addWidget(self.file1Play)
        self.gridLayout_2.addLayout(self.verticalLayout_9, 1, 1, 1, 1)
        self.tab2_plotWidget3 = PlotWidget(self.tab2)
        self.tab2_plotWidget3.setObjectName("tab2_plotWidget3")
        self.gridLayout_2.addWidget(self.tab2_plotWidget3, 4, 0, 1, 1)
        self.tab2_plotWidget2 = PlotWidget(self.tab2)
        self.tab2_plotWidget2.setObjectName("tab2_plotWidget2")
        self.gridLayout_2.addWidget(self.tab2_plotWidget2, 3, 0, 1, 1)
        self.tab2_plotWidget1 = PlotWidget(self.tab2)
        self.tab2_plotWidget1.setObjectName("tab2_plotWidget1")
        self.gridLayout_2.addWidget(self.tab2_plotWidget1, 1, 0, 1, 1)
        self.tab2_plotWidget4 = PlotWidget(self.tab2)
        self.tab2_plotWidget4.setObjectName("tab2_plotWidget4")
        self.gridLayout_2.addWidget(self.tab2_plotWidget4, 5, 0, 1, 1)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.tab2_openFile2 = QtWidgets.QPushButton(self.tab2)
        self.tab2_openFile2.setMaximumSize(QtCore.QSize(200, 16777215))
        self.tab2_openFile2.setObjectName("tab2_openFile2")
        self.verticalLayout_10.addWidget(self.tab2_openFile2)
        self.file2Play = QtWidgets.QPushButton(self.tab2)
        self.file2Play.setMaximumSize(QtCore.QSize(200, 16777215))
        self.file2Play.setObjectName("file2Play")
        self.verticalLayout_10.addWidget(self.file2Play)
        self.gridLayout_2.addLayout(self.verticalLayout_10, 3, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab2, "")
        self.tab3 = QtWidgets.QWidget()
        self.tab3.setObjectName("tab3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.tab3)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget = PlotWidget(self.tab3)
        self.widget.setObjectName("widget")
        self.verticalLayout_3.addWidget(self.widget)
        self.widget_2 = PlotWidget(self.tab3)
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_3.addWidget(self.widget_2)
        self.widget_3 = PlotWidget(self.tab3)
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_3.addWidget(self.widget_3)
        self.widget_4 = PlotWidget(self.tab3)
        self.widget_4.setObjectName("widget_4")
        self.verticalLayout_3.addWidget(self.widget_4)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.tab3_openfile = QtWidgets.QPushButton(self.tab3)
        self.tab3_openfile.setMaximumSize(QtCore.QSize(120, 16777215))
        self.tab3_openfile.setObjectName("tab3_openfile")
        self.horizontalLayout_2.addWidget(self.tab3_openfile)
        self.verticalLayout_6.addLayout(self.horizontalLayout_2)
        self.tabWidget.addTab(self.tab3, "")
        self.verticalLayout_5.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "Songs into music and vocals."))
        self.tab1_openFile.setText(_translate("MainWindow", "Open File"))
        self.tab1_Song.setText(_translate("MainWindow", "Song"))
        self.tab1_vocals.setText(_translate("MainWindow", "Vocals"))
        self.tab1_music.setText(_translate("MainWindow", "Music"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1), _translate("MainWindow", "Vocal Separation"))
        self.tab2_signal1.setText(_translate("MainWindow", "Signal 1"))
        self.tab2_signal2.setText(_translate("MainWindow", "Signal 2"))
        self.tab2_openFile1.setText(_translate("MainWindow", "Open File 1"))
        self.file1Play.setText(_translate("MainWindow", "Play file1"))
        self.tab2_openFile2.setText(_translate("MainWindow", "Open File 2"))
        self.file2Play.setText(_translate("MainWindow", "Play file2"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab2), _translate("MainWindow", "Cocktail Party"))
        self.tab3_openfile.setText(_translate("MainWindow", "Choose Signal"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab3), _translate("MainWindow", "ECG detector"))
from pyqtgraph import PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
