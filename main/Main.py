# -*- encoding: utf-8 -*-
import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from MyMainWindow import MyMainWindow
from MainWindow import Ui_MainWindow

'''
@File    :   MainWindow.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/18           WuJunQi      1.0     Internet+project
'''
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), "plugins"))
    window = MyMainWindow()
    window.settime(10, 7, 2)
    window.show()
    sys.exit(app.exec_())
