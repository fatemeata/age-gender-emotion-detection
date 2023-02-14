import sys
import os
from emotions import AGEDetection
from random import randint
from PySide2.QtGui import QPainter, QPen, QBrush
from PySide2.QtCore import Qt
from PySide2 import QtGui, QtCore, QtWidgets
import cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


class CustomRect(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.yellow, Qt.SolidPattern))
        # painter.setBrush(QBrush(Qt.green, Qt.DiagCrossPattern))
        painter.drawRect(100, 15, 100, 15)


class QtCapture(QtWidgets.QWidget):
    def __init__(self, parent, *args):
        super().__init__(parent)

        self.rectangles = []

        self.fps = 24
        self.cap = cv2.VideoCapture(*args)

        lay = QtWidgets.QWidget()
        lay.layout = QtWidgets.QVBoxLayout(lay)
        # lay.setStretch(1, 2)
        lay.layout.setMargin(0)

        self.video_frame = QtWidgets.QLabel()
        self.video_frame.setGeometry(300, 50, 1200, 800)
        # self.video_frame.resize(1200, 900)
        print("video frame size: ", self.video_frame.size())

        self.emotion_label = QtWidgets.QLabel()
        self.space = QtWidgets.QLabel()
        # self.emotion_label.setText("")
        # self.emotion_label.setStyleSheet("font-weight: bold; font-size: 20px;")

        self.emotion_groupbox = QtWidgets.QGroupBox()
        self.emotion_container = QtWidgets.QWidget(lay)
        self.emotion_container.layout = QtWidgets.QGridLayout(self.emotion_container)
        self.emotion_container.setContentsMargins(30, 30, 30, 30)
        # self.emotion_container.setLayout(self.emotion_groupbox)
        self.emotion_groupbox.setLayout(self.emotion_container.layout)

        self.info_container = QtWidgets.QWidget(lay)
        self.info_container.layout = QtWidgets.QVBoxLayout(self.info_container)
        self.info_container.setContentsMargins(0, 0, 0, 0)

        self.second_info_container = QtWidgets.QWidget(lay)
        self.second_info_container.layout = QtWidgets.QVBoxLayout(self.second_info_container)
        self.second_info_container.setContentsMargins(0, 0, 0, 0)

        self.angry_label = QtWidgets.QLabel(self.info_container)
        self.angry_label.setText("Angry (Wut) ")
        self.angry_rect = CustomRect(self.info_container)
        self.happy_label = QtWidgets.QLabel(self.info_container)
        self.happy_label.setText("Happy (Freude)")
        self.sad_label = QtWidgets.QLabel(self.info_container)
        self.sad_label.setText("Sad (Trauer)")
        self.disgust_label = QtWidgets.QLabel(self.info_container)
        self.disgust_label.setText("Disgust (Ekel)")
        self.fear_label = QtWidgets.QLabel(self.second_info_container)
        self.fear_label.setText("Fear (Angst)")
        self.surprise_label = QtWidgets.QLabel(self.second_info_container)
        self.surprise_label.setText("Surprised (überrascht)")
        self.neutral_label = QtWidgets.QLabel(self.second_info_container)
        self.neutral_label.setText("Neutral (Neutral)")

        self.bar1 = QtWidgets.QProgressBar(self.second_info_container)
        self.bar1.setMaximum(100)
        self.bar1.setValue(0)

        self.bar2 = QtWidgets.QProgressBar(self.second_info_container)
        self.bar2.setMaximum(100)
        self.bar2.setValue(0)

        self.bar3 = QtWidgets.QProgressBar(self.second_info_container)
        self.bar3.setMaximum(100)
        self.bar3.setValue(0)

        self.bar4 = QtWidgets.QProgressBar(self.second_info_container)
        self.bar4.setMaximum(100)
        self.bar4.setValue(0)

        self.bar5 = QtWidgets.QProgressBar(self.second_info_container)
        self.bar5.setMaximum(100)
        self.bar5.resize(30, 10)
        self.bar5.setValue(0)

        self.bar6 = QtWidgets.QProgressBar(self.second_info_container)
        self.bar6.setMaximum(100)
        self.bar6.setValue(0)

        self.bar7 = QtWidgets.QProgressBar(self.second_info_container)
        self.bar7.setMaximum(100)
        self.bar7.setValue(0)

        self.emotion_container.layout.addWidget(self.angry_label, 0, 0)
        self.emotion_container.layout.addWidget(self.bar1, 0, 2)
        self.emotion_container.layout.addWidget(self.disgust_label, 0, 4)
        self.emotion_container.layout.addWidget(self.bar2, 0, 6)
        self.emotion_container.layout.addWidget(self.fear_label, 1, 0)
        self.emotion_container.layout.addWidget(self.bar3, 1, 2)
        self.emotion_container.layout.addWidget(self.happy_label, 1, 4)
        self.emotion_container.layout.addWidget(self.bar4, 1, 6)
        self.emotion_container.layout.addWidget(self.sad_label, 2, 0)
        self.emotion_container.layout.addWidget(self.bar5, 2, 2)
        self.emotion_container.layout.addWidget(self.surprise_label, 2, 4)
        self.emotion_container.layout.addWidget(self.bar6, 2, 6)
        self.emotion_container.layout.addWidget(self.neutral_label, 3, 0)
        self.emotion_container.layout.addWidget(self.bar7, 3, 2)

        lay.layout.addWidget(self.space, 0, QtCore.Qt.AlignCenter)
        lay.layout.addWidget(self.video_frame, 0, QtCore.Qt.AlignCenter)
        lay.layout.addWidget(self.emotion_groupbox, 0, QtCore.Qt.AlignHCenter)
        lay.layout.addWidget(self.emotion_label, 0, QtCore.Qt.AlignCenter)

        self.setLayout(lay.layout)
        self.detector = AGEDetection()

        self.emotion_labels_list = self.detector.get_emotion_labels_list()

        self.get_coordinates()
        # ------ Modification ------ #
        self.isCapturing = False
        self.ith_frame = 1
        # ------ Modification ------ #

    def get_coordinates(self):
        self.angry_x = self.angry_label.x()
        self.angry_y = self.angry_label.y()
        print("angry_x:", self.angry_x, "angry_y: ", self.angry_y)

    def setFPS(self, fps):
        self.fps = fps

    def nextFrameSlot(self):
        ret, frame = self.cap.read()

        # ------ Modification ------ #
        if self.isCapturing:
            self.detector.detection(frame)
            self.update(self.detector.get_emotion_values_list(), self.emotion_labels_list)
        # ------ Modification ------ #

        # My webcam yields frames in BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (800, 600))
        # print("frame shape: ", frame.shape[1], frame.shape[0],)
        img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000./self.fps)

    def stop(self):
        self.timer.stop()

    # ------ Modification ------ #
    def capture(self):
        if not self.isCapturing:
            self.isCapturing = True
        else:
            self.isCapturing = False
    # ------ Modification ------ #

    def deleteLater(self):
        self.cap.release()
        super(QtWidgets.QWidget, self).deleteLater()

    def update_plot_data(self):
        self.x = self.x[1:]  # Remove the first y element.
        self.x.append(self.x[-1] + 1)  # Add a new value 1 higher than the last.

        self.y = self.y[1:]  # Remove the first
        self.y.append(randint(0,100))  # Add a new random value.

        self.data_line.setData(self.x, self.y)  # Update the data.

    def update(self, values_list, label_list):
        self.bar1.setValue(values_list[0] * 100)
        self.bar2.setValue(values_list[1] * 100)
        self.bar3.setValue(values_list[2] * 100)
        self.bar4.setValue(values_list[3] * 100)
        self.bar5.setValue(values_list[4] * 100)
        self.bar6.setValue(values_list[5] * 100)
        self.bar7.setValue(values_list[6] * 100)


class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.capture = None

        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.clicked.connect(self.startCapture)
        self.quit_button = QtWidgets.QPushButton('End')
        self.quit_button.clicked.connect(self.endCapture)
        self.end_button = QtWidgets.QPushButton('Stop')

        # ------ Modification ------ #
        self.capture_button = QtWidgets.QPushButton('Capture')
        self.capture_button.clicked.connect(self.saveCapture)
        # ------ Modification ------ #

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.quit_button)

        # ------ Modification ------ #
        vbox.addWidget(self.capture_button)
        # ------ Modification ------ #

        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        self.setGeometry(100,100,200,200)
        self.show()

    def startCapture(self):
        if not self.capture:
            self.capture = QtCapture(self, 0)
            self.end_button.clicked.connect(self.capture.stop)
            # self.capture.setFPS(1)
            # self.capture.setParent(self)
            self.capture.setWindowFlags(QtCore.Qt.Tool)
        self.capture.start()
        self.capture.show()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    # ------ Modification ------ #
    def saveCapture(self):
        if self.capture:
            self.capture.capture()
    # ------ Modification ------ #


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())