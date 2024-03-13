from PyQt6.QtWidgets import QFileDialog, QWidget, QApplication, QPushButton, QVBoxLayout
import cv2
import sys
from qt_material import apply_stylesheet

class VideoSelect(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Video Select')
        self.setGeometry(100, 100, 300, 200)
        self.layout = QVBoxLayout() 
        self.button = QPushButton('Select Video')
        self.button.clicked.connect(self.select_video)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

    def select_video(self):
        self.fname = QFileDialog.getOpenFileName(self, 'Open file', 
             'c:\\',"Video files (*.mp4 *.mov)")[0]
        self.close()

    

# app = QApplication(sys.argv)
# window = VideoSelect()
# apply_stylesheet(app, theme='dark_purple.xml', invert_secondary=False)
# window.show()
# try:
#     sys.exit(app.exec())
# except:
#     pass