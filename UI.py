from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from models.inference import *


class ImageViewer(QLabel):

    imageChanged = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
        self.setAcceptDrops(True)

    
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()
        
    
    def dropEvent(self, e):
        paths = [url.toLocalFile() for url in e.mimeData().urls()]
        paths = [paths[0]]
        self.setImage(paths[0])


    def initUI(self):
        self.setAlignment(Qt.AlignCenter)

        pal = QPalette(self.palette())
        pal.setColor(QPalette.Background, Qt.white)
        self.setAutoFillBackground(True)
        self.setPalette(pal)
        
        self.setFont(QFont("Arial", 12))
        self.setText("Drag image here")


    def setImage(self, img_path):
        img = QPixmap(img_path)
        img = img.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(img)
        self.imageChanged.emit(img_path)


class StartButton(QPushButton):

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
    
    def initUI(self):
        self.setFixedWidth(200)
        self.setText("Start")
        self.setFont(QFont("Arial", 12))
        self.setIcon(QIcon("./assets/play-64.ico"))


class BaseResultGroupBox(QGroupBox):
    
    def __init__(self, parent, title, keyLabels, valueLabels):
        super().__init__(parent)
        self.parent = parent
        self.title = title
        self.keyLabels = keyLabels
        self.valueLabels = valueLabels

    
    def initUI(self):
        self.setTitle(self.title)
        self.setFont(QFont("Arial", 12))

        self.setLayout(QVBoxLayout())
        
        for key in self.keyLabels.keys():
            itemLayout = QHBoxLayout()
            itemLayout.addWidget(self.keyLabels[key])
            itemLayout.addWidget(self.valueLabels[key])
            self.layout().addLayout(itemLayout)


class PathGroupBox(BaseResultGroupBox):
    
    def __init__(self, parent):
        self.keyLabels = {
            'path': QLabel("Path:")
        }
        self.valueLabels = {
            'path': QLabel()
        }

        super().__init__(parent, "Path", self.keyLabels, self.valueLabels)
        self.parent = parent

        self.initUI()


    def updatePath(self, path):
        self.valueLabels['path'].setText(path)


class PredictionGroupBox(BaseResultGroupBox):
    
    def __init__(self, parent):
        self.keyLabels = {
            'class': QLabel("Predicted tumor class:"),
            'type': QLabel("Predicted tumor type:")
        }
        self.valueLabels = {
            'class': QLabel(),
            'type': QLabel()
        }

        super().__init__(parent, "Prediction", self.keyLabels, self.valueLabels)
        self.parent = parent

        self.initUI()


    def updatePrediction(self, tumorClass, tumorType):
        self.valueLabels['class'].setText(tumorClass)
        self.valueLabels['type'].setText(tumorType)


class ProbabilityGroupBox(BaseResultGroupBox):

    def __init__(self, parent):
        self.keyLabels = {
            'prob_B': QLabel("Benign: "),
            'prob_M': QLabel("Malignant: "),
            'prob_A': QLabel("Adenosis: "),
            'prob_F': QLabel("Fibroadenoma: "),
            'prob_PT': QLabel("Phyllodes Tumor: "),
            'prob_TA': QLabel("Tubular Adenoma: "),
            'prob_DC': QLabel("Ductal Carcinoma: "),
            'prob_LC': QLabel("Lobular Carcinoma: "),
            'prob_MC': QLabel("Mucinous Carcinoma: "),
            'prob_PC': QLabel("Papillary Carcinoma: ")
        }
        self.valueLabels = {
            'prob_B': QLabel(),
            'prob_M': QLabel(),
            'prob_A': QLabel(),
            'prob_F': QLabel(),
            'prob_PT': QLabel(),
            'prob_TA': QLabel(),
            'prob_DC': QLabel(),
            'prob_LC': QLabel(),
            'prob_MC': QLabel(),
            'prob_PC': QLabel()
        }

        super().__init__(parent, "Probability", self.keyLabels, self.valueLabels)

        self.initUI()
    

    def updateProbability(self, classProb, typeProb):
        self.valueLabels['prob_B'].setText(str(round(classProb[0],4)))
        self.valueLabels['prob_M'].setText(str(round(classProb[1],4)))
        self.valueLabels['prob_A'].setText(str(round(typeProb[0],4)))
        self.valueLabels['prob_F'].setText(str(round(typeProb[1],4)))
        self.valueLabels['prob_PT'].setText(str(round(typeProb[2],4)))
        self.valueLabels['prob_TA'].setText(str(round(typeProb[3],4)))
        self.valueLabels['prob_DC'].setText(str(round(typeProb[4],4)))
        self.valueLabels['prob_LC'].setText(str(round(typeProb[5],4)))
        self.valueLabels['prob_MC'].setText(str(round(typeProb[6],4)))
        self.valueLabels['prob_PC'].setText(str(round(typeProb[7],4)))


    def reset(self):
        self.valueLabels['prob_B'].setText('')
        self.valueLabels['prob_M'].setText('')
        self.valueLabels['prob_A'].setText('')
        self.valueLabels['prob_F'].setText('')
        self.valueLabels['prob_PT'].setText('')
        self.valueLabels['prob_TA'].setText('')
        self.valueLabels['prob_DC'].setText('')
        self.valueLabels['prob_LC'].setText('')
        self.valueLabels['prob_MC'].setText('')
        self.valueLabels['prob_PC'].setText('')
