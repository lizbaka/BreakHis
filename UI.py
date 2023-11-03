from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from models.inference import *


class ImageViewer(QLabel):

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()


    def initUI(self):
        self.setAlignment(Qt.AlignCenter)

        pal = QPalette(self.palette())
        pal.setColor(QPalette.Background, Qt.white)
        self.setAutoFillBackground(True)
        self.setPalette(pal)
        
        self.setFont(QFont("Arial", 12))
        self.setText("Image Viewer")


    def setImage(self, img_path):
        img = QPixmap(img_path)
        img = img.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(img)


class IconTextButton(QPushButton):

    def __init__(self, parent, iconPath, text):
        super().__init__(parent)
        self.parent = parent
        self.initUI(iconPath, text)
    
    def initUI(self, iconPath, text):
        self.setFixedWidth(150)
        self.setText(text)
        self.setFont(QFont("Arial", 12))
        self.setIcon(QIcon(iconPath))


class BaseResultGroupBox(QGroupBox):
    
    def __init__(self, parent, title, keyLabels, valueLabels):
        super().__init__(parent)
        self.parent = parent
        self.title = title
        self.keyLabels = keyLabels
        self.valueLabels = valueLabels

    
    def initUI(self):
        self.setTitle(self.title)
        self.setFont(QFont("Arial", 10))
        self.setFixedWidth(400)

        self.setLayout(QVBoxLayout())
        
        for key in self.keyLabels.keys():
            itemLayout = QHBoxLayout()
            itemLayout.addWidget(self.keyLabels[key])
            itemLayout.addWidget(self.valueLabels[key])
            self.layout().addLayout(itemLayout)


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


    def updatePredictionIndex(self, tumorClass, tumorType):
        self.valueLabels['class'].setText(BackendModel.get_label('binary', tumorClass))
        self.valueLabels['type'].setText(BackendModel.get_label('subtype', tumorType))


    def updatePrediction(self, tumorClass, tumorType):
        self.valueLabels['class'].setText(tumorClass)
        self.valueLabels['type'].setText(tumorType)

    
    def reset(self):
        self.updatePrediction('', '')


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


class ImageTableWidget(QTableWidget):
        
    imported = pyqtSignal(list)
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
        self.setAcceptDrops(True)

        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(['Image', 'Path', 'Class', 'Type'])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.setColumnWidth(0, 120)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.setColumnWidth(2, 50)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.setColumnWidth(3, 50)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)


    def initUI(self):
        self.setFixedWidth(400)
        self.verticalHeader().setVisible(False)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()


    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
            paths = []
            for url in event.mimeData().urls():
                paths.append(url.toLocalFile())
            self.imported.emit(paths)
        else:
            event.ignore()
    

    def addImage(self, imgPath):
        row = self.rowCount()
        self.insertRow(row)
        imgLabel = QLabel()
        imgLabel.setPixmap(QPixmap(imgPath).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        imgLabel.setAlignment(Qt.AlignCenter)
        self.setCellWidget(row, 0, imgLabel)
        self.setItem(row, 1, QTableWidgetItem(imgPath))

    
    def addImages(self, imgPaths):
        for imgPath in imgPaths:
            self.addImage(imgPath)


    def getSelectedImagePath(self):
        if self.selectedItems():
            row = self.selectedItems()[0].row()
            return self.item(row, 1).text()
        else:
            return None
        
    
    def updateResult(self, results):
        for i in range(self.rowCount()):
            imgPath = self.item(i, 1).text()
            if imgPath in results.keys():
                tumorClass = BackendModel.get_label('binary', results[imgPath]['pred']['binary'], abbrev=True)
                tumorType = BackendModel.get_label('subtype', results[imgPath]['pred']['subtype'], abbrev=True)
                self.setItem(i, 2, QTableWidgetItem(tumorClass))
                self.setItem(i, 3, QTableWidgetItem(tumorType))

    def reset(self):
        for _ in range(self.rowCount()):
            self.removeRow(0)
            