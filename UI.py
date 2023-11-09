import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from models.inference import *
from PIL import Image
from torchcam.utils import overlay_mask


def draw_CAM(img, cam):
    # convert cam to PIL image
    cam = Image.fromarray(cam)
    return overlay_mask(img, cam, alpha=0.5)


class ImageViewer(QLabel):

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self._initUI()


    def _initUI(self):
        self.setAlignment(Qt.AlignCenter)

        pal = QPalette(self.palette())
        pal.setColor(QPalette.Background, Qt.white)
        self.setAutoFillBackground(True)
        self.setPalette(pal)
        
        self.setFont(QFont("Arial", 12))
        self.setText("Image Viewer")


    def setImage(self, img_path, cam = None):
        img = Image.open(img_path).convert('RGB')
        if cam is not None:
            img = draw_CAM(img, cam)
        img = img.toqimage()
        img = img.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(img))


class IconTextButton(QPushButton):

    def __init__(self, parent, iconPath, text):
        super().__init__(parent)
        self.parent = parent
        self._initUI(iconPath, text)
    
    def _initUI(self, iconPath, text):
        self.setFixedWidth(150)
        self.setText(text)
        self.setFont(QFont("Arial", 12))
        self.setIcon(QIcon(iconPath))


class BaseResultGroupBox(QGroupBox):
    
    def __init__(self, parent, title, keyLabels, valueLabels):
        super().__init__(parent)
        self.parent = parent
        self.title = title
        self._keyLabels = keyLabels
        self._valueLabels = valueLabels

    
    def _initUI(self):
        self.setTitle(self.title)
        self.setFont(QFont("Arial", 11))
        self.setFixedWidth(400)

        self.setLayout(QVBoxLayout())
        
        for key in self._keyLabels.keys():
            itemLayout = QHBoxLayout()
            itemLayout.addWidget(self._keyLabels[key])
            itemLayout.addWidget(self._valueLabels[key])
            self.layout().addLayout(itemLayout)


class PredictionGroupBox(BaseResultGroupBox):
    
    def __init__(self, parent):
        self._keyLabels = {
            'class': QLabel("Predicted tumor class:"),
            'type': QLabel("Predicted tumor type:")
        }
        self._valueLabels = {
            'class': QLabel(),
            'type': QLabel()
        }

        super().__init__(parent, "Prediction", self._keyLabels, self._valueLabels)
        self.parent = parent

        self._initUI()


    def updatePredictionIndex(self, tumorClassId, tumorTypeId):
        tumorClass = BaseBackendModel.get_label('binary', tumorClassId)
        tumorType = BaseBackendModel.get_label('subtype', tumorTypeId)
        self._updatePrediction(tumorClass, tumorType, BackendModel.checkConflict(tumorClassId, tumorTypeId))


    def _updatePrediction(self, tumorClass, tumorType, conflict=False):
        if tumorClass == 'reject' or conflict:
            self._valueLabels['class'].setStyleSheet('color: red; font-size: 11pt; font-family: Arial')
        else:
            self._valueLabels['class'].setStyleSheet('color: black; font-size: 11pt; font-family: Arial')
        if tumorType == 'reject' or conflict:
            self._valueLabels['type'].setStyleSheet('color: red; font-size: 11pt; font-family: Arial')
        else:
            self._valueLabels['type'].setStyleSheet('color: black; font-size: 11pt; font-family: Arial')
        self._valueLabels['class'].setText(tumorClass)
        self._valueLabels['type'].setText(tumorType)

    
    def reset(self):
        self._updatePrediction('', '')


class ProbabilityGroupBox(BaseResultGroupBox):

    def __init__(self, parent):
        self._keyLabels = {
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
        self._valueLabels = {
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
        self._valueSliders = {
            'prob_B': QSlider(Qt.Horizontal),
            'prob_M': QSlider(Qt.Horizontal),
            'prob_A': QSlider(Qt.Horizontal),
            'prob_F': QSlider(Qt.Horizontal),
            'prob_PT': QSlider(Qt.Horizontal),
            'prob_TA': QSlider(Qt.Horizontal),
            'prob_DC': QSlider(Qt.Horizontal),
            'prob_LC': QSlider(Qt.Horizontal),
            'prob_MC': QSlider(Qt.Horizontal),
            'prob_PC': QSlider(Qt.Horizontal)
        }

        super().__init__(parent, "Probability", self._keyLabels, self._valueLabels)

        self._initUI()
    

    def updateProbability(self, classProb, typeProb):
        classProb = [round(x, 4) for x in classProb]
        typeProb = [round(x, 4) for x in typeProb]
        self._valueSliders['prob_B'].setValue(int(classProb[0]*100))
        self._valueSliders['prob_M'].setValue(int(classProb[1]*100))
        self._valueSliders['prob_A'].setValue(int(typeProb[0]*100))
        self._valueSliders['prob_F'].setValue(int(typeProb[1]*100))
        self._valueSliders['prob_PT'].setValue(int(typeProb[2]*100))
        self._valueSliders['prob_TA'].setValue(int(typeProb[3]*100))
        self._valueSliders['prob_DC'].setValue(int(typeProb[4]*100))
        self._valueSliders['prob_LC'].setValue(int(typeProb[5]*100))
        self._valueSliders['prob_MC'].setValue(int(typeProb[6]*100))
        self._valueSliders['prob_PC'].setValue(int(typeProb[7]*100))
        classProb = [str(x) for x in classProb]
        typeProb = [str(x) for x in typeProb]
        self._valueLabels['prob_B'].setText(classProb[0])
        self._valueLabels['prob_M'].setText(classProb[1])
        self._valueLabels['prob_A'].setText(typeProb[0])
        self._valueLabels['prob_F'].setText(typeProb[1])
        self._valueLabels['prob_PT'].setText(typeProb[2])
        self._valueLabels['prob_TA'].setText(typeProb[3])
        self._valueLabels['prob_DC'].setText(typeProb[4])
        self._valueLabels['prob_LC'].setText(typeProb[5])
        self._valueLabels['prob_MC'].setText(typeProb[6])
        self._valueLabels['prob_PC'].setText(typeProb[7])


    def reset(self):
        for key in self._valueLabels.keys():
            self._valueLabels[key].setText('')
            self._valueSliders[key].setValue(0)


    def _initUI(self):
        self.setTitle(self.title)
        self.setFont(QFont("Arial", 11))
        self.setFixedWidth(400)

        self.setLayout(QVBoxLayout())
        
        for key in self._keyLabels.keys():
            itemLayout = QHBoxLayout()
            itemLayout.addWidget(self._keyLabels[key])
            itemLayout.addWidget(self._valueLabels[key])
            self.layout().addLayout(itemLayout)
            self._valueSliders[key].setTickPosition(QSlider.NoTicks)
            self._valueSliders[key].setRange(0, 100)
            self._valueSliders[key].setSingleStep(1)
            self._valueSliders[key].setValue(0)
            self._valueSliders[key].setEnabled(False)
            self.layout().addWidget(self._valueSliders[key])


class ImageTableWidget(QTableWidget):
        
    imported = pyqtSignal(list)
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self._pathToRow = {}
        self._initUI()
        self.setAcceptDrops(True)

        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)


    def _initUI(self):
        self.setFixedWidth(400)
        self.verticalHeader().setVisible(False)
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
        self._pathToRow[imgPath] = row
        imgLabel = QLabel()
        imgLabel.setPixmap(QPixmap(imgPath).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        imgLabel.setAlignment(Qt.AlignCenter)
        self.setCellWidget(row, 0, imgLabel)
        self.setItem(row, 1, QTableWidgetItem(imgPath))

    
    def addImages(self, imgPaths):
        for imgPath in imgPaths:
            self.addImage(imgPath)


    def selectImageByPath(self, imgPath):
        self.selectRow(self._pathToRow[imgPath])


    def getSelectedImagePath(self):
        if self.selectedItems():
            row = self.selectedItems()[0].row()
            return self.item(row, 1).text()
        else:
            return None
        
    
    def updateResult(self, results):
        for imgPath in results.keys():
            if not imgPath in self._pathToRow.keys():
                continue
            i = self._pathToRow[imgPath]
            tumorClassId = results[imgPath]['pred']['binary']
            tumorTypeId = results[imgPath]['pred']['subtype']
            tumorClass = BaseBackendModel.get_label('binary', tumorClassId, abbrev=True)
            tumorType = BaseBackendModel.get_label('subtype', tumorTypeId, abbrev=True)
            self.setItem(i, 2, QTableWidgetItem(tumorClass))
            self.setItem(i, 3, QTableWidgetItem(tumorType))
            if tumorClass == 'reject' or tumorType == 'reject' or BaseBackendModel.checkConflict(tumorClassId, tumorTypeId):
                for j in range(1, self.columnCount()):
                    self.item(i, j).setBackground(QColor(255, 0, 0, 50))
            else:
                for j in range(1, self.columnCount()):
                    self.item(i, j).setBackground(QColor(255, 255, 255, 0))
    

    def reset(self):
        self._pathToRow = {}
        for _ in range(self.rowCount()):
            self.removeRow(0)
            