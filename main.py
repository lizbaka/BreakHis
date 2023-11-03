import sys
import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import UI
from models.inference import *


class InferenceTask(QObject):

    finished = pyqtSignal(dict)

    def __init__(self, backEndModel, paths):
        QObject.__init__(self)
        self.backEndModel = backEndModel
        self.paths = paths

    
    def run(self):
        result = self.backEndModel.inference(self.paths)
        self.finished.emit(result)


class Window(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("Breast Cancer Classifier")
        self.imageViewer = UI.ImageViewer(self)
        self.startButton = UI.IconTextButton(self, 'assets/play-64.ico', 'Start')
        self.saveButton = UI.IconTextButton(self, 'assets/save-64.png', 'Save')
        self.clearButton = UI.IconTextButton(self, 'assets/clear-64.png', 'Clear')
        self.classComboBox = QComboBox(self)
        self.typeComboBox = QComboBox(self)
        self.predGroupBox = UI.PredictionGroupBox(self)
        self.probGroupBox = UI.ProbabilityGroupBox(self)
        self.imageTableWidget = UI.ImageTableWidget(self)

        self.results = {}
        self.backendModel = BackendModel()
        self.imgPaths = []
        self.selectedImgPath = None

        self.classComboBox.addItems(BackendModel.get_all_labels('binary'))
        self.classComboBox.addItem('')
        self.classComboBox.setCurrentIndex(self.classComboBox.count()-1)
        self.typeComboBox.addItems(BackendModel.get_all_labels('subtype'))
        self.typeComboBox.addItem('')
        self.typeComboBox.setCurrentIndex(self.typeComboBox.count()-1)

        self.initUI()
        self.connectSignals()


    def connectSignals(self):
        

        def changeCurrentImage():
            if self.selectedImgPath is None:
                self.imageViewer.clear()
                self.predGroupBox.reset()
                self.probGroupBox.reset()
                self.classComboBox.setCurrentIndex(self.classComboBox.count()-1)
                self.typeComboBox.setCurrentIndex(self.typeComboBox.count()-1)
                return
            self.imageViewer.setImage(self.selectedImgPath)
            if self.selectedImgPath in self.results.keys():
                classPredIdx = self.results[self.selectedImgPath]['pred']['binary']
                typePredIdx = self.results[self.selectedImgPath]['pred']['subtype']
                self.predGroupBox.updatePredictionIndex(classPredIdx, typePredIdx)
                self.probGroupBox.updateProbability(self.results[self.selectedImgPath]['prob']['binary'], self.results[self.selectedImgPath]['prob']['subtype'])
                self.classComboBox.setCurrentIndex(classPredIdx if classPredIdx is not None else self.classComboBox.count()-1)
                self.typeComboBox.setCurrentIndex(typePredIdx if typePredIdx is not None else self.typeComboBox.count()-1)
            else:
                self.predGroupBox.reset()
                self.probGroupBox.reset()
                self.classComboBox.setCurrentIndex(self.classComboBox.count()-1)
                self.typeComboBox.setCurrentIndex(self.typeComboBox.count()-1)
                
        def selectImage(imgPath):
            self.selectedImgPath = imgPath
            changeCurrentImage()

        def imported(imgPaths):
            imgPaths = list(set(imgPaths) - set(self.imgPaths))
            self.imgPaths.extend(imgPaths)
            self.imageTableWidget.addImages(imgPaths)
        
        def inferenceFinished(results):
            self.startButton.setText('Start')
            self.startButton.setEnabled(True)
            self.saveButton.setEnabled(True)
            self.clearButton.setEnabled(True)
            self.classComboBox.setEnabled(True)
            self.typeComboBox.setEnabled(True)
            self.imageTableWidget.setAcceptDrops(True)
            self.results.update(results)
            self.imageTableWidget.updateResult(results)
            changeCurrentImage()

        def startInference():
            toInfer = list(set(self.imgPaths) - set(self.results.keys()))
            if len(toInfer) == 0:
                return
            self.startButton.setText('inferencing')
            self.startButton.setEnabled(False)
            self.saveButton.setEnabled(False)
            self.clearButton.setEnabled(False)
            self.classComboBox.setEnabled(False)
            self.typeComboBox.setEnabled(False)
            self.imageTableWidget.setAcceptDrops(False)
            self.task = InferenceTask(self.backendModel, toInfer)
            self.workerThread = QThread()
            self.task.moveToThread(self.workerThread)
            self.workerThread.started.connect(self.task.run)
            self.task.finished.connect(self.workerThread.quit)
            self.task.finished.connect(inferenceFinished)
            self.task.finished.connect(self.task.deleteLater)
            self.workerThread.finished.connect(self.workerThread.deleteLater)
            self.workerThread.start()

        def saveResults():
            file_path = QFileDialog.getSaveFileName(self, 'Save Results', './', 'CSV (*.csv)')
            if file_path[0] == '':
                return
            df = pd.DataFrame(columns=['image_path', 'tumor_class', 'tumor_type'])
            for imgPath in self.results.keys():
                tumorClass = BackendModel.get_label('binary', self.results[imgPath]['pred']['binary'])
                tumorType = BackendModel.get_label('subtype', self.results[imgPath]['pred']['subtype'])
                # append is deprecated
                df.loc[len(df)] = [imgPath, tumorClass, tumorType]

            df.to_csv(file_path[0], index=False)

        def clear():
            self.imgPaths = []
            self.results = {}
            self.selectedImgPath = None
            self.imageTableWidget.reset()
            changeCurrentImage()

        def classSelected(index):
            if index == self.classComboBox.count()-1 or self.selectedImgPath is None:
                return
            if not self.selectedImgPath in self.results.keys():
                self.results[self.selectedImgPath] = BackendModel.generate_empty_result()
            self.results[self.selectedImgPath]['pred']['binary'] = index
            self.imageTableWidget.updateResult(self.results)
            changeCurrentImage()
        
        def typeSelected(index):
            if index == self.typeComboBox.count()-1 or self.selectedImgPath is None:
                return
            if not self.selectedImgPath in self.results.keys():
                self.results[self.selectedImgPath] = BackendModel.generate_empty_result()
            self.results[self.selectedImgPath]['pred']['subtype'] = index
            self.imageTableWidget.updateResult(self.results)
            changeCurrentImage()

        self.imageTableWidget.itemSelectionChanged.connect(lambda: selectImage(self.imageTableWidget.getSelectedImagePath()))
        self.imageTableWidget.imported.connect(imported)

        self.startButton.clicked.connect(startInference)
        self.saveButton.clicked.connect(saveResults)
        self.clearButton.clicked.connect(clear)

        self.classComboBox.activated.connect(classSelected)
        self.typeComboBox.activated.connect(typeSelected)


    def initUI(self):
        self.resize(1500,800)
        self.setFixedSize(self.size())

        controllPanel = QWidget(self)
        controllPanel.setLayout(QGridLayout())
        controllPanel.layout().addWidget(self.clearButton, 0, 0, 1, 2, alignment=Qt.AlignHCenter)
        controllPanel.layout().addWidget(self.startButton, 1, 0, alignment=Qt.AlignHCenter)
        controllPanel.layout().addWidget(self.saveButton, 2, 0, alignment=Qt.AlignHCenter)
        controllPanel.layout().addWidget(self.classComboBox, 1, 1)
        controllPanel.layout().addWidget(self.typeComboBox, 2, 1)
        
        leftPanel = QWidget(self)
        leftPanel.setLayout(QVBoxLayout())
        spImgList = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spImgList.setVerticalStretch(1)
        self.imageTableWidget.setSizePolicy(spImgList)
        leftPanel.layout().addWidget(self.imageTableWidget)
        leftPanel.layout().addWidget(controllPanel)


        middlePanel = QWidget(self)
        middlePanel.setLayout(QVBoxLayout())
        middlePanel.layout().addWidget(self.imageViewer)
        spMiddle = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spMiddle.setHorizontalStretch(1)
        middlePanel.setSizePolicy(spMiddle)

        rightPanel = QWidget(self)
        rightPanel.setLayout(QVBoxLayout())
        spPred = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spPred.setVerticalStretch(1)
        self.predGroupBox.setSizePolicy(spPred)
        rightPanel.layout().addWidget(self.predGroupBox)
        spProb = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spProb.setVerticalStretch(4)
        self.probGroupBox.setSizePolicy(spProb)
        rightPanel.layout().addWidget(self.probGroupBox)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(leftPanel)
        self.layout().addWidget(middlePanel)
        self.layout().addWidget(rightPanel)


def main():
    app = QApplication(sys.argv)
    window = Window()

    window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
