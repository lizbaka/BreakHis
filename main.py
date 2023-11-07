import sys
import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import UI
from models.inference import *


class InferenceTask(QObject):

    finished = pyqtSignal()
    progress = pyqtSignal(dict, int)

    def __init__(self, backEndModel, paths):
        QObject.__init__(self)
        self.backEndModel = backEndModel
        self.paths = paths

    
    def run(self):
        for i in range(0, len(self.paths), 16):
            result = self.backEndModel.inference(self.paths[i:i+16])
            self.progress.emit(result, i+16 if i+16 < len(self.paths) else len(self.paths))
        self.finished.emit()


class Window(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("Breast Cancer Classifier")
        self.imageViewer = UI.ImageViewer(self)
        self.importButton = UI.IconTextButton(self, 'assets/import-64.png', 'Import')
        self.startButton = UI.IconTextButton(self, 'assets/play-64.ico', 'Start')
        self.saveButton = UI.IconTextButton(self, 'assets/save-64.png', 'Save')
        self.clearButton = UI.IconTextButton(self, 'assets/clear-64.png', 'Clear')
        self.progressBar = QProgressBar(self)
        self.classComboBox = QComboBox(self)
        self.typeComboBox = QComboBox(self)
        self.camComboBox = QComboBox(self)
        self.predGroupBox = UI.PredictionGroupBox(self)
        self.probGroupBox = UI.ProbabilityGroupBox(self)
        self.imageTableWidget = UI.ImageTableWidget(self)

        self.results = {}
        self.backendModel = BackendModel()
        self.imgPaths = []
        self.selectedImgPath = None

        self.classComboBox.addItems(BaseBackendModel.get_all_labels('binary'))
        self.classComboBox.addItem('')
        self.classComboBox.setCurrentIndex(self.classComboBox.count()-1)
        self.typeComboBox.addItems(BaseBackendModel.get_all_labels('subtype'))
        self.typeComboBox.addItem('')
        self.typeComboBox.setCurrentIndex(self.typeComboBox.count()-1)
        self.camComboBox.addItems(['Disable CAM', 'Binary CAM', 'Subtype CAM'])
        self.progressBar.setValue(0)
        

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
            if self.selectedImgPath in self.results.keys():
                self.imageViewer.setImage(self.selectedImgPath, 
                                          self.results[self.selectedImgPath]['cam']['binary'] if self.camComboBox.currentIndex() == 1 
                                          else self.results[self.selectedImgPath]['cam']['subtype'] if self.camComboBox.currentIndex() == 2 
                                          else None)
                classPredIdx = self.results[self.selectedImgPath]['pred']['binary']
                typePredIdx = self.results[self.selectedImgPath]['pred']['subtype']
                self.predGroupBox.updatePredictionIndex(classPredIdx, typePredIdx)
                self.probGroupBox.updateProbability(self.results[self.selectedImgPath]['prob']['binary'], self.results[self.selectedImgPath]['prob']['subtype'])
                self.classComboBox.setCurrentIndex(classPredIdx if classPredIdx is not None else self.classComboBox.count()-1)
                self.typeComboBox.setCurrentIndex(typePredIdx if typePredIdx is not None else self.typeComboBox.count()-1)
            else:
                self.imageViewer.setImage(self.selectedImgPath)
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

        def importDialog():
            file_paths = QFileDialog.getOpenFileNames(self, 'Import Images', './', 'Images (*.png *.jpg *.jpeg)')
            if len(file_paths[0]) == 0:
                return
            imported(file_paths[0])

        def freezeWidgetWhenInfer(freeze):
            enabled = not freeze
            self.startButton.setText('Start' if enabled else 'inferencing')
            self.importButton.setEnabled(enabled)
            self.startButton.setEnabled(enabled)
            self.saveButton.setEnabled(enabled)
            self.clearButton.setEnabled(enabled)
            self.classComboBox.setEnabled(enabled)
            self.typeComboBox.setEnabled(enabled)
            self.imageTableWidget.setAcceptDrops(enabled)

        def inferenceProgress(results, progress):
            self.results.update(results)
            self.imageTableWidget.updateResult(results)
            self.progressBar.setValue(progress)
            changeCurrentImage()
        
        def inferenceFinished():
            freezeWidgetWhenInfer(False)

        def startInference():
            toInfer = list(set(self.imgPaths) - set(self.results.keys()))
            if len(toInfer) == 0:
                return
            freezeWidgetWhenInfer(True)
            self.progressBar.setValue(0)
            self.progressBar.setMaximum(len(toInfer))
            self.task = InferenceTask(self.backendModel, toInfer)
            self.task.progress.connect(inferenceProgress)
            self.workerThread = QThread()
            self.task.moveToThread(self.workerThread)
            self.workerThread.started.connect(self.task.run)
            self.task.finished.connect(inferenceFinished)
            self.task.finished.connect(self.workerThread.quit)
            self.task.finished.connect(self.task.deleteLater)
            self.workerThread.finished.connect(self.workerThread.deleteLater)
            self.workerThread.start()

        def saveResults():
            file_path = QFileDialog.getSaveFileName(self, 'Save Results', './', 'CSV (*.csv)')
            if file_path[0] == '':
                return
            df = pd.DataFrame(columns=['image_path', 'tumor_class', 'tumor_type'])
            for imgPath in self.results.keys():
                isConflict = BackendModel.checkConflict(self.results[imgPath]['pred']['binary'], self.results[imgPath]['pred']['subtype'])
                if isConflict or self.results[imgPath]['pred']['binary'] is None or self.results[imgPath]['pred']['subtype'] is None:
                    # display warning dialog
                    tumorClass = BaseBackendModel.get_label('binary', self.results[imgPath]['pred']['binary'])
                    tumorType = BaseBackendModel.get_label('subtype', self.results[imgPath]['pred']['subtype'])
                    QMessageBox.warning(self, 'Warning', 
                                        f'Conflict detected in image: {imgPath}\n' + 
                                        f'class {tumorClass} is incompatible with type {tumorType}\n' +
                                        'Please reselect the class and type for this image.')
                    self.imageTableWidget.selectImageByPath(imgPath)
                    return
                tumorClass = BaseBackendModel.get_label('binary', self.results[imgPath]['pred']['binary'])
                tumorType = BaseBackendModel.get_label('subtype', self.results[imgPath]['pred']['subtype'])
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
                self.results[self.selectedImgPath] = BaseBackendModel.generate_empty_result()
            self.results[self.selectedImgPath]['pred']['binary'] = index
            updResult = {self.selectedImgPath: self.results[self.selectedImgPath]}
            self.imageTableWidget.updateResult(updResult)
            changeCurrentImage()
        
        def typeSelected(index):
            if index == self.typeComboBox.count()-1 or self.selectedImgPath is None:
                return
            if not self.selectedImgPath in self.results.keys():
                self.results[self.selectedImgPath] = BaseBackendModel.generate_empty_result()
            self.results[self.selectedImgPath]['pred']['subtype'] = index
            updResult = {self.selectedImgPath: self.results[self.selectedImgPath]}
            self.imageTableWidget.updateResult(updResult)
            changeCurrentImage()

        def camSelected(index):
            changeCurrentImage()

        self.imageTableWidget.itemSelectionChanged.connect(lambda: selectImage(self.imageTableWidget.getSelectedImagePath()))
        self.imageTableWidget.imported.connect(imported)

        self.importButton.clicked.connect(importDialog)
        self.startButton.clicked.connect(startInference)
        self.saveButton.clicked.connect(saveResults)
        self.clearButton.clicked.connect(clear)

        self.classComboBox.activated.connect(classSelected)
        self.typeComboBox.activated.connect(typeSelected)
        self.camComboBox.activated.connect(camSelected)


    def initUI(self):
        self.resize(1500,800)
        self.setFixedSize(self.size())

        controllPanel = QWidget(self)
        controllPanel.setLayout(QGridLayout())
        controllPanel.layout().addWidget(self.importButton, 0, 0, alignment=Qt.AlignHCenter)
        controllPanel.layout().addWidget(self.startButton, 1, 0, alignment=Qt.AlignHCenter)
        controllPanel.layout().addWidget(self.saveButton, 2, 0, alignment=Qt.AlignHCenter)
        controllPanel.layout().addWidget(self.clearButton, 3, 0, alignment=Qt.AlignHCenter)
        controllPanel.layout().addWidget(QLabel('or drag files above'), 0, 1, alignment=Qt.AlignLeft)
        controllPanel.layout().addWidget(self.camComboBox, 1, 1)
        controllPanel.layout().addWidget(self.classComboBox, 2, 1)
        controllPanel.layout().addWidget(self.typeComboBox, 3, 1)
        
        leftPanel = QWidget(self)
        leftPanel.setLayout(QVBoxLayout())
        spImgList = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spImgList.setVerticalStretch(1)
        self.imageTableWidget.setSizePolicy(spImgList)
        leftPanel.layout().addWidget(self.imageTableWidget)
        leftPanel.layout().addWidget(self.progressBar)
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
