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
        self._imageViewer = UI.ImageViewer(self)
        self._importButton = UI.IconTextButton(self, 'assets/import-64.png', 'Import')
        self._startButton = UI.IconTextButton(self, 'assets/play-64.ico', 'Start')
        self._saveButton = UI.IconTextButton(self, 'assets/save-64.png', 'Save')
        self._clearButton = UI.IconTextButton(self, 'assets/clear-64.png', 'Clear')
        self._progressBar = QProgressBar(self)
        self._classComboBox = QComboBox(self)
        self._typeComboBox = QComboBox(self)
        self._camComboBox = QComboBox(self)
        self._predGroupBox = UI.PredictionGroupBox(self)
        self._probGroupBox = UI.ProbabilityGroupBox(self)
        self._imageTableWidget = UI.ImageTableWidget(self)

        self._results = {}
        self._backendModel = BackendModel()
        self._imgPaths = []
        self._selectedImgPath = None

        self._classComboBox.addItems(BaseBackendModel.get_all_labels('binary'))
        self._classComboBox.addItem('')
        self._classComboBox.setCurrentIndex(self._classComboBox.count()-1)
        self._typeComboBox.addItems(BaseBackendModel.get_all_labels('subtype'))
        self._typeComboBox.addItem('')
        self._typeComboBox.setCurrentIndex(self._typeComboBox.count()-1)
        self._camComboBox.addItems(['Disable CAM', 'Binary CAM', 'Subtype CAM'])
        self._progressBar.setValue(0)
        

        self._initUI()
        self._connectSignals()


    def _connectSignals(self):
        
        def changeCurrentImage():
            if self._selectedImgPath is None:
                self._imageViewer.clear()
                self._predGroupBox.reset()
                self._probGroupBox.reset()
                self._classComboBox.setCurrentIndex(self._classComboBox.count()-1)
                self._typeComboBox.setCurrentIndex(self._typeComboBox.count()-1)
                return
            if self._selectedImgPath in self._results.keys():
                self._imageViewer.setImage(self._selectedImgPath, 
                                          self._results[self._selectedImgPath]['cam']['binary'] if self._camComboBox.currentIndex() == 1 
                                          else self._results[self._selectedImgPath]['cam']['subtype'] if self._camComboBox.currentIndex() == 2 
                                          else None)
                classPredIdx = self._results[self._selectedImgPath]['pred']['binary']
                typePredIdx = self._results[self._selectedImgPath]['pred']['subtype']
                self._predGroupBox.updatePredictionIndex(classPredIdx, typePredIdx)
                self._probGroupBox.updateProbability(self._results[self._selectedImgPath]['prob']['binary'], self._results[self._selectedImgPath]['prob']['subtype'])
                self._classComboBox.setCurrentIndex(classPredIdx if classPredIdx is not None else self._classComboBox.count()-1)
                self._typeComboBox.setCurrentIndex(typePredIdx if typePredIdx is not None else self._typeComboBox.count()-1)
            else:
                self._imageViewer.setImage(self._selectedImgPath)
                self._predGroupBox.reset()
                self._probGroupBox.reset()
                self._classComboBox.setCurrentIndex(self._classComboBox.count()-1)
                self._typeComboBox.setCurrentIndex(self._typeComboBox.count()-1)
                
        def selectImage(imgPath):
            self._selectedImgPath = imgPath
            changeCurrentImage()


        def imported(imgPaths):
            imgPaths = list(set(imgPaths) - set(self._imgPaths))
            if len(imgPaths) < 300:
                self._imageTableWidget.addImages(imgPaths)
                self._imgPaths.extend(imgPaths)
                return
            dialog = QProgressDialog('Importing images...', 'Cancel', 0, len(imgPaths), self, Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
            dialog.setWindowTitle('Importing Images')
            dialog.setWindowModality(Qt.WindowModal)
            dialog.setFixedSize(400, 100)
            dialog.show()
            for i, imgPath in enumerate(imgPaths):
                self._imgPaths.append(imgPath)
                self._imageTableWidget.addImage(imgPath)
                dialog.setLabelText(f'Importing {imgPath}')
                dialog.setValue(i+1)
                if dialog.wasCanceled():
                    break
            dialog.close()


        def importDialog():
            file_paths = QFileDialog.getOpenFileNames(self, 'Import Images', './', 'Images (*.png *.jpg *.jpeg)')
            if len(file_paths[0]) == 0:
                return
            imported(file_paths[0])

        def freezeWidgetWhenInfer(freeze):
            enabled = not freeze
            self._startButton.setText('Start' if enabled else 'inferencing')
            self._importButton.setEnabled(enabled)
            self._startButton.setEnabled(enabled)
            self._saveButton.setEnabled(enabled)
            self._clearButton.setEnabled(enabled)
            self._classComboBox.setEnabled(enabled)
            self._typeComboBox.setEnabled(enabled)
            self._imageTableWidget.setAcceptDrops(enabled)

        def inferenceProgress(results, progress):
            self._results.update(results)
            self._imageTableWidget.updateResult(results)
            self._progressBar.setValue(progress)
            changeCurrentImage()
        
        def inferenceFinished():
            freezeWidgetWhenInfer(False)

        def startInference():
            toInfer = list(set(self._imgPaths) - set(self._results.keys()))
            if len(toInfer) == 0:
                return
            freezeWidgetWhenInfer(True)
            self._progressBar.setValue(0)
            self._progressBar.setMaximum(len(toInfer))
            self.task = InferenceTask(self._backendModel, toInfer)
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
            df = pd.DataFrame(columns=['image_path', 'tumor_class', 'tumor_type'])
            for imgPath in self._results.keys():
                isConflict = BackendModel.checkConflict(self._results[imgPath]['pred']['binary'], self._results[imgPath]['pred']['subtype'])
                if isConflict or self._results[imgPath]['pred']['binary'] is None or self._results[imgPath]['pred']['subtype'] is None:
                    # display warning dialog
                    tumorClass = BaseBackendModel.get_label('binary', self._results[imgPath]['pred']['binary'])
                    tumorType = BaseBackendModel.get_label('subtype', self._results[imgPath]['pred']['subtype'])
                    QMessageBox.warning(self, 'Warning', 
                                        f'Conflict detected in image: {imgPath}\n' + 
                                        f'class {tumorClass} is incompatible with type {tumorType}\n' +
                                        'Please reselect the class and type for this image.')
                    self._imageTableWidget.selectImageByPath(imgPath)
                    return
                tumorClass = BaseBackendModel.get_label('binary', self._results[imgPath]['pred']['binary'])
                tumorType = BaseBackendModel.get_label('subtype', self._results[imgPath]['pred']['subtype'])
                # append is deprecated
                df.loc[len(df)] = [imgPath, tumorClass, tumorType]
            file_path = QFileDialog.getSaveFileName(self, 'Save Results', './', 'CSV (*.csv)')
            if file_path[0] == '':
                return
            df.to_csv(file_path[0], index=False)

        def clear():
            self._imgPaths = []
            self._results = {}
            self._selectedImgPath = None
            self._imageTableWidget.reset()
            changeCurrentImage()

        def classSelected(index):
            if index == self._classComboBox.count()-1 or self._selectedImgPath is None:
                return
            if not self._selectedImgPath in self._results.keys():
                self._results[self._selectedImgPath] = BaseBackendModel.generate_empty_result()
            self._results[self._selectedImgPath]['pred']['binary'] = index
            updResult = {self._selectedImgPath: self._results[self._selectedImgPath]}
            self._imageTableWidget.updateResult(updResult)
            changeCurrentImage()
        
        def typeSelected(index):
            if index == self._typeComboBox.count()-1 or self._selectedImgPath is None:
                return
            if not self._selectedImgPath in self._results.keys():
                self._results[self._selectedImgPath] = BaseBackendModel.generate_empty_result()
            self._results[self._selectedImgPath]['pred']['subtype'] = index
            updResult = {self._selectedImgPath: self._results[self._selectedImgPath]}
            self._imageTableWidget.updateResult(updResult)
            changeCurrentImage()

        def camSelected(index):
            changeCurrentImage()

        self._imageTableWidget.itemSelectionChanged.connect(lambda: selectImage(self._imageTableWidget.getSelectedImagePath()))
        self._imageTableWidget.imported.connect(imported)

        self._importButton.clicked.connect(importDialog)
        self._startButton.clicked.connect(startInference)
        self._saveButton.clicked.connect(saveResults)
        self._clearButton.clicked.connect(clear)

        self._classComboBox.activated.connect(classSelected)
        self._typeComboBox.activated.connect(typeSelected)
        self._camComboBox.activated.connect(camSelected)


    def _initUI(self):
        self.resize(1500,800)
        self.setFixedSize(self.size())

        controllPanel = QWidget(self)
        controllPanel.setLayout(QGridLayout())
        controllPanel.layout().addWidget(self._importButton, 0, 0, alignment=Qt.AlignHCenter)
        controllPanel.layout().addWidget(self._startButton, 1, 0, alignment=Qt.AlignHCenter)
        controllPanel.layout().addWidget(self._saveButton, 2, 0, alignment=Qt.AlignHCenter)
        controllPanel.layout().addWidget(self._clearButton, 3, 0, alignment=Qt.AlignHCenter)
        controllPanel.layout().addWidget(QLabel('or drag files above'), 0, 1, alignment=Qt.AlignLeft)
        controllPanel.layout().addWidget(self._camComboBox, 1, 1)
        controllPanel.layout().addWidget(self._classComboBox, 2, 1)
        controllPanel.layout().addWidget(self._typeComboBox, 3, 1)
        
        leftPanel = QWidget(self)
        leftPanel.setLayout(QVBoxLayout())
        spImgList = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spImgList.setVerticalStretch(1)
        self._imageTableWidget.setSizePolicy(spImgList)
        leftPanel.layout().addWidget(self._imageTableWidget)
        leftPanel.layout().addWidget(self._progressBar)
        leftPanel.layout().addWidget(controllPanel)


        middlePanel = QWidget(self)
        middlePanel.setLayout(QVBoxLayout())
        middlePanel.layout().addWidget(self._imageViewer)
        spMiddle = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spMiddle.setHorizontalStretch(1)
        middlePanel.setSizePolicy(spMiddle)

        rightPanel = QWidget(self)
        rightPanel.setLayout(QVBoxLayout())
        spPred = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spPred.setVerticalStretch(1)
        self._predGroupBox.setSizePolicy(spPred)
        rightPanel.layout().addWidget(self._predGroupBox)
        spProb = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spProb.setVerticalStretch(4)
        self._probGroupBox.setSizePolicy(spProb)
        rightPanel.layout().addWidget(self._probGroupBox)

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
