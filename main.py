import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import UI


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
        self.startButton = UI.StartButton(self)
        self.predGroupBox = UI.PredictionGroupBox(self)
        self.probGroupBox = UI.ProbabilityGroupBox(self)
        self.imageList = UI.ImageList(self)

        self.results = {}
        self.backendModel = UI.BackendModel()
        self.imgPaths = []
        self.selectedImgPath = None

        self.connectSignals()
        self.initUI()


    def connectSignals(self):

        def selectImage(imgPath):
            self.selectedImgPath = imgPath
            self.imageViewer.setImage(imgPath)
            if imgPath in self.results.keys():
                bin_pred = self.backendModel.get_label('binary', self.results[imgPath]['pred']['binary'])
                sub_pred = self.backendModel.get_label('subtype', self.results[imgPath]['pred']['subtype'])
                self.predGroupBox.updatePrediction(bin_pred, sub_pred)
                self.probGroupBox.updateProbability(self.results[imgPath]['prob']['binary'], self.results[imgPath]['prob']['subtype'])
            else:
                self.predGroupBox.updatePrediction('', '')
                self.probGroupBox.reset()
        
        self.imageList.itemSelectionChanged.connect(lambda: selectImage(self.imageList.getSelectedImagePath()))

        def inferenceFinished(results):
            self.startButton.setEnabled(True)
            self.startButton.setText('Start')
            self.imageList.setAcceptDrops(True)
            self.results.update(results)
            self.imageList.updateResult(results)

        def startInference():
            if len(self.imgPaths) == 0:
                QMessageBox.warning(self, "No image imported", "Please import at least one image to start inferencing.")
                return
            self.startButton.setText('inferencing')
            self.startButton.setEnabled(False)
            self.imageList.setAcceptDrops(False)
            self.task = InferenceTask(self.backendModel, self.imgPaths)
            self.workerThread = QThread()
            self.task.moveToThread(self.workerThread)
            self.workerThread.started.connect(self.task.run)
            self.task.finished.connect(self.workerThread.quit)
            self.task.finished.connect(inferenceFinished)
            self.task.finished.connect(self.task.deleteLater)
            self.workerThread.finished.connect(self.workerThread.deleteLater)
            self.workerThread.start()

        self.startButton.clicked.connect(startInference)

        def imported(imgPaths):
            self.imgPaths = imgPaths
            self.imageList.addImages(imgPaths)

        self.imageList.imported.connect(imported)


    def initUI(self):
        self.resize(1500,800)
        self.setFixedSize(self.size())

        self.setLayout(QHBoxLayout())

        self.leftPanel = QWidget(self)
        self.leftPanel.setLayout(QVBoxLayout())
        spImgList = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spImgList.setVerticalStretch(1)
        self.imageList.setSizePolicy(spImgList)
        self.leftPanel.layout().addWidget(self.imageList)
        self.leftPanel.layout().addWidget(self.startButton, alignment=Qt.AlignHCenter)


        self.middlePanel = QWidget(self)
        self.middlePanel.setLayout(QVBoxLayout())
        self.middlePanel.layout().addWidget(self.imageViewer)
        spMiddle = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spMiddle.setHorizontalStretch(1)
        self.middlePanel.setSizePolicy(spMiddle)

        self.rightPanel = QWidget(self)
        self.rightPanel.setLayout(QVBoxLayout())
        spPred = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spPred.setVerticalStretch(1)
        self.predGroupBox.setSizePolicy(spPred)
        self.rightPanel.layout().addWidget(self.predGroupBox)
        spProb = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spProb.setVerticalStretch(4)
        self.probGroupBox.setSizePolicy(spProb)
        self.rightPanel.layout().addWidget(self.probGroupBox)

        self.layout().addWidget(self.leftPanel)
        self.layout().addWidget(self.middlePanel)
        self.layout().addWidget(self.rightPanel)


def main():
    app = QApplication(sys.argv)
    window = Window()

    window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
