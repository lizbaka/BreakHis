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
        print('run')
        result = self.backEndModel.inference(self.paths)
        self.finished.emit(result)


class Window(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle("Breast Cancer Classifier")
        self.imageViewer = UI.ImageViewer(self)
        self.startButton = UI.StartButton(self)
        self.pathGroupBox = UI.PathGroupBox(self)
        self.predGroupBox = UI.PredictionGroupBox(self)
        self.probGroupBox = UI.ProbabilityGroupBox(self)

        self.results = {}
        self.backendModel = UI.BackendModel()
        self.imgPath = None

        self.initUI()

        def imageChanged(img_path):
            self.imgPath = img_path
            self.pathGroupBox.updatePath(img_path)
            self.predGroupBox.updatePrediction('', '')
            self.probGroupBox.reset()
        
        self.imageViewer.imageChanged.connect(imageChanged)

        def inferenceFinished(results):
            self.startButton.setEnabled(True)
            self.startButton.setText('Start')
            self.imageViewer.setAcceptDrops(True)
            self.results = results
            bin_pred = self.backendModel.get_label('binary', results[self.imgPath]['pred']['binary'])
            subtype_pred = self.backendModel.get_label('subtype', results[self.imgPath]['pred']['subtype'])
            self.predGroupBox.updatePrediction(bin_pred, subtype_pred)
            self.probGroupBox.updateProbability(results[self.imgPath]['prob']['binary'], results[self.imgPath]['prob']['subtype'])

        def startInference():
            if self.imgPath is None:
                return
            self.startButton.setText('inferencing')
            self.startButton.setEnabled(False)
            self.imageViewer.setAcceptDrops(False)
            self.task = InferenceTask(self.backendModel, [self.imgPath])
            self.workerThread = QThread()
            self.task.moveToThread(self.workerThread)
            self.workerThread.started.connect(self.task.run)
            self.task.finished.connect(self.workerThread.quit)
            self.task.finished.connect(inferenceFinished)
            self.task.finished.connect(self.task.deleteLater)
            self.workerThread.finished.connect(self.workerThread.deleteLater)
            self.workerThread.start()

        self.startButton.clicked.connect(startInference)


    def initUI(self):
        self.resize(1500,800)
        self.setFixedSize(self.size())

        self.setLayout(QHBoxLayout())
        self.leftPanel = QWidget(self)
        self.leftPanel.setFixedWidth(1000)
        self.leftPanel.setLayout(QVBoxLayout())
        self.leftPanel.layout().addWidget(self.imageViewer)
        self.leftPanel.layout().addWidget(self.startButton, alignment=Qt.AlignHCenter)

        self.rightPanel = QWidget(self)
        self.rightPanel.setLayout(QVBoxLayout())
        spPath = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spPath.setVerticalStretch(1)
        self.pathGroupBox.setSizePolicy(spPath)
        self.rightPanel.layout().addWidget(self.pathGroupBox)
        spPred = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spPred.setVerticalStretch(2)
        self.predGroupBox.setSizePolicy(spPred)
        self.rightPanel.layout().addWidget(self.predGroupBox)
        spProb = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        spProb.setVerticalStretch(8)
        self.probGroupBox.setSizePolicy(spProb)
        self.rightPanel.layout().addWidget(self.probGroupBox)

        self.layout().addWidget(self.leftPanel)
        self.layout().addWidget(self.rightPanel)


def main():
    app = QApplication(sys.argv)
    window = Window()

    window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
