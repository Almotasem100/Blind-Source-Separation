
import numpy as np
import librosa

import librosa.display
import sounddevice as sd
import pandas as pd
from separation import *
from sklearn.decomposition import FastICA


class ApplicationWindow(Ui_MainWindow):
    def __init__ (self, MainWindow):
        super(ApplicationWindow, self).setupUi(MainWindow)
        #### tab1
        self.tab1Widgets = [self.tab1_Song,self.tab1_vocals,self.tab1_music,self.tab1_show]
        for i in range(3):
            self.tab1Widgets[i].hide()
        self.tab1_openFile.clicked.connect(self.openFile)
        ####tab2
        self.file1Flag = False
        self.tab2PlotWidgets = [self.tab2_plotWidget1,self.tab2_plotWidget2,self.tab2_plotWidget3,self.tab2_plotWidget4]
        self.tab2Buttons = [self.tab2_openFile1, self.file1Play,self.tab2_openFile2,self.file2Play,self.tab2_signal1,self.tab2_signal2]
        for i in range(5):
            if i < 3:
                self.tab2PlotWidgets[i+1].hide()
            self.tab2Buttons[i+1].hide()
        self.tab2_openFile1.clicked.connect(self.openFile)
        self.tab2_openFile2.clicked.connect(self.openFile)

        ####tab3
        self.tab3_openfile.clicked.connect(self.openFile)


    def play(self,data,samplingRate):
        sd.play(data,samplingRate)



    def openFile (self):
        fname = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', "D:\Sys & Bio\Sys & Bio 3rd year\2nd Semester\DSP\assignments\task_3\First Version") #,"WAV (*.wav)"
        if fname[0]:
            if self.tabWidget.currentIndex() == 0:
                for i in range(3):
                    self.tab1Widgets[i].show()
                self.data , self.samplingRate = librosa.load(fname[0])
                self.time = self.get_Time(self.data,self.samplingRate)
                self.tab1_show.plotItem.plot(self.time,self.data)
                S_full, phase = librosa.magphase(librosa.stft(self.data))
                S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.median,
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(2, sr=self.samplingRate)))
                S_filter = np.minimum(S_full, S_filter)
                margin_i, margin_v = 2, 10
                power = 2

                mask_i = librosa.util.softmask(S_filter,
                                            margin_i * (S_full - S_filter),
                                            power=power)

                mask_v = librosa.util.softmask(S_full - S_filter,
                                            margin_v * S_filter,
                                            power=power)
                S_foreground = mask_v * S_full
                S_background = mask_i * S_full
                self.vocals = librosa.istft(S_foreground)
                self.music = librosa.istft(S_background)
                self.tab1Data = [self.data , self.vocals ,self.music]
                self.playingData= [lambda:self.play(self.data,self.samplingRate),lambda:self.play(self.vocals,self.samplingRate),lambda:self.play(self.music,self.samplingRate)]
                for i in range(3):
                    self.tab1Widgets[i].clicked.connect(self.playingData[i])
            if self.tabWidget.currentIndex() == 1:
                if self.file1Flag == False:
                    self.file1Data , self.file1SamplingRate = librosa.load(fname[0])
                    time = self.get_Time(self.file1Data,self.file1SamplingRate)
                    self.tab2PlotWidgets[0].plotItem.plot(time,self.file1Data)
                    self.file1Flag = True
                    for i in range(2):
                        self.tab2Buttons[i+1].show()
                    self.tab2Buttons[1].clicked.connect(lambda:self.play(self.file1Data,self.file1SamplingRate))
                else:
                    self.file2Data , self.file2SamplingRate = librosa.load(fname[0])
                    self.tab2PlotWidgets[1].show()
                    time = self.get_Time(self.file2Data,self.file2SamplingRate)
                    # self.tab2Buttons[3].clicked.connect(self.play(self.file2Data,self.file2SamplingRate))
                    self.tab2PlotWidgets[1].plotItem.plot(time,self.file2Data)
                    data = np.c_[self.file1Data,self.file2Data]
                    ica = FastICA(n_components=2,max_iter=1000)
                    dataSeparating = ica.fit_transform(data)  # Reconstruct signals
                    self.tab2PlayingData = [lambda:self.play(self.file2Data,self.file2SamplingRate),lambda:self.play(dataSeparating[:,0],self.file1SamplingRate),lambda:self.play(dataSeparating[:,1],self.file1SamplingRate)]
                    for i in range(3):
                        if i <2:
                            self.tab2PlotWidgets[i+2].show()
                            self.tab2PlotWidgets[i+2].plotItem.plot(time,dataSeparating[:,i])
                        self.tab2Buttons[i+3].clicked.connect(self.tab2PlayingData[i])
                        self.tab2Buttons[i+3].show()

            if self.tabWidget.currentIndex() == 2:
                if fname[0].endswith('.csv'):
                    self.data2=pd.read_csv(fname[0])
                    self.widget.plotItem.clear()
                    self.widget_2.plotItem.clear()
                    self.widget_3.plotItem.clear()
                    self.widget_4.plotItem.clear()
                    self.xAxis = self.data2.iloc[:,0].values
                    self.yAxis = self.data2.iloc[:,1].values
                    self.zAxis = self.data2.iloc[:,2].values
                    self.widget.plotItem.plot(self.xAxis[0:1200],self.yAxis[0:1200])
                    self.widget_2.plotItem.plot(self.xAxis[0:1200],self.zAxis[0:1200])
                    data = np.c_[self.yAxis,self.zAxis]
                    ica = FastICA(n_components=2,max_iter=1000)
                    sep = ica.fit_transform(data)  # Reconstruct signals
                    self.widget_3.plotItem.plot(self.xAxis[0:1200],sep[0:1200,0])
                    self.widget_4.plotItem.plot(self.xAxis[0:1200],sep[0:1200,1])


    def get_Time(self,data,rate):
        '''
        get the time of the audio 

        Parameters:
            data: numpy array
               data of the recored
            rate: int  
                the smaple rate of the wav file

        Returns:
            f: ndarray
                Array of length n containing the time of audio
        '''
        length = data.shape[0]
        ArrayOfLength = np.arange(length)
        time = ArrayOfLength / rate
        return time













if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = ApplicationWindow(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())