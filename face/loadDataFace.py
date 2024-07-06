import cv2
import numpy as np
import os
import io
from math import tau
from scipy.io import wavfile



class LoadDatas:
    def __init__(self, directory) -> None:
        self.num = [48, 68]
        self.load_datas(directory)
        #self.get_images()
        #self.check_data()
        
    def load_datas(self, directory):
        paths = os.listdir(os.path.join(os.getcwd(), directory))
        for path in paths:
            p = os.path.join(os.getcwd(), directory, path)
            if path.endswith('.npy'):
                self.data = np.load(p)
            if path.endswith('.wav'):
                samplerate, data = wavfile.read(p)
                self.audio_file = p
            if path.endswith('face.mp4'):
                self.cap = cv2.VideoCapture(p)
                
    def check_data(self):
        h, w = self.data.shape[:2]
        for i in range(h):
            for j in range(w):
                p = self.data[i][j]
                print(i, j, p)
                
   

if __name__=='__main__':
    directory =  os.path.join(os.getcwd(), 'datas')
    LoadDatas(directory)