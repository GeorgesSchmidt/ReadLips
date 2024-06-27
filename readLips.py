import cv2
import numpy as np
import os
import io
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import wavfile

class LoadDatas:
    def __init__(self, directory) -> None:
        self.num = [48, 68]
        self.load_datas(directory)
        self.get_images()
        
    def load_datas(self, directory):
        paths = os.listdir(os.path.join(os.getcwd(), directory))
        for path in paths:
            p = os.path.join(os.getcwd(), directory, path)
            if path.endswith('.npy'):
                self.data = np.load(p)
            if path.endswith('.wav'):
                samplerate, data = wavfile.read(p)
            if path.endswith('.mp4'):
                self.cap = cv2.VideoCapture(p)
            
        
    def get_images(self):
        self.arr_img = []
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.dim_video = [cap_h, cap_w]
        for i in range(frame_count):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.arr_img.append(frame)
        self.cap.release()
        
class ReadLips(LoadDatas):
    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.fig = plt.figure(figsize=(15, 5))
        self.ax0 = self.fig.add_subplot(121)
        self.ax1 = self.fig.add_subplot(122)
        self.animation = FuncAnimation(self.fig, self.update, frames=range(len(self.arr_img)), interval=100)
        plt.show()
        
    def update(self, frame):
        # Mise à jour de la visualisation 3D
        pts = self.data[frame]
        d, f = self.num
        lips = pts[d:f]
        
        self.ax0.clear()
        img = self.arr_img[frame]
        self.ax0.imshow(img)
        x = [v[0] for v in pts]
        y = [v[1] for v in pts]
        self.ax0.scatter(x, y, s=2)
        x = [v[0] for v in lips]
        y = [v[1] for v in lips]
        self.ax0.plot(x, y, color='red')
        self.ax1.clear()
        

        x = [v[0] for v in pts]
        y = [v[1] for v in pts]
        self.ax1.scatter(x, y, s=2)
        x = [v[0] for v in lips]
        y = [v[1] for v in lips]
        self.ax1.plot(x, y, color='red')
        self.ax1.invert_yaxis()
        
        if frame == 1:
            self.fig.savefig('figure_frame_1.png')
        
        
       
        
if __name__=='__main__':
    directory = 'datas'
    ReadLips(directory)