import cv2
import numpy as np
import os
import io
from math import tau
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
        
class Analyse(LoadDatas):
    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.plot_values()
        
    def plot_values(self):
        data = self.data[0]
        d, f = self.num
        data = data[d:f]
        angle = np.linspace(0, tau, len(data))
        x = [v[0] for v in data]
        y = [v[1] for v in data]
        c_x, c_y = np.mean(x), np.mean(y)
        x -= np.array(c_x)
        y -= np.array(c_y)
        
        ind, _ = self.get_index_deriv(x)
        ind = ind[0]
        x_lips, y_lips = x[ind:], y[ind:]
        _, ind_y = self.get_index_deriv(y)
        print(ind_y)
        
        
        fig = plt.figure(figsize=(15, 5))
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)
        ax0.set_title('variation en x')
        ax1.set_title('variation en y')
        ax2.set_title('x et y = lips')
        ax0.plot(angle, x)
        ax0.axvline(x=angle[ind], color='red', linestyle='--')
        ax0.set_xlabel(r'Angle $2\pi$')
        
        ax1.plot(angle, y)
       
        ax1.set_xlabel(r'Angle $2\pi$')
        #ax1.scatter(angle[ind_y], y[ind_y], s=20)
        
        ax2.plot(x, y)
        ax2.plot(x_lips, y_lips, color='red')
        ax2.invert_yaxis()
        plt.show()
        fig.savefig('coordLips.png')
        
    def get_index_deriv(self, x):
        first_derivative = np.diff(x)
        
        minima_indices = np.where((first_derivative[:-1] < 0) & (first_derivative[1:] > 0))[0] + 1
        maxima_indices = np.where((first_derivative[:-1] > 0) & (first_derivative[1:] < 0))[0] + 1
        total = np.sort(np.concatenate((minima_indices, maxima_indices)))
        
        return minima_indices, total

        
        
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
    Analyse(directory)