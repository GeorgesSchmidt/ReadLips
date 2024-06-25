import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import tau

from fourier import Fourier

class Anim(Fourier):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.fig = plt.figure(figsize=(15, 5))
        self.ax = self.fig.add_subplot(111)
        
       
        self.animation = FuncAnimation(self.fig, self.update, frames=range(len(self.matPoints)), interval=200)
        plt.show()
        
    def update(self, frame):
        self.ax.clear()
        points = self.matPoints[frame]
        y = [v[0] for v in points]
        x = [v[1] for v in points]
        x.append(x[0])
        y.append(y[0])
        self.ax.plot(x, y)
        points = self.mat_3D[frame]
        y = [v[0] for v in points]
        x = [v[1] for v in points]
        x.append(x[0])
        y.append(y[0])
        self.ax.plot(x, y)
        self.ax.axhline(y=0, color='k', linewidth=1) 
        self.ax.axvline(x=0, color='k', linewidth=1)
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.invert_yaxis()
        
    
        



if __name__=='__main__':
    path = 'ShortThomas.mp4'
    Anim(path)
