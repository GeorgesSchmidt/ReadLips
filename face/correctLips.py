import cv2
import numpy as np
import os
import io
from face.loadDataFace import *
import matplotlib.pyplot as plt
from math import tau

class CorrectLips(LoadDatas):
    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.get_lips()
        self.plot_lips()
        
    def get_lips(self):
        h, w = self.data.shape[:2]
        d, f = self.num
        self.lips = np.zeros((h, 20, 2), float)
        for i in range(h):
            lips = self.data[i][d:f]
            for j, p in enumerate(lips):
                x, y = p[:2]
                self.lips[i][j] = [x, y]
        
        
        h, w = self.lips.shape[:2]
        print('lips', h, w)
        for i in range(h):
            x, y = [], []
            for j in range(w):
                p = self.lips[i][j]
                x.append(p[0])
                y.append(p[1])
            
            
            c_x = np.mean([x[0], x[6]])
            c_y = np.mean([y[0], y[6]])
            
            x = np.array(x) - np.array(c_x)
            y = np.array(y) - np.array(c_y)
            
            p_x, p_y = x[0], y[0]
            
            angle_rad = np.arctan2(p_y, p_x)
            rotation_angle = -angle_rad 
            rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])

            rotated_x = x * np.cos(rotation_angle) - y * np.sin(rotation_angle)
            rotated_y = x * np.sin(rotation_angle) + y * np.cos(rotation_angle)
            
            for j in range(w):
                self.lips[i][j] = [rotated_x[j], rotated_y[j]]
                
        
                
    def plot_lips(self):
        h, w = self.lips.shape[:2]
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(h):
            x, y, z = [], [], []
            for j in range(w):
                p = self.lips[i][j]
                x.append(p[0])
                y.append(p[1])
                z.append(i)
            ax.plot3D(x, z, y, linewidth=0.3)
        #plt.show()
        plt.savefig('lips_3d.png')
