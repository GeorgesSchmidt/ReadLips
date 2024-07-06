import cv2
import numpy as np
import os
import io
from face.correctLips import *
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from math import tau

class Interpolation(CorrectLips):
    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.interpolate()
        #self.show_result()
        
        
    def interpolate(self):
        h, w = self.lips.shape[:2]
        arr_x, arr_y, arr_angle, arr_z = [], [], [], []
        for i in range(h):
            x, y, z = [], [], []
            for j in range(w):
                p = self.lips[i][j]
                x.append(p[0])
                y.append(p[1])
                z.append(i)
            angle = np.linspace(0, tau, len(x))
            xvals = np.linspace(0, tau, 50)
            vx = np.interp(xvals, angle, x)
            vy = np.interp(xvals, angle, y)
            vz = [i]*50
            arr_angle.append(xvals)
            arr_x.append(vx)
            arr_y.append(vy)
            arr_z.append(vz)
            
        res_x, a, z_x = self.getMatInterpol(arr_angle, arr_z, arr_x)
        res_y, a, z_y = self.getMatInterpol(arr_angle, arr_z, arr_y)
        h, w = res_x.shape[0], res_x.shape[1]
        self.matPoints = np.empty((h, w, 2), float)
        print('interpolation', self.matPoints.shape)
        
        for i in range(h):
            for j in range(w):
                x, y = res_x[i][j], res_y[i][j]
                self.matPoints[i][j] = [x, y]
                
        title = f'./datas/mat_interp.npy'
        data_path = os.path.join(os.getcwd(), 'datas', 'mat_interp.npy')
        np.save(title, self.matPoints)

                
    def getMatInterpol(self, arr_angle, arr_z, arr):
        A = np.array(arr_angle)
        Z = np.asarray(arr_z)

        arr_x = np.array(arr)
        values = arr_x.flatten()

        minA, maxA = np.min(A), np.max(A)
        minZ, maxZ = np.min(Z), np.max(Z)

        # a = np.arange(minA, maxA, 1.)
        a = np.linspace(minA, maxA, 50)
        z = np.linspace(minZ, maxZ, 326340) #for sound 326340
       
        points = np.stack((A.flatten(), Z.flatten()), axis=1)

        res = griddata(points, values, (a[None, :], z[:, None]), method='cubic')

        return res, a, z
        
    def show_result(self):
        h, w = self.lips.shape[:2]
        mat_x = np.zeros((h, w), float)
        mat_y = np.zeros((h, w), float)
        for i in range(h):
            for j in range(w):
                p = self.lips[i][j]
                mat_x[i][j] = p[0]
                mat_y[i][j] = p[1]
        
        h, w = self.matPoints.shape[:2]
        mat_x_interp = np.zeros((h, w), float)
        mat_y_interp = np.zeros((h, w), float)
        for i in range(h):
            for j in range(w):
                p = self.matPoints[i][j]
                mat_x_interp[i][j] = p[0]
                mat_y_interp[i][j] = p[1]
        fig = plt.figure(figsize=(15, 5))
        h, w = self.lips.shape[:2]
        ax0 = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122, projection='3d')
        self.plot_surface(mat_x_interp, ax0)
        self.plot_surface(mat_y_interp, ax1)
        # self.plot_mat(mat_x, ax0)
        # self.plot_mat(mat_y, ax1)
        # self.plot_mat(mat_x_interp, ax0, color='red', title='x variation')
        # self.plot_mat(mat_y_interp, ax1, color='red', title='y variation')
        ax0.set_box_aspect([10, 10, 5])
        ax1.set_box_aspect([10, 10, 5])
        plt.show()
        fig.savefig('interpolation.png')
        
    def plot_surface(self, mat, ax):
        h, w = mat.shape[:2]
        angle = np.linspace(0, tau, w)
        X, Y = np.meshgrid(angle, range(h))
        ax.plot_surface(X, Y, mat, cmap='viridis', alpha=0.4)
        
    def plot_mat(self, mat, ax, color='black', title=''):
        h, w = mat.shape[:2]
        angle = np.linspace(0, tau, w)
        time = np.linspace(0, len(self.lips), h)
        for i in range(h):
            x, y, z = [], [], []
            for j in range(w):
                v = mat[i][j]
                x.append(angle[j])
                y.append(time[i])
                z.append(v)
            ax.plot3D(x, y, z, color=color)
        ax.set_xlabel(r'Angle $2\pi$')
        ax.set_ylabel('images')
        ax.set_zlabel(title)
        ax.set_title(title)
                
 