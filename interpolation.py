import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from math import tau

from InterviewLips import DetectFace
from InterviewLips import Point3DRotation

class Interpolate(DetectFace):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.normalyse_values()
        self.interpolate()
        self.plot_3d()
        
    def normalyse_values(self):
        for ind, lips in enumerate(self.lips):
            z = [v[0] for v in lips]
            x = [v[1] for v in lips]
            y = [v[2] for v in lips]
            rotator = Point3DRotation(x, y, z)
            rotator.rotate_90_deg_x()
            x, y, z = rotator.get_points()
            # Normaliser les points entre -1 et 1
            x = self.normalize(x)
            y = self.normalize(y)
            self.lips[ind] = np.array(list(zip(x, y)))
            
    def normalize(self, data):
        data_min = np.min(data)
        data_max = np.max(data)
        normalized_data = 2 * (data - data_min) / (data_max - data_min) - 1
        return normalized_data
    
    def interpolate(self):
        h, w = len(self.lips), 100
        self.mat_lips = np.zeros((h, w, 3), float)
        arr_x, arr_y, arr_angle, arr_z = [], [], [], []
        for i, lips in enumerate(self.lips):
            x = [v[0] for v in lips]
            y = [v[1] for v in lips]
            angle = np.linspace(0, tau, len(x))
            xvals = np.linspace(0, tau, 100)
            vx = np.interp(xvals, angle, x)
            vy = np.interp(xvals, angle, y)
            vz = [i]*w
            self.mat_lips[i] = np.array(list(zip(vx, vy, vz)))
            arr_angle.append(xvals)
            arr_x.append(vx)
            arr_z.append(vz)
            arr_y.append(vy)

        res_x, a, z_x = self.getMatInterpol(arr_angle, arr_z, arr_x)
        res_y, a, z_y = self.getMatInterpol(arr_angle, arr_z, arr_y)
        h, w = res_x.shape[0], res_x.shape[1]
        self.matPoints = np.empty((h, w, 3), float)
        print('interpolation', self.matPoints.shape)
        
        for i in range(h):
            z = z_x[i]
            line = []
            for j in range(w):
                x, y = res_x[i][j], res_y[i][j]
                self.matPoints[i][j] = [x, y, z]

        
            
    def getMatInterpol(self, arr_angle, arr_z, arr):
        A = np.array(arr_angle)
        Z = np.asarray(arr_z)

        arr_x = np.array(arr)
        values = arr_x.flatten()

        minA, maxA = np.min(A), np.max(A)
        minZ, maxZ = np.min(Z), np.max(Z)

        # a = np.arange(minA, maxA, 1.)
        a = np.linspace(minA, maxA, 100)
        z = np.linspace(minZ, maxZ, 100)
       
        points = np.stack((A.flatten(), Z.flatten()), axis=1)

        res = griddata(points, values, (a[None, :], z[:, None]), method='cubic')

        return res, a, z

    def plot_3d(self):
        fig = plt.figure(figsize=(15, 5))
        self.ax0 = fig.add_subplot(121, projection='3d')
        self.ax1 = fig.add_subplot(122, projection='3d')
        self.plot_mat(self.mat_lips, color='green')
        self.plot_mat(self.matPoints, color='red')
        plt.show()
    
    def plot_mat(self, mat, color='black'):
        h, w = mat.shape[:2]
        angle = np.linspace(0, tau, w)
        for i in range(h):
            x, y, z = [], [], []
            for j in range(w):
                p = mat[i][j]
                x.append(angle[j])
                y.append(p[2])
                z.append(p[0])
            self.ax0.plot3D(x, y, z, color=color)
            x, y, z = [], [], []
            for j in range(w):
                p = mat[i][j]
                x.append(angle[j])
                y.append(p[2])
                z.append(p[1])
            self.ax1.plot3D(x, y, z, color=color)
        


if __name__=='__main__':
    path = 'ShortThomas.mp4'
    Interpolate(path)

