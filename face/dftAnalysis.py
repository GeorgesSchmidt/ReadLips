import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import tau
import os

class AnalyseMat:
    def __init__(self) -> None:
        path = './datas/mat_interp.npy'
        self.mat = np.load(path)
        print('data', self.mat.shape)
        self.order = [5, 50000]
        #self.show_3d()
        self.fourier()
        self.get_datas()
        
        
    def fourier(self):
        mat_x, mat_y = cv2.split(self.mat)
        h, w = mat_x.shape
        mid_h = h//2
        mid_w = w//2
        ord_x, ord_y = self.order
        mask = np.zeros((h, w, 2), float)
        x0, x1 = mid_w-ord_x, mid_w+ord_x
        y0, y1 = mid_h-ord_y, mid_h+ord_y
        mask[y0:y1, x0:x1] = 1
       
        shift_x = self.calculCoef2D(mat_x)
        shift_y = self.calculCoef2D(mat_y)
        ifft_x = shift_x*mask
        ifft_y = shift_y*mask
        self.mat_four_x = ifft_x[y0:y1, x0:x1]
        self.mat_four_y = ifft_y[y0:y1, x0:x1]
        print('mat_four_x', self.mat_four_x.shape)
        
        
        fig = plt.figure(figsize=(15, 5))
        ax0 = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122, projection='3d')
        ax0.set_title('magnitude x not filtered')
        ax1.set_title('magnitude x filtered (with mask)')
        self.plot_mag(shift_x, ax0)
        self.plot_mag(ifft_x, ax1)
        self.plot_mask(ax1)
    
        #plt.show()
        plt.savefig('./pictures/magnitude_x.png')
        
    def plot_mask(self, ax):
        h, w = self.mat.shape[:2]
        mid_h = h//2
        mid_w = w//2
        ord_x, ord_y = self.order
        x0, x1 = mid_w-ord_x, mid_w+ord_x
        y0, y1 = mid_h-ord_y, mid_h+ord_y
        x = [x0, x1, x1, x0, x0]
        y = [y0, y0, y1, y1, y0]
        z = [0, 0, 0, 0, 0]
        ax.plot3D(y, x, z, color='black', linewidth=2)
        
        
    def calculCoef2D(self, img):
        dft_result = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        shift = np.fft.fftshift(dft_result)
        return shift

    def get_datas(self):
        h, w = self.mat.shape[:2]
        mat_x, mat_y = cv2.split(self.mat)
        mid_h = h//2
        mid_w = w//2
        ord_x, ord_y = self.order
        x0, x1 = mid_w-ord_x, mid_w+ord_x
        y0, y1 = mid_h-ord_y, mid_h+ord_y
        four_x = np.zeros((h, w, 2), float)
        four_x[y0:y1, x0:x1] = self.mat_four_x
        res_x = self.getImageWithCoef(four_x)
        
        four_y = np.zeros((h, w, 2), float)
        four_y[y0:y1, x0:x1] = self.mat_four_y
        res_y = self.getImageWithCoef(four_y)
        
        mat_res = cv2.merge((res_x, res_y))
        
        fig = plt.figure(figsize=(15, 5))
        ax0 = fig.add_subplot(131, projection='3d')
        ax1 = fig.add_subplot(132, projection='3d')
        ax2 = fig.add_subplot(133, projection='3d')
        self.plot_mat(mat_x, ax0, title='x variation')
        self.plot_surface(res_x, ax0)
        self.plot_mat(mat_y, ax1, title='y variation')
        self.plot_surface(res_y, ax1)
        self.plot_lips(mat_res, ax2)
        #plt.show()
        plt.savefig('./pictures/result_DFT.png')
        
        
    def plot_surface(self, mat, ax):
        h, w = mat.shape[:2]
        angle = np.linspace(0, tau, w)
        X, Y = np.meshgrid(angle, range(h))
        ax.plot_surface(X, Y, mat, cmap='viridis', alpha=0.8)
 
        
    
    def getImageWithCoef(self, coef):
        fft_ifft_shift = np.fft.ifftshift(coef)
        result = cv2.idft(fft_ifft_shift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        
        return result

    def plot_mag(self, shift, ax):
        mat = cv2.magnitude(shift[:,:,0], shift[:,:,1])
        h, w = mat.shape
        
        for i in range(0, h, 5000):
            x, y, z = [], [], []
            for j in range(w):
                v = mat[i][j]
                x.append(i)
                y.append(j)
                z.append(v)
            ax.plot3D(x, y, z)

    def show_3d(self):
        fig = plt.figure(figsize=(15, 5))
        ax0 = fig.add_subplot(131, projection='3d')
        ax1 = fig.add_subplot(132, projection='3d')
        ax2 = fig.add_subplot(133, projection='3d')
        mat_x, mat_y = cv2.split(self.mat)
        self.plot_mat(mat_x, ax0, title='x variation')
        self.plot_mat(mat_y, ax1, title='y variation')
        self.plot_lips(self.mat)
        #plt.show()
        plt.savefig('./pictures/split_lips.png')
        
    def plot_derivate(self, mat, ax):
        h, w = mat.shape
        angle = np.linspace(0, tau, w)
        arr_x, arr_y, arr_z = [], [], []
        for i in range(0, h, 1000):
            x, y, z = [], [], []
            for j in range(w):
                v = mat[i][j]
                x.append(angle[j])
                y.append(i)
                z.append(v)
            pics, cuvs = self.get_indexes(z)
            print(len(cuvs))
            
        
    def plot_mat(self, mat, ax, color='black', title='variation'):
        ax.set_title(title)
        h, w = mat.shape
        angle = np.linspace(0, tau, w)
        for i in range(0, h, 1000):
            x, y, z = [], [], []
            for j in range(w):
                v = mat[i][j]
                x.append(angle[j])
                y.append(i)
                z.append(v)
            ax.plot3D(x, y, z, linewidth=0.3)
        ax.set_xlabel('angle')
        ax.set_ylabel(f'{h} images')
        ax.set_zlabel(title)
       

                
    def plot_lips(self, mat, ax):
        h, w = mat.shape[:2]
        ax.set_title('lips')
        w = 30
        for i in range(0, h, 5000):
            x, y, z = [], [], []
            for j in range(w):
                p = self.mat[i][j]
                x.append(p[0])
                y.append(p[1])
                z.append(i)
            ax.plot3D(x, z, y, color='red', linewidth=0.3)
            
    def get_indexes(self, values):
        derivative = np.diff(values)
        sign_changes = np.diff(np.sign(derivative))
        peaks = np.where(sign_changes == -2)[0] + 1
        troughs = np.where(sign_changes == 2)[0] + 1

        return peaks, troughs

        
        
if __name__=='__main__':
    AnalyseMat()
        
        
        