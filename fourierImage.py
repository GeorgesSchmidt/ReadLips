import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import tau

from interpolation import Interpolation

class Fourier(Interpolation):
    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.fourier()
        
    def fourier(self, order_x=5, order_y=5):
        h, w = self.matPoints.shape[:2]
        print('order', order_x, order_y)
        mat_x = np.zeros((h, w), float)
        mat_y = np.zeros((h, w), float)
        for i in range(h):
            for j in range(w):
                p = self.matPoints[i][j]
                mat_x[i][j] = p[0]
                mat_y[i][j] = p[1]
                
        coef_x, idft_x = self.calculCoef2D(mat_x, order=order_x)
        coef_y, idft_y = self.calculCoef2D(mat_y, order=order_y)
        self.mat_fourier = [coef_x, coef_y]
        self.mat_3D = np.zeros((h, w, 2), float)
        for i in range(h):
            for j in range(w):
                x = idft_x[i][j]
                y = idft_y[i][j]
                self.mat_3D[i][j] = [x, y]
        # fig = plt.figure(figsize=(15, 5))
        # ax0 = fig.add_subplot(121, projection='3d')
        # ax1 = fig.add_subplot(122, projection='3d')
        # self.plot_matrice(mat_x, ax0, title='X variation')
        # self.plot_matrice(mat_y, ax1, title='Y variation')
        # self.plot_surface(idft_x, ax0)
        # self.plot_surface(idft_y, ax1)
        # plt.show()
        # fig.savefig('fourier.png')
        
    def calculCoef2D(self, img, order=4):
        dft_result = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Repositionner le composant de fréquence zéro au milieu du spectre
        shift = np.fft.fftshift(dft_result)
        row, col = img.shape
        center_row, center_col = row // 2, col // 2

        # Créer un masque avec un carré centré de 1s
        mask = np.zeros((row, col, 2), np.uint8)
        rad = order
        mask[center_row - rad:center_row + rad, center_col - rad:center_col + rad] = 1

        # Appliquer le masque à la DFT
        fft_shift = shift * mask

        # Inverser le décalage pour obtenir la DFT originale
        fft_ifft_shift = np.fft.ifftshift(fft_shift)

        # Calculer l'inverse de la transformée de Fourier discrète 2D avec le masque
        idft_result = cv2.idft(fft_ifft_shift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        #print('coef', idft_result.shape)

        nonzero_indices = np.nonzero(mask[:, :, 0])
        #print(fft_ifft_shift.shape)

        # Extraire les coefficients non nuls de la DFT
        nonzero_coefficients = dft_result[nonzero_indices]
        modified_coefficients = [(x, y, fft_shift[x, y, 0] + 1j * fft_shift[x, y, 1]) for x, y in zip(nonzero_indices[0], nonzero_indices[1])]
        #print(len(modified_coefficients))

        return fft_shift, idft_result
    
    
    def plot_surface(self, mat, ax):
        h, w = mat.shape[:2]
        angle = np.linspace(0, tau, w)
        X, Y = np.meshgrid(angle, range(h))
        ax.plot_surface(X, Y, mat, cmap='viridis', alpha=0.4)
        
    def plot_matrice(self, mat, ax, title):
        ax.set_title(title)
        h, w = mat.shape[:2]
        angle = np.linspace(0, tau, w)
        for i in range(h):
            x, y, z = [], [], []
            for j in range(w):
                v = mat[i][j]
                x.append(angle[j])
                y.append(i)
                z.append(v)
            ax.plot3D(x, y, z, color='red')
        ax.set_xlabel(r'Angle $2\pi$')
        ax.set_ylabel('images')
        ax.set_zlabel(title)

        
   
if __name__=='__main__':
    directory = 'datas'
    Fourier(directory)