import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import tau

from fourierImage import Fourier

class AnalyseLips(Fourier):
    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.show_2d()
        self.show_time()
        
    def show_2d(self):
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        lips = self.lips[0]
        x = [v[0] for v in lips]
        y = [v[1] for v in lips]

        for i in range(4):
            order = i+2
            self.fourier(order_y=order, show=False)
            pts = self.mat_3D[0]
            vx = [v[0] for v in pts]
            vy = [v[1] for v in pts]
            title = f'order {order}'
            ax[i].set_title(title)
            ax[i].plot(x, y)
            ax[i].plot(vx, vy)
            ax[i].invert_yaxis()
            
        plt.show()
        fig.suptitle('order_w : lissage du contour des lèvres', fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.8])  

        fig.savefig('order_w.png')
        
    def show_time(self):
        fig, ax = plt.subplots(4, 1, figsize=(15, 8))
        amp_init = self.get_values(self.lips)
        angle_init = range(len(amp_init))
        for i in range(4):
            order = i+7
            self.fourier(order_x=order, show=False)
            amp_dft = self.get_values(self.mat_3D)
            angle_dft = range(len(amp_dft))
            ax[i].plot(angle_init, amp_init)
            ax[i].plot(angle_dft, amp_dft)
            title = f'order {order}'
            ax[i].set_title(title)
            
        fig.suptitle('order_h : lissage dans le temps', fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  
        plt.show()
        fig.savefig('order_h.png')
            
        
    def get_values(self, mat):
        h, w = mat.shape[:2]
        result = []
        for i in range(h):
            maxi, mini = np.max(mat[i]), np.min(mat[i])
            value = maxi-mini
            result.append(maxi)
        return result
        
    
        
        
 
if __name__=='__main__':
    directory = 'datas'
    AnalyseLips(directory)