import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import tau

from fourierImage import Fourier

class AnalyseLips(Fourier):
    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.show_2d()
        
    def show_2d(self):
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        lips = self.lips[0]
        x = [v[0] for v in lips]
        y = [v[1] for v in lips]

        for i in range(4):
            order = i+2
            self.fourier(order_x=order, order_y=order)
            pts = self.mat_3D[0]
            vx = [v[0] for v in pts]
            vy = [v[1] for v in pts]
            title = f'order {order}'
            ax[i].set_title(title)
            ax[i].plot(x, y)
            ax[i].plot(vx, vy)
            ax[i].invert_yaxis()
            
        plt.show()
        fig.savefig('orderFourier.png')
        
    def get_values(self, mat):
        h, w = mat.shape[:2]
        for i in range(h):
            maxi = np.max(mat[i])
        
    
        
        
 
if __name__=='__main__':
    directory = 'datas'
    AnalyseLips(directory)