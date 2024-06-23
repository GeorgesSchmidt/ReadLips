import cv2
import numpy as np
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import tau

class DetectFace:
    def __init__(self, path) -> None:
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.path = path
        self.read_video()
        self.interpolation()
        self.analyse()
        
    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        
        n = 0
        self.pts_3D = []
        self.arr_lips = []
        self.height = -1
        self.width = -1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                pred = self.get_pred(frame)
                face = None
                if pred is not None:
                    lips, pts, box = pred
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                    if self.height < h:
                        self.height = h
                    if self.width < w:
                        self.width = w
                    
                    hull = cv2.convexHull(lips, returnPoints=True)
                    arr = []
                    for [p] in hull:
                        x, y = (p[0]-box[0])/box[2], (p[1]-box[1])/box[3]
                        arr.append([x, y])
                    
                    self.arr_lips.append(arr)
                    
                    cv2.polylines(frame, [hull], True, (255, 255, 0), 4)
                    
                    for p in pts:
                        cv2.circle(frame, p, 2, (255, 255, 255), -1)
                    for p in lips:
                        cv2.circle(frame, p, 5, (255, 0, 255), -1)
                    
                cv2.imshow('', frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                if n == frame_count-1:
                    break
                n += 1
        cap.release()
        cv2.destroyAllWindows()
    
        
    def get_pred(self, frame):
        pred = self.app.get(frame)
        if len(pred) == 1:
            pred = pred[0]
            box = np.array(pred['bbox']).astype(int)
            pts = np.array(pred['landmark_2d_106']).astype(int)
            lips =  pts[52:72]
            return lips, pts, box
        
    def interpolation(self):
        h = len(self.arr_lips)
        w = 20
        self.mat_lips = np.zeros((h, w, 2), float)
        for ind, line in enumerate(self.arr_lips):
            x = [v[0] for v in line]
            y = [v[1] for v in line]
            angle = np.linspace(0, tau, len(x))
            xvals = np.linspace(0, tau, w)
            vx = np.interp(xvals, angle, x)
            vy = np.interp(xvals, angle, y)
            self.mat_lips[ind] = np.column_stack((vx, vy))
            
            
    def analyse(self):
        h, w = self.mat_lips.shape[:2]
        mat_x = np.zeros((h, w), float)
        mat_y = np.zeros((h, w), float)
        for i in range(h):
            for j in range(w):
                p = self.mat_lips[i][j]
                mat_x[i][j] = p[1]
                mat_y[i][j] = p[0]
                
        coef_x, idft_x = self.calculCoef2D(mat_x, order=5)
        coef_y, idft_y = self.calculCoef2D(mat_y, order=4)
        mat_3D = np.zeros((h, w, 2), float)
        for i in range(h):
            for j in range(w):
                x = idft_x[i][j]
                y = idft_y[i][j]
                mat_3D[i][j] = [y, x]
        fig = plt.figure(figsize=(15, 5))
        ax0 = fig.add_subplot(131, projection='3d')
        ax1 = fig.add_subplot(132, projection='3d')
        ax2 = fig.add_subplot(133, projection='3d')
        ax2.set_title('Lips amplitude during time')
        
        self.plot_mat(mat_x, ax0, title='X variation')
        self.plot_mat(mat_y, ax1, title='Y variation')
        self.plot_surface(idft_x, ax0)
        self.plot_surface(idft_y, ax1)
        self.plot_lips(self.mat_lips, ax2, color='red')
        self.plot_lips(mat_3D, ax2, color='black')
        fig.savefig('3d_plot_mounth.png')
        plt.show()
        
    def plot_lips(self, mat, ax, color):
        h, w = mat.shape[:2]
        for i in range(h):
            x, y, z = [], [], []
            for j in range(w):
                p = mat[i][j]
                x.append(p[0])
                y.append(i)
                z.append(p[1])
            ax.plot3D(x, y, z, color)
        ax.set_xlabel('lips shape')
        ax.set_ylabel('time (images)')
        ax.set_zlabel('lips amplitude')
                
            
    def plot_surface(self, mat, ax):
        h, w = mat.shape[:2]
        angle = np.linspace(0, tau, w)
        X, Y = np.meshgrid(angle, range(h))
        ax.plot_surface(X, Y, mat, cmap='viridis', alpha=0.4)
        
    def plot_mat(self, mat, ax, title):
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
            ax.plot3D(x, y, z)
        ax.set_xlabel(r'Angle $2\pi$')
        ax.set_ylabel('images')
        ax.set_zlabel(title)
       
        

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
        print('coef', idft_result.shape)

        nonzero_indices = np.nonzero(mask[:, :, 0])
        print(fft_ifft_shift.shape)

        # Extraire les coefficients non nuls de la DFT
        nonzero_coefficients = dft_result[nonzero_indices]
        modified_coefficients = [(x, y, fft_shift[x, y, 0] + 1j * fft_shift[x, y, 1]) for x, y in zip(nonzero_indices[0], nonzero_indices[1])]
        print(len(modified_coefficients))

        return fft_shift, idft_result
    
    
    def plot_lips1(self):
        self.fig = plt.figure(figsize=(15, 5))
        self.ax = self.fig.add_subplot(211)
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        h, w = self.mat_lips.shape[:2]
        self.animation = FuncAnimation(self.fig, self.update, frames=range(h), interval=200)
        plt.show()
        
    def update(self, frame):
        self.ax.clear()
        pts = self.mat_lips[frame]
        x, y = [v[0] for v in pts], [v[1] for v in pts]
        self.ax.scatter(x, y)
        
  
       
        
if __name__=='__main__':
    path = 'output_video.mp4'
    DetectFace(path)
