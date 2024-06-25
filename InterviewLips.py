import cv2
import numpy as np
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
import io
from moviepy.editor import VideoFileClip
import imageio
from tqdm import tqdm

class Point3DRotation:
    def __init__(self, x, y, z):
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)

    def rotate_90_deg_x(self):
        """Rotate points 90 degrees around the X axis."""
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        self._apply_rotation(rotation_matrix)

    def rotate_90_deg_y(self):
        """Rotate points 90 degrees around the Y axis."""
        rotation_matrix = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        self._apply_rotation(rotation_matrix)

    def rotate_90_deg_z(self):
        """Rotate points 90 degrees around the Z axis."""
        rotation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        self._apply_rotation(rotation_matrix)

    def _apply_rotation(self, rotation_matrix):
        """Apply the given rotation matrix to the points."""
        points = np.array([self.x, self.y, self.z])
        rotated_points = np.dot(rotation_matrix, points)
        self.x, self.y, self.z = rotated_points[0], rotated_points[1], rotated_points[2]

    def get_points(self):
        """Return the current points as lists."""
        return self.x.tolist(), self.y.tolist(), self.z.tolist()
    
    

    
class DetectFace:
    def __init__(self, path) -> None:
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.path = os.path.join(os.getcwd(), 'videos', path)
        self.get_sound()
        self.read_video()
        
    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #frame_count = 50
        pbar = tqdm(total=frame_count, desc="Traitement des frames")
        n = 0
        self.arr_img, self.pts_3D, self.lips = [], [], []
        for i in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                pred = self.get_pred(frame)
                if pred is not None:
                    lips, pts, box, pts_3D, lips_3d = pred
                    self.pts_3D.append(pts_3D)
                    self.lips.append(lips_3d)
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                    for p in lips:
                        x, y = p
                        cv2.circle(frame, p, 5, (0, 0, 255), -1)
                    for p in pts:
                        cv2.circle(frame, p, 2, (155, 255, 255), -1)
                    self.arr_img.append(frame)
                    pbar.update(1)
                n += 1
        cap.release()
        pbar.close()
        

    def get_pred(self, frame):
        pred = self.app.get(frame)
        if len(pred) == 1:
            pred = pred[0]
            box = np.array(pred['bbox']).astype(int)
            pts = np.array(pred['landmark_2d_106']).astype(int)
            pts_3d = np.array(pred['landmark_3d_68']).astype(float)
            pose = np.array(pred['pose']).astype(float)
            d = 48
            f = 68
            lips =  pts[d:f]
            lips_3d = pts_3d[d:f]
            return lips, pts, box, pts_3d, lips_3d

    def get_sound(self):
        video = VideoFileClip(self.path)
        audio = video.audio
        audio_path = os.path.join(os.getcwd(), 'videos', 'short_output_audio.wav')
        audio.write_audiofile(audio_path, codec='pcm_s16le')
        
    


class CreateAnim(DetectFace):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.arr_figure = []
        self.fig = plt.figure(figsize=(15, 5))
        self.ax0 = self.fig.add_subplot(121)
        self.ax1 = self.fig.add_subplot(122)
       
        self.animation = FuncAnimation(self.fig, self.update, frames=range(len(self.arr_img)), interval=200)
        plt.show()
        print('arr figure', len(self.arr_figure))
        
    
    def update(self, frame):
        self.ax0.clear()
        title = f'image {frame} / {len(self.arr_img)}'
        img = self.arr_img[frame]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax0.imshow(img)
        self.ax0.set_title(title)  # Add title to the image subplot
        
        self.ax1.clear()
        points = self.pts_3D[frame]
        self.plot_points(points)
       

        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')  # Sauvegarde la figure dans le tampon mémoire
        buffer.seek(0)  # Déplace le curseur au début du tampon

        data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)  # Convertit le tampon en tableau numpy
        image = cv2.imdecode(data, 1)
        self.arr_figure.append(image)
        if frame == 50:
            self.animation.event_source.stop()
            plt.close()


    def normalize(self, data):
        data_min = np.min(data)
        data_max = np.max(data)
        normalized_data = 2 * (data - data_min) / (data_max - data_min) - 1
        return normalized_data
    
    def plot_points(self, points):
        z = [v[0] for v in points]
        x = [v[1] for v in points]
        y = [v[2] for v in points]

        rotator = Point3DRotation(x, y, z)
        rotator.rotate_90_deg_x()
        x, y, z = rotator.get_points()
        # Normaliser les points entre -1 et 1
        x = self.normalize(x)
        y = self.normalize(y)
        z = self.normalize(z)
       

        self.ax1.scatter(y, x, color='black', s=5)
        
        d = 48
        f = 68
        x = x[d:f]
        y = y[d:f]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        self.ax1.plot(y, x, color='red')
        self.ax1.invert_yaxis()
        


if __name__=='__main__':
    path = 'ShortThomas.mp4'
    DetectFace(path)
