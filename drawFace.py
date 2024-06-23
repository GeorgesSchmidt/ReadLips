import cv2
import numpy as np
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt

class DrawLandmarks:
    def __init__(self, path) -> None:
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.image = cv2.imread(path)
        self.draw_image()
    
    def draw_image(self):
        lips, pts, box = self.get_pred(self.image)
        x, y, w, h = box
        cv2.rectangle(self.image, (x, y), (w, h), (0, 255, 0), 4)
        for p in lips:
            cv2.circle(self.image, p, 4, (0, 0, 255), -1)
        mouth = cv2.convexHull(lips, returnPoints=True)
        cv2.polylines(self.image, [mouth], True, (255, 255, 0), 4)
        cv2.imshow('', self.image)
        cv2.waitKey(0)
        title = 'result_img.jpg'
        cv2.imwrite(title, self.image)
        
    
            
    def get_pred(self, frame):
        pred = self.app.get(frame)
        if len(pred) == 1:
            pred = pred[0]
            box = np.array(pred['bbox']).astype(int)
            pts = np.array(pred['landmark_2d_106']).astype(int)
            lips =  pts[52:72]
            return lips, pts, box

if __name__=='__main__':
    path = 'face.jpg'
    DrawLandmarks(path)