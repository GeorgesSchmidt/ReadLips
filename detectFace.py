import cv2
import numpy as np
from insightface.app import FaceAnalysis

class DetectFace:
    def __init__(self, path) -> None:
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.path = path
        self.read_video()
        
    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.mat_lips = np.zeros((frame_count, 20, 2), int)
        n = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                img = self.get_pred(frame)
                cv2.imshow('', img)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                if n == frame_count-1:
                    break
                n += 1
        cap.release()
            
        cap.release()
        
    def get_pred(self, frame):
        h, w = frame.shape[:2]
        black = np.zeros((h, w, 1), dtype=np.uint8)
        pred = self.app.get(frame)
        print(pred)
        for p in pred:
            box = np.array(p['bbox']).astype(int)
            pts = np.array(p['landmark_2d_106']).astype(int)
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            
            
            lips =  pts[52:72]
            for p in lips:
                cv2.circle(black, p, 12, (255), -1)
                
            rep = [pts[33], pts[72]]
            for p in rep:
                cv2.circle(frame, p, 5, (0, 0, 255), -1)
            
            left_eye = pts[33:52]
            
                
            rep1 = pts[72:75]
            
            rep2 = pts[76:79]
            
            
        
        contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==1:
            contour = contours[0]
            hull = cv2.convexHull(contour, returnPoints=True)
            cv2.polylines(frame, [contour], True, (0, 0, 255), 5)
            cv2.polylines(frame, [hull], True, (0, 255, 255), 5)
        
        return frame
            
            
  
       
        
if __name__=='__main__':
    path = 'output_video.mp4'
    DetectFace(path)
