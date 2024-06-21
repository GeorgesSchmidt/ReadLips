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
        self.arr_lips = []
        self.pts_3D = []
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
            
        cv2.destroyAllWindows()
        
    def get_pred(self, frame):
        pred = self.app.get(frame)
        for p in pred:
            box = np.array(p['bbox']).astype(int)
            pts_3D = p['landmark_3d_68']
            self.pts_3D.append(pts_3D)
            pts = np.array(p['landmark_2d_106']).astype(int)
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            
            
            lips =  pts[52:72]
            lips = cv2.convexHull(lips, returnPoints=True)
            cv2.polylines(frame, [lips], True, (255, 255, 0), 5)
            
            
            zero = pts[72]
            cv2.circle(frame, zero, 10, (0, 255, 0), -1)
            sub = []
            for [p] in lips:
                x, y = p[0]-zero[0], p[1]-zero[1]
                x /= frame.shape[1]
                y /= frame.shape[0]
                sub.append([x, y])
            self.arr_lips.append(sub)
                
            # rep = [pts[33], pts[72]]
            # for p in rep:
            #     cv2.circle(frame, p, 5, (0, 0, 255),)
            
            # left_eye = pts[33:52]
            
                
            # rep1 = pts[72:75]
            
            # rep2 = pts[76:79]
            
            
        
    
        
        return frame
            
            
  
       
        
if __name__=='__main__':
    path = 'output_video.mp4'
    DetectFace(path)
