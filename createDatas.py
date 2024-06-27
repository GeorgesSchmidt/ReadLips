import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import io
from moviepy.editor import VideoFileClip
import imageio
from tqdm import tqdm
    
class DetectFace:
    def __init__(self, input_path, output_path) -> None:
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.num = [48, 68]
        self.path = os.path.join(os.getcwd(), 'videos', input_path)
        start = self.check_data()
        if start:
            self.get_sound(title=output_path)
            self.read_video()
            data_path = os.path.join(os.getcwd(), 'datas', 'pts_face_thomas.npy')
            np.save(data_path, self.pts)
        
    def check_data(self):
        paths = os.listdir(os.path.join(os.getcwd(), 'videos'))
        title = self.path.split('/')[-1]
        if title in paths:
            print('videos is doawnloaded')
            return True
        else:
            print('this video not exist')
            return False
        
    def read_video(self):
        cap = cv2.VideoCapture(self.path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
        cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.dim = [cap_h, cap_w]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour le format MP4
        video_path = os.path.join(os.getcwd(), 'datas', 'thomas_face.mp4')
        out = cv2.VideoWriter(video_path, fourcc, cap_fps, (cap_w, cap_h))

        pbar = tqdm(total=frame_count, desc="Traitement des frames")
        num = 0
        self.pts = []
        for i in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                pts = self.get_pred(frame)
                self.pts.append(pts)
                out.write(frame)
                # if len(pts)>0:
                #     self.pts.append(pts)
                #     d, f = self.num
                #     self.lips = pts[d:f]
                #     for ind, p in enumerate(pts):
                #         x, y = p[:2]
                #         color = (255, 255, 255)
                #         if ind>=d and ind<f:
                #             color = (0, 0, 255)
                #         cv2.circle(frame, (x, y), 2, color, -1)
                    
                #     num += 1
            pbar.update(1)
            
        cap.release()
        out.release()
        pbar.close()


    def get_pred(self, frame):
        pred = self.app.get(frame)
        pts = []
        if len(pred) == 1:
            pred = pred[0]
            pts = np.array(pred['landmark_3d_68']).astype(int)
        return pts

    def get_sound(self, title='_.mp4'):
        video = VideoFileClip(self.path)
        audio = video.audio
        audio_path = os.path.join(os.getcwd(), 'datas', title)
        audio.write_audiofile(audio_path, codec='pcm_s16le')
        

    

if __name__=='__main__':
    path = 'short_thoma.mp4'
    audio = 'short_thom_audio.wav'
    DetectFace(input_path=path, output_path=audio)
