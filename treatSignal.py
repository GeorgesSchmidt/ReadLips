import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from speechReco import SpeechRecognizer
from detectFace import DetectFace

class TreatSignal:
    def __init__(self, audio_file, video_file):
        self.video = DetectFace(video_file)
        self.sound = SpeechRecognizer(audio_file)
        self.recognized_text = self.sound.recognize_from_audio_file()
        self.values = self.sound.plot_audio_waveform()
        print('values', self.values)
        self.fig = plt.figure(figsize=(15, 5))
        
        # # Ajout du subplot pour le texte reconnu avec une barre de défilement
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title(self.recognized_text)
        
        
        # # Ajout du subplot pour la visualisation 3D
        self.ax0 = self.fig.add_subplot(212, projection='3d')

        # # Configuration de l'animation 3D
        self.time = len(self.video.pts_3D)
        self.animation = FuncAnimation(self.fig, self.update, frames=range(len(self.video.pts_3D)), interval=200)
        self.ax0.set_box_aspect(aspect=(1, 1, 1))
        
        plt.show()
        
    def update(self, frame):
        # Mise à jour de la visualisation 3D
        self.ax0.clear()
        pts = self.video.pts_3D[frame]
        x = [v[0] for v in pts]
        y = [v[1] for v in pts]
        z = [v[2] for v in pts]
        self.ax0.scatter3D(x, y, z)
        
        # Mise à jour du texte reconnu avec la barre de défilement
        
        self.ax1.clear()
        self.ax1.set_title(self.recognized_text)
        time, samples = self.values
        self.ax1.plot(time, samples)
        
        self.ax1.axvline(0, color='red', linestyle='--')  # Exemple de ligne horizontale juste au-dessus du texte
    
        self.fig.canvas.draw_idle()  # Rafraîchir la figure
        
if __name__ == '__main__':
    audio = 'output_audio.wav'
    video = 'output_video.mp4'
    TreatSignal(audio_file=audio, video_file=video)
