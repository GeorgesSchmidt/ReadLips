import cv2
import numpy as np
import os

from sound.soundAnalysis import SoundTreat

directory = os.path.join(os.getcwd(), 'datas')
sound = SoundTreat(directory)
param = sound.param_sound
print('sound rate', param[0])
print('sound samples', len(param[1]))


from face.interpolation import Interpolation

Interpolation(directory)

# from face.fourierImage import Fourier

# Fourier(directory)

# path = os.path.join(os.getcwd(), 'datas', 'thomas_face.mp4')
# cap = cv2.VideoCapture(path)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
# cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
# print('video rate', cap_fps)
# print('video count', frame_count)


