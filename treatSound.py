import cv2
import numpy as np
import os
from speechReco import SpeechRecognizer
class TreatSound(SpeechRecognizer):
    def __init__(self, path) -> None:
        self.path = path
        
        
        
        
if __name__=='__main__':
    
    path = 'videos/short_output_audio.wav'
    TreatSound(path)