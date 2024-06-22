# ReadLips
AI model to read lips. 



# Collecting the data : detectAudio.py

Module opening a window (webCam) and recording frames and sound. 
input = webcam
output = 2 files : one .avi for images one .wav for soundtrack. 

# Data analyse for images : detectFace.py

Module to detect lips on images. 
input = image from video. 
output = points of the lips. 
![lips points](result_img.png)


This module use InsigthFace to get the landmarks of the lips produce by the predictions. 
It analyse this points with a DFT2D to get the function representative of the mouvements of the mounth during speacking. 

# Data analyse for sound = speechReco.py

Module to interpret sound 

input = wav file. 
output = texte (strings). 

This module use pydub to detect the sentence of the soundtrack. This sentence is a string containing all the words of the sound. 
The soundtrack is plot in matplotlib to visualise the data. 
It use a Fourier analysis to interpret this data. 


