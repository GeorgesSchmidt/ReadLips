# ReadLips
AI model to read lips.

## Collecting the Data : detectAudio.py

This module opens a window (webcam) and records frames and sound.

- **Input**: webcam
- **Output**: two files, one `.avi` for images and one `.wav` for soundtrack

## Data Analysis for Images : detectFace.py

This module detects lips on images.

- **Input**: image from video
- **Output**: points of the lips

![Lips Points](result_img.jpg)

This module uses InsightFace to get the landmarks of the lips produced by the predictions. It analyzes these points with a 2D DFT (Discrete Fourier Transform) to get the function representative of the movements of the mouth during speaking.

![Curve Points](3d_plot_mounth.png)


## Data Analysis for Sound : speechReco.py

This module interprets sound.

- **Input**: wav file
- **Output**: text (strings)

This module uses pydub to detect the sentence of the soundtrack. The sentence is a string containing all the words of the sound. The soundtrack is plotted in matplotlib to visualize the data. It uses a Fourier analysis to interpret this data.
