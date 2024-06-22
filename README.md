# ReadLips
AI model for lip reading.

## Collecting the Data: detectAudio.py

This module opens a window (webcam) and records frames and sound.

- **Input**: webcam
- **Output**: two files, one `.avi` for images and one `.wav` for soundtrack.

## Data Analysis for Images: detectFace.py

This module detects lips in images.

- **Input**: image from video
- **Output**: points of the lips

![Lips Points](result_img.jpg)

This module uses InsightFace to obtain landmarks of the lips from predictions. It analyzes these points using a 2D DFT (Discrete Fourier Transform) to generate a function that represents mouth movements during speech.

- **X variation**: Variation in the x-values of the points.
- **Y variation**: Variation in the y-values of the points.
- The plotted surface represents the result of the 2D DFT.

![DFT Plot](3d_plot_mouth.png)

These two surfaces are saved along with the Fourier coefficients.

## Data Analysis for Sound: speechReco.py

This module interprets sound.

- **Input**: WAV file
- **Output**: text (strings)

This module uses pydub to detect the sentence from the soundtrack. The sentence is a string containing all the words spoken in the audio. The soundtrack is visualized using matplotlib. Fourier analysis is employed to interpret the data.

![Soundwave Plot](soundtrack.png)
