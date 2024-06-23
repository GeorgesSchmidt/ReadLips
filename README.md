# ReadLips
AI model for lip reading.

## Collecting the Data: detectAudio.py

This module opens a webcam window and records frames and sound.

- **Input**: Webcam
- **Output**: Two files, one `.avi` for images and one `.wav` for the soundtrack.

## Data Analysis for Images: detectFace.py

This module detects lips in images.

- **Input**: Image from video
- **Output**: Points of the lips

![Lips Points](result_img.jpg)

This module uses InsightFace to obtain landmarks of the lips from predictions. A convex hull of the lip points and an interpolation of this shape are used to maintain a constant number of points for this shape. It analyzes these points using 2D DFT (Discrete Fourier Transform) to generate a function that represents mouth movements during speech.

- **X variation**: Variation in the x-values of the points.
- **Y variation**: Variation in the y-values of the points.
- The plotted surface represents the result of the 2D DFT.
- **Lips amplitude**: Graphs of the lips during the movie in red (original points) and black (result of the DFT).

![DFT Plot](3d_plot_mouth.png)

These plots are saved along with the Fourier coefficients.

## Data Analysis for Sound: speechReco.py

This module interprets sound.

- **Input**: WAV file
- **Output**: Text (strings)

This module uses pydub to detect sentences from the soundtrack. The sentence is a string containing all the words spoken in the audio. The soundtrack is visualized using matplotlib. Fourier analysis is employed to interpret the data.

![Soundwave Plot](soundtrack.png)

## Deep Learning: treatSignal.py

Now we have two functions:

f(lips, t) represents the movement of the lips during the movie (time).

f(sound, t) represents the sound wave during the movie (time).

These two functions are Fourier transforms: 2D DFT for lips, 1D FFT for sound.

Because it's a movie, we are using a sequential AI model, specifically LSTM.

- **Input**: (row, col, 2) where row is the number of images (time), col is the number of points (10), and 2 is the number of values x and y.
- **Output**: The sound signal.
