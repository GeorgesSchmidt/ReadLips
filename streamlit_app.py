import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import time
from PIL import Image

st.title('App de inteligencia por Marta solo por 2 personas')

#use_webcam = st.button('Use Webcam')

video = cv2.VideoCapture(0)

drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

i = 0
stframe = st.empty()
with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5

    ) as face_mesh:

            prevTime = 0

            while video.isOpened():
                i +=1
                ret, frame = video.read()
                if not ret:
                    continue

                results = face_mesh.process(frame)
                frame.flags.writeable = True

                face_count = 0
                if results.multi_face_landmarks:

                    #Face Landmark Drawing
                    for face_landmarks in results.multi_face_landmarks:
                        face_count += 1

                        mp.solutions.drawing_utils.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec
                        )

                # FPS Counter
                currTime = time.time()
                fps = 1/(currTime - prevTime)
                prevTime = currTime


                # Dashboard
                # kpil_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                # kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
                # kpil3_text.write(f"<h1 style='text-align: center; color:red;'>{width*height}</h1>",
                #                  unsafe_allow_html=True)

                # frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
                # frame = image_resize(image=frame, width=640)
                stframe.image(frame,channels='BGR', use_column_width=True)

