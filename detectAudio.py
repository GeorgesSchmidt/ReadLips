import cv2
import sounddevice as sd
import soundfile as sf
import threading
import numpy as np
import os


class VideoAudioRecorder:
    def __init__(self, video_filename='output_video.mp4', audio_filename='output_audio.wav',
                 frame_width=640, frame_height=480, fps=25, audio_channels=1, audio_rate=44100,
                 audio_dtype='int16'):
        self.clean_files(os.getcwd())
        self.video_filename = video_filename
        self.audio_filename = audio_filename
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.audio_channels = audio_channels
        self.audio_rate = audio_rate
        self.audio_dtype = audio_dtype

        # Variables pour contrôler l'enregistrement
        self.record_audio = True

        # Initialise l'enregistrement audio
        self.audio_frames = []
        self.audio_stream = sd.InputStream(channels=self.audio_channels, samplerate=self.audio_rate,
                                           dtype=self.audio_dtype)

        # Capture vidéo à partir de la webcam
        self.cap = cv2.VideoCapture(0)
        cap_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        cap_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Paramètres pour l'enregistrement vidéo
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.video_filename, self.fourcc, cap_fps, (cap_w, cap_h))

        # Démarrer l'enregistrement audio dans un thread séparé
        self.audio_thread = threading.Thread(target=self._audio_capture)
        self.audio_thread.start()

    def clean_files(self, dossier):
        try:
        # Parcours des fichiers dans le dossier
            for fichier in os.listdir(dossier):
                # Vérifier si le fichier est un .wav, .avi ou .mp4
                if fichier.endswith(".wav") or fichier.endswith(".avi") or fichier.endswith(".mp4"):
                    # Construire le chemin complet du fichier
                    chemin_fichier = os.path.join(dossier, fichier)
                    # Supprimer le fichier
                    os.remove(chemin_fichier)
                    print(f"Fichier supprimé : {chemin_fichier}")
        except Exception as e:
            print(f"Erreur lors de la suppression des fichiers : {e}")
                
    def _audio_capture(self):
        with self.audio_stream:
            while self.record_audio:
                data, _ = self.audio_stream.read(self.audio_rate // self.fps)
                self.audio_frames.append(data)

    def start_recording(self):
        # Capture vidéo et enregistrement simultanés
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.out.write(frame)
            cv2.imshow('Frame', frame)

            # Sortir de la boucle si 'q' est pressé
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Arrêt de l'enregistrement audio
        self.record_audio = False
        self.audio_thread.join()

        # Arrêt et nettoyage
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

        # Écriture des données audio dans un fichier WAV
        sf.write(self.audio_filename, np.concatenate(self.audio_frames), self.audio_rate, subtype='PCM_16')

# Exemple d'utilisation de la classe VideoAudioRecorder
if __name__ == "__main__":
    recorder = VideoAudioRecorder()
    recorder.start_recording()
