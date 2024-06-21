import sounddevice as sd
import wavio

class AudioRecorder:
    def __init__(self, channels=1, rate=44100, duration=5):
        self.channels = channels
        self.rate = rate
        self.duration = duration
        self.frames = []

    def record(self):
        print("Enregistrement en cours...")
        self.frames = sd.rec(int(self.duration * self.rate), samplerate=self.rate, channels=self.channels, dtype='int16')
        sd.wait()  # Attend la fin de l'enregistrement
        print("Enregistrement terminé.")

    def save_recording(self, output_filename):
        wavio.write(output_filename, self.frames, self.rate, sampwidth=2)
        print(f"Fichier sauvegardé sous {output_filename}")

# Utilisation de la classe
if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.record()  # Enregistre pendant la durée définie
    recorder.save_recording("output.wav")
