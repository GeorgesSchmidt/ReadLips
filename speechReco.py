import speech_recognition as sr
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np

class SpeechRecognizer:
    def __init__(self, audio_file):
        self.audio_file = audio_file

    def recognize_from_audio_file(self):
        # Créer un objet Recognizer
        r = sr.Recognizer()

        try:
            # Utiliser le fichier temporaire pour la reconnaissance vocale
            with sr.AudioFile(self.audio_file) as source:
                audio_data = r.record(source)  # Lecture du fichier audio

            # Reconnaissance vocale avec Google Web Speech API
            text = r.recognize_google(audio_data, language="fr-FR")
            return text
        except sr.UnknownValueError:
            print("Google Web Speech API n'a pas pu comprendre l'audio")
        except sr.RequestError as e:
            print(f"Erreur lors de la demande à Google Web Speech API : {e}")
        
        return None
    
    def plot_audio_waveform(self):
        # Charger le fichier audio avec PyDub
        sound = AudioSegment.from_wav(self.audio_file)
        
        # Convertir le son en tableau numpy d'int16 (amplitudes)
        samples = np.array(sound.get_array_of_samples())
        
        # Obtenir les échantillons par seconde et les temps associés
        sample_rate = sound.frame_rate
        times = np.linspace(0, len(samples) / sample_rate, num=len(samples))
        
        # # Créer le graphique
        # plt.figure(figsize=(10, 4))
        # plt.plot(times, samples, color='b')
        # plt.xlim(0, times[-1])
        # plt.xlabel('Temps (s)')
        # plt.ylabel('Amplitude')
        # plt.title('Forme d\'onde audio')
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        
        return times, samples

# Exemple d'utilisation de la classe SpeechRecognizer avec la méthode ajoutée
if __name__ == "__main__":
    audio_file = "output_audio.wav"
    recognizer = SpeechRecognizer(audio_file)
    
    # Afficher le texte reconnu
    recognized_text = recognizer.recognize_from_audio_file()
    if recognized_text:
        print(f"Texte reconnu : {recognized_text}")
    
    # Afficher la forme d'onde audio
    recognizer.plot_audio_waveform()
