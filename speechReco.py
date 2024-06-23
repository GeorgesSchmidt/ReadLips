import speech_recognition as sr
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np

class SpeechRecognizer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.sentence = None

    def recognize_from_audio_file(self):
        # Créer un objet Recognizer
        r = sr.Recognizer()

        try:
            # Utiliser le fichier temporaire pour la reconnaissance vocale
            with sr.AudioFile(self.audio_file) as source:
                audio_data = r.record(source)  # Lecture du fichier audio

            # Reconnaissance vocale avec Google Web Speech API
            text = r.recognize_google(audio_data, language="fr-FR")
            self.sentence = text
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
        
        # Calculer la FFT
        fft_result = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), 1/sample_rate)
        
        # Calculer l'amplitude du signal
        amplitude = np.abs(fft_result)

        # Filtrer les fréquences pour garder seulement la moitié positive
        positive_freqs = freqs[:len(freqs)//2]
        positive_amplitudes = amplitude[:len(amplitude)//2]

        # Appliquer un seuil pour nettoyer le signal
        threshold = np.max(amplitude) * 0.5  # Par exemple, 1% de l'amplitude maximale
        fft_result_cleaned = np.where(amplitude > threshold, fft_result, 0)
        
        # Calculer l'IDFT avec les coefficients nettoyés
        reconstructed_signal_cleaned = np.fft.ifft(fft_result_cleaned)
        reconstructed_signal_cleaned = np.real(reconstructed_signal_cleaned)
        
        # Créer le graphique de la forme d'onde
        plt.figure(figsize=(15, 4))
        title = f'sentence detected : {self.sentence}'
        plt.suptitle(title, fontsize=14)
        plt.subplot(1, 3, 1)
        plt.plot(times, samples, color='b')
        plt.title('Waveform')
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # Créer le graphique du spectre de fréquences
        plt.subplot(1, 3, 2)
        plt.plot(positive_freqs, positive_amplitudes, color='r')
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # Créer le graphique du signal reconstruit nettoyé
        plt.subplot(1, 3, 3)
        plt.plot(times, reconstructed_signal_cleaned, color='g')
        plt.title('Reconstructed Signal (Cleaned)')
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('soundtrack.png')
        plt.show()
        
        return times, samples
    
    def calculcoefFourier(self, signal, T):
        N = len(signal)
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N, T)
        return fft_result, freqs

    def calculResult(self, fft_result):
        # Utiliser la transformée inverse pour reconstruire le signal
        reconstructed_signal = np.fft.ifft(fft_result)
        return np.real(reconstructed_signal)


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
