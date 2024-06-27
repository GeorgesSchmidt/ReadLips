import speech_recognition as sr
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import re

from readLips import LoadDatas

class TreatSound(LoadDatas):
    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.sentence = self.recognize_from_audio_file()
        if self.sentence is not None:
            print('sentence =', self.sentence)
            self.get_numbers_words()
            self.plot_audio_waveform()
            self.plot_fourier()
        
        
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
    
    def get_numbers_words(self):
        parts = re.findall(r'\S+|\s', self.sentence)
        self.nb_words = len(parts)
        n = 0
        for w in parts:
            if w == ' ':
                n += 1
        print('words', self.nb_words, 'silence', n)
        
        
    
    def plot_audio_waveform(self):
        values, _, _ = self.get_datas_sound()
        times, samples = values
        angle = np.linspace(times[0], times[-1], len(times))
        t = np.linspace(times[0], times[-1], self.nb_words)
        vx = np.interp(t, angle, samples)

        vy = np.interp(t, angle, times)

        
        fig = plt.figure(figsize=(15, 4))
        title = f'{self.sentence} ({self.nb_words} words)'
        fig.suptitle(title, fontsize=14)
        ax = fig.add_subplot(111)
        
        ax.plot(times, samples, label='Original')
        
        ax.vlines(x=vy, ymin=0, ymax=np.max(samples), colors='r', linestyles='dashed', label='Interpolated Points')

        
        ax.set_xlabel('Time')  # Label de l'axe des x
        
        plt.savefig('soundtrack_interp.png')
        plt.show()
        
        
    def plot_fourier(self):
        values, four, reconstructed_signal_cleaned = self.get_datas_sound()
        times, samples = values
        positive_freqs, positive_amplitudes = four
    

        plt.figure(figsize=(15, 4))
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
        
       
        
    def get_datas_sound(self):
        # Charger le fichier audio avec PyDub
        sound = AudioSegment.from_wav(self.audio_file)
        
        # Convertir le son en tableau numpy d'int16 (amplitudes)
        samples = np.array(sound.get_array_of_samples())
        
        # Obtenir les échantillons par seconde et les temps associés
        sample_rate = sound.frame_rate
        times = np.linspace(0, len(samples) / sample_rate, num=len(samples))
        self.values = [times, samples]
        # Calculer la FFT
        fft_result = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), 1/sample_rate)
        
        # Calculer l'amplitude du signal
        amplitude = np.abs(fft_result)

        # Filtrer les fréquences pour garder seulement la moitié positive
        val = len(freqs)//2
        positive_freqs = freqs[:len(freqs)//2]
        positive_amplitudes = amplitude[:len(amplitude)//2]

        # Appliquer un seuil pour nettoyer le signal
        threshold = np.max(amplitude) * 0.01  # Par exemple, 1% de l'amplitude maximale
        fft_result_cleaned = np.where(amplitude > threshold, fft_result, 0)
        
        # Calculer l'IDFT avec les coefficients nettoyés
        reconstructed_signal_cleaned = np.fft.ifft(fft_result_cleaned)
        reconstructed_signal_cleaned = np.real(reconstructed_signal_cleaned)
        
        return (times, samples), (positive_freqs, positive_amplitudes), reconstructed_signal_cleaned
        
    
    def calculcoefFourier(self, signal, T):
        N = len(signal)
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N, T)
        return fft_result, freqs

    def calculResult(self, fft_result):
        # Utiliser la transformée inverse pour reconstruire le signal
        reconstructed_signal = np.fft.ifft(fft_result)
        return np.real(reconstructed_signal)


        
    

        
if __name__=='__main__':
    directory = 'datas'
    TreatSound(directory)