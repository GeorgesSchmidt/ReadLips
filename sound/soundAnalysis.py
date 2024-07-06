import speech_recognition as sr
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os
from scipy.io import wavfile
import scipy.signal as signal


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face.loadDataFace import LoadDatas

class SoundTreat(LoadDatas):
    def __init__(self, directory) -> None:
        super().__init__(directory)
        self.sentence = self.get_sentence()
        if len(self.sentence)>0:
            print('sentence', self.sentence)
            self.nb_words = len(self.sentence.split(' '))
            self.analyse_sound()
        
    def get_sentence(self):
        r = sr.Recognizer()
        try:
            with sr.AudioFile(self.audio_file) as source:
                audio_data = r.record(source)
            text = r.recognize_google(audio_data, language="fr-FR")
            return text
        except sr.UnknownValueError:
            print("Google Web Speech API n'a pas pu comprendre l'audio")
        except sr.RequestError as e:
            print(f"Erreur lors de la demande à Google Web Speech API : {e}")
        return None
    
    def analyse_sound(self):
        # Lire le fichier audio
        rate, data = wavfile.read(self.audio_file)
        if len(data.shape) == 2:
            data = data.mean(axis=1)  # Convertir en mono

        # Afficher le signal audio
        plt.figure(figsize=(12, 6))
        plt.plot(data)
        plt.title('Signal Audio')
        plt.xlabel('Temps (échantillons)')
        plt.ylabel('Amplitude')
        plt.show()

        # Détecter les silences et les paroles
        threshold = np.std(data) / 2  # Définir un seuil d'amplitude pour détecter les silences
        silence = np.abs(data) < threshold
        word = np.abs(data) >= threshold

        transitions = np.where(np.diff(silence.astype(int)) != 0)[0]
        transitions = self.clean_transitions(transitions)

        print('transitions', len(transitions))
        
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Signal Audio')
        
        for d, f in transitions:
            plt.axvline(x=d, color='red', linestyle='-')
            
        plt.title('Signal Audio avec Silences et Paroles')
        plt.xlabel('Temps (échantillons)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
        
    def clean_transitions(self, tr):
        result = []
        for i in range(len(tr)-1):
            n0, n1 = tr[i], tr[i+1]
            diff = n1-n0
            if diff > 500:
                print(diff)
                result.append([n0, n1])
        return result
    
    def fourier_analysis(self):
        rate, data = wavfile.read(self.audio_file)
        if len(data.shape) == 2:
            data = data.mean(axis=1)
        # Calcul de la FFT
        n = len(data)
        fft_result = np.fft.fft(data)
        freq = np.fft.fftfreq(n, d=1/rate)

        # Filtrage des hautes fréquences (par exemple, suppression des fréquences au-dessus de 4000 Hz)
        cutoff_freq = 4000  # Fréquence de coupure en Hz
        b, a = signal.butter(4, cutoff_freq / (rate / 2), 'low')
        filtered_data = signal.lfilter(b, a, data)

        # Calcul de la FFT du signal filtré
        filtered_fft_result = np.fft.fft(filtered_data)
        
 
    def fourier_analysis1(self):
        rate, data = wavfile.read(self.audio_file)
        self.param_sound = [rate, data]
        if len(data.shape) == 2:
            data = data.mean(axis=1)
            
        speech_intervals = self.detect_speech_intervals(data)
        change_indices = np.where(np.diff(speech_intervals))[0] + 1
        
        # Créer la liste des intervalles de parole [(début, fin), (début, fin), ...]
        speech_intervals_list = []
        start_idx = 0
        for end_idx in change_indices:
            if speech_intervals[start_idx]:
                speech_intervals_list.append((start_idx, end_idx - 1))
            start_idx = end_idx
        
        # Ajouter le dernier intervalle s'il se termine à la fin du signal
        if speech_intervals[start_idx]:
            self.speech_intervals_list.append((start_idx, len(speech_intervals) - 1))
            
        self.plot_intervals(rate, data, speech_intervals, speech_intervals_list)
            
            
    def frame_energy_fft(self, frame):
        # Appliquer la FFT au segment
        fft_result = np.fft.fft(frame)
        # Calculer l'énergie comme la somme des carrés des amplitudes
        energy = np.sum(np.abs(fft_result) ** 2) / len(fft_result)
        return energy
    
    def detect_speech_intervals(self, data):
        # Diviser le signal en segments et calculer l'énergie avec la FFT
        num_frames = int(np.ceil((len(data) - self.frame_size) / self.hop_size)) + 1
        energies = []

        for i in range(num_frames):
            start_idx = i * self.hop_size
            end_idx = start_idx + self.frame_size
            if end_idx > len(data):
                end_idx = len(data)
            frame = data[start_idx:end_idx]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)), 'constant')
            
            # Calculer l'énergie du segment avec la FFT
            energy = self.frame_energy_fft(frame)
            energies.append(energy)
        
        # Convertir les énergies en un tableau NumPy
        energies = np.array(energies)
        
        # Définir un seuil pour distinguer la parole du silence
        threshold = np.mean(energies) * self.threshold_ratio
        
        # Identifier les segments de parole et de silence
        speech_intervals = energies > threshold
        
        return speech_intervals

    def plot_intervals(self, rate, data, speech_intervals, speech_intervals_list):
        print(speech_intervals_list)
        # Afficher les résultats
        maxi = np.max(data)
        mini = np.min(data)
        time = np.arange(0, len(data)) / rate
        plt.figure(figsize=(14, 7))
        plt.plot(time, data, label='Signal')
        frame_times = np.arange(0, len(speech_intervals)) * self.hop_size / rate
        # for d, f in speech_intervals_list:
        #     duration = frame_times[f] - frame_times[d]
        #     if duration > 0.05:
        #         plt.hlines(y=maxi, xmin=frame_times[d], xmax=frame_times[f], color='g', linestyle='-')
        #         plt.hlines(y=mini, xmin=frame_times[d], xmax=frame_times[f], color='g', linestyle='-')
        #         plt.vlines(frame_times[d], min(data), max(data), color='r')
        #         plt.vlines(frame_times[f], min(data), max(data), color='b')
        plt.xlabel('Temps [s]')
        plt.ylabel('Amplitude')
        #plt.title(f'{self.sentence} = {self.nb_words} words')
        plt.title('waveform')
        plt.tight_layout()
        #plt.show()
        plt.savefig('waveform.png')
        
        
        

if __name__ == '__main__':
    directory = 'datas'
    SoundTreat(directory)
