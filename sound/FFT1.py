import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

class SpeechDetector:
    def __init__(self, file_path, frame_size=1024, hop_size=512, threshold_ratio=0.05):
        self.file_path = file_path
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.threshold_ratio = threshold_ratio
        
        # Appel de la méthode principale pour détecter les intervalles de parole et afficher le plot
        self.detect_and_plot_intervals()
    
    def read_wav_file(self):
        try:
            rate, data = wavfile.read(self.file_path)
            if len(data.shape) == 2:
                data = data.mean(axis=1)
            return rate, data
        except Exception as e:
            print(f"Erreur lors du chargement du fichier audio : {str(e)}")
            return None, None
    
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
    
    def detect_and_plot_intervals(self):
        # Lire le fichier WAV
        rate, data = self.read_wav_file()
        
        if rate is None or data is None:
            return
        
        # Détecter les intervalles de parole
        speech_intervals = self.detect_speech_intervals(data)
        
        # Identifier les changements dans les intervalles de parole
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
            speech_intervals_list.append((start_idx, len(speech_intervals) - 1))
        
        # Afficher les résultats
        self.plot_intervals(rate, data, speech_intervals, speech_intervals_list)
        
        self.intervalles = speech_intervals_list
        
        # Sauvegarder les intervalles de parole en fichiers audio
        #self.save_speech_intervals(speech_intervals_list, rate, data)
        
    def plot_intervals(self, rate, data, speech_intervals, speech_intervals_list):
        print(speech_intervals_list)
        # Afficher les résultats
        maxi = np.max(data)
        mini = np.min(data)
        time = np.arange(0, len(data)) / rate
        plt.figure(figsize=(14, 7))
        plt.plot(time, data, label='Signal')
        frame_times = np.arange(0, len(speech_intervals)) * self.hop_size / rate
        for d, f in speech_intervals_list:
            duration = frame_times[f] - frame_times[d]
            if duration > 0.1:
                plt.hlines(y=maxi, xmin=frame_times[d], xmax=frame_times[f], color='g', linestyle='-')
                plt.hlines(y=mini, xmin=frame_times[d], xmax=frame_times[f], color='g', linestyle='-')
                plt.vlines(frame_times[d], min(data), max(data), color='r')
                plt.vlines(frame_times[f], min(data), max(data), color='b')
        plt.xlabel('Temps [s]')
        plt.ylabel('Amplitude')
        plt.title('Détection de la parole')
        plt.tight_layout()
        #plt.show()
        plt.savefig('./pictures/soundIntervals.png')
    
    def save_speech_intervals(self, speech_intervals_list, rate, data):
        # Créer les fichiers audio pour chaque intervalle de parole détecté
        for ind, interv in enumerate(speech_intervals_list):
            start_idx, end_idx = interv
            interval_data = data[start_idx:end_idx]
            if len(interval_data) > 10:
                print(len(interval_data))
                filename = f'./interval_audio/morceau_extrait_{ind}.wav'
                wavfile.write(filename, rate, interval_data)
            
            # if duration > 10:
            #     print(duration)
            #     interval_data = data[debut_echantillon:fin_echantillon]
            #     filename = f'morceau_extrait_{ind}.wav'
            #     print('data', len(interval_data))
            #     #wavfile.write(filename, rate, interval_data)

# Exemple d'utilisation de la classe SpeechDetector
if __name__ == "__main__":
    detector = SpeechDetector(file_path='./datas/audio_short_thoma.wav')
