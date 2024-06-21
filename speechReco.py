import speech_recognition as sr
from pydub import AudioSegment

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

# Exemple d'utilisation de la classe SpeechRecognizer
if __name__ == "__main__":
    audio_file = "output_audio.wav"
    recognizer = SpeechRecognizer(audio_file)
    recognized_text = recognizer.recognize_from_audio_file()
    
    if recognized_text:
        print(f"Texte reconnu : {recognized_text}")
