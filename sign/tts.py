from gtts import gTTS
import pygame
import os
# from ttsmms import TTS
from scipy.io import wavfile
import numpy as np
import urllib3
import requests

def run(word):
    try:
        requests.get('https://www.google.com', timeout=5)
        ontts(word)
    except requests.ConnectionError:
        pass
        # offtts(word)

def ontts(word):
    language = 'en'
    myobj = gTTS(text=word, lang=language, slow=False)
    myobj.save("audio.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("audio.mp3")
    pygame.mixer.music.play()

# model_path = 'data/eng'
# tts = TTS(model_path)

# def synthesize_speech(text):
#     wav = tts.synthesis(text)
#     return wav

# def save_audio(wav, output_path):
#     wavfile.write(output_path, wav["sampling_rate"], np.array(wav["x"]))
#     print(f"Audio saved to {output_path}")

# def play_audio():
#     pygame.mixer.init()
#     pygame.mixer.music.load("output_speech.wav")
#     pygame.mixer.music.play()

# def offtts(content):
#     wav = synthesize_speech(content)
#     output_wav_path = "output_speech.wav"
#     save_audio(wav, output_wav_path)
#     play_audio()