import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from matplotlib import pyplot as plt

mp3_file = AudioSegment.from_mp3(r"C:\Users\henry\OneDrive\Documents\GitHub\aquazik\code\Ecossaise_Both.mp3") #add your audio path
wav_file = mp3_file.export("output.wav", format="wav")

#Load the WAV file
sample_rate, audio_data = wavfile.read("output.wav")
audio_data = audio_data / 32768.0

#
plt.plot(audio_data)
plt.show()