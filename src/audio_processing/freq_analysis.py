from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
# from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import os
import tqdm
from datetime import datetime

DEBUG_OUTPUT_FILES = False

WINDOW_TIME = 0.125

FREQ_MIN = 100
FREQ_MAX = 7_000

TOP_NOTES = 5

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]

def freq_to_number(f): return 69 + 12*np.log2(f/440.0)
def number_to_freq(n): return 440 * 2.0**((n-69)/12.0)
def note_name(n): return NOTE_NAMES[n % 12] + str(int(n/12 - 1))



mp3_file = AudioSegment.from_mp3("audio_in/PinkPanther_Piano_Only.mp3")
wav_file = mp3_file.export("output.wav", format="wav")

sample_rate, audio_data = wavfile.read("output.wav")
AUDIO_LENGTH = audio_data.shape[0] / sample_rate


audio_data = audio_data.mean(axis=1) # Convert to mono

SAMPLES_PER_WINDOW = int(sample_rate * WINDOW_TIME)
NUMBER_OF_WINDOWS = int(audio_data.size / SAMPLES_PER_WINDOW) # + 1

FPS = NUMBER_OF_WINDOWS / AUDIO_LENGTH



xf = np.fft.rfftfreq(audio_data.size, 1/sample_rate) # Frequency bins for the FFT

def get_top_notes(fft, xf, mx, top_n=TOP_NOTES) -> list:
    if np.max(fft.real)<0.001:
        return []

    lst = [x for x in enumerate(fft.real)]
    lst = sorted(lst, key=lambda x: x[1],reverse=True)

    idx = 0
    found = []
    found_note = set()
    #local_mx = np.max(fft.real)
    while( (idx<len(lst)) and (len(found)<top_n) ):
        try:
            f = xf[lst[idx][0]]
            y = lst[idx][1]/mx
            if y < 0.05 :  # Ignore very low magnitudes
                idx += 1
                continue
            n = freq_to_number(f)
            n0 = int(round(n))
            name = note_name(n0)

            if name not in found_note:
                found_note.add(name)
                s = [f,name,y]
                found.append(s)

        except :
            pass
        idx += 1
        
    print(f"Found notes: {found}")
    return found


def build_fig_matplotlib(p, xf, notes, filename, dimensions=(16, 8)):
    if not DEBUG_OUTPUT_FILES:
        return
    plt.figure(figsize=dimensions)
    plt.plot(xf, p/mx, color='steelblue')
    plt.xlim(FREQ_MIN, FREQ_MAX)
    plt.ylim(0, 1)
    plt.xlabel("Frequency (note)")
    plt.ylabel("Magnitude")
    plt.title("frequency spectrum")

    for note in notes:
        plt.annotate(
            text=note[1], 
            xy=(note[0], note[2]), 
            fontsize=12, 
            color='red',
        )
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


mx = 0.0

# Calculate FFT for each frame and find the maximum value
for frame_num in range(NUMBER_OF_WINDOWS):
    start = frame_num * SAMPLES_PER_WINDOW
    end = start + SAMPLES_PER_WINDOW
    if end > len(audio_data):
        end = len(audio_data)

    frame_audio = audio_data[start:end]
    
    # Skip empty frames
    if len(frame_audio) == 0:
        continue
    mx = max(mx, np.max(np.abs(np.fft.rfft(frame_audio))))



# Create folder with current date and time
current_time_formated = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
frames_folder = f"frames_{current_time_formated}"
output_videos_folder = f"output_videos_{current_time_formated}"

if DEBUG_OUTPUT_FILES:
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(output_videos_folder, exist_ok=True)

notes_array = []

# Process each frame and save the FFT plot
for frame_num in tqdm.tqdm(range(NUMBER_OF_WINDOWS)):
    start = frame_num * SAMPLES_PER_WINDOW
    end = start + SAMPLES_PER_WINDOW
    if end > len(audio_data):
        end = len(audio_data)
    frame_audio = audio_data[start:end]

    # Skip empty frames
    if len(frame_audio) == 0:
        continue

    print(end)
    fft = np.fft.rfft(frame_audio)
    fft = np.abs(fft) 

    frame_xf = np.fft.rfftfreq(len(frame_audio), 1/sample_rate)
    
    top_notes = get_top_notes(fft, frame_xf, mx)

    dominant_note = min(top_notes, key=lambda x: x[0]) if top_notes else None

    if dominant_note is None: continue

    notes_array.append({WINDOW_TIME * frame_num:dominant_note[1]})

    # draw and save the figure
    build_fig_matplotlib(fft, frame_xf, get_top_notes(fft, frame_xf, mx), f"{frames_folder}/fft_frame_{frame_num:04d}.png")

print(notes_array)

if DEBUG_OUTPUT_FILES:

    # Combine frames into a video using ffmpeg
    import subprocess
    video_no_audio = f"{output_videos_folder}/output_video_no_audio.mp4"
    final_video = f"{output_videos_folder}/final_video.mp4"

    # Step 1: Create video from frames
    subprocess.run([
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-framerate", str(FPS),
        "-i", f"{frames_folder}/fft_frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_no_audio
    ])

    # Step 2: Combine video with audio
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", video_no_audio,
        "-i", "output.wav",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        final_video
    ])

    print(f"Video with audio saved as: {final_video}")




# Perform ICA and PCA on the audio data (only test)
'''
best_kurtosis = -np.inf
best_signals = None
best_H = None



F = np.fft.fft(audio_data, axis=0)  # Ensure audio_data is in the correct shape for FFT
freq = np.fft.fftfreq(audio_data.shape[0], d=1/sample_rate)
plt.plot(freq, F.real, freq, F.imag)
plt.show()


for i in range(5):
    ica = FastICA(n_components=2, whiten="arbitrary-variance")
    signals = ica.fit_transform(audio_data)
    pca = PCA(n_components=2)
    H = pca.fit_transform(audio_data)

    # Use sum of absolute kurtosis as a separation quality metric
    kurtosis = np.sum(np.abs(np.apply_along_axis(lambda x: np.mean((x - np.mean(x))**4) / (np.var(x)**2), 0, signals)))
    if kurtosis > best_kurtosis:
        best_kurtosis = kurtosis
        best_signals = signals
        best_H = H

# Use best_signals and best_H for saving and plotting
signals = best_signals
H = best_H



import matplotlib.pyplot as plt

plt.figure()

models = [audio_data, signals, H]
names = [
    "Observations (mixed signal)",
    "ICA recovered signals",
    "PCA recovered signals",
]
colors = ["red", "steelblue", "orange"]

num_samples = audio_data.shape[0]
time = np.arange(num_samples) / sample_rate  # Time in seconds

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(time, sig, color=color)
    plt.xlabel("Time (s)")

# Save each ICA component as its own WAV file
for idx, sig in enumerate(signals.T):
    # Normalize to int16 range
    sig_int16 = np.int16(sig / np.max(np.abs(sig)) * 32767)
    wavfile.write(f"output_ica_signal_{idx+1}.wav", sample_rate, sig_int16)

# Optionally, do the same for PCA components
for idx, sig in enumerate(H.T):
    sig_int16 = np.int16(sig / np.max(np.abs(sig)) * 32767)
    wavfile.write(f"output_pca_signal_{idx+1}.wav", sample_rate, sig_int16)

wavfile.write("output_ica_signals.wav", sample_rate, np.int16(signals * 32767))
wavfile.write("output_pca_signals.wav", sample_rate, np.int16(H * 32767))


plt.tight_layout()
plt.show()

'''