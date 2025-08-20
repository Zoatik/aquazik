import librosa 
import librosa.feature.rhythm
import numpy as np
from scipy.io import wavfile
# from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import os
import tqdm
from datetime import datetime


DEBUG_OUTPUT_FILES = True  # Set to True to enable output files
if not os.path.exists("output"):
    os.makedirs("output")
if not os.path.exists("audio_in"):
    os.makedirs("audio_in")
OUTPUT_FOLDER = os.path.abspath("output")
INPUT_FOLDER = os.path.abspath("audio_in")

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


class Note:
    def __init__(self, frequency, name, magnitude, variation=0.0):
        self.frequency = frequency
        self.name = name
        self.magnitude = magnitude
        self.maximum = magnitude  # Store the maximum magnitude observed for this note
        self.variation = variation # Indicates if the note is fading in or out (positive for fading in, negative for fading out)

    def __repr__(self):
        return f"Note(frequency={self.frequency}, name='{self.name}', magnitude={self.magnitude})" 
    
# Load audio file
audio_name = "PinkPanther_Trumpet_Only.mp3"
audio_path = os.path.join(INPUT_FOLDER, audio_name) 
audio_data, sample_rate = librosa.load(audio_path, mono=True)
AUDIO_LENGTH = librosa.get_duration(y=audio_data, sr=sample_rate)
AUDIO_BPM = librosa.feature.rhythm.tempo(y=audio_data, sr=sample_rate)[0]


SAMPLES_PER_WINDOW = int(sample_rate * WINDOW_TIME)
NUMBER_OF_WINDOWS = int(audio_data.size / SAMPLES_PER_WINDOW) # + 1

FPS = NUMBER_OF_WINDOWS / AUDIO_LENGTH



xf = np.fft.rfftfreq(audio_data.size, 1/sample_rate) # Frequency bins for the FFT

def get_top_notes(fft, xf, mx, prev_notes, top_n=TOP_NOTES) -> list:
    if np.max(fft.real)<0.001:
        return []

    lst = [x for x in enumerate(fft.real)]
    lst = sorted(lst, key=lambda x: x[1],reverse=True)

    idx = 0
    found = []
    found_note_name = set()
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

            if name not in found_note_name:
                found_note_name.add(name)
                found_note = Note(frequency=f, name=name, magnitude=y)

                for prev_note in prev_notes:
                    if prev_note.name == name:
                        found_note.variation = found_note.magnitude - prev_note.magnitude
                        found_note.maximum = max(found_note.maximum, prev_note.maximum)
                        break

                found.append(found_note)

        except :
            pass
        idx += 1
        
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
            text=note.name, 
            xy=(note.frequency, note.magnitude), 
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


music_name_folder = audio_name.replace(".mp3", "").replace(".wav", "")
frames_folder = os.path.join(OUTPUT_FOLDER, music_name_folder, f"frames_{current_time_formated}")
output_videos_folder = os.path.join(OUTPUT_FOLDER, music_name_folder, f"output_videos_{current_time_formated}")


if DEBUG_OUTPUT_FILES:
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(output_videos_folder, exist_ok=True)

notes_array = []
prev_top_notes = []

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

    fft = np.fft.rfft(frame_audio)
    fft = np.abs(fft) 
    frame_xf = np.fft.rfftfreq(len(frame_audio), 1/sample_rate)

    top_notes = get_top_notes(fft, frame_xf, mx, prev_top_notes)

    top_notes = sorted(top_notes, key=lambda x: x.frequency)

    if top_notes:

        dominant_note = top_notes[0]
        if top_notes[0].variation < 0 and top_notes[0].magnitude < 0.2 * top_notes[0].maximum: # note is fading out
            dominant_note = top_notes[1] if len(top_notes) > 1 else top_notes[0]

        notes_array.append({WINDOW_TIME * frame_num:dominant_note.name})
        
    else: 
        notes_array.append({WINDOW_TIME * frame_num:"No Note"})

    # draw and save the figure
    build_fig_matplotlib(fft, frame_xf, top_notes, f"{frames_folder}/fft_frame_{frame_num:04d}.png")

print(notes_array)

if DEBUG_OUTPUT_FILES:

    # Combine frames into a video using ffmpeg
    import subprocess
    video_no_audio = os.path.join(output_videos_folder, "output_video_no_audio.mp4")
    final_video = os.path.join(output_videos_folder, "final_video.mp4")

    # Step 1: Create video from frames
    subprocess.run([
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-framerate", str(FPS),
        "-i", os.path.join(frames_folder, "fft_frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_no_audio
    ])

    # Step 2: Combine video with audio
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", video_no_audio,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        final_video
    ])

    print(f"Video with audio saved as: {final_video}")

