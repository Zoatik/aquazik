import librosa
import librosa.feature.rhythm
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

# from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import statistics
import os
import tqdm
from datetime import datetime
#from raw_signal_analysis import is_local_maximum

#temp
def is_local_maximum(x, arr):
    if len(arr) < 1:
        return False
    y = arr[x]
    left_y = arr[x - 1] if x > 1 else y - 1
    right_y = arr[x + 1] if len(arr) > x + 1 else y - 1

    l = 1
    r = 1
    while True:
        if y < left_y or y < right_y:
            return False
        if y > left_y and y > right_y:
            return True
        if y == right_y:
            r += 1
        if y == left_y:
            r += 1

        left_y = arr[x - l] if x - l > 0 else y + 1
        right_y = arr[x + r] if x + r < len(arr) else y + 1

if not os.path.exists("output"):
    os.makedirs("output")
if not os.path.exists("audio_in"):
    os.makedirs("audio_in")
OUTPUT_FOLDER = os.path.abspath("output")
INPUT_FOLDER = os.path.abspath("audio_in")

WINDOW_TIME = 0.125  # window time of each analysis

FREQ_MIN = 100
FREQ_MAX = 7_000

TOP_NOTES = 10  # how many notes are stored per analysis

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


INSTRUMENTS_HARMONICS = {
    "Piano": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Trumpet": [12, 7, 5, 4, 3, 3],
}


class Tools:
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    # Mapping of note names to MIDI numbers
    NOTE_TO_MIDI = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }

    @staticmethod
    def note_to_midi(note):
        note_name = note[:-1]  # Extract the note name (e.g., 'A', 'B', etc.)
        octave = int(note[-1])  # Extract the octave (e.g., '0', '2', etc.)
        return Tools.NOTE_TO_MIDI[note_name] + 12 * (octave + 1)

    @staticmethod
    def seconds_to_beat(sec, bpm):
        return sec * (bpm / 60.0)

    @staticmethod
    def freq_to_number(f):
        return int(round(69 + 12 * np.log2(f / 440.0)))

    @staticmethod
    def number_to_freq(n):
        return 440 * 2.0 ** ((n - 69) / 12.0)

    @staticmethod
    def note_name(n):
        return NOTE_NAMES[n % 12] + str(int(n / 12 - 1)) 
    
    @staticmethod
    def freq_to_note(f):
        return Tools.note_name(Tools.freq_to_number(f))


class Note:
    def __init__(self, frequency, magnitude, start = 0.0, length = 0.0, variation=0.0):
        self.frequency = frequency
        self.start_time = start
        self.start_bpm = 0.0
        self.length = length
        self.length_bpm = 0.0
        self.midi_number = Tools.freq_to_number(self.frequency)
        self.name = Tools.note_name(self.midi_number)
        self.magnitude = magnitude
        self.maximum = magnitude  # Store the maximum magnitude observed for this note
        self.variation = variation  # Indicates if the note is fading in or out (positive for fading in, negative for fading out)
        self.instrument = "Unknown"

    def __repr__(self):
        return f"Note(frequency={self.frequency}, name='{self.name}', start={self.start_time}, bpm start={self.start_bpm} , bpm length={self.length_bpm}, magnitude={self.magnitude})\n"


class AudioAnalyzer:
    def __init__(
        self, audio_name="PinkPanther_Trumpet_cut.mp3", DEBUG_OUTPUT_FILES=False, duration = None
    ):
        self.audio_name = audio_name
        self.audio_path = os.path.join(INPUT_FOLDER, audio_name)
        self.audio_data, self.sample_rate = librosa.load(self.audio_path, mono=True, duration=duration)
        self.audio_length = librosa.get_duration(y=self.audio_data, sr=self.sample_rate)
        self.audio_bpm = 120
        self.beat_unit = 60 / self.audio_bpm
        self.samples_per_window = int(self.sample_rate * WINDOW_TIME)
        self.number_of_windows = int(
            self.audio_data.size / self.samples_per_window
        )  # + 1
        self.fps = self.number_of_windows / self.audio_length
        self.xf = np.fft.rfftfreq(self.samples_per_window, 1 / self.sample_rate)
        self.freq_dist_per_bin = (
            abs(self.xf[0] - self.xf[1]) if len(self.xf) > 1 else 8.0
        )
        self.debug_output_files = DEBUG_OUTPUT_FILES

        est_bpm = librosa.feature.rhythm.tempo(y=self.audio_data, sr=self.sample_rate)[
            0
        ]
        if est_bpm % 10 > 5:
            self.audio_bpm = est_bpm - est_bpm % 10 + 10
        else:
            self.audio_bpm = est_bpm - est_bpm % 10

    def convert_to_notes(self):
        mx = 0.0
        mx_buckets = 0.0

        # Calculate FFT for each frame and find the maximum value
        for frame_num in range(self.number_of_windows):
            start = frame_num * self.samples_per_window
            end = start + self.samples_per_window
            if end > len(self.audio_data):
                end = len(self.audio_data)

            frame_audio = self.audio_data[start:end]

            # Skip empty frames
            if len(frame_audio) == 0:
                continue

            frame_fft = np.fft.rfft(frame_audio)
            frame_fft = np.abs(frame_fft)
            mx = max(mx, np.max(np.abs(frame_fft)))
            mx_buckets = max(mx_buckets, np.max(self.__fft_to_notes_buckets(frame_fft)[1]))

        # Prepares output files directory
        current_time_formated = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        music_name_folder = self.audio_name.replace(".mp3", "").replace(".wav", "")
        frames_folder = os.path.join(
            OUTPUT_FOLDER, music_name_folder, f"frames_{current_time_formated}"
        )
        output_videos_folder = os.path.join(
            OUTPUT_FOLDER, music_name_folder, f"output_videos_{current_time_formated}"
        )

        # Creates the directory if debug_output_files is enabled
        if self.debug_output_files:
            os.makedirs(frames_folder, exist_ok=True)
            os.makedirs(output_videos_folder, exist_ok=True)

        notes_array = []
        prev_top_notes = []
        # Process each frame and analyse the frequencies
        for frame_num in tqdm.tqdm(range(self.number_of_windows)):
            start = frame_num * self.samples_per_window
            end = start + self.samples_per_window
            if end > len(self.audio_data):
                end = len(self.audio_data)
            frame_audio = self.audio_data[start:end]

            # Skip empty frames
            if len(frame_audio) == 0:
                continue

            fft = np.fft.rfft(frame_audio)
            fft = np.abs(fft)
            midi_notes, notes_buckets = self.__fft_to_notes_buckets(fft)
            notes_buckets = notes_buckets/mx_buckets

            top_notes = self.__get_top_notes(midi_notes, notes_buckets, prev_top_notes)
            top_notes = sorted(top_notes, key=lambda x: x.frequency)
            print("before: \n", top_notes) 
            top_notes = self.__filter_notes_and_harmonics(top_notes)
            print("after: \n", top_notes)


            if top_notes:
                for top_note in top_notes:
                    top_note.start_time = frame_num * WINDOW_TIME
                    top_note.start_bpm = Tools.seconds_to_beat(top_note.start_time, self.audio_bpm)

                notes_array.append(top_notes)
            
            """if top_notes:

                dominant_note = top_notes[0]
                if (
                    top_notes[0].variation < 0
                    and top_notes[0].magnitude < 0.2 * top_notes[0].maximum
                ):  # note is fading out
                    dominant_note = top_notes[1] if len(top_notes) > 1 else top_notes[0]

                notes_array.append(dominant_note.name)

            else:
                notes_array.append(None)"""

            self.__build_fig_matplotlib(midi_notes, notes_buckets, top_notes, f"{frames_folder}/fft_frame_{frame_num:04d}.png")
            # draw and save the figure
            """self.__build_fig_matplotlib(
                fft, mx, top_notes, f"{frames_folder}/fft_frame_{frame_num:04d}.png"
            )"""
        
        #exit(0)
        # Create the mp4 video if debug_output_files is enabled
        if self.debug_output_files:

            # Combine frames into a video using ffmpeg
            import subprocess

            video_no_audio = os.path.join(
                output_videos_folder, "output_video_no_audio.mp4"
            )
            final_video = os.path.join(output_videos_folder, "final_video.mp4")

            # Step 1: Create video from frames
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # Overwrite output files without asking
                    "-framerate",
                    str(self.fps),
                    "-i",
                    os.path.join(frames_folder, "fft_frame_%04d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    video_no_audio,
                ]
            )

            # Step 2: Combine video with audio
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_no_audio,
                    "-i",
                    self.audio_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    final_video,
                ]
            )

            print(f"Video with audio saved as: {final_video}")

        #print(self.__find_notes_length(notes_array))
        
        return self.audio_bpm, self.__find_notes_length(notes_array)
        # print(notes_array)

    def __filter_notes_and_harmonics(self, top_notes_sorted: list[Note]):
        if len(top_notes_sorted) < 1:
            return None
        nb_of_harmonics = 4
        min_good_harmonics = 2
        max_mag = np.max([note.magnitude for note in top_notes_sorted])
        top_notes_isorted_midi = [note.midi_number for note in top_notes_sorted]
        base_notes = []
        already_used_harmonics = []
        for base_note in top_notes_sorted:
            if base_note.magnitude < 0.4 * max_mag or base_note.midi_number in already_used_harmonics:
                continue
            th_harmonics = self.__get_harmonics(base_note, nb_of_harmonics)

            nb_wrong_harm = 0
            for th_harmonic in th_harmonics:
                if (not th_harmonic in top_notes_isorted_midi) or th_harmonic in already_used_harmonics:
                    nb_wrong_harm += 1
                else:
                    already_used_harmonics.append(th_harmonic)
                    print(Tools.note_name(th_harmonic))


            if nb_of_harmonics - nb_wrong_harm >= min_good_harmonics:
                base_notes.append(base_note)
    
        
        return base_notes
            
    def __get_harmonics(self, note: Note, nb_of_harmonics = 2):
        harmonics = []
        for i in range(2, nb_of_harmonics + 2):
            harmonics.append(Tools.freq_to_number(note.frequency*i))

        return harmonics

    def __find_instrument(self, notes):
        distances = []
        for i in range(len(notes) - 1):
            dist = abs(Tools.note_to_midi(notes[i + 1]) - Tools.note_to_midi(notes[i]))
            distances.append(dist)

        distance_mean = np.mean(distances)
        distance_median = np.median(distances)

    def __find_notes_length(self, notes: list[list[Note]]):
        if len(notes) < 1:
            return
        prev_notes = notes[0] # first list of played note(s)
        all_notes_with_duration:list[Note] = []
        for note in prev_notes:
            note.length = WINDOW_TIME
            all_notes_with_duration.append(note)

        for simult_notes in notes[1:]:
            for note in simult_notes:
                for prev_note in all_notes_with_duration[-5:]:
                    if (note.start_time == prev_note.start_time + prev_note.length
                        and prev_note.name == note.name):
                        prev_note.length += WINDOW_TIME
                        break
                else:
                    note.length = WINDOW_TIME
                    all_notes_with_duration.append(note)

        for note in all_notes_with_duration:
            beat_duration = Tools.seconds_to_beat(note.length, self.audio_bpm)
            snapped_beat_duration = round(beat_duration * 8) / 8
            note.length_bpm = snapped_beat_duration

        #print(all_notes_with_duration)
        return all_notes_with_duration

    def __fft_to_notes_buckets(self, fft):
        all_midi_notes = [i for i in range (0, 127)]
        notes_buckets = [0.0]*len(all_midi_notes) 
        for i in range(1, len(fft)):
            if fft[i] > 0:
                note_number = Tools.freq_to_number(self.xf[i]) - 12
                notes_buckets[note_number] += fft[i]

        return all_midi_notes, notes_buckets
    
    def __get_top_notes(self, midi_notes, notes_buckets, prev_notes, top_n=TOP_NOTES) -> list:
        if np.max(notes_buckets) < 0.1:
            return []
        peak_indexes = self.__magnitude_filter(notes_buckets)


        top_notes = [Note(Tools.number_to_freq(midi_notes[i]), notes_buckets[i]) for i in peak_indexes]

        
        # peak_freq = self.xf[fft[peak_indexes]]
        """print(peak_indexes)

        plt.plot(fft.real)
        # plt.plot(peak_indexes, self.xf[peak_indexes], color="red")

        plt.plot(peak_indexes, fft[peak_indexes], "x", color="red")
        plt.vlines(
            x=peak_indexes,
            ymin=fft[peak_indexes] - properties["prominences"],
            ymax=fft[peak_indexes],
            color="C1",
        )
        plt.show()"""

        # top_notes = [Note(self.xf[i], fft[i] / mx) for i in peak_indexes]

        #print(top_notes)

        return top_notes

        print(f"fisrt freq: {self.xf[0]}, last freq: {self.xf[-1]}")

        lst = [x for x in enumerate(fft.real)]
        lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)

        idx = 0
        found_notes = []
        found_notes_probabilities = {}
        while (idx < len(lst_sorted)) and (len(found_notes) < top_n):
            try:
                f = self.xf[lst_sorted[idx][0]]
                y = lst_sorted[idx][1] / mx

                if y < 0.02:  # Ignore very low magnitudes
                    idx += 1
                    continue

                n = freq_to_number(f)

                name = note_name(n)

                note_above_freq = number_to_freq(n + 1)
                note_below_freq = number_to_freq(n - 1)
                upper_bound_offset = int(
                    (note_above_freq - f - self.freq_dist_per_bin / 2)
                    // self.freq_dist_per_bin
                )
                lower_bound_offset = int(
                    (f - note_below_freq - self.freq_dist_per_bin / 2)
                    // self.freq_dist_per_bin
                )
                upper_index = lst_sorted[idx][0] + upper_bound_offset
                lower_index = lst_sorted[idx][0] - lower_bound_offset

                if upper_index >= len(lst_sorted) or lower_index < 0:
                    idx += 1
                    continue

                local_max = (
                    max(lst[lower_index:upper_index], key=lambda x: x[1])[1] / mx
                )
                """print(f"y: {y}, before: {lst[idx-1][1]/mx}, after: {lst[idx+1][1]/mx}")
                if y < lst[idx-1][1]/mx or y < lst[idx+1][1]/mx:
                    idx += 1
                    continue"""

                # print(f"current index: {idx} of freq: {xf[idx]}, lower_index: {lower_index} of freq: {xf[lower_index]}, upper_index: {upper_index} of freq: {xf[upper_index]}")
                # print(f"y: {y}, local_max: {local_max}")
                """if y < local_max:  # Ignore if the current value is not a local maximum
                    idx += 1
                    continue"""

                # diff = np.diff(lst[lower_index:lst_sorted[idx][0]][1])
                x_indices = np.arange(lower_index, lst_sorted[idx][0] + 1)
                y_values = [
                    item[1] / mx for item in lst[lower_index : lst_sorted[idx][0] + 1]
                ]
                up_slope, intercept = np.polyfit(x_indices, y_values, 1)

                x_indices = np.arange(lst_sorted[idx][0], upper_index)
                y_values = [
                    item[1] / mx for item in lst[lst_sorted[idx][0] : upper_index]
                ]

                down_slope, intercept = np.polyfit(x_indices, y_values, 1)

                # stdev = statistics.stdev(lst[lower_index:upper_index][1]/mx)

                print(
                    f"idx: {idx}, freq: {f}, number: {n}, name: {name}, magnitude: {y}, up_slope: {up_slope}, down_slope: {down_slope}"
                )
                if up_slope < 0.0 or down_slope > 0.0:
                    idx += 1
                    continue
                """if stdev > 2.0:
                    idx += 1
                    continue"""

                square_area = len(x_indices) * y
                signal_area = np.trapezoid(y_values, x_indices)

                area_difference = (square_area - signal_area) / square_area

                print(
                    f"area_difference: {area_difference}, square_area: {square_area}, signal_area: {signal_area}"
                )

                found_note = Note(frequency=f, magnitude=y)
                note_probability = area_difference

                if name not in found_notes_probabilities:
                    found_notes_probabilities[name] = [note_probability]
                    for prev_note in prev_notes:
                        if prev_note.name == name:
                            found_note.variation = (
                                found_note.magnitude - prev_note.magnitude
                            )
                            found_note.maximum = max(
                                found_note.maximum, prev_note.maximum
                            )
                            break

                    found_notes.append(found_note)
                """else:
                    found_notes_probabilities[name].append(note_probability)"""

            except Exception as e:
                pass
            idx += 1

        # Calculate the average probability for each note
        for note in found_notes:
            if note.name in found_notes_probabilities:
                mean_proba = np.mean(found_notes_probabilities[note.name])
                print(f"Note: {note.name}, Mean Probability: {mean_proba}")
                if mean_proba < 0.01:
                    found_notes.remove(note)

        return found_notes

    def __magnitude_filter(self, signal: np.ndarray):
        peaks_indexes = []
        norm_signal = signal.copy() / np.max(signal)

        for i in range(len(norm_signal)):
            if norm_signal[i] > 0.1 :
                peaks_indexes.append(i)
        return peaks_indexes

    def __build_fig_matplotlib(
        self,
        xf: np.ndarray,
        yf: np.ndarray,
        notes: list[Note],
        filename: str,
        dimensions: tuple[int, int] = (16, 8),
    ):
        """Creates a graph and saves it

        Args:
            fft (np.ndarray): fast fourrier transform array
            mx (int): max magnitude of the fft
            notes (list[Note]): list of notes
            filename (str): path to save the output image
            dimensions (tuple[int, int], optional): figure dimensions (in inch). Defaults to (16, 8).
        """
        if not self.debug_output_files:
            return
        plt.figure(figsize=dimensions)
        plt.bar(xf, yf, color="steelblue")
        plt.ylim(0, 1)
        plt.xlabel("Frequency (note)")
        plt.ylabel("Magnitude")
        plt.title("frequency spectrum")

        if notes:
            for note in notes:
                plt.annotate(
                    text=note.name,
                    xy=(note.midi_number, note.magnitude),
                    fontsize=12,
                    color="red",
                )
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


#audioAnalyzer = AudioAnalyzer("PinkPanther_Both.mp3", True, 5.0)
#audioAnalyzer.convert_to_notes()


"""
{timestamp: [notes]}
None durée

notes:
    velocity, durée, C#
"""
