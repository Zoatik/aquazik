import librosa
import os
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, interpolate
from scipy.signal import savgol_filter, medfilt, detrend


def rough_start_stop(audio_name="PinkPanther_Trumpet_Only.mp3"):

    INPUT_FOLDER = os.path.abspath("audio_in")
    audio_data, sr = librosa.load(
        os.path.join(INPUT_FOLDER, audio_name), mono=True
    )

    audio_duration = librosa.get_duration(y=audio_data, sr=sr)

    times = np.arange(len(audio_data)) / sr
    max_audio = np.max(np.abs(audio_data))
    norm_audio_data = (audio_data / max_audio) ** 2

    window_time = 0.01
    samples_per_chunk = int(sr * window_time)
    energies = []

    nb_of_chunks = 0
    for start in range(0, len(norm_audio_data), samples_per_chunk):
        end = start + samples_per_chunk
        chunk = norm_audio_data[start:end]

        # Ex: énergie de la fenêtre
        energy = np.sum(chunk) / len(chunk)
        energies.append(energy)
        nb_of_chunks += 1

    chunk_x = np.arange(0, len(norm_audio_data), samples_per_chunk) / sr
    max_energy = np.max(energies)


    print(len(norm_audio_data))
    print(samples_per_chunk * nb_of_chunks)

    sample_trend = medfilt(norm_audio_data, kernel_size=samples_per_chunk * 100 + 1)
    sample_without_trend = norm_audio_data - sample_trend
    sample_formatted_y = []
    for el in sample_without_trend:
        if el-0.001 > 0.0:
            sample_formatted_y.append(el)

    sample_formatted_x = np.arange(0, len(sample_formatted_y)) / sr
    # look for chunks start / end
    smoothed_energies = medfilt(energies, kernel_size=11)
    #smoothed_energies_without_trend = smoothed_energies - sample_trend
    # smoothed_energies = medfilt(smoothed_energies, kernel_size=21)
    # smoothed_energies = savgol_filter(energies, window_length=11, polyorder=2)

    #plt.plot(times, norm_audio_data)
    #plt.plot(chunk_x, smoothed_energies)
    #plt.plot(times, sample_trend)
    plt.plot(sample_formatted_x, sample_formatted_y)
    #plt.plot(times, sample_without_trend)
    plt.show()
    exit(0)


    starts_x = []
    starts_y = []
    ends_x = []
    ends_y = []
    sound_treshhold = 0.01 * max_energy  # originally 0.01
    min_prominence = 0.1 * max_energy
    sound_low_lim = 0.01 * sound_treshhold
    min_dist = 3 * audio_duration / len(chunk_x)

    for i in range(len(chunk_x) - 19):
        if (
            smoothed_energies[i] < sound_treshhold
            and np.max(smoothed_energies[i + 1 : i + 5]) > sound_treshhold
        ):
            if len(starts_x) - len(ends_x) < 1:  # must be in pairs
                if len(starts_x) < 1 or chunk_x[i] - chunk_x[starts_x[-1]] > min_dist:
                    starts_x.append(i)

        elif (
            smoothed_energies[i + 1] < sound_treshhold
            and smoothed_energies[i] > sound_treshhold
        ):
            if len(ends_x) - len(starts_x) < 0:  # must be in pairs
                if len(ends_x) < 1 or chunk_x[i] - chunk_x[ends_x[-1]] > min_dist:
                    j = i
                    while True:
                        j += 1
                        if (
                            j + 1 > len(smoothed_energies)
                            or smoothed_energies[j + 1] > smoothed_energies[j]
                            or smoothed_energies[j] <= sound_low_lim
                        ):
                            ends_x.append(j)
                            break

    print("sound tresh: ", sound_treshhold)
    return ((times, norm_audio_data), (chunk_x, smoothed_energies), starts_x, ends_x)


def enhanced_start_stop(x, raw_data):  # TODO
    window = 100
    instability = np.array(
        [
            np.var(
                raw_data[max(0, i - window // 2) : min(len(raw_data), i + window // 2)]
            )
            for i in range(len(raw_data))
        ]
    )

    threshold = np.mean(instability) + 2 * np.std(instability)
    outliers = np.where(instability > threshold)[0]

    plt.plot(x, raw_data)
    plt.plot(x[outliers], raw_data[outliers], "ro")


def estimate_bpm(x_axis, starts_x, ends_x):
    print(len(starts_x))
    if len(starts_x) < 2:
        return None
    total_length = x_axis[starts_x[-1]] - x_axis[starts_x[0]]
    distances = [
        x_axis[starts_x[i + 1]] - x_axis[starts_x[i]] for i in range(len(starts_x) - 1)
    ]
    distances = sorted(distances)

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    print(min_dist)
    print(max_dist)

    #### It does not matter if the bpm is n or 2*n or x*n.
    #### It will be adjusted with the beats length (1/n * length)

    max_error_for_1_beat = 0.05
    min_notes_per_bucket = 0
    bucket_x = 0.01
    print("bucket x: ", bucket_x)
    x_arr = np.arange(min_dist, max_dist, bucket_x)
    y_arr = np.zeros_like(x_arr)
    for el in distances:
        prev_dist = -1
        fitting_idx = 0
        for i in range(len(x_arr)):
            dist = abs(x_arr[i] - el)
            if dist < prev_dist or prev_dist < 0:
                fitting_idx = i
                prev_dist = dist

        y_arr[fitting_idx] += 1

    new_x = [x_arr[i] for i in range(len(x_arr)) if y_arr[i] >= min_notes_per_bucket]
    new_y = [y_arr[i] for i in range(len(x_arr)) if y_arr[i] >= min_notes_per_bucket]

    max = None
    max_idx = -1
    for i in range(len(new_x)):
        if max == None or new_y[i] > max:
            max = new_y[i]
            max_idx = i

    quarter_note = None
    mult_notes = []
    for i in range(len(new_x)):
        if quarter_note == None:
            if is_local_maximum(i, new_y):
                quarter_note = new_x[i]
        else:
            i = (i) * 2
            if i > len(new_x):
                break
            lower_bound = i - 3 if i > 2 else 0
            upper_bound = i + 3 if len(new_x) > i else len(new_x)
            loc_max = find_local_maximums(
                new_x[lower_bound:upper_bound], new_y[lower_bound:upper_bound]
            )
            print("loc maxs: ", loc_max[0])
            if len(loc_max[0]) > 0:
                most_sig_y = loc_max[1][0]
                most_sig_x = loc_max[0][0]
                for j in range(1, len(loc_max[0])):
                    if loc_max[1][j] > most_sig_y:
                        most_sig_y = loc_max[1][j]
                        most_sig_x = loc_max[0][j]
                if not most_sig_x in mult_notes and most_sig_x != quarter_note:
                    mult_notes.append(most_sig_x)

    print("quarter note : ", quarter_note)
    print("mult notes :", mult_notes)
    coeffs = [int(x / quarter_note) for x in mult_notes if x % quarter_note < 0.1]
    mult_notes = [x for x in mult_notes if x % quarter_note < 0.1]
    print("coeffs: ", coeffs)
    m = np.mean(
        [quarter_note] + [mult_notes[i] / coeffs[i] for i in range(len(mult_notes))]
    )
    print("mean: ", m)
    est_bpm = 1 / m * 60.0
    print(f"bpm: {est_bpm}, noire: {m}")
    # plt.plot(new_x, new_y)
    # plt.show()
    # exit(0)

    return quarter_note, est_bpm


def is_local_minimum(x, arr):
    if len(arr) < 1:
        return False
    y = arr[x]
    left_y = arr[x - 1] if x > 1 else y + 1
    right_y = arr[x + 1] if len(arr) > x + 1 else y + 1

    l = 1
    r = 1
    while True:
        if y > left_y or y > right_y:
            return False
        if y < left_y and y < right_y:
            return True
        if y == right_y:
            r += 1
        if y == left_y:
            r += 1

        left_y = arr[x - l] if x - l > 0 else y - 1
        right_y = arr[x + r] if x + r < len(arr) else y - 1


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


def find_local_minimums(xs, ys, count_borders=False):
    if len(ys) < 1:
        return [], []
    mins_x = []
    mins_y = []
    for i in range(1, len(ys) - 1):
        if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]:
            mins_x.append(xs[i])
            mins_y.append(ys[i])
    if count_borders and len(ys) > 1:
        if ys[-1] < ys[-2]:
            mins_x.append(xs[len(ys) - 1])
            mins_y.append(ys[-1])
        if ys[0] < ys[1]:
            mins_x.append(xs[0])
            mins_y.append(ys[0])

    return mins_x, mins_y


def find_local_maximums(xs, ys, count_borders=False):
    if len(ys) < 1:
        return [], []
    maxs_x = []
    maxs_y = []
    for i in range(1, len(ys) - 1):
        if ys[i] > ys[i - 1] and ys[i] > ys[i + 1]:
            maxs_x.append(xs[i])
            maxs_y.append(ys[i])
    if count_borders and len(ys) > 1:
        if ys[-1] > ys[-2]:
            maxs_x.append(xs[len(ys) - 1])
            maxs_y.append(ys[-1])
        if ys[0] > ys[1]:
            maxs_x.append(xs[0])
            maxs_y.append(ys[0])

    return maxs_x, maxs_y


def find_local_extremums(smoothed_array, x_array):  # resources heavy
    local_max_x = []
    local_max_y = []
    local_min_x = []
    local_min_y = []
    for i in range(len(smoothed_array)):
        if is_local_maximum(i, smoothed_array):
            local_max_x.append(x_array[i])
            local_max_y.append(smoothed_array[i])

        elif is_local_minimum(i, smoothed_array):
            local_min_x.append(x_array[i])
            local_min_y.append(smoothed_array[i])

    return (local_min_x, local_min_y), (local_max_x, local_max_y)


norm_signal, smoothed_data, starts_x, ends_x = rough_start_stop("PinkPanther_Both.mp3")
#  local_min, local_max = find_local_extremums(smoothed_data[1], smoothed_data[0])

for i in range(min(len(ends_x), len(starts_x))):
    print(f"start: {smoothed_data[0][starts_x[i]]}, end: {smoothed_data[0][ends_x[i]]}")

bpm_data = estimate_bpm(smoothed_data[0], starts_x, ends_x)
quarter_note = None
bpm = None
if bpm_data is not None:
    quarter_note, bpm = bpm_data

print(f"temps de la noire: {quarter_note}, bpm : {bpm}")

plt.plot(norm_signal[0], norm_signal[1])
plt.plot(smoothed_data[0], smoothed_data[1])
plt.plot(smoothed_data[0][starts_x], smoothed_data[1][starts_x], "x")
plt.plot(smoothed_data[0][ends_x], smoothed_data[1][ends_x], "x")


# enhanced_start_stop(norm_signal[0], norm_signal[1])
# plt.plot(local_min[0], local_min[1], "o")
# plt.plot(local_max[0], local_max[1], "o")


plt.show()

"""
rising_edges_x = []
rising_edges_y = []


# look for raising edges
for i in range(len(chunk_x) - 1):
    enthropy = energies[i - 1] / max_energy if i > 0 else 0.0

    if (energies[i + 1] - energies[i]) / max_energy > enthropy:
        rising_edges_x.append(chunk_x[i])
        rising_edges_x.append(chunk_x[i + 1])
        rising_edges_y.append(energies[i])
        rising_edges_y.append(energies[i + 1])

    elif (
        i + 2 < len(chunk_x)
        and (energies[i + 2] - energies[i]) / (2 * max_energy) > enthropy
    ):
        rising_edges_x.append(chunk_x[i])
        rising_edges_x.append(chunk_x[i + 1])
        rising_edges_y.append(energies[i])
        rising_edges_y.append(energies[i + 1])
        print("new rising edges found !")

segments = [
    [
        (rising_edges_x[i], rising_edges_y[i]),
        (rising_edges_x[i + 1], rising_edges_y[i + 1]),
    ]
    for i in range(0, len(rising_edges_x), 2)
]

"""

"""
# plt.plot(times, abs(norm_audio_data), color="blue")
plt.plot(times, np.pow(norm_audio_data, 2), color="green")


plt.plot(chunk_x, energies, color="orange")
smoothed_energy = savgol_filter(energies, window_length=11, polyorder=2)
plt.plot(chunk_x, smoothed_energy)


ax = plt.gca()  # axe courant (ne crée PAS de nouvelle figure)
if segments:  # éviter d'ajouter une collection vide
    lc = LineCollection(
        segments, colors="r", linewidths=2, label="Rising edges", zorder=5
    )
    ax.add_collection(lc)
    ax.autoscale_view()  # au cas où les segments dépassent la vue actuelle
ax.autoscale()

plt.plot(chunk_x, smoothed_energies, label="Énergie médiane")
plt.plot(starts_x, starts_y, "x", color="violet")
plt.plot(ends_x, ends_y, "x", color="black")


plt.show()
"""
