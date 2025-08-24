import librosa
import os
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.signal import savgol_filter, medfilt


def rough_start_stop(audio_name="PinkPanther_Piano_Only.mp3"):

    INPUT_FOLDER = os.path.abspath("audio_in")
    audio_data, sr = librosa.load(
        os.path.join(INPUT_FOLDER, audio_name), mono=True, sr=None
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

    # look for chunks start / end
    smoothed_energies = medfilt(energies, kernel_size=11)
    smoothed_energies = medfilt(smoothed_energies, kernel_size=21)
    # smoothed_energies = savgol_filter(energies, window_length=11, polyorder=2)

    starts_x = []
    starts_y = []
    ends_x = []
    ends_y = []
    sound_treshhold = 0.01 * max_energy
    sound_low_lim = 0.01 * sound_treshhold
    min_dist = 3 * audio_duration / len(chunk_x)

    for i in range(len(chunk_x) - 1):
        if (
            smoothed_energies[i] < sound_treshhold
            and smoothed_energies[i + 1] > sound_treshhold
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


norm_signal, smoothed_data, starts_x, ends_x = rough_start_stop()
#  local_min, local_max = find_local_extremums(smoothed_data[1], smoothed_data[0])

for i in range(len(starts_x)):
    print(f"start: {smoothed_data[0][starts_x[i]]}, end: {smoothed_data[0][ends_x[i]]}")


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
