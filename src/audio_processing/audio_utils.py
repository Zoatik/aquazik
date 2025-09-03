import os
import numpy as np
from enum import Enum


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

INPUT_FOLDER = os.path.abspath("audio_in")
OUTPUT_FOLDER = os.path.abspath("audio_out")


class Instrument(Enum):
    PIANO = 11
    TRUMPET = 2
    UNKNOWN = -1


class Tools:
    NOTE_NAMES = NOTE_NAMES
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
    def note_to_midi(note: str) -> int:
        # "C#4" → name="C#", octave=4 → MIDI = 1 + 12*(4+1) = 61
        note_name = note[:-1]
        octave = int(note[-1])
        return Tools.NOTE_TO_MIDI[note_name] + 12 * (octave + 1)

    @staticmethod
    def seconds_to_beat(sec: float, bpm: float) -> float:
        return sec * (bpm / 60.0)

    @staticmethod
    def freq_to_number(f: float) -> int:
        # MIDI number
        return int(np.round(69 + 12 * np.log2(f / 440.0)))

    @staticmethod
    def number_to_freq(n: int) -> float:
        return 440.0 * (2.0 ** ((n - 69) / 12.0))

    @staticmethod
    def note_name(n: int) -> str:
        return NOTE_NAMES[n % 12] + str(int(n / 12 - 1))

    @staticmethod
    def freq_to_note(f: float) -> str:
        return Tools.note_name(Tools.freq_to_number(f))


class Note:
    def __init__(
        self,
        frequency: float,
        magnitudes: list[float],
        times: list[float],
        start=0.0,
        length=0.0,
    ):
        self.frequency = frequency
        self.start_time = start
        self.start_bpm = 0.0
        self.length = length
        self.length_bpm = 0.0
        self.midi_number = Tools.freq_to_number(self.frequency)
        self.name = Tools.note_name(self.midi_number)
        self.magnitudes = magnitudes  # List of magnitudes over time for this note
        self.times = times  # Times (in seconds) corresponding to each magnitude
        self.maximum = np.max(
            magnitudes
        )  # Store the maximum magnitude observed for this note
        self.instrument = Instrument.PIANO.value

        # Enveloppes
        self.h2_h1: float = 0.0
        self.h3_h1: float = 0.0
        self.odd_even: float = 0.0
        self.slope: float = 0.0
        self.attack: float = 0.0
        self.half_life: float = 0.0
        self.vib_rate: float = 0.0
        self.vib_depth: float = 0.0
        self.energy_median: float = 0.0
        self.energy_p95: float = 0.0

    def __repr__(self):
        return f"Note(frequency={self.frequency}, name='{self.name}', start={self.start_time}, bpm start={self.start_bpm} , bpm length={self.length_bpm}, magnitude={self.magnitudes})\n"

    def print_features(self):
        print(f"Note: {self.name} ({self.midi_number})")
        print(f"  Frequency: {self.frequency:.2f} Hz")
        print(f"  Start time: {self.start_time:.2f} s")
        print(f"  Length: {self.length:.2f} s")
        print("  Envelopes:")
        print(f"    h2/h1: {self.h2_h1:.2f}")
        print(f"    h3/h1: {self.h3_h1:.2f}")
        print(f"    odd/even: {self.odd_even:.2f}")
        print(f"    slope: {self.slope:.2f}")
        print(f"    attack: {self.attack:.2f}")
        print(f"    half-life: {self.half_life:.2f}")
        print(f"    vibrato rate: {self.vib_rate:.2f}")
        print(f"    vibrato depth: {self.vib_depth:.2f}")
        print(f"    energy median: {self.energy_median:.2f}")
        print(f"    energy p95: {self.energy_p95:.2f}")

    def set_bpm(self, bpm: int):
        self.start_bpm = round(Tools.seconds_to_beat(self.start_time, bpm) * 8) / 8
        self.length_bpm = round(Tools.seconds_to_beat(self.length, bpm) * 8) / 8
