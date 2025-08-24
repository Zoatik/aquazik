import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import midiutil
from difflib import SequenceMatcher
from freq_analysis import AudioAnalyzer
import matplotlib.pyplot as plt
import MidiV2 as mdi
import os


class MidiNote:
    def __init__(self, pitch, start: float, end: float):
        self.pitch = pitch
        self.start = start
        self.end = end
        self.length = end - start

    def __repr__(self) -> str:
        return f"pitch: {self.pitch}, start: {self.start}, end: {self.end}, length: {self.length}"


def midi_comparison(
    created_midi_path: str, reference_midi_path: str, results_output_path: str
):

    with open(reference_midi_path, "r") as rmn:
        reference_content = rmn.read()

    with open(created_midi_path, "r") as cmn:
        created_content = cmn.read()

    seq_match = SequenceMatcher(None, created_content, reference_content)
    print(seq_match.ratio)


def integration_comparison():
    pass


def note_relative_erors(ref_note: MidiNote, gen_note: MidiNote):
    start_diff = ref_note.start - gen_note.start
    end_diff = ref_note.end - gen_note.end
    length_diff = ref_note.length - gen_note.length

    rel_start_diff = start_diff / ref_note.length
    rel_end_diff = end_diff / ref_note.length
    rel_length_diff = length_diff / ref_note.length

    if rel_start_diff > 1.0:
        print("NOT GOOD: ", rel_start_diff)
        print(f"start ref : {ref_note.start}, start gen : {gen_note.start}")

    return rel_start_diff, rel_end_diff, rel_length_diff


audioAnalyzer = AudioAnalyzer("PinkPanther_Trumpet_Only.mp3", True)
bpm, notes_data = audioAnalyzer.convert_to_notes()

midi_out = mdi.midi_maker(notes_data, bpm=bpm)

ref_midi_path = os.path.join(
    os.path.abspath("audio_in"), "PinkPanther_Trumpet_Only.mid"
)
output_path = os.path.abspath("output")

# midi_comparison(midi_out, ref_midi_path, output_path)

import pretty_midi

midi_data = pretty_midi.PrettyMIDI(ref_midi_path)
ref_notes = {}
gen_notes = {}
print("*******************************************************")
print("duration:", midi_data.get_end_time())
print(f'{"note":>10} {"start":>10} {"end":>10}')
ref_array_x = []
ref_array_y = []
for instrument in midi_data.instruments:
    print("instrument:", instrument.program)
    for note in instrument.notes:
        ref_array_x.append(note.start)
        ref_array_x.append(note.end)
        ref_array_y.append(note.pitch)
        ref_array_y.append(note.pitch)
        if note.pitch not in ref_notes:
            ref_notes[note.pitch] = [MidiNote(note.pitch, note.start, note.end)]
        else:
            ref_notes[note.pitch].append(MidiNote(note.pitch, note.start, note.end))
        print(f"{note.pitch:10} {note.start:10} {note.end:10}")

plt.plot(ref_array_x, ref_array_y, color="blue")

print("*******************************************************")
midi_data = pretty_midi.PrettyMIDI(midi_out)
print("duration:", midi_data.get_end_time())
print(f'{"note":>10} {"start":>10} {"end":>10}')
gen_array_x = []
gen_array_y = []

for instrument in midi_data.instruments:
    print("instrument:", instrument.program)
    for note in instrument.notes:
        gen_array_x.append(note.start)
        gen_array_x.append(note.end)
        gen_array_y.append(note.pitch)
        gen_array_y.append(note.pitch)
        if note.pitch not in gen_notes:
            gen_notes[note.pitch] = [MidiNote(note.pitch, note.start, note.end)]
        else:
            gen_notes[note.pitch].append(MidiNote(note.pitch, note.start, note.end))
        print(f"{note.pitch:10} {note.start:10} {note.end:10}")

plt.plot(gen_array_x, gen_array_y, color="red")

missing_notes = []
notes_relative_start_errors = 0.0
notes_relative_end_errors = 0.0
notes_relative_length_errors = 0.0
notes_absolute_start_errors = 0.0
notes_absolute_end_errors = 0.0
notes_absolute_length_errors = 0.0
nb_of_samples = 0

for pitch in ref_notes.keys():
    for i in range(len(ref_notes[pitch])):
        ref_note = ref_notes[pitch][i]

        if pitch in gen_notes.keys() and i < len(gen_notes[pitch]):
            gen_note = gen_notes[pitch][i]
            start_error, end_error, length_error = note_relative_erors(
                ref_note, gen_note
            )
            notes_relative_start_errors += start_error
            notes_relative_end_errors += end_error
            notes_relative_length_errors += length_error
            notes_absolute_start_errors += abs(start_error)
            notes_absolute_end_errors += abs(end_error)
            notes_absolute_length_errors += abs(length_error)
            nb_of_samples += 1

        else:
            missing_notes.append(ref_note)


print("rel start error (sum): ", notes_relative_start_errors)
print("rel end error (sum): ", notes_relative_end_errors)
print("rel length error (sum): ", notes_relative_length_errors)
print("abs start error (sum): ", notes_absolute_start_errors)
print("abs end error (sum): ", notes_absolute_end_errors)
print("abs length error (sum): ", notes_absolute_length_errors)
print("rel start error (mean): ", notes_relative_start_errors / nb_of_samples)
print("rel end error (mean): ", notes_relative_end_errors / nb_of_samples)
print("rel length error (mean): ", notes_relative_length_errors / nb_of_samples)
print(
    "missing notes : ",
    sorted(
        [missing_notes[i] for i in range(len(missing_notes))], key=lambda x: x.start
    ),
)


# plt.show()
