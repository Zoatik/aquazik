import mido
from mido import MidiFile, MidiTrack, Message
import os

# Mapping of note names to MIDI numbers
NOTE_TO_MIDI = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

# Function to convert note and octave to MIDI number
def note_to_midi(note):
    note_name = note[:-1]  # Extract the note name (e.g., 'A', 'B', etc.)
    octave = int(note[-1])  # Extract the octave (e.g., '0', '2', etc.)
    return NOTE_TO_MIDI[note_name] + 12 * (octave + 1)

# Function to convert length notation to MIDI ticks
def length_to_ticks(length, ticks_per_beat):
    if length == 1:
        return ticks_per_beat * 4  # Whole note
    elif length == 2:
        return ticks_per_beat * 2  # Half note
    elif length == 4:
        return ticks_per_beat  # Quarter note
    elif length == 8:
        return ticks_per_beat // 2  # Eighth note
    elif length == 16:
        return ticks_per_beat // 4  # Sixteenth note
    return ticks_per_beat  # Default to quarter note

# Function to generate MIDI file based on the given sequence and tempo
def generate_piano_midi(sequence, tempo, output_file="output/music_lent.mid", instrument_program=0):
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    track.append(Message('program_change', program=instrument_program))
    microseconds_per_beat = 60 * 1000000 // tempo
    track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))

    ticks_per_beat = 60
    current_time = 0

    for i in range(0, len(sequence), 3):
        note = sequence[i]  # Note name (e.g., 'A0')
        length = sequence[i + 1]  # Length (e.g., 4 for quarter note)
        velocity = sequence[i + 2]  # Velocity (e.g., 100 for volume)

        if note == 'X':  # Handle pauses (no sound)
            track.append(Message('note_on', note=0, velocity=0, time=current_time))
            track.append(Message('note_off', note=0, velocity=0, time=length_to_ticks(length, ticks_per_beat)))
            current_time = 0
            continue

        midi_note = note_to_midi(note)
        duration = length_to_ticks(length, ticks_per_beat)

        track.append(Message('note_on', note=midi_note, velocity=velocity, time=current_time))
        track.append(Message('note_off', note=midi_note, velocity=velocity, time=duration))

        current_time = 0

    midi.save(output_file)
    print(f"MIDI file '{output_file}' has been saved!")

# Function to expand macros
def expand_macros(sequence, macros):
    expanded_sequence = []
    i = 0
    while i < len(sequence):
        if isinstance(sequence[i], str) and sequence[i].startswith('M'):  # Macro starts with 'M'
            macro_name = sequence[i]
            if macro_name in macros:
                expanded_sequence.extend(macros[macro_name])  # Expand the macro
            i += 1  # Skip the macro name
        else:
            expanded_sequence.append(sequence[i])
            i += 1
    return expanded_sequence

# Informations to fill 

macros = {
    # M001 : motif sur C (C majeur)
    "M001": [
        "C4", 8, 96,  "E4", 8, 96,  "G4", 8, 96,  "C5", 8, 100,
        "E5", 8, 96,  "G4", 8, 94,  "E4", 8, 92,  "C4", 8, 90
    ],
    # M002 : réponse sur F
    "M002": [
        "F4", 8, 96,  "A4", 8, 96,  "C5", 8, 96,  "F5", 8, 100,
        "A4", 8, 94,  "C5", 8, 92,  "A4", 8, 90,  "F4", 8, 88
    ],
    # M003 : dominante G (pré-cadence)
    "M003": [
        "G3", 8, 92,  "D4", 8, 96,  "G4", 8, 98,  "B4", 8, 100,
        "D5", 8, 96,  "G4", 8, 94,  "D4", 8, 92,  "G3", 8, 90
    ],
    # M004 : cadence respirée (noires)
    "M004": [
        "E4", 4, 96,  "D4", 4, 94,  "C4", 4, 100,  "G3", 4, 92
    ],
    # M005 : fin (demi + 2 noires) = 1 mesure
    "M005": [
        "C4", 2, 104,  "G3", 4, 92,  "C4", 4, 112
    ]
}

# --- PARTITION (enchaînement des mesures)
sequence = [
    "M001", "M002", "M003", "M004",   # 4 premières mesures
    "M001", "M002", "M003", "M005"    # reprise + conclusion
]

expanded_sequence = expand_macros(sequence=sequence, macros=macros) 

generate_piano_midi(sequence= expanded_sequence, tempo=110)