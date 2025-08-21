from midiutil import MIDIFile
import librosa

# Mapping of note names to MIDI numbers
NOTE_TO_MIDI = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

# Function to convert note and octave to MIDI number
def note_to_midi(note):
    note_name = note[:-1]  # Extract the note name (e.g., 'A', 'B', etc.)
    octave = int(note[-1])  # Extract the octave (e.g., '0', '2', etc.)
    return NOTE_TO_MIDI[note_name] + 12 * (octave + 1)

# Informations to fill
degrees = {
    "M0" : [60, 2, "G4"], "M1" : [None],  "M2" : [80, 1, "C4", 90, 2, "D4"]
}                                               # MIDI note number
track = 0                                       # Track number (if 1 => 0)
channel = 0                                     # Channel number (btwn 0 and 15)
time = 0                                        # Time of the first note (In Beats)
duration = 1                                    # Duration of each note (In Beats)
tempo =  110 #librosa.feature.rhythm.tempo("audio_in/Ecossaise_Both.mp3")
volume = 100                                    # 0-127

'''for i in note:
    degrees.append(note_to_midi(i))'''

MyMIDI = MIDIFile(1)

MyMIDI.addTempo(track, time, tempo)

for i, event in enumerate(degrees):
    for j in range(0, len(degrees[event]), 3):
        l = degrees[event]
        print(l)
        if l[j] is None:
            continue
        else: 
            volume = l[j]
            duration = l[j + 1]
            note = note_to_midi(l[j + 2])
            MyMIDI.addNote(track, channel, note, time + i, duration, volume)

with open("music.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)