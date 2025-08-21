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


def midi_maker(macro, track = 0, bpm = 110):
    """
    Create a .mid file as music.mid

    Args:
        macro : Dictionnary with the velocity, the duration and the note ("M0" : [60, 2, "G4"])
        track : Number of tracks (if 1 => 0)
        bpm : The tempo of the music
    
    Returns:
        The string of the path of the music.mid
    """

    channel = 0                                     # Channel number (btwn 0 and 15)
    time = 0                                        # Time of the first note (In Beats)
    duration = 1                                    # Duration of each note (In Beats)
    volume = 100                                    # 0-127

    MyMIDI = MIDIFile(1)

    MyMIDI.addTempo(track, time, bpm)

    # Adding the notes to the sheet music 
    for i, event in enumerate(macro):
        for j in range(0, len(macro[event]), 3):
            l = macro[event]
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
    
    return