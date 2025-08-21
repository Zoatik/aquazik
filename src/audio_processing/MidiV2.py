from midiutil import MIDIFile
import librosa
from freq_analysis import AudioAnalyzer

baba = AudioAnalyzer("PinkPanther_Trumpet_Only.mp3")
bpm, mama = baba.convert_to_notes() # mama = [[C4, 1], [...]]

# Mapping of note names to MIDI numbers
NOTE_TO_MIDI = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

# Function to convert note and octave to MIDI number
def note_to_midi(note):
    note_name = note[:-1]  # Extract the note name (e.g., 'A', 'B', etc.)
    octave = int(note[-1])  # Extract the octave (e.g., '0', '2', etc.)
    return NOTE_TO_MIDI[note_name] + 12 * (octave + 1)


def midi_maker(macro, track = 0, bpm = 110, channel = 0):
    """
    Create a .mid file as music.mid

    Args:
        macro : list made with the velocity, the duration and the note ([60, 2, "G4"], [None])
        track : Number of tracks (if 1 => 0)
        bpm : The tempo of the music
        channel : Channel number (btwn 0 and 15)
    
    Returns:
        The string of the path of the music.mid
    """

    time = 0                                        # Time of the first note (In Beats)
    volume = 100                                    # 0-127

    MyMIDI = MIDIFile(1)

    MyMIDI.addTempo(track, time, bpm)

    # Adding the notes to the sheet music 
    for element in macro:
        for j in range(0, len(element), 3):
            if element[j] is None:
                time += element[j + 1]
                print(f"pause: {time}")
                continue
            else:
                print(f"start: {time}") 
                #volume = l[j + 2]
                duration = element[j + 1]
                print(f"duration : {duration}")
                note = note_to_midi(element[j])
                MyMIDI.addNote(track, channel, note, time, duration, volume)
                time += duration

    with open("music.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)
    
    return "music.mid"
print(mama)
midi_maker(macro= mama, bpm=bpm)