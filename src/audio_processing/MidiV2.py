from midiutil import MIDIFile
from audio_processing.freq_analysis import Note
from audio_processing.midi_reader import Instrument

#from freq_analysis import AudioAnalyzer


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


# Function to convert note and octave to MIDI number
def note_to_midi(note):
    note_name = note[:-1]  # Extract the note name (e.g., 'A', 'B', etc.)
    octave = int(note[-1])  # Extract the octave (e.g., '0', '2', etc.)
    return NOTE_TO_MIDI[note_name] + 12 * (octave + 1)

# (bpm, [(0, [["G4", 2, 60],["C4", 2, 60]]),(0, [None, 2]], [1, [None , 2])])

def midi_maker(macro : list[Note], bpm, outfile="music.mid"):
    """Create a .mid file as music.mid

    Args:
        macro (List[Tuple]): List made with the velocity, the duration and the note ([60, 2, "G4"], [None, 2])
        bpm (Float): The tempo of the music

    Returns:
        String: The path of the created file (music.mid)
    """

    # Creating a midi file and adding the tempo
    track = 0
    MyMIDI = MIDIFile(1)
    MyMIDI.addTempo(track, 0, bpm)

    # Adding the notes to the sheet music => Note(start, duration, velocity, note_name)
    sheet_music = sorted(macro, key=lambda note: note.start_bpm)
    for note in sheet_music:
        try: 
            channel = 0 if note.instrument == Instrument.PIANO else 1
        except:
            channel = 0
        midi_note = note_to_midi(note.name)
        time = note.start_bpm
        duration = note.length_bpm
        volume = int(note.magnitude) * 127
        MyMIDI.addNote(track, channel, midi_note, time, duration, volume)
    
    #MyMIDI.addNote(track, channel, note, time, duration, volume)
    
    # create the .mid file
    with open(f"music2.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)

    return "music2.mid"


#baba = AudioAnalyzer("PinkPanther_Trumpet_Only.mp3")
#bpm, mama = baba.convert_to_notes()  # mama = [[C4, 1], [...]]
#babar = [(1, mama)]
#midi_maker(babar, bpm=bpm)