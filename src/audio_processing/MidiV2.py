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
def note_to_midi(note : str):
    """Finding the pitch according to the note

    Args:
        note (str): The note (ex. C4)

    Returns:
        Int: the pitch (ex. 60)
    """
    note_name = note[:-1] 
    octave = int(note[-1])
    return NOTE_TO_MIDI[note_name] + 12 * (octave + 1)

def midi_maker(macro : list[Note], bpm : int, outfile : str ="music.mid"):
    """Create a .mid file as music.mid

    Args:
        macro (list[Note]): List made with the velocity, the duration and the note ([60, 2, "G4"], [None, 2])
        bpm (float): The tempo of the music

    Returns:
        str: The path of the created file (music.mid)
    """

    # Initialising the midi file and adding the tempo
    track = 0
    MyMIDI = MIDIFile(1)
    MyMIDI.addTempo(track, 0, bpm)
    

    # Adding the notes to the sheet music => Note(channel = instrument, pitch = midi note , time = starting time, duration, volume)
    sheet_music = sorted(macro, key=lambda note: note.start_bpm)
    for note in sheet_music:
        channel = note.instrument
        midi_note = note_to_midi(note.name)
        time = note.start_bpm
        duration = note.length_bpm
        volume = int((note.magnitude) * 127)
        MyMIDI.addNote(track=track, channel=channel, pitch=midi_note, time=time, duration=duration, volume=volume)
    
    # Creating the .mid file
    with open(f"music.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)

    return "music.mid"

