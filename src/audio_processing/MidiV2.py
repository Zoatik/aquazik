from midiutil import MIDIFile
import librosa

from freq_analysis import AudioAnalyzer


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

# (bpm, [(0, [60, 2, "G4"],[60, 2, "C4"]),(0, [None, 2]], [1, [None , 2])])

def midi_maker(macro, bpm):
    """Create a .mid file as music.mid

    Args:
        macro (List[Tuple]): List made with the velocity, the duration and the note ([60, 2, "G4"], [None, 2])
        bpm (Float): The tempo of the music

    Returns:
        String: The path of the created file (ex. music_piano.mid)
    """

    # Separate in 2 tracks according to the instrument
    track_piano = []
    track_trumpet = [] 
    for i in macro:
        match i[0]:
            case 0: track_piano.append(i)
            case 1: track_trumpet.append(i), print(f"track trumpet: {track_trumpet}")
            case _: continue


    volume = 100  # 0-127

    # Detect wich instrument(s) are selected
    track_selected = [0 if len(track_piano) == 0 else 1, 0 if len(track_trumpet) == 0 else 1]    

    def sheet_music(track):
        """Compose the midi file with the notes given

        Args:
            track (List[Tuple]): A track of a specific instrument (ex. [(0, [67, 1.5, "C4"],[None, 1]),(0, [80, 0.5, "G4"])]

        Returns:
            String: The path of the created file (ex. music_piano.mid)
        """

        nb_of_the_track = 0 
        time = 0

        MyMIDI = MIDIFile(1)
        MyMIDI.addTempo(nb_of_the_track, time, bpm)

        nb_channel = len(track)

        # Adding the notes to the sheet music
        # track = [(0, [[60, 2, "G4"],[60, 2, "C4"]]),(0, [None, 2]], [1, [None , 2])]
        for c in range(0, nb_channel):
            time = 0
            channel = c
            print(f"channel : {channel}")
            track_channel = track[c]
            print(f"track channel:{track_channel}")
            channel_notes = track_channel[1]
            print(f"channel notes: {channel_notes}")
            for element in channel_notes:
                for j in range(0, len(element), 3):
                    if element[j] is None:
                        time += element[j + 1]
                        continue
                    else:
                        duration = element[j + 1]
                        note = note_to_midi(element[j])
                        MyMIDI.addNote(nb_of_the_track, channel, note, time, duration, volume)
                        time += duration

        # Extract the name of the instrument (0 = piano, 1 = trumpet)
        ftuple = track[0]
        n = ftuple[0]
        name = "piano" if n == 0 else "trumpet"


        with open(f"music_{name}.mid", "wb") as output_file:
            MyMIDI.writeFile(output_file)

        return f"music_{name}.mid"

    match track_selected:
        case [1,1]: return sheet_music(track_piano), sheet_music(track_trumpet)
        case [1,0]: return sheet_music(track_piano)
        case [0,1]: return sheet_music(track_trumpet)
        case _: return 


baba = AudioAnalyzer("PinkPanther_Trumpet_Only.mp3")
bpm, mama = baba.convert_to_notes()  # mama = [[C4, 1], [...]]
babar = [(1, mama)]
midi_maker(babar, bpm=bpm)
