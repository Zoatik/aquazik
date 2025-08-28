from midiutil import MIDIFile

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

def midi_maker(macro, bpm):
    """Create a .mid file as music.mid

    Args:
        macro (List[Tuple]): List made with the velocity, the duration and the note ([60, 2, "G4"], [None, 2])
        bpm (Float): The tempo of the music

    Returns:
        String: The path of the created file (music.mid)
    """
    time = 0
    track = 0
    MyMIDI = MIDIFile(1)
    MyMIDI.addTempo(track, time, bpm)

    # Adding the notes to the sheet music
    # track = [(0, [["G4", 2, 60],["C4", 2, 60]]),(0, [None, 2]], [1, [None , 2])]
    for i in range(0, len(macro)):
        time = 0
        element = macro[i]                                  #(0, [["G4", 2, 60],["C4", 2, 60]])
        print(f"element:{element}")
        channel_notes = element[1]                          #[["G4", 2, 60],["C4", 2, 60]]
        print(f"channel notes: {channel_notes}")
        channel = int(element[0])                           # 0 = piano / 1 = trumpet
        print(f"channel: {channel}")
        for parameters in channel_notes:
            try:
                length = len(parameters)
            except:
                length = 1
            for j in range(0, length, 3):
                if parameters[j] is None:
                    time += parameters[j + 1]
                    continue
                else:
                    try:
                        volume = parameters[j + 2]
                    except:
                        volume = 100  # 0-127
                    duration = parameters[j + 1]
                    note = note_to_midi(parameters[j])
                    MyMIDI.addNote(track, channel, note, time, duration, volume)
                    time += duration


    # create the .mid file
    with open(f"music.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)

    return "music.mid"


#baba = AudioAnalyzer("PinkPanther_Trumpet_Only.mp3")
#bpm, mama = baba.convert_to_notes()  # mama = [[C4, 1], [...]]
#babar = [(1, mama)]
#midi_maker(babar, bpm=bpm)


'''
    # Separate in 2 tracks according to the instrument
    track_piano = []
    track_trumpet = [] 
    for i in macro:
        match i[0]:
            case 0: track_piano.append(i)
            case 1: track_trumpet.append(i), print(f"track trumpet: {track_trumpet}")
            case _: continue
    '''
'''
    # Detect wich instrument(s) are selected
    track_selected = [0 if len(track_piano) == 0 else 1, 0 if len(track_trumpet) == 0 else 1]    
    '''
    
    #def sheet_music(track):
"""Compose the midi file with the notes given

    Args:
        track (List[Tuple]): A track of a specific instrument (ex. [(0, [67, 1.5, "C4"],[None, 1]),(0, [80, 0.5, "G4"])]

    Returns:
        String: The path of the created file (music.mid)
    """


'''
    match track_selected:
        case [1,1]: return sheet_music(track_piano), sheet_music(track_trumpet)
        case [1,0]: return sheet_music(track_piano)
        case [0,1]: return sheet_music(track_trumpet)
        case _: return 
    '''

partition = [
    # Slice 1 : le piano tient un accord 2 temps, la trompette est en silence
    (0, [["C4", 2, 80], ["G4", 2, 80]]),
    (1, [[None, 2]]),

    # Slice 2 : les deux jouent 2 temps
    (0, [["E4", 2, 78]]),
    (1, [["C5", 2, 88]]),

    # Slice 3 : accord bref au piano (1 temps), trompette 1 temps
    (0, [["F4", 1, 80], ["A4", 1, 80], ["C5", 1, 80]]),
    (1, [["D5", 1, 90]]),

    # Slice 4 : silence piano, trompette 1 temps
    (0, [[None, 1]]),
    (1, [["E5", 1, 92]]),

    # Slice 5 : piano accord de 2 temps, trompette silence
    (0, [["G4", 2, 82], ["B4", 2, 82]]),
    (1, [[None, 2]]),

    # Slice 6 : les deux 1 temps
    (0, [["E4", 1, 76]]),
    (1, [["C5", 1, 86]]),

    # Slice 7 : les deux 1 temps
    (0, [["D4", 1, 76]]),
    (1, [["A#4", 1, 84]]),

    # Slice 8 : cadences finales tenues 4 temps
    (0, [["C4", 4, 85], ["E4", 4, 85], ["G4", 4, 85]]),
    (1, [["C5", 4, 90]]),
]
midi_maker(partition, bpm=120)