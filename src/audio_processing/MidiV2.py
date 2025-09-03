from typing import Union
from midiutil import MIDIFile
import numpy as np
from audio_processing.audio_utils import Tools, Instrument, Note

import matplotlib.pyplot as plt


# from freq_analysis import AudioAnalyzer



def midi_maker(macro: list[Note], bpm: int, outfile: str = "music.mid") -> str:
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

    def get_slope(note : Note):
        # list note.magnitudes note.times
        y0 = note.magnitudes[0]
        x0 = note.times[0]
        y = note.maximum
        index_x = note.magnitudes.index(y)
        x = note.times[index_x]
        print(f"note magnitudes : {note.magnitudes}")
        print(f"note time : {note.times}")
        #print(f"y : {y}, y0 : {y0}, x : {x}, x0 : {x0}")
        return (y - y0)/(x - x0)

        #slope_list.append(slope)

    def get_instrument(note : Note, piano_range : tuple = (0,0), trumpet_range : tuple = (0,0)):
        slope = get_slope(note)
        try:
            if piano_range[1] >= slope >= piano_range[0]:
                note.instrument = Instrument.PIANO.value
            elif trumpet_range[1] >= slope >= trumpet_range[0]:
                note.instrument = Instrument.TRUMPET.value
            else: note.instrument
        except:
            print("Ranges are not well defined")

        return
    
    
    #slope_list = []

    # Adding the notes to the sheet music => Note(channel = instrument, pitch = midi note , time = starting time, duration, volume)
    sheet_music = sorted(macro, key=lambda note: note.start_bpm)


    for note in sheet_music:

        midi_note = Tools.note_to_midi(note.name)
        time = note.start_bpm
        duration = note.length_bpm
        volume = int((note.maximum) * 127)
        get_instrument(note)
        channel = note.instrument
        MyMIDI.addNote(
            track=track,
            channel=channel,
            pitch=midi_note,
            time=time,
            duration=duration,
            volume=volume,
        )
    
    
    

    
        
    # Creating the .mid file
    with open(outfile, "wb") as output_file:
        MyMIDI.writeFile(output_file)

    return outfile


