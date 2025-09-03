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

    # slope_list = []

    # Adding the notes to the sheet music => Note(channel = instrument, pitch = midi note , time = starting time, duration, volume)
    sheet_music = sorted(macro, key=lambda note: note.start_bpm)

    for note in sheet_music:

        midi_note = Tools.note_to_midi(note.name)
        time = note.start_bpm
        duration = note.length_bpm
        volume = int((note.maximum) * 127)
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
