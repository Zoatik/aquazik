from enum import Enum, auto
import mido
from constants import NOTE_NAMES

DEBUG = False

class Instrument(Enum):
    PIANO = auto()
    TRUMPET = auto()


class MidiFile:
    note_list = []

    def __init__(self, midiFile):
        unfinished_notes = []

        mid = mido.MidiFile(midiFile)

        self.ticks_per_beat = mid.ticks_per_beat

        for track in mid.tracks:
            current_tick = 0
            for msg in track:
                if DEBUG:
                    print("midi_reader DBG: ", msg)
                match msg.type:
                    case "note_on":
                        unfinished_notes.append(
                            MidiNote(
                                self,
                                msg.note,
                                msg.velocity,
                                msg.channel,
                                current_tick + msg.time,
                            )
                        )
                    case "note_off":
                        note: MidiNote = next(
                            x for x in unfinished_notes if x.noteIndex == msg.note
                        )
                        unfinished_notes.remove(note)
                        note.set_end_ticks(current_tick + msg.time)
                        self.note_list.append(note)
                        if DEBUG:
                            print("midi_reader DBG:  note_off : " + note.get_real_note())
                    case "set_tempo":
                        self.tempo = msg.tempo
                current_tick += msg.time
        self.totalTime = self.note_list[-1].endSeconds

    def find_note(self, time_seconds: float):
        ret = []
        for x in self.note_list:
            if time_seconds >= x.startSeconds and time_seconds <= x.endSeconds:
                ret.append(x)
        return ret

    # returns a list of unique used notes in the midi file
    def get_used_notes(self) -> list[str]:
        return list(set([x.get_real_note()[:-1] for x in self.note_list]))

class MidiNote:
    endTicks = -1
    endSeconds = -1

    def __init__(self, midiFile, noteIndex, velocity, channel, timeTicks):
        self.parent = midiFile
        self.noteIndex = noteIndex
        self.velocity = velocity
        self.channel = channel
        self.startTicks = timeTicks
        self.startSeconds = mido.tick2second(
            timeTicks, self.parent.ticks_per_beat, self.parent.tempo
        )

    def get_instrument(self):
        if self.channel == 2:
            return Instrument.TRUMPET
        if self.channel == 11:
            return Instrument.PIANO

    def set_end_ticks(self, endTicks):
        self.endTicks = endTicks
        self.endSeconds = mido.tick2second(
            endTicks, self.parent.ticks_per_beat, self.parent.tempo
        )

    def get_real_note(self) -> str:
        octave = (self.noteIndex // 12) - 1
        note = NOTE_NAMES[self.noteIndex % 12]
        return f"{note}{octave}"

# test case
"""
mdi = MidiFile("audio_in/PinkPanther.midi")
for n in mdi.find_note(5.2):
    print(n.startSeconds, n.endSeconds)
"""