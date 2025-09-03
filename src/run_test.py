import os
from audio_processing.baba import convert_to_midi
from audio_processing.audio_utils import INPUT_FOLDER, OUTPUT_FOLDER, Instrument
from audio_processing.instrument_split import analyse_features, assign_instrument
from audio_processing.MidiV2 import midi_maker


audio_name = "SuperMario"
audio_in_name = f"{audio_name}.mp3"
audio_out_name = f"{audio_name}.mid"

json_out_path = os.path.abspath("instruments_features.json")

audio_in_path = os.path.join(INPUT_FOLDER, audio_in_name)
audio_out_path = os.path.join(OUTPUT_FOLDER, audio_out_name)


bpm, notes = convert_to_midi(audio_in_path, audio_out_path, debug=False)

"""analyse_features(audio_in_path, Instrument.PIANO, json_path=json_out_path)

audio_name = "PinkPanther_Trumpet_Only"
audio_in_name = f"{audio_name}.mp3"
audio_out_name = f"{audio_name}.mid"

audio_in_path = os.path.join(INPUT_FOLDER, audio_in_name)
audio_out_path = os.path.join(OUTPUT_FOLDER, audio_out_name)

analyse_features(audio_in_path, Instrument.TRUMPET, json_path=json_out_path)"""


notes_with_instrument = assign_instrument(notes, features_used=["slope"])

midi_maker(notes_with_instrument, bpm, audio_out_path)
