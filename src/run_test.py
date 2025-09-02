import os
from audio_processing.baba import convert_to_midi
from audio_processing.audio_utils import INPUT_FOLDER, OUTPUT_FOLDER


audio_name = "Gamme"
audio_in_name = f"{audio_name}.mp3"
audio_out_name = f"{audio_name}.mid"

audio_in_path = os.path.join(INPUT_FOLDER, audio_in_name)
audio_out_path = os.path.join(OUTPUT_FOLDER, audio_out_name)


convert_to_midi(audio_in_path, audio_out_path, debug=True)
