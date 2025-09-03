from typing import Union
from midiutil import MIDIFile
import numpy as np
from audio_processing.audio_utils import Tools, Instrument, Note

import matplotlib.pyplot as plt


# from freq_analysis import AudioAnalyzer

def feature_to_gaussian(values: Union[list[float], np.ndarray]) -> dict:
        """
        Calcule des statistiques (moyenne, médiane, std, etc.) 
        pour résumer une feature comme une distribution gaussienne approx.

        Args:
            values (list[float] | np.ndarray): les valeurs de la feature.

        Returns:
            dict: {
                "mean": float,
                "median": float,
                "std": float,
                "var": float,
                "min": float,
                "max": float,
                "q25": float,
                "q75": float,
                "iqr": float
            }
        """
        if values is None or len(values) == 0:
            return {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "var": np.nan,
                "min": np.nan,
                "max": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "iqr": np.nan,
            }

        arr = np.asarray(values, dtype=float)

        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)),   # écart-type corrigé (n-1)
            "var": float(np.var(arr, ddof=1)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Densité de la loi normale N(mu, sigma^2) sans dépendre de scipy."""
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros_like(x, dtype=float)
    v = sigma ** 2
    return (1.0 / np.sqrt(2.0 * np.pi * v)) * np.exp(-0.5 * (x - mu) ** 2 / v)


def plot_feature_gaussian(
    features: dict[str, list[float]],
    *,
    bins: int = 30,
    save_dir: str | None = None,
    show: bool = True,
):
    """
    Trace un histogramme + gaussienne ajustée pour chaque feature.

    Args:
        features: dict {nom_feature: liste_de_valeurs}
        bins: nombre de binnings pour l'histogramme
        save_dir: si fourni, sauvegarde chaque figure en PNG dans ce dossier
        show: si False, ne fait pas plt.show(); utile pour l'appel batch/tests
    """
    if features is None:
        return

    for name, vals in features.items():
        if vals is None or len(vals) == 0:
            continue
        arr = np.asarray(vals, dtype=float)
        stats = feature_to_gaussian(arr)
        mu = stats["mean"]
        sd = stats["std"]
        med = stats["median"]

        fig, ax = plt.subplots(figsize=(6, 4))
        # Histogramme normalisé
        ax.hist(arr, bins=bins, density=True, alpha=0.6, edgecolor="black")

        # Courbe gaussienne si écart-type valide
        if np.isfinite(mu) and np.isfinite(sd) and sd > 0:
            x_min = np.min(arr)
            x_max = np.max(arr)
            # élargir légèrement les bornes
            span = (x_max - x_min) if np.isfinite(x_max - x_min) else 1.0
            x = np.linspace(x_min - 0.1 * span, x_max + 0.1 * span, 512)
            y = gaussian_pdf(x, mu, sd)
            ax.plot(x, y, linewidth=2)

        # Repères mean/median
        if np.isfinite(mu):
            ax.axvline(mu, linestyle="--", linewidth=1.5, label=f"mean={mu:.3f}")
        if np.isfinite(med):
            ax.axvline(med, linestyle=":", linewidth=1.5, label=f"median={med:.3f}")

        ax.set_title(f"Distribution de {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("densité")
        ax.legend(loc="best")
        fig.tight_layout()

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(out_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)    
    

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

    attacks = []
    slopes = []
    h2_h1s = []
    h3_h1s = []
    odd_evens = []

    for note in sheet_music:
        attacks.append(note.attack)
        slopes.append(note.slope)
        h2_h1s.append(note.h2_h1)
        h3_h1s.append(note.h3_h1)
        odd_evens.append(note.odd_even)

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

    """feat_attack = feature_to_gaussian(attacks)
    feat_slope = feature_to_gaussian(slopes)
    feat_h2_h1 = feature_to_gaussian(h2_h1s)
    feat_h3_h1 = feature_to_gaussian(h3_h1s)
    feat_odd_even = feature_to_gaussian(odd_evens)"""

    features = {
        "attack": attacks,
        "slope": slopes,
        "H2_H1": h2_h1s,
        "H3_H1": h3_h1s,
        "odd_even": odd_evens,
    }
    plot_feature_gaussian(features, bins=30, save_dir="figs", show=True)
    
    
    

    
        
    # Creating the .mid file
    with open(outfile, "wb") as output_file:
        MyMIDI.writeFile(output_file)

    return "music.mid"


''' 
print(f"piano slopes : {slope_list}")
print(f"max : {max(slope_list)}")
print(f"min : {min(slope_list)}")
    '''
