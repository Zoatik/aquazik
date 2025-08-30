
import os
import librosa
import numpy as np
from audio_processing.freq_analysis import Tools
import matplotlib.pyplot as plt
from audio_processing.midi_reader import Instrument
from scipy.signal import find_peaks

from audio_processing.MidiV2 import midi_maker

WINDOW_TIME = 0.01
FREQ_MIN = 100
FREQ_MAX = 7000

class Note:
    def __init__(self, frequency, magnitude, start = 0.0, length = 0.0, variation=0.0):
        self.frequency = frequency
        self.start_time = start
        self.start_bpm = 0.0
        self.length = length
        self.length_bpm = 0.0
        self.midi_number = Tools.freq_to_number(self.frequency)
        self.name = Tools.note_name(self.midi_number)
        self.magnitude = magnitude
        self.maximum = magnitude  # Store the maximum magnitude observed for this note
        self.variation = variation  # Indicates if the note is fading in or out (positive for fading in, negative for fading out)
        self.instrument = Instrument.PIANO.value

    def __repr__(self):
        return f"Note(frequency={self.frequency}, name='{self.name}', start={self.start_time}, bpm start={self.start_bpm} , bpm length={self.length_bpm}, magnitude={self.magnitude})\n"


def get_harmonics(note: Note, nb_of_harmonics = 2):
        harmonics = []
        for i in range(2, nb_of_harmonics + 2):
            harmonics.append(Tools.freq_to_number(note.frequency*i))

        return harmonics

def filter_notes_and_harmonics(notes: list[Note],
                               max_harmonic: int = 6,
                               alpha: float = 0.9,
                               strict_drop_octaves: bool = True,
                               consider_evidence: bool = True,
                               promotion_ratio_octave: float = 0.9,
                               promotion_ratio_other: float = 1.0) -> list[Note]:
    """
    Filtre avec "évidence harmonique" :
      - Calcule pour chaque pic un *score de support* = magnitude + Σ w_k·mag(k·f) (k≥2),
        avec w_k = 1/k**alpha.
      - Lorsqu'un candidat i peut être l'harmonique (k·f) d'une fondamentale b:
          * Par défaut on supprime i (comportement classique),
          * MAIS si `consider_evidence=True` et que i possède un score de support
            suffisamment élevé par rapport à b (>= promotion_ratio_* × score(b)),
            alors i est **promu** en fondamentale indépendante et n'est pas supprimé.
        → Ceci permet de conserver une octave (ou une autre harmonique) si elle a
          réellement sa propre famille d'harmoniques (ex: doublage à l'octave, voix distincte).

    Paramètres clés:
      - promotion_ratio_octave: seuil relatif pour promouvoir 2f (octave). Ex: 0.9.
      - promotion_ratio_other : seuil relatif pour promouvoir 3f,4f,... Ex: 1.0 (plus strict).
      - strict_drop_octaves   : si False, les octaves sont acceptées même sans promotion.
    """
    if not notes:
        return []

    # --- Préparation triée par fréquence
    idx_by_freq = sorted(range(len(notes)), key=lambda i: notes[i].frequency)
    freqs = np.array([float(notes[i].frequency) for i in idx_by_freq], dtype=float)
    mags  = np.array([float(notes[i].magnitude) for i in idx_by_freq], dtype=float)

    def cents_diff(f1: float, f2: float) -> float:
        if f1 <= 0 or f2 <= 0:
            return 1e9
        return abs(1200.0 * np.log2(f1 / f2))

    # tolérance dynamique (inharmonicité croissante avec k)
    def tol_for_harmonic(k: int) -> float:
        if k == 2: return 15.0
        elif k <= 5: return 25.0
        elif k <= 8: return 40.0
        else: return 60.0

    # Cherche l'index de bin le plus proche d'une cible, avec tolérance dépendant de k
    def find_near_idx(target: float, k: int) -> int | None:
        if target <= 0:
            return None
        pos = int(np.searchsorted(freqs, target))
        cand = []
        for j in (pos-2, pos-1, pos, pos+1, pos+2):
            if 0 <= j < len(freqs):
                cand.append(j)
        if not cand:
            return None
        best, best_c = None, 1e9
        for j in cand:
            c = cents_diff(freqs[j], target)
            if c < best_c:
                best, best_c = j, c
        return best if (best is not None and best_c <= tol_for_harmonic(k)) else None

    # --- Score de "famille harmonique" pour chaque candidat (somme pondérée des multiples)
    weights = [0.0] + [1.0 / (k ** alpha) for k in range(1, max_harmonic + 1)]
    fam_score = np.zeros_like(mags)
    for i in range(len(freqs)):
        s = mags[i]
        f0 = freqs[i]
        for k in range(2, max_harmonic + 1):
            j = find_near_idx(f0 * k, k)
            if j is not None:
                s += weights[k] * mags[j]
        fam_score[i] = s

    # --- Sélection gloutonne informée par le support
    order = list(np.argsort(-fam_score))  # du plus soutenu au moins soutenu
    removed = np.zeros(len(freqs), dtype=bool)
    kept_idx_sorted_space: list[int] = []

    for i in order:
        if removed[i]:
            continue

        # Vérifie si i est déjà l'harmonique d'une fondamentale gardée
        promoted = False
        is_harm = False
        for b in kept_idx_sorted_space:
            if freqs[i] <= freqs[b]:
                continue
            ratio = freqs[i] / freqs[b]
            k = int(round(ratio))
            if 2 <= k <= max_harmonic:
                # i est (≈) k·b ?
                if cents_diff(freqs[i], freqs[b] * k) <= tol_for_harmonic(k):
                    is_harm = True
                    # Tenter une promotion si i a suffisamment de support propre
                    if consider_evidence:
                        if k == 2:
                            need = promotion_ratio_octave * fam_score[b]
                        else:
                            need = promotion_ratio_other * fam_score[b]
                        if fam_score[i] >= need:
                            promoted = True
                            is_harm = False  # i devient fondamentale indépendante
                    if is_harm and (k == 2) and (not strict_drop_octaves):
                        # autoriser les octaves si demandé
                        is_harm = False
                    if is_harm:
                        break  # on peut arrêter la boucle b → i sera supprimé
        if is_harm and not promoted:
            continue  # i supprimé (considéré harmonique d'une fondamentale gardée)

        # Garder i
        kept_idx_sorted_space.append(i)

        # Supprimer ses harmoniques (classique NMS harmonique)
        for k in range(2, max_harmonic + 1):
            if (k == 2) and (not strict_drop_octaves):
                continue
            j = find_near_idx(freqs[i] * k, k)
            if j is not None:
                removed[j] = True

    # --- Conversion indices → notes
    kept_orig_idx = [idx_by_freq[i] for i in kept_idx_sorted_space]
    kept_notes = [notes[i] for i in kept_orig_idx]
    kept_notes.sort(key=lambda n: n.frequency)
    return kept_notes

def find_notes_length(notes: list[list[Note]], audio_bpm):
        """Fusionne les activations frame-par-frame en notes continues.
        Hypothèses:
          - Chaque élément de `notes` est la liste des `Note` actives à cette frame.
          - La durée d'une frame vaut WINDOW_TIME.
        Stratégie:
          - On maintient un dictionnaire d'"actives" indexé par midi_number.
          - Si une note est présente à la frame courante, on prolonge sa durée;
            sinon on incrémente un compteur de "trou". Au-delà d'un petit seuil
            (join_gap_frames), on clôt la note.
        """
        if not notes:
            return []

        join_gap_frames = 2  # autorise un trou de 1 frame pour recoller une même note
        snap_fraction = 8    # arrondi à la 1/8 de beat (comme avant)

        active: dict[int, Note] = {}
        gaps: dict[int, int] = {}
        result: list[Note] = []

        for frame_notes in notes:
            # Ensemble des notes (par midi) présentes sur cette frame
            present = set(n.midi_number for n in frame_notes)

            # 1) Prolonger (ou démarrer) les notes présentes
            for n in frame_notes:
                key = n.midi_number
                if key in active:
                    # prolonge
                    cur = active[key]
                    cur.length += WINDOW_TIME
                    # garder le max de magnitude observé
                    if n.magnitude > cur.maximum:
                        cur.maximum = n.magnitude
                    gaps[key] = 0  # reset trou
                else:
                    # démarre une nouvelle note
                    new = Note(frequency=n.frequency, magnitude=n.magnitude, start=n.start_time)
                    new.midi_number = n.midi_number
                    new.name = n.name
                    new.start_bpm = n.start_bpm
                    new.length = WINDOW_TIME
                    active[key] = new
                    gaps[key] = 0

            # 2) Gérer les notes absentes (compter les trous, éventuellement clôturer)
            for key in list(active.keys()):
                if key not in present:
                    gaps[key] += 1
                    if gaps[key] > join_gap_frames:
                        # clôture
                        result.append(active[key])
                        del active[key]
                        del gaps[key]

        # 3) Clôturer les notes encore actives en fin de séquence
        for key, note in active.items():
            result.append(note)

        # 4) Calcul des longueurs en beats avec snapping
        for note in result:
            beat_duration = Tools.seconds_to_beat(note.length, audio_bpm)
            snapped = round(beat_duration * snap_fraction) / snap_fraction
            note.length_bpm = snapped

        # Tri par temps de début (puis par fréquence) pour stabilité
        result.sort(key=lambda n: (n.start_time, n.midi_number))
        return result

# Helper: filtre les notes trop courtes selon min_seconds et/ou min_beats
def filter_short_notes(notes: list[Note],
                       min_seconds: float | None = None,
                       min_beats: float | None = None) -> list[Note]:
    """Filtre les notes trop courtes.
    Conserve une note si elle satisfait *toutes* les contraintes renseignées:
      - min_seconds: longueur en secondes minimale (si None, ignoré)
      - min_beats:   longueur en temps musical (beats) minimale (si None, ignoré)
    """
    if not notes:
        return []
    kept = []
    for n in notes:
        ok_sec = (min_seconds is None) or (n.length >= float(min_seconds))
        ok_beat = (min_beats is None) or (n.length_bpm >= float(min_beats))
        if ok_sec and ok_beat:
            kept.append(n)
    return kept

def estimate_bpm_from_wavelet(audio_data, sample_rate, use_vqt=False, bins_per_octave=12):
    """
    Estime le BPM en s'appuyant sur une transformée type wavelet (CQT/VQT).
    Met à jour audio_bpm et retourne (bpm_estime, beat_times_seconds).
    """
    n_bins = bins_per_octave * 9         # C0..B8
    fmin = librosa.note_to_hz("C0")
    hop_length = max(1, int(sample_rate * WINDOW_TIME))

    # 1) CQT/VQT
    if use_vqt:
        C = librosa.vqt(audio_data, sr=sample_rate, hop_length=hop_length,
                        fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    else:
        C = librosa.cqt(audio_data, sr=sample_rate, hop_length=hop_length,
                        fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)

    # 2) Magnitude + log-compression (meilleur SNR pour l'onset)
    S = np.abs(C)
    if S.size == 0 or np.max(S) == 0:
        return audio_bpm, np.array([])

    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # 3) Enveloppe d'attaques (agrégation sur les hauteurs)
    onset_env = librosa.onset.onset_strength(
        S=S_db, sr=sample_rate, hop_length=hop_length, aggregate=np.mean
    )

    # 4) Estimation de tempo + temps des battements
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sample_rate, hop_length=hop_length
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length)

    # 5) Option de “snapping” esthétique (comme tu faisais)
    est_bpm = float(tempo)
    lower_bpm = np.floor(est_bpm) 
    upper_bpm = lower_bpm + 2
    if abs(est_bpm - lower_bpm) > abs(est_bpm - upper_bpm):
        est_bpm = upper_bpm
    else:
        est_bpm = lower_bpm
    

    audio_bpm = est_bpm
    return est_bpm, beat_times

def wavelet_mag(audio_data, sample_rate, use_vqt=False, bins_per_octave=12):
    """Retourne (mag, times, note_labels) où:
    - mag: matrice (n_bins, n_frames) normalisée [0..1]
    - times: centres temporels des frames (en secondes)
    - note_labels: liste de labels 'C0'..'B8' alignés aux lignes de mag
    """
    n_bins = bins_per_octave * 9           # octaves 0..8 → 9 * 12 = 108
    fmin = librosa.note_to_hz("C0")
    hop_length = max(1, int(sample_rate * WINDOW_TIME))

    if use_vqt:
        C = librosa.vqt(audio_data, sr=sample_rate, hop_length=hop_length,
                        fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    else:
        C = librosa.cqt(audio_data, sr=sample_rate, hop_length=hop_length,
                        fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)

    mag = np.abs(C)
    mx = np.max(mag) if mag.size else 0.0
    mag = mag / mx if mx > 0 else mag

    # Temps au centre de chaque frame
    times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sample_rate, hop_length=hop_length)

    # Labels de notes pour chaque bin CQT
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    note_labels = [Tools.freq_to_note(f) for f in freqs]  # ex: 'C#4', 'A3', ...

    return mag, times, note_labels



def fft_note_buckets_mag(audio_data, sample_rate, use_hann=True):
    """
    Construit un piano-roll (notes x frames) à partir de la FFT par trames.
    Retourne (mag_notes, times, note_labels) où:
      - mag_notes: (108, n_frames) normalisé [0..1], lignes = C0..B8
      - times: centres des trames (s)
      - note_labels: ['C0','C#0',...,'B8'] (108 entrées)
    """
    # Grille de notes C0..B8
    note_labels = [f"{name}{oct}" for oct in range(0, 9) for name in Tools.NOTE_NAMES]
    n_note_bins = len(note_labels)  # 108

    # Fenêtrage et tramage
    samples_per_window = int(sample_rate * WINDOW_TIME)
    if samples_per_window <= 0:
        raise ValueError("WINDOW_TIME trop petit")
    n_frames = int(len(audio_data) / samples_per_window)
    if n_frames <= 0:
        return np.zeros((n_note_bins, 0)), np.array([]), note_labels

    freqs = np.fft.rfftfreq(samples_per_window, d=1.0 / sample_rate)

    # masque bande utile
    in_band = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)

    # tableau de sortie rempli avec -999
    midi_numbers = np.full(freqs.shape, -999, dtype=int)

    # ne convertir qu'aux indices True du masque
    valid_freqs = freqs[in_band]
    midi_vals = np.array([Tools.freq_to_number(float(f)) for f in valid_freqs], dtype=int)
    midi_numbers[in_band] = midi_vals
    note_idx = midi_numbers - 12  # C0 (MIDI 12) -> 0
    valid_bins = (note_idx >= 0) & (note_idx < n_note_bins)

    # Fenêtre (réduit les fuites spectrales)
    window = np.hanning(samples_per_window) if use_hann else None

    # Accumulateur (notes x frames)
    mag_notes = np.zeros((n_note_bins, n_frames), dtype=float)

    # Boucle sur les trames
    for t in range(n_frames):
        start = t * samples_per_window
        end = start + samples_per_window
        frame = audio_data[start:end]
        if len(frame) < samples_per_window:
            # ignore les trames incomplètes en fin d'audio
            break
        if window is not None:
            frame = frame * window

        spec = np.fft.rfft(frame)
        spec_mag = np.abs(spec)

        # Accumulation dans les buckets de notes (vectorisée)
        # On ne prend que les bins valides
        idx_bins = note_idx[valid_bins]
        vals = spec_mag[valid_bins]
        # np.add.at gère les collisions (plusieurs bins sur la même note)
        np.add.at(mag_notes[:, t], idx_bins, vals)

    # Normalisation globale
    mx = mag_notes.max()
    if mx > 0:
        mag_notes /= mx

    # Temps (centre de trame)
    times = (np.arange(n_frames) + 0.5) * (samples_per_window / sample_rate)

    return mag_notes, times, note_labels

def plot_pianoroll_fft(audio_data, sample_rate, audio_bpm, show_beats=False, save_path=None):
    """
    Affiche le piano-roll basé sur la FFT (notes x temps).
    - show_beats=True : axe X en beats (utilise audio_bpm)
    - save_path : si renseigné, sauvegarde le PNG
    """
    mag_notes, times, note_labels = fft_note_buckets_mag(audio_data, sample_rate)
    if mag_notes.size == 0:
        print("Aucune donnée FFT pour le piano-roll.")
        return

    # Axe X: secondes ou beats
    if show_beats:
        x = np.array([Tools.seconds_to_beat(t, audio_bpm) for t in times])
        x_label = f"Temps (beats) — BPM ≈ {int(audio_bpm)}"
    else:
        x = times
        x_label = "Temps (s)"

    fig, ax = plt.subplots(figsize=(14, 6))

    # imshow avec extent pour positionner correctement les axes
    n_bins, n_frames = mag_notes.shape
    x_end = x[-1] if len(x) > 1 else (x[0] + WINDOW_TIME)
    extent = [x[0], x_end, 0, n_bins]
    im = ax.imshow(mag_notes, aspect='auto', origin='lower', extent=extent)  # (pas de couleurs spécifiées)

    # Ticks Y: un label par octave (sur les C)
    yticks_idx = list(range(0, n_bins, 12))
    yticks_lbl = [note_labels[i] for i in yticks_idx]
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels(yticks_lbl)

    # Grilles légères
    for y in yticks_idx:
        ax.axhline(y, linewidth=0.5)
    step = 1.0
    x_max = x[-1] if len(x) else 0
    for gx in np.arange(0, np.ceil(x_max) + 1e-6, step):
        ax.axvline(gx, linewidth=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Notes (octaves)")
    ax.set_title("Piano-roll (FFT → buckets de notes)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Piano-roll FFT sauvegardé → {save_path}")
    else:
        plt.show() 

def plot_pianoroll(audio_data, sample_rate, audio_bpm, use_vqt=False, show_beats=False, save_path=None,
                   threshold=None,           # ex: 0.02  -> masque mag < 2% du max
                   percentile=None):          # ex: 75    -> masque mag < 75e percentile
    """
    Affiche un piano-roll CQT/VQT avec masque sur faibles magnitudes.
      - threshold : seuil absolu (dans [0,1]) appliqué sur la magnitude normalisée
      - percentile : si fourni, on calcule le seuil comme np.percentile(mag, percentile)
      - show_beats : axe X en beats
      - save_path  : si fourni, sauvegarde le PNG
    """
    mag, times, note_labels = wavelet_mag(audio_data, sample_rate, use_vqt=use_vqt)
    if mag.size == 0:
        print("Aucune donnée pour le piano-roll.")
        return

    # Axe X: secondes ou beats
    if show_beats:
        x = np.array([Tools.seconds_to_beat(t, audio_bpm) for t in times])
        x_label = f"Temps (beats) — BPM ≈ {int(audio_bpm)}"
    else:
        x = times
        x_label = "Temps (s)"

    # Détermination du seuil
    if percentile is not None:
        thr = float(np.percentile(mag, percentile))
    elif threshold is not None:
        thr = float(threshold)
    else:
        thr = None

    # Masquage des faibles magnitudes
    data = mag
    cmap = None
    if thr is not None:
        data = np.ma.masked_less(mag, thr)
        import matplotlib.pyplot as plt
        cmap = plt.cm.viridis.copy()   # n'impose pas de couleurs spécifiques
        cmap.set_bad(alpha=0.0)        # valeurs masquées -> transparentes

    # Tracé
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 6))

    n_bins = data.shape[0]
    x_end = x[-1] if len(x) > 1 else (x[0] + WINDOW_TIME)
    extent = [x[0], x_end, 0, n_bins]
    im = ax.imshow(data, aspect='auto', origin='lower', extent=extent, cmap=cmap)

    # Ticks Y: un label par octave (sur chaque C)
    yticks_idx = list(range(0, n_bins, 1))
    yticks_lbl = [note_labels[i] for i in yticks_idx]
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels(yticks_lbl)

    # Grilles légères
    for y in yticks_idx:
        ax.axhline(y, linewidth=0.5)
    step = 1.0
    x_max = x[-1] if len(x) else 0
    for gx in np.arange(0, np.ceil(x_max) + 1e-6, step):
        ax.axvline(gx, linewidth=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Notes (octaves)")
    title = "Piano-roll Wavelet"
    if thr is not None:
        title += f" (masque < {thr:.3f}{' (perc.)' if percentile is not None else ''})"
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Piano-roll sauvegardé → {save_path}")
    else:
        plt.show()

def detect_notes_from_wavelet(audio_data,
                              sample_rate,
                              audio_bpm,
                              use_vqt=False,
                              bins_per_octave=12,
                              threshold=0.02,
                              percentile=None,
                              temporal_smooth=0,
                              suppress_harmonics=True,
                              min_length_seconds=None,
                              min_length_beats=None):
    """
    Détecte les notes actives à chaque frame via CQT/VQT, avec sélection itérative
    et 'carving' harmonique (masquage des multiples 2f..Kf). Regroupe ensuite
    les notes identiques consécutives en durées.
    Retourne: (bpm, notes_avec_durées)
    """
    # --- 1) CQT/VQT → magnitude normalisée + temps + fréquences de bins
    n_bins = bins_per_octave * 9           # C0..B8
    fmin = librosa.note_to_hz("C0")
    hop_length = max(1, int(sample_rate * WINDOW_TIME))

    if use_vqt:
        C = librosa.vqt(audio_data, sr=sample_rate, hop_length=hop_length,
                        fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    else:
        C = librosa.cqt(audio_data, sr=sample_rate, hop_length=hop_length,
                        fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)

    mag = np.abs(C)                           # (n_bins, n_frames)
    if mag.size == 0:
        return audio_bpm, []

    mx = mag.max()
    if mx == 0:
        return audio_bpm, []
    mag = mag / mx

    times = librosa.frames_to_time(np.arange(mag.shape[1]),
                                   sr=sample_rate, hop_length=hop_length)
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin,
                                    bins_per_octave=bins_per_octave)

    # --- 2) Option: lissage temporel léger (réduction du flicker)
    if temporal_smooth and temporal_smooth > 1:
        k = int(temporal_smooth)
        ker = np.ones(k) / k
        mag = np.apply_along_axis(lambda row: np.convolve(row, ker, mode="same"),
                                  axis=1, arr=mag)

    # --- 3) Détermination du seuil (sur la matrice normalisée)
    base_for_thr = mag
    base_thr = float(threshold) if threshold is not None else 0.0
    if percentile is not None:
        perc_thr = float(np.percentile(base_for_thr, percentile))
        # Seuil combiné: on garde le plus contraignant des deux
        thr = max(base_thr, perc_thr)
    else:
        thr = base_thr

    # --- 4) Sélection itérative + 'carving' harmonique par frame
    notes_by_frame: list[list[Note]] = []

    tol_cents_oct = 12.0
    tol_cents_oth = 20.0
    K_CAP = 12                   # plafond absolu d’harmoniques à considérer
    max_fundamentals = 6
    carve_halfwidth_bins = 1
    f_max = min(FREQ_MAX, 0.98 * sample_rate * 0.5)  # limite freq utile (bande + Nyquist)

    def cents_diff(f1: float, f2: float) -> float:
        if f1 <= 0 or f2 <= 0:
            return 1e9
        return abs(1200.0 * np.log2(f1 / f2))

    def find_bin_for_freq(target_hz: float, tol_cents: float):
        """Index de bin CQT le plus proche de target_hz si dans la tolérance, sinon None."""
        import bisect
        pos = bisect.bisect_left(freqs, target_hz)
        candidates = [k for k in (pos-2, pos-1, pos, pos+1, pos+2) if 0 <= k < len(freqs)]
        if not candidates:
            return None
        best, best_c = None, 1e9
        for k in candidates:
            c = cents_diff(freqs[k], target_hz)
            if c < best_c:
                best_c, best = c, k
        return best if (best is not None and best_c <= tol_cents) else None

    for t in range(mag.shape[1]):
        col_pref = mag[:, t].copy()   # on détecte directement sur la magnitude normalisée
        col_real = mag[:, t]

        # Applique le seuil sur la colonne
        col_pref[col_pref < thr] = 0.0

        frame_notes: list[Note] = []

        # Boucle NMS harmonique: prend le meilleur pic, puis masque ses multiples
        for _ in range(max_fundamentals):
            i0 = int(np.argmax(col_pref))
            if col_pref[i0] <= 0.0:
                break  # plus de pics utiles

            f0 = float(freqs[i0])

            # Enregistrer la fondamentale (avec magnitude réelle)
            n = Note(frequency=f0, magnitude=float(col_real[i0]), start=times[t])
            n.start_bpm = Tools.seconds_to_beat(n.start_time, audio_bpm)
            frame_notes.append(n)

            # K dynamique: limite par bande passante
            K_dyn = int(np.floor(f_max / max(f0, 1e-9)))
            K_dyn = max(2, min(K_CAP, K_dyn))  # borne dans [2 .. K_CAP]

            # Masquer 2f..K_dyn·f0 (± bande)
            for k_h in range(2, K_dyn + 1):
                tol = tol_cents_oct if k_h == 2 else tol_cents_oth
                j = find_bin_for_freq(f0 * k_h, tol)
                if j is not None:
                    j0 = max(0, j - carve_halfwidth_bins)
                    j1 = min(len(col_pref), j + carve_halfwidth_bins + 1)
                    col_pref[j0:j1] = 0.0

            # Masquer la fondamentale et ses voisins pour éviter les doublons proches
            j0 = max(0, i0 - carve_halfwidth_bins)
            j1 = min(len(col_pref), i0 + carve_halfwidth_bins + 1)
            col_pref[j0:j1] = 0.0

        # Sécurité: passe encore le filtre d’harmoniques strict (au cas où)
        if suppress_harmonics and frame_notes:
            frame_notes = filter_notes_and_harmonics(
                sorted(frame_notes, key=lambda x: x.frequency),
                max_harmonic = 3,
                alpha = 0.3,
                strict_drop_octaves = True,
                consider_evidence = True,
                promotion_ratio_octave = 0.9,
                promotion_ratio_other = 1.0
            ) or []

        notes_by_frame.append(frame_notes)

    # --- 5) Regrouper en durées (réutilise ton snapping en bpm)
    notes_with_len = find_notes_length(notes_by_frame, audio_bpm)

    # Filtrage final des notes trop courtes (parasites)
    notes_with_len = filter_short_notes(notes_with_len,
                                        min_seconds=min_length_seconds,
                                        min_beats=min_length_beats)
    return audio_bpm, notes_with_len

def plot_notes_timeline(audio_bpm, notes: list[Note], show_beats=False, save_path=None):
    """
    Affiche chaque note détectée sous forme de barre (start → start+length).
    L'axe Y est en demi-tons (C0..B8), avec ticks aux C d'octave.
    """
    if not notes:
        print("Aucune note à afficher.")
        return

    # Convertit en coordonnées
    xs = []
    spans = []  # (x, width, y, height)
    ymin = 0
    ymax = 12*9 - 1  # C0..B8 -> 108 bins (0..107)
    h = 0.9          # hauteur visuelle d'une barre

    for n in notes:
        # x et largeur (secondes ou beats)
        start = n.start_bpm if show_beats else n.start_time
        dur   = n.length_bpm if show_beats else n.length
        if dur <= 0:
            continue  # ignore longueurs nulles
        # y: index de note (C0=0)
        y = n.midi_number - 12
        if 0 <= y <= ymax:
            spans.append((start, dur, y - h/2, h))

    if not spans:
        print("Aucune note valide dans la plage C0..B8.")
        return

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 6))

    # Dessin des barres
    for (x, w, y, hh) in spans:
        ax.add_patch(plt.Rectangle((x, y), w, hh, linewidth=0.5, edgecolor='k', facecolor='tab:blue', alpha=0.7))

    # Axes
    # X: secondes ou beats
    x_label = f"Temps (beats) — BPM ≈ {int(audio_bpm)}" if show_beats else "Temps (s)"
    ax.set_xlabel(x_label)

    # Y: ticks tous les 12 bins (C de chaque octave)
    yticks_idx = list(range(0, 12*9, 12))
    yticks_lbl = [f"C{o}" for o in range(0, 9)]
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels(yticks_lbl)
    ax.set_ylim(-1, 12*9)  # un peu de marge

    # Grilles
    for y in yticks_idx:
        ax.axhline(y, linewidth=0.5, color='gray', alpha=0.3)
    # lignes verticales chaque 1 s / 1 beat
    import numpy as np
    x_max = max(x + w for (x, w, *_ ) in spans)
    grid_xs = np.arange(0, np.ceil(x_max) + 1e-6, 1.0)
    for gx in grid_xs:
        ax.axvline(gx, linewidth=0.5, color='gray', alpha=0.2)

    ax.set_title("Notes détectées dans le temps")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Timeline sauvegardée → {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    music_name = "Ecossaise_Both.mp3"
    p = os.path.abspath(f"../audio_in/{music_name}")
    audio_data, sr = librosa.load(p)
    #print(audio_data)
    bpm, beat_times = estimate_bpm_from_wavelet(audio_data, sr, use_vqt=True)
    print("BPM estimé:", bpm)
    print("Battements (s):", beat_times[:10])


    bpm, notes = detect_notes_from_wavelet(
        audio_data, sr, bpm,
        use_vqt=False,
        bins_per_octave=12,
        threshold=0.05,      # plancher absolu
        percentile=80,       # seuil relatif par contenu
        temporal_smooth=5,
        suppress_harmonics=True,
        min_length_beats=0.15
    )

    print(f"file saved ({music_name}): {midi_maker(notes, bpm)}")
    #plot_notes_timeline(bpm, notes)

    #plot_pianoroll_fft(audio_data, sr, bpm)
    #plot_pianoroll(audio_data, sr, bpm, threshold=0.05)

