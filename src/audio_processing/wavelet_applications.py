# spec_compare.py
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Iterable, Dict, Tuple, Optional, Any
import numpy as np

from baba import wavelet_mag

# ==========================
# Réglages généraux
# ==========================
# WINDOW_TIME = 0.01  # secondes (résolution temporelle)

NOTE_NAMES_12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
WINDOW_TIME = 0.025  # secondes (résolution temporelle)


class Tools:
    NOTE_NAMES = NOTE_NAMES_12
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

    @staticmethod
    def note_to_midi(note: str) -> int:
        # "C#4" → name="C#", octave=4 → MIDI = 1 + 12*(4+1) = 61
        note_name = note[:-1]
        octave = int(note[-1])
        return Tools.NOTE_TO_MIDI[note_name] + 12 * (octave + 1)

    @staticmethod
    def seconds_to_beat(sec: float, bpm: float) -> float:
        return sec * (bpm / 60.0)

    @staticmethod
    def freq_to_number(f: float) -> int:
        # MIDI number
        return int(np.round(69 + 12 * np.log2(f / 440.0)))

    @staticmethod
    def number_to_freq(n: int) -> float:
        return 440.0 * (2.0 ** ((n - 69) / 12.0))

    @staticmethod
    def note_name(n: int) -> str:
        return NOTE_NAMES_12[n % 12] + str(int(n / 12 - 1))

    @staticmethod
    def freq_to_note(f: float) -> str:
        return Tools.note_name(Tools.freq_to_number(f))


# ===========================================
# CQT (wavelet musical) magnitude par trame
# ===========================================
def detect_pitches_cqt(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    fmin_note: str = "A0",
    fmax_note: str = "C8",
    bins_per_octave: int = 24,
    threshold: float = 0.02,
    smooth_time: int = 5,
    hysteresis_hi: float = 0.6,
    hysteresis_lo: float = 0.4,
    min_duration: float = 0.05,
    use_hpss: bool = True,
):
    """
    Retourne (events, debug) :
      - events : liste d'événements {midi, note, onset, offset, strength}
      - debug  : dict avec matrices et paramètres intermédiaires
    """
    # Prétraitement
    y = librosa.to_mono(y) if y.ndim > 1 else y
    y = librosa.util.normalize(y)
    if use_hpss:
        y_h, _ = librosa.effects.hpss(y)
    else:
        y_h = y

    fmin_hz = librosa.note_to_hz(fmin_note)
    fmax_hz = librosa.note_to_hz(fmax_note)
    n_oct = int(np.ceil(np.log2(fmax_hz / fmin_hz)))
    n_bins = n_oct * bins_per_octave

    # CQT hybride
    C = np.abs(
        librosa.cqt(
            y_h,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin_hz,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    )

    # Normalisations
    C /= C.max() + 1e-12
    band_med = np.median(C, axis=1, keepdims=True)
    Cn = C / (band_med + 1e-9)

    # Lissage temporel
    if smooth_time > 1:
        k = np.ones(smooth_time) / smooth_time
        Cn = np.apply_along_axis(lambda x: np.convolve(x, k, mode="same"), 1, Cn)

    # Grille MIDI (énergie agrégée par demi-ton)
    freqs = librosa.cqt_frequencies(
        n_bins=n_bins, fmin=fmin_hz, bins_per_octave=bins_per_octave
    )
    midi_min, midi_max = 21, 108  # A0..C8
    midi_grid = np.arange(midi_min, midi_max + 1)
    centers_hz = librosa.midi_to_hz(midi_grid)
    bin_to_midi = np.argmin(np.abs(freqs[:, None] - centers_hz[None, :]), axis=1)

    frames = Cn.shape[1]
    note_energy = np.zeros((len(midi_grid), frames), dtype=np.float32)
    for b, m in enumerate(bin_to_midi):
        note_energy[m] += Cn[b]

    # Normalisation par pas MIDI (pour stabilité des seuils)
    note_energy /= note_energy.max(axis=1, keepdims=True) + 1e-9

    # Hystérésis & durée minimale → événements
    times = librosa.frames_to_time(np.arange(frames), sr=sr, hop_length=hop_length)
    active = np.zeros_like(note_energy, dtype=bool)
    for i in range(note_energy.shape[0]):
        on = False
        for t in range(frames):
            v = note_energy[i, t]
            if not on and v >= hysteresis_hi:
                on = True
            elif on and v < hysteresis_lo:
                on = False
            active[i, t] = on

    min_len = int(np.ceil(min_duration * sr / hop_length))
    events = []
    for i, m in enumerate(midi_grid):
        edges = np.flatnonzero(np.diff(np.r_[0, active[i].view(np.int8), 0]))
        starts, ends = edges[::2], edges[1::2]
        for s, e in zip(starts, ends):
            if e - s < min_len:
                continue
            seg = note_energy[i, s:e]
            events.append(
                {
                    "midi": int(m),
                    "note": librosa.midi_to_note(m),
                    "onset": float(times[s]),
                    "offset": float(times[e] if e < len(times) else times[-1]),
                    "strength": float(seg.max() if seg.size else 0.0),
                }
            )

    debug = dict(
        C=C,
        Cn=Cn,
        note_energy=note_energy,  # ex- "piano"
        times=times,
        midi=midi_grid,
        params=dict(
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            threshold=threshold,
            hysteresis=(hysteresis_hi, hysteresis_lo),
        ),
    )
    return events, debug


# ==========================
# Estimation BPM via CQT
# ==========================
def estimate_bpm_from_wavelet(audio_data: np.ndarray, sample_rate: int):
    hop_length = max(1, int(sample_rate * WINDOW_TIME))
    n_bins = 12 * 9
    fmin = librosa.note_to_hz("C0")

    C = librosa.cqt(
        audio_data,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=12,
    )

    S = np.abs(C)
    if S.size == 0 or np.max(S) == 0:
        return None, np.array([])

    S_db = librosa.amplitude_to_db(S, ref=np.max)
    onset_env = librosa.onset.onset_strength(
        S=S_db, sr=sample_rate, hop_length=hop_length, aggregate=np.mean
    )
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sample_rate, hop_length=hop_length
    )
    beat_times = librosa.frames_to_time(
        beat_frames, sr=sample_rate, hop_length=hop_length
    )
    return float(tempo), beat_times


def filter_fundamentals_from_grid(
    note_energy: np.ndarray,
    times: np.ndarray,
    midi: np.ndarray,
    *,
    max_harmonic: int = 8,
    cents_tolerance: float = 35.0,
    harmonic_relax: float = 1.25,
    suppress_subharmonics: bool = True,
    min_duration_s: float = 0.04,
    hysteresis_hi: float = 0.5,
    hysteresis_lo: float = 0.3,
):
    """
    note_energy : [n_pitches, n_frames] énergie par pas MIDI (grille demi-ton)
    Retourne:
      - fundamentals_mask : bool [n_pitches, n_frames]
      - events_fund       : liste d'événements fondamentaux
    """
    n_notes, n_frames = note_energy.shape
    fundamentals = np.zeros_like(note_energy, dtype=bool)

    ks = np.arange(2, max_harmonic + 1)
    semitone_offsets = 12.0 * np.log2(ks)
    tol_semitones = cents_tolerance / 100.0

    midi = np.asarray(midi)
    midi_min, midi_max = midi[0], midi[-1]

    # Sélection par frame (greedy)
    for t in range(n_frames):
        col = note_energy[:, t]
        if not np.any(col > 0):
            continue
        order = np.argsort(col)[::-1]
        accepted = []

        for idx in order:
            m = midi[idx]
            e = col[idx]
            if e <= 0:
                continue

            # Rejeter si harmonique d'une fondamentale déjà acceptée
            is_harm = False
            for f_idx in accepted:
                f_m = midi[f_idx]
                f_e = col[f_idx]
                targets = f_m + semitone_offsets
                if np.any(np.abs(m - targets) <= tol_semitones):
                    if e <= harmonic_relax * (f_e + 1e-12):
                        is_harm = True
                        break
            if is_harm:
                continue

            # Optionnel : rejeter sous-harmoniques d'une note plus haute forte
            if suppress_subharmonics:
                is_subharm = False
                for off in semitone_offsets:
                    m_high = m + off
                    if m_high < midi_min or m_high > midi_max:
                        continue
                    j = np.argmin(np.abs(midi - m_high))
                    if (
                        abs(midi[j] - m_high) <= tol_semitones
                        and col[j] >= harmonic_relax * e
                    ):
                        is_subharm = True
                        break
                if is_subharm:
                    continue

            fundamentals[idx, t] = True
            accepted.append(idx)

    # Hystérésis + durée minimale par pas MIDI
    fund_clean = np.zeros_like(fundamentals)
    # approx frames per second via median Δt
    fps = 1.0 / np.median(np.diff(times + 1e-12))
    min_len = max(1, int(np.ceil(min_duration_s * fps)))

    for i in range(n_notes):
        row = note_energy[i]
        v = row / (row.max() + 1e-9) if row.max() > 0 else row
        on = False
        active = np.zeros(n_frames, dtype=bool)
        for t in range(n_frames):
            if not on and (fundamentals[i, t] and v[t] >= hysteresis_hi):
                on = True
            elif on and v[t] < hysteresis_lo:
                on = False
            active[t] = on

        edges = np.flatnonzero(np.diff(np.r_[0, active.view(np.int8), 0]))
        starts, ends = edges[::2], edges[1::2]
        for s, e in zip(starts, ends):
            if (e - s) >= min_len:
                fund_clean[i, s:e] = True

    # Événements issus des fondamentales
    events_fund = []
    for i, m in enumerate(midi):
        edges = np.flatnonzero(np.diff(np.r_[0, fund_clean[i].view(np.int8), 0]))
        starts, ends = edges[::2], edges[1::2]
        for s, e in zip(starts, ends):
            seg = note_energy[i, s:e]
            events_fund.append(
                {
                    "midi": int(m),
                    "note": librosa.midi_to_note(m),
                    "onset": float(times[s]),
                    "offset": float(times[e] if e < len(times) else times[-1]),
                    "strength": float(seg.max() if seg.size else 0.0),
                }
            )
    return fund_clean, events_fund


def plot_note_events(
    events,
    *,
    ax=None,
    y_range=(21, 108),  # A0..C8
    show_note_names=True,  # graduations sur les 'C' (Do)
    title="Notes détectées",
    grid=True,
):
    """
    Trace des notes détectées (onset/offset) en fonction du temps.

    Params
    ------
    events : iterable de dicts
        Chaque dict doit contenir au minimum:
          - 'midi'   (int)
          - 'onset'  (float, s)
          - 'offset' (float, s)
        Optionnel: 'strength' (float dans [0,1]) et 'note' (str)
    ax : matplotlib.axes.Axes ou None
        Axe existant. Si None, crée une figure.
    y_range : (lo, hi)
        Plage MIDI affichée (par défaut A0..C8).
    show_note_names : bool
        Si True, affiche les noms de notes sur les degrés 'C' (Do) de chaque octave.
    title : str
        Titre du graphique.
    grid : bool
        Active la grille principale.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    if not events:
        ax.text(
            0.5,
            0.5,
            "Aucun événement",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    lo, hi = int(y_range[0]), int(y_range[1])

    # Nettoyage et extraction
    ev = [
        e
        for e in events
        if lo <= int(e["midi"]) <= hi and float(e["offset"]) > float(e["onset"])
    ]
    if not ev:
        ax.text(
            0.5,
            0.5,
            "Aucun événement dans la plage sélectionnée",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig, ax

    mids = np.array([int(e["midi"]) for e in ev])
    onsets = np.array([float(e["onset"]) for e in ev])
    offsets = np.array([float(e["offset"]) for e in ev])
    durs = offsets - onsets
    strengths = np.array([float(e.get("strength", 1.0)) for e in ev])

    # Normalisation des forces -> alpha
    smin, smax = strengths.min(), strengths.max()
    if smax > smin:
        alphas = 0.3 + 0.7 * (strengths - smin) / (smax - smin)  # [0.3..1.0]
    else:
        alphas = np.full_like(strengths, 0.8, dtype=float)

    # Tracé: une barre par note
    for m, t0, dur, a in zip(mids, onsets, durs, alphas):
        # broken_barh: ((start, width), (y, height))
        ax.broken_barh([(t0, dur)], (m - 0.45, 0.9), linewidth=0, alpha=float(a))

    # Axes
    ax.set_xlim(0, max(offsets))
    ax.set_ylim(lo - 1, hi + 1)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("MIDI")

    # Graduations lisibles sur les Do (C)
    if show_note_names:
        # Trouve le premier MIDI == C (mod 12 == 0) dans la plage
        start_C = lo + ((12 - (lo % 12)) % 12)
        ticks_C = np.arange(start_C, hi + 1, 12, dtype=int)
        ax.set_yticks(ticks_C)
        ax.set_yticklabels([librosa.midi_to_note(m, octave=True) for m in ticks_C])
        # Ticks mineurs à chaque demi-ton
        ax.set_yticks(np.arange(lo, hi + 1, dtype=int), minor=True)
    else:
        # Un tick par octave
        start = lo + ((12 - (lo % 12)) % 12)
        oct_ticks = np.arange(start, hi + 1, 12, dtype=int)
        ax.set_yticks(oct_ticks)

    if grid:
        ax.grid(True, which="major", linestyle=":")
        ax.grid(True, which="minor", linestyle=":", alpha=0.3)

    if title:
        ax.set_title(title)

    return fig, ax


def events_to_midi(
    events: Iterable[Dict[str, Any]],
    out_path: str,
    *,
    tempo: float = 120.0,
    default_program: int = 0,  # 0 = Acoustic Grand, mais tu peux changer
    velocity_range: Tuple[int, int] = (24, 112),
    clip_times: Optional[Tuple[float, float]] = None,  # (t_min, t_max) pour tronquer
    group_key: Optional[
        str
    ] = None,  # ex: "track" / "source" pour créer plusieurs pistes
    assume_drums_key: Optional[str] = None,  # si présent et True, piste en mode drum
) -> str:
    """
    Convertit une liste d'événements en fichier MIDI.

    events: iterable de dicts avec au minimum:
        - 'midi'   : int (0..127)
        - 'onset'  : float (s)
        - 'offset' : float (s)
      optionnels:
        - 'strength': float ~ [0..1] -> mappée en vélocité
        - 'program' : int (0..127)   -> instrument MIDI (si absent, default_program)
        - group_key : la clé choisie (ex 'track') pour regrouper les événements en pistes
        - assume_drums_key : bool (si True -> piste drum, canal indépendant côté pretty_midi)

    Retourne: chemin du fichier écrit.
    """
    import pretty_midi

    # Crée l'objet MIDI avec un tempo initial (utile si tu re-quantifies plus tard)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Prépare les pistes (instruments) dynamiquement
    tracks = {}  # key -> pretty_midi.Instrument

    def get_track_key(e):
        g = e.get(group_key) if group_key else "main"
        prog = int(e.get("program", default_program))
        is_drum = bool(e.get(assume_drums_key, False)) if assume_drums_key else False
        return (g, prog, is_drum)

    def get_or_create_track(key):
        if key not in tracks:
            _, prog, is_drum = key
            inst = pretty_midi.Instrument(
                program=int(prog), is_drum=is_drum, name=str(key[0])
            )
            tracks[key] = inst
            pm.instruments.append(inst)
        return tracks[key]

    vmin, vmax = int(velocity_range[0]), int(velocity_range[1])
    vmin = max(1, min(126, vmin))
    vmax = max(vmin + 1, min(127, vmax))  # bornes sûres

    # Tri par temps pour un MIDI propre
    events_sorted = sorted(
        events, key=lambda e: (float(e.get("onset", 0.0)), int(e.get("midi", 0)))
    )

    # Conversion
    for e in events_sorted:
        pitch = int(e["midi"])
        t0 = float(e["onset"])
        t1 = float(e["offset"])
        if t1 <= t0:
            continue
        if clip_times is not None:
            t_min, t_max = clip_times
            if t1 < t_min or t0 > t_max:
                continue
            t0 = max(t0, t_min)
            t1 = min(t1, t_max)
            if t1 <= t0:
                continue

        # Vélocité à partir de strength (optionnelle)
        s = float(e.get("strength", 1.0))
        s = float(np.clip(s, 0.0, 1.0))
        vel = int(round(vmin + (vmax - vmin) * s))
        vel = max(1, min(127, vel))

        trk = get_or_create_track(get_track_key(e))
        trk.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=t0, end=t1))

    # Écrit le fichier
    pm.write(out_path)
    return out_path


def apply_wavelet_mask_to_cqt_debug(
    debug_cqt: dict,
    mag: np.ndarray,
    times_w: np.ndarray,
    note_labels,
    *,
    threshold: float = 0.2,  # seuil sur la magnitude VQT/CQT pour "laisser passer"
    mode: str = "binary",  # "binary" ou "soft"
    softness: float = 2.0,  # pour mode="soft": pente (plus grand = transition plus douce)
    freq_blend_bins: int = 0,  # mélange +/- N bins voisins côté wavelet (0 = nearest seul)
    time_smooth: float = 0.0,  # lissage temporel (s) sur la magnitude wavelet (0 = off)
    pad_mode: str = "edge",
):
    """
    Construit un masque à partir du triplet (mag, times_w, note_labels) et l'applique
    à debug_cqt["note_energy"]. Retourne (note_energy_masked, mask_grid).

    - Recalage fréquentiel: chaque pas MIDI de la grille CQT est associé au bin 'note_labels'
      le plus proche (option: moyenne locale +/- freq_blend_bins).
    - Recalage temporel: interpolation linéaire de mag sur debug_cqt["times"].
    - Masquage: binaire (>= threshold) ou doux (rampe contrôlée par 'softness').
    """
    note_energy = debug_cqt["note_energy"]  # [n_midi, n_tc]
    times_c = debug_cqt["times"]  # [n_tc]
    midi_grid = debug_cqt["midi"]  # [n_midi]
    n_midi, n_tc = note_energy.shape

    # --- 1) Convertir note_labels -> MIDI (ex: 'C#4' -> 61)
    # Si tes labels sont déjà en MIDI, remplace ce bloc par un np.asarray(...)
    midi_w = np.array([librosa.note_to_midi(lbl) for lbl in note_labels], dtype=float)
    n_wbins, n_tw = mag.shape
    if n_wbins != len(midi_w):
        raise ValueError("note_labels et mag ont des tailles incompatibles.")

    # --- 2) (Optionnel) petit lissage temporel côté wavelet (évite le jitter)
    if time_smooth and n_tw > 1:
        dt_w = float(np.median(np.diff(times_w))) if times_w.size > 1 else 0.0
        L = max(1, int(round(time_smooth / max(dt_w, 1e-6))))
        if L % 2 == 0:
            L += 1
        r = (L - 1) // 2
        # padding
        mag_pad = np.pad(mag, ((0, 0), (r, r)), mode=pad_mode)
        k = np.ones(L, dtype=float) / L
        mag = np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 1, mag_pad)

    # --- 3) Recalage fréquentiel: pour chaque MIDI CQT -> index wavelet le plus proche
    idx_w_nearest = np.argmin(
        np.abs(midi_w[:, None] - midi_grid[None, :]), axis=0
    )  # [n_midi]

    # (Option) mélange local en fréquence (moyenne symétrique +/- freq_blend_bins)
    if freq_blend_bins > 0:
        mask_rows = np.empty((n_midi, n_tw), dtype=float)
        for i, j0 in enumerate(idx_w_nearest):
            j1 = max(0, j0 - freq_blend_bins)
            j2 = min(n_wbins, j0 + freq_blend_bins + 1)
            mask_rows[i] = mag[j1:j2].mean(axis=0)
        mag_sel = mask_rows
    else:
        mag_sel = mag[idx_w_nearest, :]  # [n_midi, n_tw]

    # --- 4) Recalage temporel: interpole sur les temps CQT
    # On interpole ligne par ligne: linéaire, valeurs de bord étendues
    mask_time = np.empty((n_midi, n_tc), dtype=float)
    # gérer cas dégénéré (un seul temps wavelet)
    if times_w.size <= 1:
        mask_time[:] = mag_sel[:, -1][:, None]
    else:
        tmin, tmax = float(times_w[0]), float(times_w[-1])
        for i in range(n_midi):
            mask_time[i] = np.interp(
                times_c, times_w, mag_sel[i], left=mag_sel[i, 0], right=mag_sel[i, -1]
            )

    # --- 5) Construire le masque (binaire ou doux)
    # On suppose mag déjà normalisé [0..1] (c'est le cas dans wavelet_mag).
    if mode == "binary":
        mask_grid = (mask_time >= float(threshold)).astype(note_energy.dtype)
    elif mode == "soft":
        # rampe lisse: 0 sous le seuil, 1 bien au-dessus
        # y = clip((x - th)/(1 - th), 0, 1) ** (1/softness)
        th = float(threshold)
        x = (mask_time - th) / max(1e-9, 1.0 - th)
        x = np.clip(x, 0.0, 1.0)
        mask_grid = x ** (1.0 / max(1e-6, float(softness)))
        mask_grid = mask_grid.astype(note_energy.dtype, copy=False)
    else:
        raise ValueError("mode doit être 'binary' ou 'soft'.")

    # --- 6) Application du masque
    note_energy_masked = note_energy * mask_grid

    return note_energy_masked, mask_grid


def grid_to_events(
    grid: np.ndarray,  # [n_pitches, n_frames]  (ex: grid_masked)
    times: np.ndarray,  # [n_frames]
    midi: np.ndarray,  # [n_pitches]
    *,
    normalize_per_pitch: bool = True,
    hysteresis_hi: float = 0.6,
    hysteresis_lo: float = 0.4,
    min_duration_s: float = 0.05,
    strength_method: str = "max",  # "max", "mean", ou "area"
    drop_threshold: (
        float | None
    ) = None,  # si défini, ignore les lignes trop faibles (ex. 0.05)
):
    """
    Convertit une grille temps×pas-MIDI en liste d'événements {midi, note, onset, offset, strength}.
    - normalize_per_pitch: divise chaque ligne par son max (stabilise les seuils).
    - hysteresis_hi/lo: seuils d'activation/relâchement sur le signal (après normalisation éventuelle).
    - min_duration_s: filtre les activations trop courtes.
    - strength_method: mesure de force pour l'event.
    - drop_threshold: si fourni, on ignore complètement les pas dont le max < drop_threshold.
    """
    if grid.ndim != 2:
        raise ValueError("grid doit être de forme (n_pitches, n_frames)")
    n_pitches, n_frames = grid.shape
    if n_frames == 0 or n_pitches == 0:
        return []

    # Option: ignorer les hauteurs très faibles globalement
    line_max = grid.max(axis=1)
    if drop_threshold is not None:
        keep_rows = line_max >= float(drop_threshold)
    else:
        keep_rows = np.ones(n_pitches, dtype=bool)

    # Normalisation par pas MIDI (optionnelle)
    G = grid.copy().astype(float, copy=False)
    if normalize_per_pitch:
        denom = line_max.copy()
        denom[denom == 0.0] = 1.0
        G = (G.T / denom).T  # évite division broadcast ambiguë

    # Conversion duration -> frames
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    if dt <= 0:
        dt = (times[-1] / max(1, n_frames - 1)) if len(times) > 0 else 1.0
    min_len = max(1, int(np.ceil(min_duration_s / dt)))

    events = []
    for i in range(n_pitches):
        if not keep_rows[i]:
            continue
        v = G[i]

        # Hystérésis
        on = False
        active = np.zeros(n_frames, dtype=bool)
        for t in range(n_frames):
            x = v[t]
            if not on and x >= hysteresis_hi:
                on = True
            elif on and x < hysteresis_lo:
                on = False
            active[t] = on

        # Segments
        edges = np.flatnonzero(np.diff(np.r_[0, active.view(np.int8), 0]))
        starts, ends = edges[::2], edges[1::2]

        for s, e in zip(starts, ends):
            if (e - s) < min_len:
                continue

            seg_raw = grid[i, s:e]  # force calculée sur la grille NON normalisée
            if strength_method == "max":
                strength = float(seg_raw.max() if seg_raw.size else 0.0)
            elif strength_method == "mean":
                strength = float(seg_raw.mean() if seg_raw.size else 0.0)
            elif strength_method == "area":
                strength = float(seg_raw.sum() * dt if seg_raw.size else 0.0)
            else:
                raise ValueError("strength_method doit être 'max', 'mean' ou 'area'.")

            onset = float(times[s])
            offset = float(times[e] if e < len(times) else times[-1])
            m = int(midi[i])
            events.append(
                {
                    "midi": m,
                    "note": librosa.midi_to_note(m),
                    "onset": onset,
                    "offset": offset,
                    "strength": strength,
                }
            )

    # Tri (facultatif, pour propreté)
    events.sort(key=lambda e: (e["onset"], e["midi"]))
    return events


# ==========================
# Exemple d’utilisation
# ==========================
def main():
    # Modifie le nom du fichier ici
    audio_name = "PinkPanther_Both"
    audio_path = f"audio_in/{audio_name}.mp3"
    midi_out = f"audio_out/{audio_name}.mid"
    if not os.path.exists(audio_path):
        print(f"⚠️ Fichier introuvable: {audio_path}")
        return

    y, sr = librosa.load(audio_path, mono=True, duration=None)

    bpm_est, _ = estimate_bpm_from_wavelet(y, sr)

    if bpm_est is not None:
        print(f"BPM estimé (wavelet): {bpm_est:.1f}")

    events, debug = detect_pitches_cqt(
        y=y,
        sr=sr,
        bins_per_octave=48,
        smooth_time=5,
        min_duration=0.05,
        threshold=0.03,
        hysteresis_hi=0.5,
        hysteresis_lo=0.3,
    )

    grid = debug["note_energy"]
    times = debug["times"]
    midi = debug["midi"]

    mag, times_w, note_labels = wavelet_mag(y, sr, bins_per_octave=24)

    """fund_mask, fund_events = filter_fundamentals_from_grid(
        grid, times, midi, max_harmonic=6
    )"""

    grid_masked, mask_grid = apply_wavelet_mask_to_cqt_debug(
        debug,
        mag,
        times_w,
        note_labels,
        threshold=0.25,  # durcis si des sous-harmoniques passent encore
        mode="soft",  # "binary" si tu veux couper net
        softness=2.0,
        freq_blend_bins=1,  # moyenne des voisins +/-1 côté wavelet
        time_smooth=0.04,  # ~40 ms de lissage temporel wavelet
    )

    events2_from_grid = grid_to_events(
        mag,
        times_w,
        [Tools.note_to_midi(note_label) for note_label in note_labels],
        normalize_per_pitch=True,
        hysteresis_hi=0.6,
        hysteresis_lo=0.4,
        min_duration_s=0.05,
        strength_method="max",
        drop_threshold=0.02,  # optionnel
    )

    events_from_grid = grid_to_events(
        grid_masked,
        debug["times"],
        debug["midi"],
        normalize_per_pitch=True,
        hysteresis_hi=0.6,
        hysteresis_lo=0.4,
        min_duration_s=0.05,
        strength_method="max",
        drop_threshold=0.02,  # optionnel
    )

    # Crée une figure avec 2 sous-graphiques horizontaux
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    out = events_to_midi(
        events_from_grid,
        midi_out,
        tempo=bpm_est if bpm_est is not None else 120.0,
        default_program=0,
        velocity_range=(28, 112),
        group_key=None,  # ou 'source' si tu as taggé les événements
        assume_drums_key=None,  # ou 'is_drum' si certains events sont percussifs
    )
    print("MIDI écrit ->", out)

    # Tracé des notes détectées (toutes)
    plot_note_events(events, y_range=(21, 108), title="Notes détectées (CQT)", ax=ax1)
    plot_note_events(
        events_from_grid,
        y_range=(21, 108),
        title="Notes détectées (CQT + Wavelet)",
        ax=ax2,
    )
    plot_note_events(
        events2_from_grid,
        y_range=(21, 108),
        title="Notes détectées (Wavelet seule)",
        ax=ax3,
    )

    # Ajuste la mise en page
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
