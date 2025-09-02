import math
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from midiutil import MIDIFile
from audio_processing.audio_utils import Tools, Instrument, Note
from audio_processing.MidiV2 import midi_maker


WINDOW_TIME = 0.025
FREQ_MIN = 100
FREQ_MAX = 7000


def estimate_bpm_from_wavelet(
    audio_data, sample_rate, use_vqt=False, bins_per_octave=12
):
    """
    Estime le BPM en s'appuyant sur une transformée type wavelet (CQT/VQT).
    Met à jour audio_bpm et retourne (bpm_estime, beat_times_seconds).
    """
    n_bins = bins_per_octave * 9  # C0..B8
    fmin = librosa.note_to_hz("C0")
    hop_length = max(1, int(sample_rate * WINDOW_TIME))

    # 1) CQT/VQT
    if use_vqt:
        C = librosa.vqt(
            audio_data,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    else:
        C = librosa.cqt(
            audio_data,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )

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
    beat_times = librosa.frames_to_time(
        beat_frames, sr=sample_rate, hop_length=hop_length
    )

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
    n_bins = bins_per_octave * 9  # octaves 0..8 → 9 * 12 = 108
    fmin = librosa.note_to_hz("C0")
    hop_length = int(sample_rate * WINDOW_TIME)
    audio_data = librosa.util.normalize(audio_data)

    if use_vqt:
        C = librosa.vqt(
            audio_data,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    else:
        C = librosa.hybrid_cqt(
            audio_data,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )

    mag = np.abs(C)
    mx = np.max(mag) if mag.size else 0.0
    mag = mag / mx if mx > 0 else mag

    # Temps au centre de chaque frame
    times = librosa.frames_to_time(
        np.arange(mag.shape[1]), sr=sample_rate, hop_length=hop_length
    )

    # Labels de notes pour chaque bin CQT
    freqs = librosa.cqt_frequencies(
        n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave
    )
    note_labels = [Tools.freq_to_note(f) for f in freqs]  # ex: 'C#4', 'A3', ...

    return mag, times, note_labels


def smooth_wavelet_mag(
    mag: np.ndarray,
    times: np.ndarray,
    note_labels,  # non utilisé dans le calcul (fourni pour compatibilité / debug)
    *,
    time_s: float = 0.05,  # largeur de fenêtre temporelle (secondes)
    freq_bins: int = 3,  # largeur de fenêtre fréquentielle (en bins)
    method: str = "gaussian",  # "gaussian" ou "box"
    renormalize: bool = False,  # remet l'échelle sur [0,1] après lissage
    pad_mode: str = "reflect",  # "reflect", "edge"…
) -> np.ndarray:
    """
    Lissage séparable de la matrice d'amplitude 'mag' (n_bins, n_frames).

    - Lissage temporel (le long de l'axe frames) avec une fenêtre de 'time_s' secondes.
    - Lissage fréquentiel (le long de l'axe bins) avec 'freq_bins' bins.
    - 'method' contrôle la forme de la fenêtre : gaussienne (douce) ou boîte (moyenne glissante).

    Retourne:
      mag_smooth : np.ndarray de même shape que 'mag'.
    """
    if mag.ndim != 2:
        raise ValueError("mag doit être 2D (n_bins, n_frames)")

    n_bins, n_frames = mag.shape
    if n_frames == 0 or n_bins == 0:
        return mag.copy()

    # --- Tailles de fenêtres (oddes) ---
    # pas temporel moyen (s) entre frames
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    if dt <= 0.0:
        # fallback: estime via longueur / nb frames si possible
        dt = (times[-1] / max(1, n_frames - 1)) if len(times) > 0 else 1.0

    L_t = max(1, int(round(time_s / dt)))  # en frames
    if L_t % 2 == 0:
        L_t += 1

    L_f = max(1, int(freq_bins))  # en bins
    if L_f % 2 == 0:
        L_f += 1

    # --- Kernels 1D normalisés ---
    def gaussian_kernel(L):
        # sigma ~ L/3 pour une gaussienne "classique"
        sigma = max(1e-6, L / 3.0)
        r = (L - 1) // 2
        x = np.arange(-r, r + 1, dtype=np.float64)
        k = np.exp(-(x * x) / (2.0 * sigma * sigma))
        k /= k.sum()
        return k

    def box_kernel(L):
        return np.full(L, 1.0 / L, dtype=np.float64)

    if method.lower() == "gaussian":
        kt = gaussian_kernel(L_t)
        kf = gaussian_kernel(L_f)
    elif method.lower() == "box":
        kt = box_kernel(L_t)
        kf = box_kernel(L_f)
    else:
        raise ValueError("method doit être 'gaussian' ou 'box'")

    # --- Convolutions séparables (temps puis fréquence) ---
    def convolve_along_axis(X, kernel, axis: int, pad_mode: str = "reflect"):
        """
        Convolution 1D le long d'un axe avec padding symétrique.
        Garantit une sortie de même longueur que l'entrée sur l'axe ciblé.
        """
        X = np.asarray(X)
        L = len(kernel)
        r = (L - 1) // 2
        expected_N = X.shape[axis]

        # Padding symétrique
        pad_width = [(0, 0)] * X.ndim
        pad_width[axis] = (r, r)
        Xp = np.pad(X, pad_width, mode=pad_mode)

        # Amène l'axe ciblé en dernier, reshape en (rows, N_pad)
        Xp = np.moveaxis(Xp, axis, -1)  # (..., N + 2r)
        rows = int(np.prod(Xp.shape[:-1])) if Xp.ndim > 1 else 1
        N_pad = Xp.shape[-1]
        Xw = Xp.reshape(rows, N_pad)

        # Conteneur de sortie (même shape que X)
        out = np.empty_like(X, dtype=np.float64)
        Yw = out.reshape(rows, expected_N)

        # Convolution ligne par ligne
        for i in range(rows):
            y = np.convolve(Xw[i], kernel, mode="valid")  # longueur = expected_N + 1
            # Corrige l'excès d'un élément (off-by-one) dû à r=r(L)
            if y.shape[0] == expected_N + 1:
                y = y[:-1]
            elif y.shape[0] != expected_N:
                # Sécurité: recadrage centré si jamais (ne devrait pas arriver)
                start = (y.shape[0] - expected_N) // 2
                y = y[start : start + expected_N]
            Yw[i] = y

        # Restaure la forme et la position d'axe
        out = Yw.reshape(*Xp.shape[:-1], expected_N)
        out = np.moveaxis(out, -1, axis)
        return out

    # applique d'abord le lissage temporel (axis=1), puis fréquentiel (axis=0)
    tmp = convolve_along_axis(mag.astype(np.float64, copy=False), kt, axis=1)
    smoothed = convolve_along_axis(tmp, kf, axis=0)

    # Optionnel: renormalisation à [0,1] (utile si tu relies à un seuil fixe)
    if renormalize:
        mx = smoothed.max() if smoothed.size else 0.0
        if mx > 0:
            smoothed /= mx

    return smoothed.astype(mag.dtype, copy=False)


def reinforce_fundamentals(
    mag,
    *,
    bins_per_octave=12,
    max_harm=6,  # on cherche les harmoniques 2..max_harm
    presence_thresh=0.15,  # seuil relatif (à la crête de la frame) pour “compter” une harmonique
    neigh_bins=1,  # tolérance d’intonation ±neigh_bins (max-pooling après alignement)
    boost_strength=0.8,  # intensité de renfort : 0..1 (≈ poids appliqué au nombre d’harmoniques)
    use_amplitude=False,  # False = on compte; True = on pondère par l’amplitude alignée
    renormalize=True,  # re-normaliser chaque frame sur [0,1] à la fin
    eps=1e-12,
):
    """
    Renforce les fondamentales selon le nombre d'harmoniques détectées.

    mag : np.ndarray de forme (n_bins, n_frames), normalisée [0,1] (comme ta sortie CQT).
    Retourne une matrice de même forme.

    Principe :
    - pour chaque harmonique k (2..max_harm), on décale mag de ~log2(k)*BPO vers le bas
      (pour aligner l’harmonique k sur le bin de la fondamentale),
    - on fait un petit max-pooling ±neigh_bins pour tolérer un léger désaccord,
    - on “compte” l’harmonique si son amplitude alignée dépasse presence_thresh * max_de_la_frame,
    - plus un bin cumule d’harmoniques, plus on le renforce.
    """
    if mag is None or mag.size == 0:
        return mag

    n_bins, n_frames = mag.shape
    X = mag

    # Pré-calcul des décalages (en bins CQT) pour aligner k*f0 -> f0
    offsets = []
    for k in range(2, max_harm + 1):
        off = int(round(bins_per_octave * math.log2(k)))
        if off < n_bins:  # si l’harmonique sort de bande, on l’ignore
            offsets.append(off)
    if not offsets:
        return mag.copy()

    # max par frame pour un seuil relatif robuste
    frame_max = np.maximum(np.max(X, axis=0, keepdims=True), eps)

    # Accumulateurs
    harmonic_count = np.zeros_like(X)  # nombre d'harmoniques présentes par (bin, frame)
    harmonic_power = np.zeros_like(
        X
    )  # somme des amplitudes alignées (si use_amplitude)

    for off in offsets:
        # Aligner l’harmonique k: on “rabaisse” la bande haute (i+off) sur (i)
        aligned = np.zeros_like(X)
        aligned[:-off, :] = X[off:, :]  # padding zéro en haut

        # Tolérance d’intonation : max-pooling ±neigh_bins (après alignement)
        if neigh_bins > 0:
            pooled = aligned.copy()
            for d in range(1, neigh_bins + 1):
                # roll + masque zéro pour éviter les déversements aux bords
                up = np.zeros_like(aligned)
                up[d:, :] = aligned[:-d, :]
                dn = np.zeros_like(aligned)
                dn[:-d, :] = aligned[d:, :]
                pooled = np.maximum(pooled, np.maximum(up, dn))
            aligned = pooled

        # Présence au-dessus d’un seuil relatif (par frame)
        present = aligned >= (presence_thresh * frame_max)

        harmonic_count += present.astype(X.dtype)
        if use_amplitude:
            harmonic_power += aligned

    # Score d’harmoniques (compte, éventuellement mixé avec l'amplitude)
    if use_amplitude:
        # normaliser la puissance par (max_harm-1) pour rester dans [0,1] environ
        score = (harmonic_count + harmonic_power) / max(1, (len(offsets)))
    else:
        score = harmonic_count / max(1, (len(offsets)))

    # Facteur de renfort (seulement sur les fondamentales candidates)
    # 1 + boost_strength * score  ∈ [1, 1+boost_strength]
    boost = 1.0 + boost_strength * score
    Y = X * boost

    if renormalize:
        max_per_frame = np.maximum(np.max(Y, axis=0, keepdims=True), eps)
        Y = Y / max_per_frame

    return Y


def plot_pianoroll(
    mag, times, note_labels, title, threshold: float | None = None
):  # ex: 75    -> masque mag < 75e percentile
    """
    Affiche un piano-roll CQT/VQT avec masque sur faibles magnitudes.
      - threshold : seuil absolu (dans [0,1]) appliqué sur la magnitude normalisée
      - percentile : si fourni, on calcule le seuil comme np.percentile(mag, percentile)
      - show_beats : axe X en beats
      - save_path  : si fourni, sauvegarde le PNG
    """
    if mag.size == 0:
        print("Aucune donnée pour le piano-roll.")
        return

    # Axe X: secondes ou beats
    x = times
    x_label = "Temps (s)"

    if threshold is not None:
        thr = float(threshold)
    else:
        thr = None

    # Masquage des faibles magnitudes
    data = mag
    cmap = None
    if thr is not None:
        data = np.ma.masked_less(mag, thr)

        cmap = plt.cm.viridis.copy()  # n'impose pas de couleurs spécifiques
        cmap.set_bad(alpha=0.0)  # valeurs masquées -> transparentes

    fig, ax = plt.subplots(figsize=(14, 6))

    n_bins = data.shape[0]
    x_end = x[-1] if len(x) > 1 else (x[0] + WINDOW_TIME)
    extent = [x[0], x_end, 0, n_bins]
    im = ax.imshow(data, aspect="auto", origin="lower", extent=extent, cmap=cmap)

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
    if thr is not None:
        title += f" (masque < {thr:.3f})"
    ax.set_title(title)

    plt.tight_layout()


def track_notes(
    mag,
    times,
    *,
    bins_per_octave=12,
    freqs=None,  # np.array(n_bins) en Hz ; si None et librosa dispo, sera déduit
    fmin_note="C0",  # utilisé seulement si freqs=None (via librosa.note_to_hz)
    max_harm=6,  # harmoniques 2..max_harm pour scorer les fondamentales
    peak_rel_thresh=0.20,  # seuil relatif (à max de frame) pour garder un pic
    peak_neighborhood=1,  # pic local : strictement plus grand que ±peak_neighborhood
    harm_neigh_bins=1,  # tolérance d’accordage pour l’alignement harmonique
    w_pitch=1.0,  # poids coût |Δpitch|
    w_harm=0.7,  # poids bonus harmonique (soustrait au coût)
    w_energy=0.15,  # petite pénalité si l’énergie diffère trop
    max_jump_bins=3,  # gating : écart max piste→candidat (en bins)
    gap_max=2,  # frames de “miss” tolérées pour garder la piste active
    k_confirm=3,  # confirmations nécessaires (sur une fenêtre courte) pour promouvoir une piste
    min_duration_frames=4,  # rejette les segments plus courts que ça (bruit)
    gap_join=3,  # joint deux segments si le trou ≤ gap_join
    thr_join_bins=1.5,  # et si l’écart moyen de pitch < thr_join_bins
    penalty_octave=0.5,  # pénalité ajoutée si candidat ≈ piste ±12 bins (évite erreurs d’octave)
    use_energy_ratio=True,  # compare énergie en ratio (sinon différence absolue)
    random_seed=0,  # reproductibilité du tie-breaking
):
    """
    Entrées
    -------
    mag : (n_bins, n_frames) magnitudes CQT normalisées (0..1).
    times : (n_frames,) centres temporels en secondes.

    Sortie
    ------
    tracks : liste de dicts, chacun contient :
        - 'onset_idx', 'offset_idx'          (indices de frames, inclusif/exclusif)
        - 'onset_time', 'offset_time'        (en secondes)
        - 'pitch_bins'                       (np.array des bins suivis)
        - 'pitch_hz' (si freqs fourni ou déductible)
        - 'energy'                           (np.array énergie par frame)
        - 'harm_score'                       (np.array score harmonique)
        - 'confirmed'                        (bool)
    """

    rng = np.random.RandomState(random_seed)

    if mag is None or mag.size == 0:
        return []

    n_bins, n_frames = mag.shape

    # --- fréquences par bin (optionnel pour la sortie) ---
    if freqs is None:
        try:
            import librosa

            fmin = librosa.note_to_hz(fmin_note)
            freqs = librosa.cqt_frequencies(
                n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave
            )
        except Exception:
            freqs = None  # on pourra quand même renvoyer les bins

    # --- offsets (en bins) pour aligner les harmoniques k sur la fondamentale ---
    harm_offsets = []
    for k in range(2, max_harm + 1):
        off = int(round(bins_per_octave * math.log2(k)))
        if 0 < off < n_bins:
            harm_offsets.append(off)
    if not harm_offsets:
        harm_offsets = [
            bins_per_octave
        ]  # au moins l’octave comme “harmonique” si configuration étroite

    # --- extraction de pics par frame ---
    frame_max = np.maximum(mag.max(axis=0), 1e-12)
    thr = peak_rel_thresh * frame_max  # seuil relatif par frame

    def find_frame_peaks(col, thr_val):
        # candidats = pics locaux au-dessus du seuil
        if np.all(col <= thr_val):
            return []
        # comparaison locale ±peak_neighborhood
        peaks = []
        for i in range(n_bins):
            v = col[i]
            if v <= thr_val:
                continue
            ok = True
            for d in range(1, peak_neighborhood + 1):
                if i - d >= 0 and col[i - d] >= v:
                    ok = False
                    break
                if i + d < n_bins and col[i + d] > v:
                    ok = False
                    break
            if ok:
                peaks.append(i)
        return peaks

    # --- score harmonique d’un bin (par frame) ---
    def harmonic_score_at(bin_idx, t):
        base = mag[bin_idx, t]
        if base <= 0.0:
            return 0.0, 0.0
        count = 0
        amp_sum = 0.0
        for off in harm_offsets:
            j = bin_idx + off
            if j >= n_bins:  # hors bande
                continue
            # tolérance d’accordage
            vals = [mag[j, t]]
            for d in range(1, harm_neigh_bins + 1):
                if j - d >= 0:
                    vals.append(mag[j - d, t])
                if j + d < n_bins:
                    vals.append(mag[j + d, t])
            a = max(vals)
            amp_sum += a
            # on compte comme "présent" si relatif au max de frame
            if a >= 0.1 * frame_max[t]:  # seuil doux pour la présence harmonique
                count += 1
        # normalisation douce
        if len(harm_offsets) > 0:
            count_norm = count / float(len(harm_offsets))
            amp_norm = amp_sum / float(len(harm_offsets))
        else:
            count_norm, amp_norm = 0.0, 0.0
        # combiner présence + amplitude
        score = 0.6 * count_norm + 0.4 * (amp_norm / (frame_max[t] + 1e-12))
        return score, amp_norm

    # --- conteneur piste ---
    class Track:
        __slots__ = (
            "id",
            "p",
            "v",
            "last_t",
            "miss",
            "buf_bins",
            "buf_energy",
            "buf_harm",
            "alive",
            "confirmed",
            "hits",
            "avg_energy",
            "avg_harm",
        )

        def __init__(self, tid, p0, e0, h0, t0):
            self.id = tid
            self.p = float(p0)
            self.v = 0.0
            self.last_t = t0
            self.miss = 0
            self.buf_bins = [p0]
            self.buf_energy = [e0]
            self.buf_harm = [h0]
            self.alive = True
            self.confirmed = False
            self.hits = 1
            self.avg_energy = float(e0)
            self.avg_harm = float(h0)

        def predict(self):
            return self.p + self.v

        def update(self, p_new, e_new, h_new, t):
            dt = max(1, t - self.last_t)
            v_new = (p_new - self.p) / dt
            # lissage léger de la vitesse
            self.v = 0.6 * self.v + 0.4 * v_new
            self.p = float(p_new)
            self.last_t = t
            self.miss = 0
            self.buf_bins.append(p_new)
            self.buf_energy.append(e_new)
            self.buf_harm.append(h_new)
            self.hits += 1
            # moyennes exponentielles
            self.avg_energy = 0.8 * self.avg_energy + 0.2 * e_new
            self.avg_harm = 0.8 * self.avg_harm + 0.2 * h_new
            # promotion si assez d’évidence
            if not self.confirmed and self.hits >= k_confirm:
                self.confirmed = True

        def miss_one(self):
            self.miss += 1
            # on “prolonge” la prédiction pour garder l’alignement
            self.buf_bins.append(self.predict())
            self.buf_energy.append(self.avg_energy)
            self.buf_harm.append(self.avg_harm)

        def kill(self):
            self.alive = False

    # --- boucle principale d’association ---
    tracks_active = []
    all_tracks_finished = []
    next_id = 1

    for t in range(n_frames):
        col = mag[:, t]
        peaks = find_frame_peaks(col, thr[t])
        if not peaks:
            # pas de candidats : tous les actifs prennent un miss
            for tr in list(tracks_active):
                tr.miss_one()
                if tr.miss > gap_max:
                    tr.kill()
                    all_tracks_finished.append(tr)
                    tracks_active.remove(tr)
            continue

        # fabriquer la liste de candidats [{bin, energy, harm_score}]
        cand_bins = np.array(peaks, dtype=float)
        cand_energy = col[peaks]
        cand_harm = np.zeros(len(peaks), dtype=float)
        for idx, b in enumerate(peaks):
            sc, _ = harmonic_score_at(b, t)
            cand_harm[idx] = sc

        # Si aucune piste active, spawn tout comme pistes tentées
        if not tracks_active:
            for i in range(len(peaks)):
                tr = Track(next_id, cand_bins[i], cand_energy[i], cand_harm[i], t)
                tracks_active.append(tr)
                next_id += 1
            continue

        # matrice de coût (tracks x candidats)
        T = len(tracks_active)
        C = len(peaks)
        cost = np.full((T, C), np.inf, dtype=float)

        for i, tr in enumerate(tracks_active):
            p_hat = tr.predict()
            for j in range(C):
                dp = abs(p_hat - cand_bins[j])
                if dp > max_jump_bins:
                    continue  # gating dur
                # pénalité octave si très proche de ±12 bins
                if abs(dp - 12.0) < 0.6:  # fenêtre autour d’une octave
                    dp += penalty_octave
                # similarité harmonique -> on soustrait (bonus)
                harm_bonus = w_harm * cand_harm[j]
                # énergie : ratio ou delta
                if use_energy_ratio:
                    ratio = cand_energy[j] / (tr.avg_energy + 1e-6)
                    e_term = abs(1.0 - np.clip(ratio, 1e-3, 1e3))
                else:
                    e_term = abs(cand_energy[j] - tr.avg_energy)
                c = w_pitch * dp + w_energy * e_term - harm_bonus
                cost[i, j] = c

        # assignment : Hongrois si possible, sinon glouton
        assigned_tracks = set()
        assigned_cands = set()
        pairs = []

        try:
            from scipy.optimize import linear_sum_assignment

            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if (
                    np.isfinite(cost[i, j])
                    and i not in assigned_tracks
                    and j not in assigned_cands
                ):
                    pairs.append((i, j))
                    assigned_tracks.add(i)
                    assigned_cands.add(j)
        except Exception:
            # glouton : on prend (i,j) par coût croissant
            flat = [
                (cost[i, j], i, j)
                for i in range(T)
                for j in range(C)
                if np.isfinite(cost[i, j])
            ]
            flat.sort(key=lambda x: (x[0], rng.rand()))
            for _, i, j in flat:
                if i in assigned_tracks or j in assigned_cands:
                    continue
                pairs.append((i, j))
                assigned_tracks.add(i)
                assigned_cands.add(j)

        # mise à jour des pistes assignées
        for i, j in pairs:
            tr = tracks_active[i]
            tr.update(cand_bins[j], cand_energy[j], cand_harm[j], t)

        # les pistes non assignées : miss
        for i, tr in enumerate(list(tracks_active)):
            if i not in assigned_tracks:
                tr.miss_one()
                if tr.miss > gap_max:
                    tr.kill()
                    all_tracks_finished.append(tr)
                    tracks_active.remove(tr)

        # les candidats non assignés : nouvelles pistes
        for j in range(C):
            if j not in assigned_cands:
                tr = Track(next_id, cand_bins[j], cand_energy[j], cand_harm[j], t)
                tracks_active.append(tr)
                next_id += 1

    # clôturer tout ce qui reste
    for tr in tracks_active:
        tr.kill()
        all_tracks_finished.append(tr)

    # --- convertir les Track en segments (onset/offset, filtrage bruit, fusion) ---
    def track_to_segment(tr):
        # enlever les samples "miss" de fin si au-delà du dernier hit
        # (ici on garde tout le buffer; le filtre de durée fera le ménage)
        bins_arr = np.asarray(tr.buf_bins, dtype=float)
        energy_arr = np.asarray(tr.buf_energy, dtype=float)
        harm_arr = np.asarray(tr.buf_harm, dtype=float)
        # indices
        onset_idx = tr.last_t - (len(bins_arr) - 1)
        offset_idx = tr.last_t + 1  # exclusif
        onset_idx = max(0, onset_idx)
        offset_idx = min(n_frames, offset_idx)
        return {
            "id": tr.id,
            "onset_idx": onset_idx,
            "offset_idx": offset_idx,
            "onset_time": float(times[onset_idx]) if len(times) > onset_idx else None,
            "offset_time": (
                float(times[offset_idx - 1])
                if len(times) >= offset_idx and offset_idx > 0
                else None
            ),
            "pitch_bins": bins_arr,
            "energy": energy_arr,
            "harm_score": harm_arr,
            "confirmed": bool(tr.confirmed),
        }

    segments = [track_to_segment(tr) for tr in all_tracks_finished]

    # filtrage : confirmation & durée minimale
    def seg_duration_frames(seg):
        return max(0, seg["offset_idx"] - seg["onset_idx"])

    segments = [
        s
        for s in segments
        if (s["confirmed"] and seg_duration_frames(s) >= min_duration_frames)
    ]

    # fusion ("join the dots") : segments proches séparés d’un petit trou
    segments.sort(key=lambda s: (s["onset_idx"], s["id"]))
    merged = []
    for s in segments:
        if not merged:
            merged.append(s)
            continue
        prev = merged[-1]
        gap = s["onset_idx"] - prev["offset_idx"]
        if 0 <= gap <= gap_join:
            # distance moyenne de pitch sur les bords
            p_prev = np.mean(prev["pitch_bins"][-min(3, len(prev["pitch_bins"])) :])
            p_s = np.mean(s["pitch_bins"][: min(3, len(s["pitch_bins"]))])
            if abs(p_prev - p_s) <= thr_join_bins:
                # fusion
                prev["offset_idx"] = s["offset_idx"]
                prev["offset_time"] = s["offset_time"]
                prev["pitch_bins"] = np.concatenate(
                    [prev["pitch_bins"], s["pitch_bins"]]
                )
                prev["energy"] = np.concatenate([prev["energy"], s["energy"]])
                prev["harm_score"] = np.concatenate(
                    [prev["harm_score"], s["harm_score"]]
                )
                continue
        merged.append(s)

    # ajouter pitch en Hz si possible
    if freqs is not None:
        for s in merged:
            # clamp indices pour échantillonnage
            idxs = np.clip(np.round(s["pitch_bins"]).astype(int), 0, n_bins - 1)
            s["pitch_hz"] = freqs[idxs]
    else:
        for s in merged:
            s["pitch_hz"] = None

    return merged


def _group_by(seq, key):
    seq_sorted = sorted(seq, key=key)
    groups = []
    cur_k = object()
    cur_group = []
    for item in seq_sorted:
        k = key(item)
        if k != cur_k:
            if cur_group:
                groups.append((cur_k, cur_group))
            cur_k = k
            cur_group = [item]
        else:
            cur_group.append(item)
    if cur_group:
        groups.append((cur_k, cur_group))
    return groups


def tracks_to_notes(
    tracks,
    times,
    *,
    # unités et conversion
    output_units="beats",  # "beats" ou "seconds"
    bpm=120.0,  # utilisé si output_units="beats"
    # nettoyage pour MIDI
    min_duration_beats=1 / 64,  # durée minimale (beats) après conversion
    epsilon_beats=1e-3,  # écart min entre fin et début sur même pitch
    overlap_policy="trim",  # "trim" | "shift" | "merge"
    # choix de la fréquence représentative
    use_energy_weight=True,  # fréquence moyenne pondérée par l’énergie
):
    """
    Convertit des 'tracks' (sortie de tracking) en liste de Note,
    en dédupliquant et empêchant les chevauchements par pitch.

    - output_units: 'beats' => start/length seront en beats (recommandé pour midiutil)
                    'seconds' => start/length resteront en secondes
    - overlap_policy:
        * 'trim'  : coupe la note précédente à (start_next - ε)
        * 'shift' : décale la note suivante à (end_prev + ε)
        * 'merge' : fusionne les deux notes (start=min starts, end=max ends)
    """
    if not tracks:
        return []

    # 1) Extraire segments “plats” (start, end, pitch_hz, energy…)
    segs = []
    for seg in tracks:
        i0, i1 = int(seg["onset_idx"]), int(seg["offset_idx"])
        if i0 >= i1 or i0 < 0 or i1 > len(times):
            continue

        # temps en secondes
        start_s = float(times[i0])
        end_s = float(times[i1 - 1]) if i1 - 1 < len(times) else float(times[-1])
        # si on veut inclure le centre de la dernière frame, rajouter demi-hop au besoin
        # ici on prend la borne supérieure i1 si disponible:
        if i1 < len(times):
            end_s = float(times[i1])

        duration_s = max(0.0, end_s - start_s)
        if duration_s == 0.0:
            continue

        # fréquence instantanée
        if "pitch_hz" in seg and seg["pitch_hz"] is not None:
            freq_series = np.asarray(seg["pitch_hz"], dtype=float)
        else:
            # fallback: approx via bins -> Hz impossible sans fmin/BPO;
            # on prend une pseudo-fréquence en mappant le bin moyen sur A440 (purement pour avoir un nombre)
            bins = np.asarray(seg["pitch_bins"], dtype=float)
            if bins.size == 0:
                continue
            # approximation: midi ~ 69 + bins (si l'échelle de bins ~ demi-tons). Ajuste si besoin.
            midi_est = 69.0 + (np.nanmean(bins))
            freq_series = np.array([440.0 * (2.0 ** ((midi_est - 69.0) / 12.0))])

        # énergie
        energy = np.asarray(seg.get("energy", []), dtype=float)
        if energy.size == 0 or energy.size != freq_series.size:
            energy = np.ones_like(freq_series)

        # fréquence représentative (pondérée ou pas)
        if use_energy_weight and np.sum(energy) > 0:
            freq = float(np.sum(freq_series * energy) / np.sum(energy))
        else:
            freq = float(np.mean(freq_series))

        # start/length selon unités demandées
        if output_units == "beats":
            start = start_s
            end = end_s
        elif output_units == "seconds":
            start = start_s
            end = end_s
        else:
            raise ValueError("output_units doit être 'beats' ou 'seconds'.")

        length = max(0.0, end - start)
        # construire élément intermédiaire
        midi_number = Tools.freq_to_number(freq)
        name = Tools.note_name(midi_number)

        segs.append(
            {
                "pitch_hz": freq,
                "midi": midi_number,
                "name": name,
                "start": start,
                "end": end,
                "length": length,
                "times": [
                    start,
                    end,
                ],  # placeholder; la classe Note attend une série de times
                "magnitudes": [
                    1.0,
                    1.0,
                ],  # placeholder; tu peux remplacer par l’énergie du track resamplée
            }
        )

    if not segs:
        return []

    # 2) Déduplication stricte: même (midi, start) => garder la plus longue durée
    segs.sort(key=lambda s: (s["midi"], s["start"], -s["length"]))
    dedup = []
    last_key = None
    for s in segs:
        key = (
            s["midi"],
            round(s["start"], 6),
        )  # arrondi pour tolérer de minuscules flottants
        if key != last_key:
            dedup.append(s)
            last_key = key
        else:
            # doublon: le premier a déjà la durée max (trié par -length), on ignore celui-ci
            pass
    segs = dedup

    # 3) Anti-overlap par pitch (trim/shift/merge) + epsilon + durée minimale
    epsilon = epsilon_beats if output_units == "beats" else (epsilon_beats * 60.0 / bpm)
    min_len = (
        min_duration_beats
        if output_units == "beats"
        else (min_duration_beats * 60.0 / bpm)
    )

    notes_clean = []
    for midi_val, group in _group_by(segs, key=lambda s: s["midi"]):
        # tri par start
        group.sort(key=lambda s: (s["start"], s["end"]))
        current = None
        for s in group:
            if current is None:
                current = s
                continue
            # Chevauchement ?
            if s["start"] < current["end"] - 1e-9:
                if overlap_policy == "trim":
                    # on coupe la note courante
                    current["end"] = max(current["start"], s["start"] - epsilon)
                    current["length"] = max(0.0, current["end"] - current["start"])
                    # si elle devient trop courte, on la jette
                    if current["length"] < min_len:
                        current = s
                        continue
                elif overlap_policy == "shift":
                    # on décale la note suivante
                    shift = (current["end"] + epsilon) - s["start"]
                    s["start"] += max(0.0, shift)
                    s["length"] = max(0.0, s["end"] - s["start"])
                    # si trop courte après décalage, on l’ignore
                    if s["length"] < min_len:
                        continue
                elif overlap_policy == "merge":
                    # on fusionne les deux
                    current["end"] = max(current["end"], s["end"])
                    current["length"] = max(0.0, current["end"] - current["start"])
                    continue
                else:
                    raise ValueError(
                        "overlap_policy doit être 'trim', 'shift' ou 'merge'."
                    )

            else:
                # pas de chevauchement; mais impose un epsilon si start == end (collés)
                if abs(s["start"] - current["end"]) < epsilon:
                    # petite marge pour éviter 'On' et 'Off' exactement au même tick
                    if overlap_policy in ("trim", "merge"):
                        current["end"] = current["end"] - 0.5 * epsilon
                        current["length"] = max(0.0, current["end"] - current["start"])
                        s["start"] = s["start"] + 0.5 * epsilon
                        s["length"] = max(0.0, s["end"] - s["start"])
                    else:  # shift
                        s["start"] = current["end"] + epsilon
                        s["length"] = max(0.0, s["end"] - s["start"])

            # valider et pousser current si assez long, puis avancer
            if current["length"] >= min_len:
                notes_clean.append(current)
            current = s

        # pousser le dernier
        if current and current["length"] >= min_len:
            notes_clean.append(current)

    # 4) Construire les objets Note (ta classe)
    # Tri final par start
    notes_clean.sort(key=lambda s: (s["start"], s["midi"]))
    notes_out = []
    for s in notes_clean:
        # times/magnitudes : si tu veux mieux, tu peux y mettre l’enveloppe d’énergie du track
        note = Note(
            frequency=s["pitch_hz"],
            magnitudes=s["magnitudes"],
            times=s["times"],
            start=s["start"],
            length=s["length"],
        )
        note.length_bpm = Tools.seconds_to_beat(note.length, bpm)
        note.start_bpm = Tools.seconds_to_beat(note.start_time, bpm)
        notes_out.append(note)

    return notes_out


def harmonic_clean_and_octave_promote(
    mag,
    *,
    bins_per_octave=12,
    max_harm=8,  # harmoniques à considérer (2..K)
    tol_bins=1,  # tolérance d’accordage ± tol_bins
    rel_thresh=0.12,  # seuil relatif (vs max de la frame) pour considérer une harmonique
    octave_factor=2.0,  # score d’octave doit être supérieur à factor * score f0
    # Atténuation des harmoniques
    attenuate_harmonics=True,
    harmonic_atten=0.35,  # multiplier les harmoniques identifiées par ce facteur (0..1)
    energy_conserve=True,  # si True, réalloue l’énergie retirée vers la fondamentale locale
    # Promotion d’octave (répare f0 trop bas)
    promote_octaves=True,
    max_octave_shift=2,  # promote +1/+2 octaves max
    octave_win_support=3,  # nb d’harmoniques considérées pour le score (min 2)
    octave_margin=0.15,  # marge de score requise pour promouvoir (anti-flap)
    promote_fraction=0.7,  # fraction d’énergie déplacée vers l’octave promue (0..1)
    eps=1e-12,
):
    """
    Pré-filtre CQT pour :
      (A) atténuer les harmoniques H2..K quand une fondamentale plausible est identifiée en dessous,
      (B) promouvoir certaines fondamentales sous-estimées (+12/+24 bins) si leur gabarit harmonique y est meilleur.

    mag : ndarray (n_bins, n_frames), valeurs quelconques (l’algo travaille en relatif au max par frame).
    Retour : ndarray (n_bins, n_frames), même shape, magnitudes ajustées.
    """

    if mag is None or mag.size == 0:
        return mag

    n_bins, n_frames = mag.shape
    Y = mag.copy()

    # Offsets (en bins log-f) des harmoniques k=2..K
    harm_off = []
    for k in range(2, max_harm + 1):
        off = int(round(bins_per_octave * math.log2(k)))
        if 0 < off < n_bins:
            harm_off.append((k, off))
    if not harm_off:
        return Y

    # --- Helpers locaux -------------------------------------------------------

    def local_max_aligned(col, idx):
        """Max sur [idx - tol_bins, idx + tol_bins] borné à [0, n_bins)."""
        j0 = max(0, idx - tol_bins)
        j1 = min(n_bins, idx + tol_bins + 1)
        if j0 >= j1:
            return 0.0
        return float(np.max(col[j0:j1]))

    def harmonic_presence(col, f0_bin, fmax):
        """
        Pour une hypothèse de fondamentale f0_bin, renvoie:
          - count_norm: proportion d’harmoniques présentes (≥ rel_thresh * fmax),
          - amp_norm  : moyenne des amplitudes harmonique (non normalisée par fmax).
        """
        count = 0
        amp_sum = 0.0
        used = 0
        for _, off in harm_off:
            j = f0_bin + off
            if j >= n_bins:
                break
            a = local_max_aligned(col, j)
            amp_sum += a
            if a >= rel_thresh * fmax:
                count += 1
            used += 1
        if used == 0:
            return 0.0, 0.0
        return (count / used), (amp_sum / used)

    def template_score(col, f0_bin, k_use, fmax):
        """
        Score d’adéquation harmonique d’une fondamentale à f0_bin,
        basé sur les k_use premières harmoniques.
        """
        cnt = 0
        power = 0.0
        used = 0
        for idx, (_, off) in enumerate(harm_off):
            if idx >= k_use:
                break
            j = f0_bin + off
            if j >= n_bins:
                break
            a = local_max_aligned(col, j)
            power += a
            if a >= rel_thresh * fmax:
                cnt += 1
            used += 1
        if used == 0:
            return 0.0
        pres = cnt / used
        pow_rel = power / (used * fmax + eps)
        return 0.6 * pres + 0.4 * pow_rel

    # --- Traitement frame par frame ------------------------------------------

    for t in range(n_frames):
        col = Y[:, t]
        if np.all(col <= eps):
            continue

        # max scalaire de la frame courante (pour les seuils relatifs)
        fmax = float(max(col.max(), eps))

        # (A) Atténuation d’harmoniques guidée par un set de pics
        if attenuate_harmonics:
            peaks = []
            thr = rel_thresh * fmax
            for i in range(n_bins):
                v = col[i]
                if v <= thr:
                    continue
                left_ok = (i == 0) or (col[i] > col[i - 1])
                right_ok = (i == n_bins - 1) or (col[i] >= col[i + 1])
                if left_ok and right_ok:
                    peaks.append(i)

            for p in peaks:
                best_f0 = None
                best_support = -1.0

                # Candidats fondamentaux en dessous: p - off(k)
                for _, off in harm_off:
                    f0 = p - off
                    if f0 < 0:
                        continue
                    base_energy = local_max_aligned(col, f0)
                    if base_energy <= eps:
                        continue
                    cnt_norm, amp_norm = harmonic_presence(col, f0, fmax)
                    # mix énergie f0 (rel.) + structure harmonique
                    score = 0.4 * (base_energy / (fmax + eps)) + 0.6 * (
                        0.6 * cnt_norm + 0.4 * (amp_norm / (fmax + eps))
                    )
                    if score > best_support:
                        best_support = score
                        best_f0 = f0

                # --- Règle spéciale "octave" (1ʳᵉ harmonique) ---
                # Si p ≈ 2 * best_f0 (± tol_bins) et que l'octave n'est pas assez forte,
                # on SUPPRIME l'harmonique (mise à zéro) au lieu de simplement l'atténuer.
                if (
                    best_f0 is not None
                    and best_support > 0.2
                    and abs((p - best_f0) - bins_per_octave) <= tol_bins
                ):
                    # Fenêtres locales robustes
                    j0_f0 = max(0, best_f0 - tol_bins)
                    j1_f0 = min(n_bins, best_f0 + tol_bins + 1)

                    oct_ix = int(round(best_f0 + bins_per_octave))
                    j0_oct = max(0, oct_ix - tol_bins)
                    j1_oct = min(n_bins, oct_ix + tol_bins + 1)

                    # Énergies max locales (tu peux utiliser .mean() si tu préfères)
                    f0_energy = col[j0_f0:j1_f0].max() if j1_f0 > j0_f0 else 0.0
                    oct_energy = col[j0_oct:j1_oct].max() if j1_oct > j0_oct else 0.0

                    # Si l'octave n'est pas >= octave_factor × fondamentale -> suppression de TOUTE la bande d'octave
                    if (
                        f0_energy > 0.0
                        and oct_energy < octave_factor * f0_energy
                        and j1_oct > j0_oct
                    ):
                        removed = float(col[j0_oct:j1_oct].sum())
                        col[j0_oct:j1_oct] = 0.0
                        if energy_conserve and removed > 0.0 and j1_f0 > j0_f0:
                            col[j0_f0:j1_f0] += removed / (j1_f0 - j0_f0)
                        # Octave traitée : passer au pic suivant (ne pas appliquer l'atténuation générique)
                        continue

                # --- Cas général: atténuation des harmoniques expliquées par une f0 plausible ---
                if best_f0 is not None and best_support > 0.2 and harmonic_atten < 1.0:
                    removed = col[p] * (1.0 - harmonic_atten)
                    col[p] *= harmonic_atten
                    if energy_conserve and removed > 0.0:
                        j0 = max(0, best_f0 - tol_bins)
                        j1 = min(n_bins, best_f0 + tol_bins + 1)
                        if j1 > j0:
                            col[j0:j1] += removed / (j1 - j0)

        # (B) Promotion d’octave (corrige f0 trop bas → +12/+24 bins)
        if promote_octaves:
            peaks2 = []
            thr = rel_thresh * fmax
            for i in range(n_bins):
                v = col[i]
                if v <= thr:
                    continue
                left_ok = (i == 0) or (col[i] > col[i - 1])
                right_ok = (i == n_bins - 1) or (col[i] >= col[i + 1])
                if left_ok and right_ok:
                    peaks2.append(i)

            for p in peaks2:
                base_score = template_score(
                    col, p, k_use=max(2, octave_win_support), fmax=fmax
                )
                best_shift = 0
                best_score = base_score

                for s in range(1, max_octave_shift + 1):
                    q = p + s * bins_per_octave
                    if q >= n_bins:
                        break
                    score_q = template_score(
                        col, q, k_use=max(2, octave_win_support), fmax=fmax
                    )
                    if score_q > best_score + octave_margin:
                        best_score = score_q
                        best_shift = s

                if best_shift > 0:
                    q = p + best_shift * bins_per_octave
                    move = col[p] * promote_fraction
                    if move > 0.0:
                        col[p] -= move
                        j0 = max(0, q - tol_bins)
                        j1 = min(n_bins, q + tol_bins + 1)
                        if j1 > j0:
                            col[j0:j1] += move / (j1 - j0)

        Y[:, t] = col

    return Y


def plot_note_tracks(
    tracks,
    *,
    times=None,  # array des centres de frames (optionnel mais recommandé)
    y_mode="midi",  # "midi" | "hz" | "bin"
    bins_per_octave=12,  # requis si y_mode="midi" et pas de pitch_hz
    fmin_hz=None,  # requis si y_mode="midi" et pas de pitch_hz (ex: librosa.note_to_hz("C0"))
    note_labels=None,  # optionnel: labels des bins pour y_mode="bin" (ex: ['C0','C#0',...])
    title="Trajectoires de notes",
):
    """
    Affiche les trajectoires de notes retournées par track_notes.

    - y_mode="midi" : affiche en demi-tons MIDI (si pitch_hz dispo, conversion directe; sinon via bin->Hz)
    - y_mode="hz"   : affiche en Hz (utilise pitch_hz si présent, sinon approx via bin->Hz)
    - y_mode="bin"  : affiche en index de bin (peut annoter l'axe avec note_labels si fournis)

    Chaque piste est tracée comme une ligne continue entre onset et offset.
    """
    if not tracks:
        print("Aucune piste à afficher.")
        return

    # récupère les vecteurs temps par piste
    def seg_time_vector(seg):
        if times is not None:
            # on prend le sous-vecteur correspondant à la longueur de la piste
            i0, i1 = seg["onset_idx"], seg["offset_idx"]
            i1 = min(i1, len(times))
            return times[i0:i1]
        else:
            # échelle en index de frames
            n = len(seg["pitch_bins"])
            return np.arange(n)

    # helpers conversions
    def hz_to_midi(f):
        f = np.maximum(f, 1e-12)
        return 69.0 + 12.0 * np.log2(f / 440.0)

    def bins_to_hz_from_meta(seg):
        # essaie d'utiliser pitch_hz s'il existe
        if "pitch_hz" in seg and seg["pitch_hz"] is not None:
            return np.asarray(seg["pitch_hz"], dtype=float)
        # sinon approximation via fmin_hz et BPO (nécessaire si l'utilisateur n'a pas passé freqs)
        if fmin_hz is None:
            raise ValueError(
                "Pour convertir bin->Hz, passez fmin_hz (ex: librosa.note_to_hz('C0'))."
            )
        # on considère bin 0 à fmin, chaque bin = 2**(1/BPO)
        b = np.asarray(seg["pitch_bins"], dtype=float)
        return fmin_hz * np.power(2.0, b / float(bins_per_octave))

    # Prépare la figure
    plt.figure(figsize=(10, 4))
    for seg in tracks:
        tvec = seg_time_vector(seg)

        if y_mode == "bin":
            y = np.asarray(seg["pitch_bins"], dtype=float)
        elif y_mode == "hz":
            y = bins_to_hz_from_meta(seg)
        elif y_mode == "midi":
            # si on a déjà pitch_hz -> conversion MIDI ; sinon bin->Hz->MIDI
            if "pitch_hz" in seg and seg["pitch_hz"] is not None:
                y = hz_to_midi(np.asarray(seg["pitch_hz"], dtype=float))
            else:
                y = hz_to_midi(bins_to_hz_from_meta(seg))
        else:
            raise ValueError("y_mode doit être 'midi', 'hz' ou 'bin'.")

        # clamp tvec à la longueur de y (selon comment les indices ont été stockés)
        n = min(len(tvec), len(y))
        if n <= 1:
            continue
        plt.plot(tvec[:n], y[:n])

    if y_mode == "bin" and note_labels is not None and len(note_labels) > 0:
        # petites graduations lisibles (p. ex. montrer 1 label sur 3 ou 4)
        step = max(1, len(note_labels) // 24)
        idxs = np.arange(0, len(note_labels), step)
        plt.yticks(idxs, [note_labels[i] for i in idxs])

    plt.xlabel("Temps (s)" if times is not None else "Frame")
    if y_mode == "midi":
        plt.ylabel("Pitch (MIDI)")
    elif y_mode == "hz":
        plt.ylabel("Fréquence (Hz)")
    else:
        plt.ylabel("Bin CQT")

    plt.title(title)
    plt.tight_layout()


def convert_to_midi(audio_path: str, output_midi_path: str | None, debug: bool = False):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Le fichier audio '{audio_path}' est introuvable.")

    audio_data, sr = librosa.load(audio_path, mono=True)

    bpm, beat_times = estimate_bpm_from_wavelet(audio_data, sr, use_vqt=False)
    bpm = round(bpm)
    print("BPM estimé:", bpm)

    print("Calcul de la CQT...")
    mag, times, note_labels = wavelet_mag(
        audio_data, sr, use_vqt=False, bins_per_octave=48
    )

    mag[mag < 0.01] = 0.0  # seuil numérique

    print("Renforcement des fondamentales...")
    mag_fund = reinforce_fundamentals(
        mag,
        bins_per_octave=48,
        max_harm=6,
        presence_thresh=0.15,
        neigh_bins=1,
        boost_strength=0.8,
        use_amplitude=False,
        renormalize=True,
    )

    mag_fund[mag_fund < 0.4] = 0.0

    print("Nettoyage harmonique et promotion d’octaves...")
    mag_h = harmonic_clean_and_octave_promote(
        mag_fund,
        bins_per_octave=48,
        max_harm=6,
        max_octave_shift=4,
        harmonic_atten=0.35,
        tol_bins=2,
        octave_factor=2.0,
    )

    mag_h[mag_h < 0.01] = 0.0

    print("Génération des pistes de notes...")
    tracks = track_notes(
        mag_h,
        times,
        bins_per_octave=48,
        max_harm=6,
        peak_rel_thresh=0.2,
        max_jump_bins=3,
        gap_max=2,
        k_confirm=3,
        min_duration_frames=4,
        gap_join=3,
        thr_join_bins=1.5,
    )

    notes = tracks_to_notes(tracks, times, bpm=bpm)

    midi_maker(notes, bpm, output_midi_path if output_midi_path else "music.mid")

    if debug:
        plot_pianoroll(mag, times, note_labels, "piano-roll brut", threshold=0.00)
        plot_pianoroll(
            mag_fund,
            times,
            note_labels,
            "piano-roll avec renforcement des fondamentales",
            threshold=0.00,
        )
        plot_pianoroll(
            mag_h,
            times,
            note_labels,
            "piano-roll avec nettoyage harmonique",
            threshold=0.00,
        )

        plot_note_tracks(
            tracks, times=times, y_mode="midi", title="Trajectoires de notes détectées"
        )

        plt.show()


if __name__ == "__main__":
    convert_to_midi("audio_in/Gamme.mp3", "test.mid", debug=True)
