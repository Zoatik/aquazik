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

    fmin = Tools.note_to_midi("A0")
    fmax = Tools.note_to_midi("C8")
    n_bins = (fmax - fmin) * bins_per_octave // 12 + 1
    fmin = librosa.note_to_hz("A0")
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
            filter_scale=2.0,
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


def reinforce_fundamentals(
    mag,
    *,
    bins_per_octave=12,
    max_harm=6,
    presence_thresh=0.15,
    neigh_bins=1,
    boost_strength=0.8,
    use_amplitude=False,
    renormalize=True,
    eps=1e-12,
    presence_ref="local",
    f0_gate_rel=0.05,
    harm_weight="1/k",
    renorm_percentile=None,
    # --- Subharmonique (inchangé) ---
    enable_subharmonic=True,
    promote_thresh=0.5,
    demote_thresh=0.2,
    sub_boost_strength=0.6,
    suppress_factor=0.25,
    presence_h2_rel=0.10,
    # --- NOUVEAU : sauver f0 même si X[i]==0 s’il a assez d’harmoniques possibles ---
    min_potential_harmonics_for_zero_f0=3,  # 0 = désactivé ; 3 recommandé
):
    if mag is None or mag.size == 0:
        return mag

    X = mag
    n_bins, n_frames = X.shape

    # offsets harmoniques k=2..K
    ks, offsets = [], []
    for k in range(2, max_harm + 1):
        off = int(round(bins_per_octave * math.log2(k)))
        if off < n_bins:
            ks.append(k)
            offsets.append(off)
    if not offsets:
        return X.copy()
    ks = np.asarray(ks)
    offsets = np.asarray(offsets)

    # poids harmoniques
    if harm_weight is None:
        w_k = np.ones_like(ks, dtype=X.dtype)
    elif harm_weight == "1/k":
        w_k = 1.0 / ks
    elif harm_weight == "1/k2":
        w_k = 1.0 / (ks.astype(X.dtype) ** 2)
    else:
        raise ValueError("harm_weight doit être None, '1/k' ou '1/k2'.")

    # --- validité par bin : poids ET COMPTE ENTIER (nouveau) ---
    valid_weight_per_bin = np.zeros(n_bins, dtype=X.dtype)
    valid_count_per_bin = np.zeros(n_bins, dtype=np.int32)  # nouveau
    for off, wk in zip(offsets, w_k):
        valid_weight_per_bin[: n_bins - off] += wk
        valid_count_per_bin[: n_bins - off] += 1
    valid_weight_per_bin = np.maximum(valid_weight_per_bin, eps)

    # Seuils par frame + gate énergie
    frame_max = np.maximum(np.max(X, axis=0, keepdims=True), eps)
    f0_gate_energy = (X >= (f0_gate_rel * frame_max)).astype(X.dtype)

    # --- NOUVEAU : bypass du gate si assez d’harmoniques potentielles ---
    if min_potential_harmonics_for_zero_f0 and min_potential_harmonics_for_zero_f0 > 0:
        potential_mask = valid_count_per_bin >= int(min_potential_harmonics_for_zero_f0)
        # alt_gate = 1 pour les bins avec suffisamment de "k" possibles, sinon gate énergie
        alt_gate = np.where(potential_mask[:, None], 1.0, f0_gate_energy)
    else:
        alt_gate = f0_gate_energy

    # Référence "locale" pour le seuil de présence
    if presence_ref == "local":
        ref_pool = X.copy()
        if neigh_bins > 0:
            for d in range(1, neigh_bins + 1):
                up = np.zeros_like(X)
                up[d:, :] = X[:-d, :]
                dn = np.zeros_like(X)
                dn[:-d, :] = X[d:, :]
                ref_pool = np.maximum(ref_pool, np.maximum(up, dn))
    elif presence_ref == "frame":
        ref_pool = None
    else:
        raise ValueError("presence_ref doit être 'local' ou 'frame'.")

    # Accumulateurs
    harmonic_count_w = np.zeros_like(X)
    harmonic_power_w = np.zeros_like(X)
    aligned_cache = []

    for off, wk in zip(offsets, w_k):
        aligned = np.zeros_like(X)
        aligned[:-off, :] = X[off:, :]

        if neigh_bins > 0:
            pooled = aligned.copy()
            for d in range(1, neigh_bins + 1):
                up = np.zeros_like(aligned)
                up[d:, :] = aligned[:-d, :]
                dn = np.zeros_like(aligned)
                dn[:-d, :] = aligned[d:, :]
                pooled = np.maximum(pooled, np.maximum(up, dn))
            aligned = pooled

        if presence_ref == "local":
            ref_aligned = np.zeros_like(X)
            ref_aligned[:-off, :] = ref_pool[off:, :]
            ref_ref = np.maximum(ref_aligned, eps)
        else:
            ref_ref = frame_max

        present = aligned >= (presence_thresh * ref_ref)

        harmonic_count_w += wk * present.astype(X.dtype)
        if use_amplitude:
            harmonic_power_w += wk * aligned

        aligned_cache.append((off, wk, aligned))

    # Score "bin comme f0" (pas de gate énergie dur : on utilise alt_gate)
    if use_amplitude:
        score = (harmonic_count_w + harmonic_power_w) / valid_weight_per_bin[:, None]
    else:
        score = harmonic_count_w / valid_weight_per_bin[:, None]

    # --- appliquer le gate assoupli (autorise X[i]==0 si assez d'harmoniques potentielles) ---
    score *= alt_gate

    # Renfort de base
    boost = 1.0 + boost_strength * np.clip(score, 0.0, 1.0)

    # ===== Promotion subharmonique (H2 -> f0) =====
    promote_map = np.ones_like(X)
    suppress_map = np.ones_like(X)

    if enable_subharmonic and bins_per_octave > 0:
        oct_off = int(round(bins_per_octave))
        present_H2 = X >= (presence_h2_rel * frame_max)

        for i in range(oct_off, n_bins):
            b0 = i - oct_off
            strong_f0 = score[b0, :] >= promote_thresh
            weak_self = score[i, :] <= demote_thresh
            cond = strong_f0 & weak_self & present_H2[i, :]

            if not np.any(cond):
                continue

            promote_map[b0, cond] *= 1.0 + sub_boost_strength
            for off in offsets:
                h = b0 + off
                if h < n_bins:
                    suppress_map[h, cond] *= suppress_factor

    Y = X * boost * promote_map * suppress_map

    if renormalize:
        if renorm_percentile is None:
            denom = np.maximum(np.max(Y, axis=0, keepdims=True), eps)
        else:
            p = float(renorm_percentile)
            denom = np.percentile(Y, p, axis=0, keepdims=True)
            denom = np.maximum(denom, eps)
        Y = Y / denom

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


def segment_features(mag, segment, bins_per_octave=12, max_k=6, neigh=1):
    """Retourne un dict de features timbraux pour un segment track_notes."""
    pitch = np.clip(np.round(segment["pitch_bins"]).astype(int), 0, mag.shape[0] - 1)
    t0, t1 = segment["onset_idx"], segment["offset_idx"]
    frames = range(max(0, t0), min(mag.shape[1], t1))
    H1s, ratios, devs = [], [], []
    odd_sum, even_sum = 0.0, 0.0

    for tt in frames:
        i = pitch[min(tt - t0, len(pitch) - 1)]
        H = []
        # H1
        H1 = mag[i, tt]
        # Hk alignés (max-pooling ±neigh)
        for k in range(2, max_k + 1):
            off = int(round(bins_per_octave * math.log2(k)))
            j = i + off
            if j >= mag.shape[0]:
                H.append(0.0)
                continue
            vals = [mag[j, tt]]
            for d in range(1, neigh + 1):
                if j - d >= 0:
                    vals.append(mag[j - d, tt])
                if j + d < mag.shape[0]:
                    vals.append(mag[j + d, tt])
            H.append(max(vals))
        # collecte
        H1s.append(H1)
        if H1 > 1e-9:
            ratios.append([h / H1 for h in H])  # [H2/H1, H3/H1, ...]
        # inharmonicité: déviation moyenne du meilleur alignement
        # (approx simple: écart binaire entre j et l’indice où le max a été pris)
        # ici on saute pour garder le code court; cf. remarque ci-dessous
        # odd/even
        for idx, h in enumerate(H, start=2):
            if idx % 2:
                odd_sum += h
            else:
                even_sum += h

    ratios = np.array(ratios) if ratios else np.zeros((0, max_k - 1))
    med_ratios = np.median(ratios, axis=0) if ratios.size else np.zeros(max_k - 1)
    # pente spectrale (log)
    ks = np.arange(2, max_k + 1, dtype=float)
    y = np.log(np.maximum(med_ratios, 1e-12))
    slope = float(np.polyfit(np.log(ks), y, 1)[0]) if ratios.size else 0.0

    # dynamiques
    e = np.array(segment["energy"], dtype=float)
    e = e[(0 if t0 == segment["onset_idx"] else 0) :]  # déjà la bonne fenêtre
    if e.size:
        e_norm = e / (np.max(e) + 1e-12)
        # attaque 10→90%
        try:
            t10 = np.argmax(e_norm >= 0.1)
            t90 = np.argmax(e_norm >= 0.9)
            attack = float(max(0, t90 - t10))
        except Exception:
            attack = float("nan")
        # demi-vie (descente 1→0.5)
        post = e_norm[np.argmax(e_norm) :]
        half = np.argmax(post <= 0.5) if post.size else 0
        half_life = float(half)
    else:
        attack = half_life = 0.0

    # vibrato (sur pitch_bins)
    pb = np.array(segment["pitch_bins"], dtype=float)
    pb = pb - np.mean(pb)
    vib_rate = 0.0
    vib_depth = 0.0
    if pb.size >= 16 and np.std(pb) > 1e-6:
        ac = np.correlate(pb, pb, mode="full")[pb.size - 1 :]
        ac = ac / (ac[0] + 1e-12)
        # chercher le 1er pic secondaire
        lag = np.argmax(ac[1 : min(64, pb.size // 2)]) + 1
        vib_rate = 1.0 / lag if lag > 0 else 0.0
        vib_depth = 2.0 * np.std(pb)  # ~ crête-à-crête en bins

    return {
        "H2_H1": float(med_ratios[0]) if med_ratios.size >= 1 else 0.0,
        "H3_H1": float(med_ratios[1]) if med_ratios.size >= 2 else 0.0,
        "odd_even": (odd_sum + 1e-9) / (even_sum + 1e-9),
        "slope": slope,
        "attack_frames": attack,
        "half_life_frames": half_life,
        "vib_rate_cyc_per_frame": vib_rate,
        "vib_depth_bins": vib_depth,
        "energy_med": float(np.median(H1s) if H1s else 0.0),
        "energy_p95": float(np.percentile(H1s, 95) if H1s else 0.0),
    }


def track_notes(
    mag,
    times,
    *,
    bins_per_octave=12,
    freqs=None,  # np.array(n_bins) en Hz ; si None et librosa dispo, sera déduit
    fmin_note="A0",  # utilisé seulement si freqs=None (via librosa.note_to_hz)
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
    mags,
    *,
    # unités et conversion
    output_units="beats",  # "beats" ou "seconds"
    bpm=120.0,  # utilisé si output_units="beats"
    # nettoyage pour MIDI
    min_duration_beats=0.125,  # durée minimale (beats) après conversion
    epsilon_beats=1e-3,  # écart min entre fin et début sur même pitch
    overlap_policy="trim",  # "trim" | "shift" | "merge"
    # choix de la fréquence représentative
    use_energy_weight=True,  # fréquence moyenne pondérée par l’énergie
) -> list[Note]:
    """
    Convertit des 'tracks' (sortie de tracking) en liste de Note,
    en dédupliquant et empêchant les chevauchements par pitch.
    Assigne aussi les descripteurs timbraux à chaque Note.
    """
    if not tracks:
        return []

    # --- utilitaires locaux ---
    def _group_by(seq, key):
        groups = {}
        for x in seq:
            k = key(x)
            groups.setdefault(k, []).append(x)
        return groups.items()

    def _duration_of_seg(s):
        return max(0.0, float(s["end"] - s["start"]))

    def _combine_features(fa, fb, wa, wb):
        # moyenne pondérée par la durée (wa, wb)
        def wmean(a, b):
            return (wa * a + wb * b) / max(1e-12, (wa + wb))

        out = {}
        keys = [
            "H2_H1",
            "H3_H1",
            "odd_even",
            "slope",
            "attack_frames",
            "half_life_frames",
            "vib_rate_cyc_per_frame",
            "vib_depth_bins",
            "energy_med",
            "energy_p95",
        ]
        for k in keys:
            out[k] = wmean(fa.get(k, 0.0), fb.get(k, 0.0))
        return out

    # 1) Aplatir les segments et calculer les features par segment
    segs = []
    for seg in tracks:
        i0, i1 = int(seg["onset_idx"]), int(seg["offset_idx"])
        if i0 >= i1 or i0 < 0 or i1 > len(times):
            continue

        # temps (secondes)
        start_s = float(times[i0])
        end_s = float(times[i1]) if i1 < len(times) else float(times[-1])
        duration_s = max(0.0, end_s - start_s)
        if duration_s == 0.0:
            continue

        # fréquence instantanée de la piste (si disponible)
        if "pitch_hz" in seg and seg["pitch_hz"] is not None:
            freq_series = np.asarray(seg["pitch_hz"], dtype=float)
        else:
            bins = np.asarray(seg["pitch_bins"], dtype=float)
            if bins.size == 0:
                continue
            midi_est = 69.0 + (np.nanmean(bins))  # fallback grossier
            freq_series = np.array([440.0 * (2.0 ** ((midi_est - 69.0) / 12.0))])

        # énergie
        energy = np.asarray(seg.get("energy", []), dtype=float)
        if energy.size == 0 or energy.size != freq_series.size:
            energy = np.ones_like(freq_series)

        # fréquence représentative
        if use_energy_weight and np.sum(energy) > 0:
            freq = float(np.sum(freq_series * energy) / np.sum(energy))
        else:
            freq = float(np.mean(freq_series))

        # unités de sortie (on garde start/end en secondes puis on convertit plus tard)
        start = start_s
        end = end_s
        length = max(0.0, end - start)

        midi_number = Tools.freq_to_number(freq)
        name = Tools.note_name(midi_number)

        # === features timbraux pour ce segment ===
        feat_env = segment_features(mag=mags, segment=seg, bins_per_octave=48, max_k=6)

        segs.append(
            {
                "pitch_hz": freq,
                "midi": midi_number,
                "name": name,
                "start": start,
                "end": end,
                "length": length,
                "times": [start, end],  # placeholder; peut être enrichi
                "magnitudes": [1.0, 1.0],  # idem; on peut y mettre l’enveloppe réelle
                "features": feat_env,
            }
        )

    if not segs:
        return []

    # 2) Déduplication stricte: même (midi, start) -> garder la plus longue
    segs.sort(key=lambda s: (s["midi"], s["start"], -s["length"]))
    dedup = []
    last_key = None
    for s in segs:
        key = (s["midi"], round(s["start"], 6))
        if key != last_key:
            dedup.append(s)
            last_key = key
        # sinon, on ignore (le premier est déjà le plus long)
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
        group.sort(key=lambda s: (s["start"], s["end"]))
        current = None
        for s in group:
            if current is None:
                current = s
                continue

            if s["start"] < current["end"] - 1e-9:
                if overlap_policy == "trim":
                    # couper la note courante
                    current["end"] = max(current["start"], s["start"] - epsilon)
                    current["length"] = max(0.0, current["end"] - current["start"])
                    if current["length"] < min_len:
                        current = s
                        continue
                elif overlap_policy == "shift":
                    # décaler la suivante
                    shift = (current["end"] + epsilon) - s["start"]
                    s["start"] += max(0.0, shift)
                    s["length"] = max(0.0, s["end"] - s["start"])
                    if s["length"] < min_len:
                        continue
                elif overlap_policy == "merge":
                    # fusion + fusion des features (pondération par durée)
                    wa = _duration_of_seg(current)
                    wb = _duration_of_seg(s)
                    f_merged = _combine_features(
                        current["features"], s["features"], wa, wb
                    )
                    current["end"] = max(current["end"], s["end"])
                    current["length"] = max(0.0, current["end"] - current["start"])
                    current["features"] = f_merged
                    continue
                else:
                    raise ValueError(
                        "overlap_policy doit être 'trim', 'shift' ou 'merge'."
                    )
            else:
                # pas de chevauchement; marge si collés
                if abs(s["start"] - current["end"]) < epsilon:
                    if overlap_policy in ("trim", "merge"):
                        current["end"] = current["end"] - 0.5 * epsilon
                        current["length"] = max(0.0, current["end"] - current["start"])
                        s["start"] = s["start"] + 0.5 * epsilon
                        s["length"] = max(0.0, s["end"] - s["start"])
                    else:  # shift
                        s["start"] = current["end"] + epsilon
                        s["length"] = max(0.0, s["end"] - s["start"])

            # valider la courante si assez longue
            if current["length"] >= min_len:
                notes_clean.append(current)
            current = s

        if current and current["length"] >= min_len:
            notes_clean.append(current)

    # 4) Construire les objets Note + assigner les features timbraux
    notes_clean.sort(key=lambda s: (s["start"], s["midi"]))
    notes_out = []
    for s in notes_clean:
        note = Note(
            frequency=s["pitch_hz"],
            magnitudes=s["magnitudes"],
            times=s["times"],
            start=s["start"],
            length=s["length"],
        )
        # conversion beats si demandé
        note.length_bpm = Tools.seconds_to_beat(note.length, bpm)
        note.start_bpm = Tools.seconds_to_beat(note.start_time, bpm)

        # === assignation des features ===
        f = s.get("features", {}) or {}
        note.h2_h1 = float(f.get("H2_H1", 0.0))
        note.h3_h1 = float(f.get("H3_H1", 0.0))
        note.odd_even = float(f.get("odd_even", 0.0))
        note.slope = float(f.get("slope", 0.0))
        note.attack = float(f.get("attack_frames", 0.0))
        note.half_life = float(f.get("half_life_frames", 0.0))
        note.vib_rate = float(f.get("vib_rate_cyc_per_frame", 0.0))
        note.vib_depth = float(f.get("vib_depth_bins", 0.0))
        note.energy_median = float(f.get("energy_med", 0.0))
        note.energy_p95 = float(f.get("energy_p95", 0.0))

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
    min_secondary_harmonics=2,
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

    def harmonic_count(col, f0_bin, fmax):
        """Compte brut (#) d'harmoniques >= rel_thresh * fmax autour de f0_bin."""
        cnt = 0
        used = 0
        for _, off_k in harm_off:
            j = f0_bin + off_k
            if j >= n_bins:
                break
            a = local_max_aligned(col, j)
            if a >= rel_thresh * fmax:
                cnt += 1
            used += 1
        return cnt, used

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

                # Déterminer si p est bien une harmonique de best_f0 et laquelle (offset utilisé)
                is_harm = False
                used_off = None
                if best_f0 is not None and best_support > 0.2:
                    delta = p - best_f0
                    for _, off_k in harm_off:
                        if abs(delta - off_k) <= tol_bins:
                            is_harm = True
                            used_off = off_k
                            break

                # --- Règle spéciale "octave" (1ʳᵉ harmonique) ---
                # Si p ≈ best_f0 + bins_per_octave (± tol_bins) et que l'octave n'est pas assez forte,
                # on SUPPRIME toute la fenêtre autour de l’octave.
                if is_harm and abs(used_off - bins_per_octave) <= tol_bins:
                    # Fenêtres locales robustes
                    j0_f0 = max(0, best_f0 - tol_bins)
                    j1_f0 = min(n_bins, best_f0 + tol_bins + 1)

                    oct_ix = int(round(best_f0 + bins_per_octave))
                    j0_oct = max(0, oct_ix - tol_bins)
                    j1_oct = min(n_bins, oct_ix + tol_bins + 1)

                    f0_energy = col[j0_f0:j1_f0].max() if j1_f0 > j0_f0 else 0.0
                    oct_energy = col[j0_oct:j1_oct].max() if j1_oct > j0_oct else 0.0

                    if (
                        f0_energy > 0.0
                        and oct_energy < octave_factor * f0_energy
                        and j1_oct > j0_oct
                    ):
                        removed = float(col[j0_oct:j1_oct].sum())
                        col[j0_oct:j1_oct] = 0.0
                        if energy_conserve and removed > 0.0 and j1_f0 > j0_f0:
                            col[j0_f0:j1_f0] += removed / (j1_f0 - j0_f0)
                        continue  # octave traitée → pic suivant

                # --- Règle "plus sévère" pour TOUTE harmonique ---
                # On considère l’harmonique p comme une fondamentale hypothétique :
                # si elle n’a pas au moins `min_secondary_harmonics` harmoniques au-dessus d’elle,
                # on la SUPPRIME (± tol_bins).
                if is_harm and min_secondary_harmonics > 0:
                    cnt_p, used = harmonic_count(
                        col, p, fmax
                    )  # compte p+off_k (p elle-même non comptée)
                    if cnt_p < min_secondary_harmonics:
                        j0_p = max(0, p - tol_bins)
                        j1_p = min(n_bins, p + tol_bins + 1)
                        removed = float(col[j0_p:j1_p].sum())
                        col[j0_p:j1_p] = 0.0
                        if energy_conserve and removed > 0.0 and best_f0 is not None:
                            j0_f0 = max(0, best_f0 - tol_bins)
                            j1_f0 = min(n_bins, best_f0 + tol_bins + 1)
                            if j1_f0 > j0_f0:
                                col[j0_f0:j1_f0] += removed / (j1_f0 - j0_f0)
                        continue  # harmonique supprimée → pic suivant

                # --- Cas général: atténuation douce des harmoniques expliquées par une f0 plausible ---
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

    mag[mag < 0.05] = 0.0  # seuil numérique

    print("Nettoyage harmonique et promotion d’octaves...")
    mag_h = harmonic_clean_and_octave_promote(
        mag,
        bins_per_octave=48,
        max_harm=6,
        max_octave_shift=4,
        harmonic_atten=0.35,
        tol_bins=4,
        octave_factor=2.0,
        promote_octaves=True,
    )

    mag_h[mag_h < 0.01] = 0.0

    """
    print("Renforcement des fondamentales...")
    mag_fund = reinforce_fundamentals(
        mag,
        bins_per_octave=48,
        max_harm=6,
        presence_ref="local",
        presence_thresh=0.12,
        neigh_bins=1,
        harm_weight="1/k2",
        use_amplitude=True,
        f0_gate_rel=0.02,
        renorm_percentile=98,
        enable_subharmonic=True,
        promote_thresh=0.35,
        demote_thresh=0.22,
        presence_h2_rel=0.06,
        sub_boost_strength=0.8,
        suppress_factor=0.01,
        min_potential_harmonics_for_zero_f0=2,
    )"""

    print("Renforcement des fondamentales...")
    mag_fund = reinforce_fundamentals(
        mag_h,
        bins_per_octave=48,
        max_harm=6,
        presence_thresh=0.15,
        neigh_bins=1,
        boost_strength=0.8,
        use_amplitude=False,
        renormalize=True,
        enable_subharmonic=False,
    )

    mag_fund[mag_fund < 0.4] = 0.0

    print("Génération des pistes de notes...")
    tracks = track_notes(
        mag_fund,
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

    notes = tracks_to_notes(tracks, times, mag, bpm=bpm)

    for note in notes:
        note.print_features()

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
