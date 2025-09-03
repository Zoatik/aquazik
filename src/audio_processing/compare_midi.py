#!/usr/bin/env python3
# compare_midi_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import pretty_midi as pm
from scipy.optimize import linear_sum_assignment

# ======== Logique de comparaison (reprend le script précédent) ========


def load_notes(path, programs=None, drums=False):
    midi = pm.PrettyMIDI(path)
    notes = []
    for inst in midi.instruments:
        if inst.is_drum and not drums:
            continue
        if programs is not None and inst.program not in programs:
            continue
        for n in inst.notes:
            notes.append(
                {
                    "pitch": int(n.pitch),
                    "start": float(n.start),
                    "end": float(n.end),
                    "vel": int(n.velocity),
                    "program": int(inst.program),
                    "is_drum": bool(inst.is_drum),
                }
            )
    notes.sort(key=lambda x: (x["start"], x["pitch"]))
    return notes


def make_cost_matrix(
    ref,
    hyp,
    onset_tol=0.05,
    offset_tol=0.08,
    max_pitch_diff=0,
    min_overlap=0.3,
    large=1e6,
):
    R, H = len(ref), len(hyp)
    C = np.full((R, H), large, dtype=float)
    for i, r in enumerate(ref):
        rs, re, rp = r["start"], r["end"], r["pitch"]
        rdur = max(0.0, re - rs)
        for j, h in enumerate(hyp):
            hs, he, hp = h["start"], h["end"], h["pitch"]
            hdur = max(0.0, he - hs)
            if abs(rp - hp) > max_pitch_diff:
                continue
            overlap = max(0.0, min(re, he) - max(rs, hs))
            min_dur = max(1e-6, min(rdur, hdur))
            onset_diff = abs(rs - hs)
            ok_time = (overlap >= min_overlap * min_dur) or (onset_diff <= onset_tol)
            if not ok_time:
                continue
            offset_diff = abs(re - he)
            C[i, j] = (onset_diff / max(onset_tol, 1e-9)) + (
                offset_diff / max(offset_tol, 1e-9)
            )
    return C


def match_notes(
    ref, hyp, onset_tol=0.05, offset_tol=0.08, max_pitch_diff=0, min_overlap=0.3
):
    if len(ref) == 0 or len(hyp) == 0:
        return [], list(range(len(ref))), list(range(len(hyp)))
    C = make_cost_matrix(ref, hyp, onset_tol, offset_tol, max_pitch_diff, min_overlap)
    row_ind, col_ind = linear_sum_assignment(C)
    pairs, used_r, used_h = [], set(), set()
    large = 1e6
    for i, j in zip(row_ind, col_ind):
        if C[i, j] >= large:
            continue
        r, h = ref[i], hyp[j]
        onset_err = abs(r["start"] - h["start"])
        offset_err = abs(r["end"] - h["end"])
        pitch_ok = r["pitch"] == h["pitch"]
        pairs.append((i, j, onset_err, offset_err, pitch_ok))
        used_r.add(i)
        used_h.add(j)
    unmatched_ref = [i for i in range(len(ref)) if i not in used_r]
    unmatched_hyp = [j for j in range(len(hyp)) if j not in used_h]
    return pairs, unmatched_ref, unmatched_hyp


def compute_metrics(
    ref, hyp, pairs, unmatched_ref, unmatched_hyp, onset_tol=0.05, offset_tol=0.08
):
    n_ref = len(ref)
    n_hyp = len(hyp)
    n_match = len(pairs)
    global_similarity = 100.0 * (n_match / n_ref) if n_ref > 0 else 0.0
    if n_match > 0:
        onset_ok = sum(1 for _, _, e_on, _, _ in pairs if e_on <= onset_tol)
        offset_ok = sum(1 for _, _, _, e_off, _ in pairs if e_off <= offset_tol)
        pitch_ok = sum(1 for _, _, _, _, p_ok in pairs if p_ok)
        onset_similarity = 100.0 * onset_ok / n_match
        offset_similarity = 100.0 * offset_ok / n_match
        pitch_accuracy = 100.0 * pitch_ok / n_match
        mae_onset = 1000.0 * np.mean([e_on for _, _, e_on, _, _ in pairs])  # ms
        mae_offset = 1000.0 * np.mean([e_off for _, _, _, e_off, _ in pairs])  # ms
    else:
        onset_similarity = offset_similarity = pitch_accuracy = 0.0
        mae_onset = mae_offset = float("nan")
    false_notes = len(unmatched_hyp)
    missed_notes = len(unmatched_ref)
    precision = 100.0 * (n_match / n_hyp) if n_hyp > 0 else 0.0
    f1 = 0.0
    if precision + global_similarity > 0:
        f1 = 2 * precision * global_similarity / (precision + global_similarity)

    return {
        "n_ref": n_ref,
        "n_hyp": n_hyp,
        "n_matched": n_match,
        "global_similarity_%": global_similarity,
        "precision_%": precision,
        "f1_%": f1,
        "onset_similarity_%": onset_similarity,
        "offset_similarity_%": offset_similarity,
        "pitch_accuracy_%": pitch_accuracy,
        "mean_abs_onset_error_ms": mae_onset,
        "mean_abs_offset_error_ms": mae_offset,
        "false_notes": false_notes,
        "missed_notes": missed_notes,
    }


def analyze_contextual_errors(ref, hyp, pairs, unmatched_ref, unmatched_hyp):
    """Analyse des erreurs contextuelles.
    - Notes manquées isolées vs dans un accord (côté référence)
    - Fausses notes isolées vs simultanées (côté hypothèse)
    - Substitutions: appariements avec mauvais pitch
    """
    matched_ref = {i for i, _, _, _, _ in pairs}
    matched_hyp = {j for _, j, _, _, _ in pairs}

    def overlaps_interval(a_start, a_end, b_start, b_end):
        return (a_start < b_end) and (a_end > b_start)

    missed_in_chord = 0
    missed_isolated = 0
    for i in unmatched_ref:
        r = ref[i]
        # chevauchement avec une autre note de référence (autre index)
        has_overlap = any(
            (k != i)
            and overlaps_interval(ref[k]["start"], ref[k]["end"], r["start"], r["end"])
            for k in range(len(ref))
        )
        if has_overlap:
            missed_in_chord += 1
        else:
            missed_isolated += 1

    false_simultaneous = 0
    false_isolated = 0
    for j in unmatched_hyp:
        h = hyp[j]
        has_overlap = any(
            (k != j)
            and overlaps_interval(hyp[k]["start"], hyp[k]["end"], h["start"], h["end"])
            for k in range(len(hyp))
        )
        if has_overlap:
            false_simultaneous += 1
        else:
            false_isolated += 1

    substitutions = sum(1 for _, _, _, _, p_ok in pairs if not p_ok)

    return {
        "missed_in_chord": missed_in_chord,
        "missed_isolated": missed_isolated,
        "false_simultaneous": false_simultaneous,
        "false_isolated": false_isolated,
        "substitutions": substitutions,
    }


# ======== UI Tkinter ========


class MidiCompareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Comparateur MIDI (réf vs généré)")
        self.geometry("760x560")

        self.ref_path = tk.StringVar()
        self.gen_path = tk.StringVar()

        # Menu
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(
            label="Ouvrir MIDI de référence…",
            command=self.pick_ref,
            accelerator="Ctrl+R / ⌘R",
        )
        filemenu.add_command(
            label="Ouvrir MIDI généré…",
            command=self.pick_gen,
            accelerator="Ctrl+G / ⌘G",
        )
        filemenu.add_separator()
        filemenu.add_command(
            label="Quitter", command=self.destroy, accelerator="Ctrl+Q / ⌘Q"
        )
        menubar.add_cascade(label="Fichier", menu=filemenu)
        self.config(menu=menubar)

        # Raccourcis clavier (Win/Linux Ctrl, macOS Command)
        self.bind_all("<Control-r>", lambda e: self.pick_ref())
        self.bind_all("<Control-g>", lambda e: self.pick_gen())
        self.bind_all("<Control-q>", lambda e: self.destroy())
        self.bind_all("<Command-r>", lambda e: self.pick_ref())
        self.bind_all("<Command-g>", lambda e: self.pick_gen())
        self.bind_all("<Command-q>", lambda e: self.destroy())

        # chemins
        frm = tk.Frame(self, padx=10, pady=10)
        frm.pack(fill="x")

        tk.Label(frm, text="MIDI de référence:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.ref_path, width=70, state="readonly").grid(
            row=0, column=1, padx=6
        )
        tk.Button(frm, text="Choisir...", command=self.pick_ref).grid(row=0, column=2)

        tk.Label(frm, text="MIDI généré:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.gen_path, width=70, state="readonly").grid(
            row=1, column=1, padx=6
        )
        tk.Button(frm, text="Choisir...", command=self.pick_gen).grid(row=1, column=2)

        # paramètres
        par = tk.LabelFrame(self, text="Paramètres", padx=10, pady=10)
        par.pack(fill="x", padx=10)

        self.onset_ms = tk.DoubleVar(value=50.0)
        self.offset_ms = tk.DoubleVar(value=80.0)
        self.min_overlap = tk.DoubleVar(value=0.30)
        self.max_pitch_diff = tk.IntVar(value=0)
        self.only_program = tk.StringVar(value="")  # vide = tous
        self.include_drums = tk.BooleanVar(value=False)

        row = 0
        tk.Label(par, text="Tolérance attaque (ms):").grid(
            row=row, column=0, sticky="w"
        )
        tk.Entry(par, textvariable=self.onset_ms, width=8).grid(
            row=row, column=1, sticky="w", padx=6
        )
        tk.Label(par, text="Tolérance fin (ms):").grid(
            row=row, column=2, sticky="w", padx=(16, 0)
        )
        tk.Entry(par, textvariable=self.offset_ms, width=8).grid(
            row=row, column=3, sticky="w", padx=6
        )

        row += 1
        tk.Label(par, text="Chevauchement min (0..1):").grid(
            row=row, column=0, sticky="w"
        )
        tk.Entry(par, textvariable=self.min_overlap, width=8).grid(
            row=row, column=1, sticky="w", padx=6
        )
        tk.Label(par, text="Écart pitch max (demi-tons):").grid(
            row=row, column=2, sticky="w", padx=(16, 0)
        )
        tk.Entry(par, textvariable=self.max_pitch_diff, width=8).grid(
            row=row, column=3, sticky="w", padx=6
        )

        row += 1
        tk.Label(par, text="Filtrer instrument (program 0-127, vide=tous):").grid(
            row=row, column=0, columnspan=2, sticky="w"
        )
        tk.Entry(par, textvariable=self.only_program, width=8).grid(
            row=row, column=2, sticky="w"
        )
        tk.Checkbutton(par, text="Inclure drums", variable=self.include_drums).grid(
            row=row, column=3, sticky="w"
        )

        # actions
        act = tk.Frame(self, padx=10, pady=6)
        act.pack(fill="x")
        tk.Button(act, text="Comparer", command=self.run_compare).pack(side="left")
        tk.Button(
            act, text="Exporter appariements (CSV)", command=self.export_pairs
        ).pack(side="left", padx=8)
        tk.Button(
            act, text="Enregistrer résultats (.txt)", command=self.save_results
        ).pack(side="left", padx=8)
        tk.Button(act, text="Effacer", command=self.clear_text).pack(
            side="left", padx=8
        )

        # zone résultats
        self.txt = tk.Text(self, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

        # stockage des derniers résultats pour export
        self._last_pairs = None
        self._last_ref = None
        self._last_hyp = None

    def pick_ref(self):
        p = filedialog.askopenfilename(
            title="Choisir le MIDI de référence",
            filetypes=[("MIDI files", "*.mid *.midi")],
        )
        if p:
            self.ref_path.set(p)

    def pick_gen(self):
        p = filedialog.askopenfilename(
            title="Choisir le MIDI généré", filetypes=[("MIDI files", "*.mid *.midi")]
        )
        if p:
            self.gen_path.set(p)

    def run_compare(self):
        ref = self.ref_path.get().strip()
        gen = self.gen_path.get().strip()
        if not ref:
            self.pick_ref()
            ref = self.ref_path.get().strip()
        if not gen:
            self.pick_gen()
            gen = self.gen_path.get().strip()
        if not ref or not gen:
            messagebox.showwarning(
                "Fichiers manquants", "Sélectionne les deux fichiers MIDI."
            )
            return

        try:
            programs = None
            if self.only_program.get().strip() != "":
                programs = [int(self.only_program.get().strip())]

            ref_notes = load_notes(
                ref, programs=programs, drums=self.include_drums.get()
            )
            hyp_notes = load_notes(
                gen, programs=programs, drums=self.include_drums.get()
            )

            onset_tol = float(self.onset_ms.get()) / 1000.0
            offset_tol = float(self.offset_ms.get()) / 1000.0
            pairs, unr, unp = match_notes(
                ref_notes,
                hyp_notes,
                onset_tol=onset_tol,
                offset_tol=offset_tol,
                max_pitch_diff=int(self.max_pitch_diff.get()),
                min_overlap=float(self.min_overlap.get()),
            )
            metrics = compute_metrics(
                ref_notes,
                hyp_notes,
                pairs,
                unr,
                unp,
                onset_tol=onset_tol,
                offset_tol=offset_tol,
            )

            ctx = analyze_contextual_errors(ref_notes, hyp_notes, pairs, unr, unp)

            self._last_pairs = pairs
            self._last_ref = ref_notes
            self._last_hyp = hyp_notes

            # Affichage
            if self.txt.index("end-1c") != "1.0":
                self.txt.insert("end", "\n\n")
            t = []
            t.append("---- Résultats ----")
            t.append(f"Fichier réf : {ref}")
            t.append(f"Fichier gen : {gen}")
            t.append(f"Notes réf      : {metrics['n_ref']}")
            t.append(f"Notes générées : {metrics['n_hyp']}")
            t.append(f"Appariées      : {metrics['n_matched']}")
            t.append(
                f"Similarité globale (rappel) : {metrics['global_similarity_%']:.1f}%"
            )
            t.append(f"Précision                       : {metrics['precision_%']:.1f}%")
            t.append(f"F1                              : {metrics['f1_%']:.1f}%")
            t.append(
                f"Début (attaque)                 : {metrics['onset_similarity_%']:.1f}%  (tol {self.onset_ms.get():.0f} ms)"
            )
            t.append(
                f"Fin   (relâchement)             : {metrics['offset_similarity_%']:.1f}%  (tol {self.offset_ms.get():.0f} ms)"
            )
            t.append(
                f"Justesse de note (pitch exact)  : {metrics['pitch_accuracy_%']:.1f}% (±{int(self.max_pitch_diff.get())} st)"
            )
            if np.isfinite(metrics["mean_abs_onset_error_ms"]):
                t.append(f"MAE attaque : {metrics['mean_abs_onset_error_ms']:.1f} ms")
                t.append(f"MAE fin     : {metrics['mean_abs_offset_error_ms']:.1f} ms")
            t.append(f"Fausses notes (FP) : {metrics['false_notes']}")
            t.append(f"Notes manquées (FN): {metrics['missed_notes']}")
            t.append(f"Substitutions (mauvais pitch apparié): {ctx['substitutions']}")
            t.append(f"Notes manquées isolées : {ctx['missed_isolated']}")
            t.append(f"Notes manquées dans accords : {ctx['missed_in_chord']}")
            t.append(f"Fausses notes isolées : {ctx['false_isolated']}")
            t.append(f"Fausses notes simultanées : {ctx['false_simultaneous']}")
            self._last_report_text = "\n".join(t)
            self.txt.insert("end", self._last_report_text)

        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def export_pairs(self):
        if self._last_pairs is None:
            messagebox.showinfo("Pas de données", "Lance d’abord une comparaison.")
            return
        p = filedialog.asksaveasfilename(
            title="Enregistrer le CSV des appariements",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not p:
            return
        try:
            import csv

            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "ref_idx",
                        "hyp_idx",
                        "ref_pitch",
                        "hyp_pitch",
                        "ref_start_s",
                        "hyp_start_s",
                        "onset_err_ms",
                        "ref_end_s",
                        "hyp_end_s",
                        "offset_err_ms",
                        "pitch_equal",
                    ]
                )
                for i, j, e_on, e_off, p_ok in self._last_pairs:
                    r, h = self._last_ref[i], self._last_hyp[j]
                    w.writerow(
                        [
                            i,
                            j,
                            r["pitch"],
                            h["pitch"],
                            f"{r['start']:.6f}",
                            f"{h['start']:.6f}",
                            f"{e_on*1000:.1f}",
                            f"{r['end']:.6f}",
                            f"{h['end']:.6f}",
                            f"{e_off*1000:.1f}",
                            int(p_ok),
                        ]
                    )
            messagebox.showinfo("Export", f"Appariements écrits dans:\n{p}")
        except Exception as e:
            messagebox.showerror("Erreur export", str(e))

    def save_results(self):
        content = self.txt.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo("Pas de données", "Aucun résultat à enregistrer.")
            return
        p = filedialog.asksaveasfilename(
            title="Enregistrer les résultats",
            defaultextension=".txt",
            filetypes=[("Texte", "*.txt"), ("All", "*.*")],
        )
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Export", f"Résultats enregistrés dans:\n{p}")
        except Exception as e:
            messagebox.showerror("Erreur export", str(e))

    def clear_text(self):
        self.txt.delete("1.0", "end")


if __name__ == "__main__":
    app = MidiCompareApp()
    app.mainloop()
