import math
from typing import Dict, List, Union
from matplotlib import pyplot as plt
import numpy as np
from audio_processing.audio_utils import Tools, Instrument, Note
from audio_processing.baba import convert_to_midi
from pathlib import Path
import json
import os


def notes_to_features(
    notes: list[Note], *, drop_nans: bool = True
) -> dict[str, list[float]]:
    keys = ("attack", "slope", "h2_h1", "h3_h1", "odd_even")
    feats: Dict[str, List[float]] = {k: [] for k in keys}

    for n in notes:
        vals = {
            "attack": n.attack,
            "slope": n.slope,
            "h2_h1": n.h2_h1,
            "h3_h1": n.h3_h1,
            "odd_even": n.odd_even,
        }
        for k, v in vals.items():
            if drop_nans and (
                v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
            ):
                continue
            feats[k].append(float(v))
    return feats


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
        "std": float(np.std(arr, ddof=1)),  # écart-type corrigé (n-1)
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
    v = sigma**2
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


def _to_python_scalar(x):
    """Assure que les valeurs sont sérialisables JSON (float natifs)."""
    try:
        # numpy scalars -> float
        import numpy as np

        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
    except Exception:
        pass
    # bool, int, float natifs ok
    return float(x) if isinstance(x, (int, float)) else x


def analyse_features(
    audio_path,
    instrument: Instrument,
    *,
    json_path: str | os.PathLike = "instruments_features.json",
    mode: str = "replace",  # "replace" ou "append"
):
    """
    Analyse un fichier mono-instrument et écrit les features dans un JSON.

    - mode="replace": remplace les features existants de cet instrument.
    - mode="append" : ajoute une entrée à la liste de cet instrument (historique).
    """
    bpm, notes = convert_to_midi(audio_path, None)
    features = notes_to_features(notes)

    # Gaussian stats pour chaque feature
    features_gaussed = {
        name: feature_to_gaussian(vals) for name, vals in features.items()
    }

    # Convertit proprement en scalaires JSON
    for name, d in features_gaussed.items():
        features_gaussed[name] = {k: _to_python_scalar(v) for k, v in d.items()}

    # Clé d'instrument lisible (Enum -> nom ou valeur)
    instr_key = instrument.name if hasattr(instrument, "name") else str(instrument)

    # Charger l'existant si présent
    json_path = Path(json_path)

    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                db = json.load(f)
        except Exception:
            db = {}
    else:
        db = {}

    # S'assurer d'une structure dict en racine
    if not isinstance(db, dict):
        db = {}

    # Appliquer la politique
    if mode == "replace":
        # Remplace les features de l'instrument
        db[instr_key] = features_gaussed
    elif mode == "append":
        # Empile sous forme de liste
        if instr_key not in db or not isinstance(db[instr_key], list):
            db[instr_key] = []
        db[instr_key].append(features_gaussed)
    else:
        raise ValueError("mode doit être 'replace' ou 'append'")

    # Sauvegarde
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    # Optionnel: impression console
    print(f"Écrit les features de '{instr_key}' dans {json_path.resolve()}")
    return features_gaussed


def _load_instrument_models(json_path: str) -> dict:
    """Charge et normalise la structure du JSON vers {instr: {feature: {mean, std, median, iqr}}}.
    Gère le mode 'append' (prend la dernière entrée pour chaque instrument)."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Fichier JSON introuvable: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    models = {}
    if not isinstance(db, dict):
        return models

    for instr, payload in db.items():
        if isinstance(payload, list) and payload:  # mode append: liste d'entrées
            feat_dict = payload[-1]  # prend la plus récente
        elif isinstance(payload, dict):  # mode replace
            feat_dict = payload
        else:
            continue
        models[instr] = feat_dict
    return models


def _extract_mu_sigma(feat_stats: dict) -> tuple[float, float]:
    """Récupère (mu, sigma) avec fallback robuste si besoin."""
    mu = feat_stats.get("mean")
    sd = feat_stats.get("std")
    # Fallback robuste si NaN/None/0
    if sd in (None, 0) or not np.isfinite(sd):
        med = feat_stats.get("median", np.nan)
        iqr = feat_stats.get("iqr", np.nan)
        if np.isfinite(iqr) and iqr > 0:
            sd = float(iqr) / 1.349
            mu = (
                float(med)
                if np.isfinite(med)
                else float(mu if np.isfinite(mu) else 0.0)
            )
        else:
            sd = 1.0  # ultime garde-fou
            mu = float(mu if np.isfinite(mu) else 0.0)
    else:
        mu = float(mu if np.isfinite(mu) else 0.0)
        sd = float(sd)
    # epsilon pour éviter divisions par 0
    sd = float(max(sd, 1e-9))
    return mu, sd


def _gauss_logpdf(x: float, mu: float, sd: float) -> float:
    v = sd * sd
    return -0.5 * math.log(2.0 * math.pi * v) - 0.5 * ((x - mu) ** 2) / v


def assign_instrument(
    notes: list[Note],
    *,
    json_path: str = "instruments_features.json",
    features_used: list[str] | None = None,
    instrument_labels: (
        dict[str, int] | None
    ) = None,  # map "PIANO" -> Instrument.PIANO.value, etc.
) -> list[Note]:
    """
    Assigne un instrument à chaque Note via vote de vraisemblance gaussienne par feature.

    - json_path : fichier JSON créé par `analyse_features`.
    - features_used : sous-ensemble de features à utiliser (par défaut: celles dispo dans le JSON).
    - instrument_labels : mapping nom -> code canal/enum value; si None utilise Instrument.<name>.value.
    """
    # Charger modèles
    models_raw = _load_instrument_models(
        json_path
    )  # { "PIANO": {"attack": {...}, ...}, "TRUMPET": {...}}
    if not models_raw:
        # Rien à comparer → on ne modifie pas les instruments existants
        return notes

    # Liste des features
    if features_used is None:
        # intersection: celles présentes dans TOUS les instruments
        common = None
        for feat_dict in models_raw.values():
            keys = set(k for k, v in feat_dict.items() if isinstance(v, dict))
            common = keys if common is None else (common & keys)
        features_used = sorted(common) if common else []

    # Construire (mu, sigma) par instrument/feature
    models = {}
    for instr_name, feat_dict in models_raw.items():
        models[instr_name] = {}
        for feat in features_used:
            stats = feat_dict.get(feat, {})
            mu, sd = _extract_mu_sigma(stats)
            models[instr_name][feat] = (mu, sd)

    # Préparer mapping nom -> valeur Instrument
    if instrument_labels is None:
        instrument_labels = {}
        for instr_name in models.keys():
            try:
                # Essaie de retrouver l'Enum Instrument par nom
                enum_val = getattr(Instrument, instr_name).value
            except Exception:
                # fallback: 0 (piano) si inconnu
                enum_val = Instrument.PIANO.value if hasattr(Instrument, "PIANO") else 0
            instrument_labels[instr_name] = enum_val

    # Fonction pour obtenir la valeur de feature d'une Note
    def _get_feature_value(note: Note, feat: str) -> float | None:
        v = getattr(note, feat, None)
        if v is None:
            return None
        try:
            vf = float(v)
            if np.isnan(vf) or np.isinf(vf):
                return None
            return vf
        except Exception:
            return None

    # Parcours des notes : vote + tie-break par log-likelihood total
    for note in notes:
        votes = {instr: 0 for instr in models.keys()}
        total_ll = {instr: 0.0 for instr in models.keys()}

        for feat in features_used:
            x = _get_feature_value(note, feat)
            if x is None:
                continue

            # log-pdf par instrument
            lls = {}
            for instr, params in models.items():
                mu, sd = params[feat]
                ll = _gauss_logpdf(x, mu, sd)
                lls[instr] = ll
                total_ll[instr] += ll

            # vote : instrument avec la plus grande log-vraisemblance
            best_instr = max(lls.items(), key=lambda kv: kv[1])[0]
            votes[best_instr] += 1

        # Choix final : le plus de votes, puis ll total
        if votes:
            max_votes = max(votes.values())
            cands = [instr for instr, c in votes.items() if c == max_votes]
            if len(cands) == 1:
                chosen = cands[0]
            else:
                # tie-break: somme des log-likelihoods
                chosen = max(cands, key=lambda instr: total_ll[instr])
        else:
            # si aucune feature exploitable, garder l'instrument actuel
            continue

        # Assigner sur la Note (canal/valeur)
        note.instrument = instrument_labels.get(chosen, note.instrument)

    return notes


def main():
    analyse_features("")


if __name__ == "__main__":
    main()
