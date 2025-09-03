from typing import Dict, List, Union
from matplotlib import pyplot as plt
import numpy as np
from audio_processing.audio_utils import Tools, Instrument, Note
from audio_processing.baba import convert_to_midi


def notes_to_features(notes: list[Note], *, drop_nans: bool = True) -> dict[str, list[float]]:
    keys = ("attack", "slope", "h2_h1", "h3_h1", "odd_even")
    feats: Dict[str, List[float]] = {k: [] for k in keys}

    for n in notes:
        vals = {
            "attack":   n.attack,
            "slope":    n.slope,
            "h2_h1":    n.h2_h1,
            "h3_h1":    n.h3_h1,
            "odd_even": n.odd_even,
        }
        for k, v in vals.items():
            if drop_nans and (v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))):
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

def get_features(audio_path):
    bpm, notes = convert_to_midi(audio_path, None)
    features = notes_to_features(notes)

    features_gaussed = {}
    for name, vals in features.items():
        features_gaussed[name] = feature_to_gaussian(vals)

    print(features_gaussed)

    
def main():
    get_features("")


if __name__ == "__main__":
    main()