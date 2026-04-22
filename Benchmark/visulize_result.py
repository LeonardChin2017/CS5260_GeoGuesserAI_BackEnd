import json
from math import nan, isnan
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from eval_points import RESULTS_FILE


def save_distr_img(data: list[int | float], name: str, n_bins: int = 100, log10: bool = False) -> None:
    # Histogram
    plt.figure()
    if log10:
        _, bins = np.histogram(data, bins=n_bins)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.hist(data, bins=logbins)
        plt.xscale('log')
    else:
        plt.hist(data, bins=n_bins)
    plt.xlabel(name)
    plt.ylabel("Count")
    plt.savefig(Path("Benchmark", f"{"Log10 " if log10 else ''}{name.lower().replace(' ', '_')}.png"))


if __name__ == "__main__":
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(RESULTS_FILE)
    with open(RESULTS_FILE, "r") as f:
        all_results: dict[str, dict[str, object]] = json.load(f)
    distance_kms: list[float] = []
    for i, result in all_results.items():
        final_guess: Optional[dict[str, object]] = result.get("final_guess")
        if final_guess is None:
            print(f"[{i}] has no final_guess, skipping")
            continue
        distance_km: float = final_guess.get("distance_km", nan)
        if isnan(distance_km):
            print(f"[{i}][final_guess] has no distance_km, skipping")
            continue
        distance_kms.append(distance_km)
    save_distr_img(distance_kms, "Distance (km)")
    save_distr_img(distance_kms, "Distance (km)", log10=True)

    iters: list[int] = []
    for i, result in all_results.items():
        iterations_used: int = result.get("iterations_used", -1)
        if iterations_used < 0:
            print(f"[{i}] has no iterations_used, skipping")
            continue
        iters.append(iterations_used)
    save_distr_img(iters, "Iterations Used", n_bins=5)