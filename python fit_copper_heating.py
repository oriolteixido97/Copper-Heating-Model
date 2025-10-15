#!/usr/bin/env python3
"""
Transient heating fit for a copper plate (lumped-capacity model).

- Reads time/temperature from a TXT/CSV (two columns: t [s], T [°C])
- Cleans trailing/blank rows, non-finite values, and non-increasing time
- Fits either:
    (a) no ambient drift, or
    (b) linear ambient drift: T_inf(t) = T_inf0 + beta * t
- Reports P_abs, h_tot, tau, q_abs, and (optionally) q_incident if alpha is given
- Plots data and fitted model (axes in °C)

Usage:
    python fit_copper_heating.py --file "path/to/your/data.txt" --alpha 0.03 --drift
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------------------
# Defaults (can be overridden via CLI args)
# -----------------------------
DEFAULT_FILE = Path(__file__).parent / "data" / "copper_temperature.txt"
DEFAULT_ALPHA = 0.03          # set None if unknown; e.g., 0.95 for matte black paint
DEFAULT_USE_DRIFT = True      # model T_inf(t) = T_inf0 + beta * t
DEFAULT_DROP_LAST = True      # drop last row (often empty/NaN in exported logs)

# Geometry: 1 cm x 1 cm x 1 mm; bottom insulated -> A_s = A
A_cm = 1.0
thickness_mm = 1.0
bottom_insulated = True

# Copper properties (typical)
rho = 8960.0                  # kg/m^3
cp  = 385.0                   # J/(kg·K)

# Initial guess for h_tot (convective + radiative)
h_guess = 8.0                 # W/m²K

# -----------------------------
# Derived geometry and capacity
# -----------------------------
A = (A_cm * 1e-2) ** 2                          # m²
V = (A_cm * 1e-2) ** 2 * (thickness_mm * 1e-3)  # m³
m = rho * V                                     # kg
C = m * cp                                      # J/K
A_s = A if bottom_insulated else 2.0 * A        # m²


def load_time_temperature(path: Path, drop_last: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Load two-column data: time [s], temperature [°C].
    - Accepts whitespace or tab separators, with optional header or lines starting with '#'
    - Cleans trailing/blank rows, non-finite values
    - Sorts by time, removes non-increasing timestamps, and re-centers t -> t - t0
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Try a tolerant load (handles spaces/tabs); then retry skipping a header if needed.
    def _try_load(skip_header: int = 0):
        return np.genfromtxt(
            path, comments="#", dtype=float, autostrip=True, skip_header=skip_header
        )

    data = _try_load(0)
    if data.ndim == 1 or (data.ndim == 2 and data.shape[1] < 2):
        data = _try_load(1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Input file must have at least 2 columns: time(s), temperature(°C).")

    if drop_last and data.shape[0] >= 1:
        data = data[:-1, :]

    t = np.asarray(data[:, 0], dtype=float)
    T = np.asarray(data[:, 1], dtype=float)

    # Filter non-finite values
    mask = np.isfinite(t) & np.isfinite(T)
    t, T = t[mask], T[mask]
    if t.size < 2:
        raise ValueError("Too few valid samples after filtering NaNs/Infs.")

    # Sort by time and keep strictly increasing times
    order = np.argsort(t)
    t, T = t[order], T[order]
    keep = [0]
    for i in range(1, len(t)):
        if t[i] > t[keep[-1]]:
            keep.append(i)
    t, T = t[keep], T[keep]

    # Re-center time to start at 0
    t = t - t[0]
    return t, T


# ---------- Lumped-capacity analytical models ----------
def model_no_drift(t, P_abs, h_tot, T_inf0, T0):
    """No ambient drift: T_inf(t) = T_inf0 (constant)."""
    tau = C / (h_tot * A_s)
    return T_inf0 + (T0 - T_inf0 - P_abs/(h_tot*A_s)) * np.exp(-t / tau) + P_abs/(h_tot*A_s)

def model_with_drift(t, P_abs, h_tot, Tinf0, beta, T0):
    """Linear ambient drift: T_inf(t) = Tinf0 + beta * t."""
    tau = C / (h_tot * A_s)
    theta0 = T0 - Tinf0
    return (Tinf0 + beta*t
            + theta0 * np.exp(-t / tau)
            + (P_abs/C - beta) * tau * (1.0 - np.exp(-t / tau)))


# ---------- Initial guesses and dynamic bounds ----------
def initial_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Robust initial slope using a small window near t=0."""
    if len(t) < 3:
        return (y[-1] - y[0]) / max(1e-9, (t[-1] - t[0]))
    n = min(max(5, int(0.05 * len(t))), len(t) - 1)
    return (y[n] - y[0]) / (t[n] - t[0])

def clamp(x, lo, hi):
    return min(max(x, lo + 1e-12), hi - 1e-12)


def main():
    parser = argparse.ArgumentParser(description="Fit transient heating of a copper plate.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE,
                        help="Path to data file (two columns: time[s], temperature[°C]).")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Absorptivity (None or negative to skip incident flux).")
    parser.add_argument("--drift", action="store_true", default=DEFAULT_USE_DRIFT,
                        help="Enable ambient drift model T_inf(t) = T_inf0 + beta*t.")
    parser.add_argument("--no-drift", dest="drift", action="store_false",
                        help="Disable ambient drift model.")
    parser.add_argument("--drop-last", action="store_true", default=DEFAULT_DROP_LAST,
                        help="Drop last row of input (useful if file has a blank trailing line).")
    parser.add_argument("--save-plot", type=Path, default=None,
                        help="Path to save the plot (e.g., report/figures/curve_copper_heating.png).")
    args = parser.parse_args()

    # Load and prepare data (input T in °C; model uses K internally)
    t, T_C = load_time_temperature(args.file, drop_last=args.drop_last)
    T_K = T_C + 273.15
    T0_K = float(T_K[0])

    # Useful ranges (in K) for dynamic bounds
    TminK, TmaxK = float(np.min(T_K)), float(np.max(T_K))
    Trange = max(5.0, TmaxK - TminK)
    Tpad_lo = 0.5 * Trange
    Tpad_hi = 0.5 * Trange

    # Initial guesses
    dTdt0 = initial_slope(t, T_K)
    P_guess = max(1e-9, C * dTdt0)        # W
    Tinf0_guess = float(np.median(T_K[-max(3, len(T_K)//10):]))
    beta_guess = 0.0
    T0_guess = T0_K

    # Dynamic bounds
    P_min, P_max = 0.0, 1e7
    h_min, h_max = 0.05, 200.0
    Tinf_min, Tinf_max = TminK - Tpad_lo, TmaxK + Tpad_hi
    T0_min, T0_max = T0_K - 5.0, T0_K + 5.0
    beta_min, beta_max = -2.0/60.0, 2.0/60.0   # ±2 K/min

    if args.drift:
        p0 = [
            clamp(P_guess, P_min, P_max),
            clamp(h_guess, h_min, h_max),
            clamp(Tinf0_guess, Tinf_min, Tinf_max),
            clamp(beta_guess, beta_min, beta_max),
            clamp(T0_guess, T0_min, T0_max),
        ]
        bounds = (
            [P_min, h_min, Tinf_min, beta_min, T0_min],
            [P_max, h_max, Tinf_max, beta_max, T0_max],
        )
        popt, pcov = curve_fit(model_with_drift, t, T_K, p0=p0, bounds=bounds, maxfev=40000)
        P_fit, h_fit, Tinf0_fit, beta_fit, T0_fit = popt
        T_fit_K = model_with_drift(t, *popt)
    else:
        p0 = [
            clamp(P_guess, P_min, P_max),
            clamp(h_guess, h_min, h_max),
            clamp(Tinf0_guess, Tinf_min, Tinf_max),
            clamp(T0_guess, T0_min, T0_max),
        ]
        bounds = (
            [P_min, h_min, Tinf_min, T0_min],
            [P_max, h_max, Tinf_max, T0_max],
        )
        popt, pcov = curve_fit(model_no_drift, t, T_K, p0=p0, bounds=bounds, maxfev=40000)
        P_fit, h_fit, Tinf0_fit, T0_fit = popt
        beta_fit = 0.0
        T_fit_K = model_no_drift(t, *popt)

    perr = np.sqrt(np.diag(pcov))
    tau_fit = C / (h_fit * A_s)
    q_abs = P_fit / A
    alpha = args.alpha if (args.alpha is not None and args.alpha > 0) else None
    q_incident = (q_abs / alpha) if alpha else None

    def pm(v, e): return f"{v:.4g} ± {e:.2g}"

    print("\n===== FIT RESULTS (input & plots in °C) =====")
    print(f"File: {args.file}")
    print(f"Samples used: {len(t)}")
    print(f"A = {A:.3e} m², A_s = {A_s:.3e} m² (bottom_insulated={bottom_insulated})")
    print(f"V = {V:.3e} m³, m = {m:.3e} kg, C = {C:.3g} J/K")
    if args.drift:
        labels = ["P_abs [W]", "h_tot [W/m²K]", "T_inf,0 [K]", "beta [K/s]", "T0 [K]"]
        for lab, v, e in zip(labels, [P_fit, h_fit, Tinf0_fit, beta_fit, T0_fit], perr):
            print(f"{lab}: {pm(v, e)}")
    else:
        labels = ["P_abs [W]", "h_tot [W/m²K]", "T_inf [K]", "T0 [K]"]
        for lab, v, e in zip(labels, [P_fit, h_fit, Tinf0_fit, T0_fit], perr):
            print(f"{lab}: {pm(v, e)}")

    print(f"tau = C/(h_tot*A_s) = {tau_fit:.3g} s")
    print(f"q_abs = P_abs/A = {q_abs:.4g} W/m²")
    if q_incident is not None:
        print(f"q_incident (alpha={alpha}) = {q_incident:.4g} W/m²")
    else:
        print("q_incident: set --alpha to estimate incident flux (q_abs/alpha).")

    # -------- Plot (always in °C for axes) --------
    T_fit_C = T_fit_K - 273.15
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(t, T_C, 'o', ms=4, label="Data (°C)")
    plt.plot(t, T_fit_C, '-', lw=2, label="Fit (model)")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.title("Copper plate heating — lumped fit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.save_plot:
        out = Path(args.save_plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"\nSaved plot to: {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
