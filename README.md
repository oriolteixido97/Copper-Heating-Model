# Estimation of incident power by transient heating of a copper plate

This project combines **experimental data acquisition** and **analytical modeling** to estimate the incident radiant flux from a **tungsten filament lamp** by observing the **transient heating** of a small copper plate.

---

## Overview
A 1×1 cm copper plate (1 mm thick) was irradiated by a tungsten lamp.  
Temperature was measured with a **Type-T thermocouple** connected to a **CR1000X datalogger**.  
From the measured transient, a **lumped-capacity model** was applied to extract:
- absorbed power \( P_\text{abs} \),
- total heat transfer coefficient \( h_\text{tot} \),
- time constant \( \tau \),
- and ambient drift parameters.

---

## Physical model
The plate is treated as a single thermal node obeying:

\[
C\frac{dT}{dt} = P_\text{abs} - h_\text{tot}A_s(T - T_\infty),
\]

whose analytical solution, including slow ambient drift, was fitted to the data using **SciPy’s `curve_fit`**.  
The model achieves **R² > 0.999**, reproducing the experimental curve with residuals below 0.3 °C.

---

## Results
| Parameter | Symbol | Value |
|------------|---------|--------|
| Absorbed power | \( P_\text{abs} \) | 0.292 W |
| Heat transfer coefficient | \( h_\text{tot} \) | 28 W/m²K |
| Time constant | \( \tau \) | 123 s |
| Absorbed flux | \( q''_\text{abs} \) | 2.9 × 10³ W/m² |
| Incident flux (α = 0.03) | \( q''_\text{incident} \) | 9.7 × 10⁴ W/m² |

The fitted model matches the experimental temperature evolution almost perfectly, validating the lumped-capacity assumption and the extracted parameters.

---

## Repository structure
├── main.tex # LaTeX report (Overleaf-ready)
├── Estimation_of_incident_power_by_transient_heating.pdf
├── python_fit_copper_heating.py # Analysis & fitting code
├── copper_temperature.txt # Experimental data
├── Curbe copper heating.png # Resulting plot
└── README.md # This file


---

## Tools and methods
- **Python 3** (`numpy`, `matplotlib`, `scipy.optimize.curve_fit`)
- **LaTeX (Overleaf)** for report generation
- **Thermocouple T-type** + **CR1000X datalogger**
- **Tungsten filament lamp** as radiant source
- **Lumped-parameter thermal modeling**

---

## Highlights
- Demonstrates the use of an **analytical differential model** to interpret experimental transients.  
- Shows **parameter estimation** and uncertainty interpretation from real measurements.  
- Integrates **data acquisition**, **numerical fitting**, and **scientific reporting** in one coherent workflow.

---

*Author:* [Oriol Teixidó]  
*Created:* 2025  
*License:* MIT (open for academic use)
