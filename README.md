# wbsim – Open python simulation model for EV charger efficiency analysis

This repository contains the simulation model **wbsim** for the performance
evaluation of residential EV charging stations (wallboxes). It is a Python
port of the original MATLAB model, developed as part of the
research project *Wallbox-Inspektion*, and was the basis for the following
studies:

- How Much Does the Sun Power Your EV? Simulation Study on Wallbox Efficiency
  https://doi.org/10.52825/pv-symposium.v2i.2650
- Performance indicator for residential solar-optimised EV chargers
  https://doi.org/10.1049/icp.2025.4140
- [Wallbox-Inspektion 2025](https://solar.htw-berlin.de/studien/wallbox-inspektion-2025/)
- Benchmarking Solar-Optimized Electric Vehicle Charging Systems: An Open and Reproducible Evaluation Method (in publication)

The model is fully parametrizable to simulate different wallboxes under
identical framework conditions. This Python export make the model openly
accessible and reproducible.

---

## 1. Model Overview

### 1.1 Description

The model represents a 1-second time-series simulation of an EV charger and
a generic, lossless EV. The simulation enables numerous parameters to be
varied and compared under otherwise identical framework conditions. This
approach trades off some generality for the highest achievable accuracy.

The input time series include PV surplus power at 1-second resolution for one year. .

Synthetic mobility profiles from **SynPro** by Fraunhofer ISE were used. 
Mobility profiles include arrival/departure times and vehicle
consumption in kWh. The example profile used here represents an EV with ~9,619 km
per year and a 70 kWh battery, driven by a city-living employee.

The EV is modeled as a simplified, lossless battery storage system with
ideal control behavior. This simplification is deliberate: it isolates the
influence of the charger itself. The model is nonetheless prepared to
integrate EV losses (on-board-charger efficiency, own consumption) if
needed for future studies.

The EV is assumed to be connected to the charger whenever it is at home. For
each charging session, a required state of charge (SOC) at departure is
specified to guarantee mobility needs. By default the charger operates in a
surplus-oriented ("solar") mode, adapting to available PV generation. Only
if the target SOC would otherwise not be reached does the charger switch to
a fallback mode, charging at full power at the latest feasible point in
time — plus a safety margin reflecting imperfect forecasts. The model is prepared
for more sophisticated plug-in behavior following the literature.

Actual charging power is determined in three sequential steps:

1. **Status evaluation** — waiting, standby, or charging.
2. **Set-point derivation** — discretizing the (dead-time-delayed) surplus
   power and incorporating steady-state control deviations.
3. **Control translation** — mapping the set-point to the physical device
   behavior, accounting for settling times, transient dynamics (PT1
   response or rate-limited ramping), and further discretization.

The result is a discrete EV-charging power profile reproducing both the
power deviations and the dynamic behavior observed in laboratory
measurements.

### 1.2 Simulation flow

```
1. Load input time series (Pd, driving profile)
2. Decode / pre-compute parameters (decode_inputs)
3. Enter 1-second time-series loop, for one simulated year:
       check status        (at home / waiting / charging)
       compute set-point    (surplus power, discretised, control deviation)
       compute charging power (settling time, dynamics, rate limiting)
4. Post-process the final charging event so that start and end
   state-of-charge match (energy balance correction)
5. Evaluate energy flows
```

### 1.3 Model assumptions

- The system charges in solar (surplus-oriented) mode as long as possible.
  It switches to maximum-power charging whenever the remaining time before
  departure would otherwise be insufficient.
  *To simulate pure maximum-power or time-dependent charging, the surplus
  power input can be manipulated accordingly.*
- Whenever the EV's battery is not sufficiently charged for its next trip,
  it is assumed to be externally recharged and returns home at 25% SOC.
- Forecasts of driving behavior are assumed to be perfect, but a safety
  margin (a fraction of the EV's battery capacity) is always added to
  reflect real-world forecast uncertainty.
- Plug-in behavior defaults to "plug in on arrival".
- The EV is lossless by default. To model charging losses, a quadratic
  loss function derived from measurements can be supplied (see e.g.
  https://doi.org/10.1016/j.seta.2023.103512).

---

## 2. Project Structure

```
wbsim/
│
├── data/
│   ├── raw/                       # Input data
│   │   ├── Pd.csv*                # Surplus power (compact or classic format), *: download  [10.5281/zenodo.21672007](https://doi.org/10.5281/zenodo.21672007)
│   │   ├── meta.json*             # Metadata for Pd.csv (compact format) *: download  [10.5281/zenodo.21672007](https://doi.org/10.5281/zenodo.21672007)
│   │   └── driving_profile.csv    # Driving profile (human-readable timestamps)
│   │
│   ├── config/                    # Parameter sets
│   │   ├── s_WB_default.py        # Default wallbox parameters
│   │   └── s_EV_default.py        # Default EV parameters
│   │
│   └── processed/                 # Simulation outputs
│
├── src/
│   ├── __init__.py
│   ├── dataclass_definitions.py   # Wallbox / EV / ForecastSettings / DriveProfile dataclasses
│   ├── utils.py                   # load_Pd, load_drive_profile, idx2timeseries, decode_inputs, find_project_root
│   ├── wbsim.py                   # wbsim_7_81() – main simulation function
│   ├── simulation_call.py         # Example entry point / script to run a simulation
│   └── graph.py                   # Loads simulation output (.pkl) and plots results
│
├── requirements.txt
├── requirements-plotting.txt      # optional: only for graph.py
├── pyproject.toml
└── README.md                      # This file
```

---

## 3. Requirements

- **Python** ≥ 3.10

There are two ways to install dependencies, depending on what you want:

**A) Reproduce the manuscript's results exactly** — use the pinned
`requirements.txt`. This is the recommended path if you just want to run
the simulation as published. It installs only the core dependencies —
plotting (`graph.py`) pulls in a noticeably heavier stack (plotly,
plotly-resampler, Jupyter widgets) and is kept separate so installation
stays fast if you don't need it; see [4.3](#43-visualizing-results) for
the extra step if you do.

```text
# core – required to run the simulation
numpy==2.4.4
pandas==3.0.2
numba==0.64.0
```

```bash
pip install -r requirements.txt
```

Versions are pinned exactly for reproducibility. numba/numpy are tightly
coupled -- if you need a different numpy version, check numba's
compatibility notes before changing the pin.

**B) Use wbsim as an installable package** — e.g. to `import wbsim`-style
modules from another project. `pyproject.toml` declares the same
dependencies as minimum-version floors (not exact pins), so installing
wbsim doesn't force version conflicts in an existing environment.

```bash
pip install .                  # core only
pip install ".[plotting]"      # + graph.py's plotting stack
pip install ".[validation]"    # + scipy, for the planned MATLAB-comparison workflow
```

> Avoid `pip install -e .` (editable install) for running the simulation
> itself — editable installs of this flat py-modules layout can, on some
> setuptools versions, misplace where files are read from/written to
> (see `utils.find_project_root()`). Use `-e` only when importing wbsim's
> modules from another project you're developing against.

Nevertheless you have to download the input data externally from  [10.5281/zenodo.21672007](https://doi.org/10.5281/zenodo.21672007) as github allows only small file sizes.

---

## 4. Quickstart

```bash
git clone <repository-url>
cd wbsim
# download input data
wget -O data/raw/meta.json "https://zenodo.org/records/21672008/files/meta.json?download=1"
wget -O data/raw/Pd.csv   "https://zenodo.org/records/21672008/files/Pd.csv?download=1"

pip install -r requirements.txt
python -m src.simulation_call
```

This runs a simulation using the default wallbox/EV parameter sets
(`data/config/s_WB_default.py`, `data/config/s_EV_default.py`) against the
bundled example input data (`data/raw/`), and stores the results in
`data/processed/`.

### 4.1 Input data

#### Surplus power (`Pd.csv` + `meta.json`)

**Option A – compact (recommended for large datasets):**
`Pd.csv` contains only power values (`Pd_W`, one per line, loaded as `float64`).
`meta.json` reconstructs the timestamps:

```json
{
  "start_timestamp": 1388534399,
  "sampling_interval": 1,
  "unit_power": "W",
  "encoding": "implicit_index"
}
```

**Option B – classic (timestamps included in the CSV):**
A column `Time` (Unix seconds) or any column containing "time" (parsed as a
human-readable datetime), plus a `Pd_W` column.

#### Driving profile (`driving_profile.csv`)

Human-readable timestamps plus energy consumption per trip:

| Column      | Type     | Description                    |
|-------------|----------|---------------------------------|
| `departure` | datetime | Departure time                  |
| `arrival`   | datetime | Arrival time                    |
| `SOE`       | float    | Energy consumed on the trip [kWh] |

> **Timestamp consistency:** `Pd.csv` and `driving_profile.csv` must share
> the same time base (e.g. both UTC), and `start_timestamp` in `meta.json`
> must correspond to the first timestamp in `Pd.csv`.

### 4.2 Running a custom simulation

```python
from utils import load_Pd, load_drive_profile, decode_inputs
from wbsim import wbsim_7_81
from dataclass_definitions import Wallbox, EV
from types import SimpleNamespace

# 1. Load surplus power and time axis
Pd, dt, dtime = load_Pd("data/raw/Pd.csv", "data/raw/meta.json")

# 2. Define wallbox and EV parameters (dt must match load_Pd's dt, in hours)
s_WB = Wallbox(P_WB2EV_min=1400, P_WB2EV_max=11000, n_phase_min=1,
                n_phase_max=3, I_step=1, t_hold_off_min=5,
                t_stop_phaseswitch=3, t_in_deep_standby=10,
                t_DEAD=5, t_SETTLING=10, dt=dt)

s_EV = EV(E_BAT=70, eta_BAT=100, min_soe=30,
           P_WB2EV_min_in=1400, P_WB2EV_max_in=11000, Pobc_max=11000)

# 3. Load driving profile (EV parameters are supplied later, in decode_inputs)
lp = load_drive_profile(
    csv_path="data/raw/driving_profile.csv",
    dtime=dtime,
)

# 4. Decode inputs (pre-compute lookup tables, unpack parameter structs)
s = SimpleNamespace(WB=s_WB, EV=s_EV)
para, ts, pl_dict, random_draws = decode_inputs(Pd, s, lp, pl=0, random_draws=None)

# 5. Run simulation
results = wbsim_7_81(para, ts, pl_dict, random_draws)

# 6. Save results
import numpy as np
np.save("data/processed/P_wb.npy", results["P_wb"])
np.save("data/processed/E_b.npy", results["E_b"])
import pickle
with open("data/processed/sim_results.pkl", "wb") as f: 
    pickle.dump(results, f)
```

> **Parameter studies:** `load_Pd()` and `load_drive_profile()` only need
> to run once. `decode_inputs()` must be re-run whenever wallbox or EV
> parameters change, but not when only re-running the same configuration.
> See `simulation_call.py` for a complete example.

### 4.3 Visualizing results

`graph.py` needs the plotting extras, which aren't in `requirements.txt`
(see [Requirements](#3-requirements)) — install them once:

```bash
pip install -r requirements-plotting.txt
```

```bash
python -m src.graph
```
The skript expects a "data/processed/sim_results.pkl" from simulation_call
and could be easily customized. If the plotting extras aren't installed,
it fails with a clear message telling you to run the command above.


---

## 5. Output Data

`simulation_call.py` saves both individual `.npy` files and a combined
`data/processed/sim_results.pkl` (used by `graph.py`). The pickle contains
all of the keys below.

| Key       | Description                                 | Unit |
|-----------|----------------------------------------------|------|
| `P_wb`    | Wallbox AC charging power                     | W    |
| `P_bat`   | Battery DC power                              | W    |
| `P_obc`   | On-board-charger loss power                   | W    |
| `E_b`     | EV battery state of energy                    | kWh  |
| `E_b_ext` | Externally charged energy (safety fallback)   | kWh  |
| `Pd`      | Surplus power input (dead-time shifted)       | W    |
| `Pdp`     | Positive part of the surplus power (`max(Pd, 0)`) | W |
| `Ppv2wb`  | PV-to-wallbox power flow                      | W    |
| `Pg2wb`   | Grid-to-wallbox power flow                    | W    |
| `dtime`   | Simulation time axis                          | `datetime64[ns]` |

---

## 6. Parameter Reference

### 6.1 Example parameter set (near-ideal wallbox)

**Power-related parameters**

| Parameter                       | Value      |
|----------------------------------|------------|
| Discretised current steps        | 0.01 A     |
| Minimum power                    | 1.4 kW     |
| Maximum power                    | 11 kW      |
| Control deviation power          | 0 W        |
| Current limiter, positive        | 3× 16 A/s  |
| Current limiter, negative        | 3× −16 A/s |
| Standby (plugged)                | 0 W        |
| Standby (unplugged)              | 0 W        |
| Deep standby (unplugged)         | 0 W        |
| Periphery                        | 0 W        |

**Time-related parameters**

| Parameter                              | Value |
|------------------------------------------|-------|
| Control delay time                        | 0 s   |
| Set time (time resolution)                | 1 s   |
| Phase-switch hold time                    | 0 s   |
| Phase-switch off time                     | 0 s   |
| Hold power-off time                       | 0 s   |
| Hold power-on time                        | 0 s   |
| Hold deep-standby time (unplugged)        | 0 s   |

An example EV charger could be found under:
```
wbsim/
├── data/
│   ├── config/                    # Parameter sets
│   │   ├── s_WB_default.py        # Default wallbox parameters
```
### 6.2 Determining charger parameters from measurements

Assuming a measurement has been conducted according to the guideline
[*Prüfrichtlinie Unidirektionales und Solares Laden*](https://wallbox-inspektion.de/publikationen/251207_WB-Test_solar_V1-1.pdf)
(eng. "Measurement guideline for unidirectional and solar charging"),
parameters can be derived as follows:

**General approach**
- Calculating parameters directly from time series has proven useful.
- **Power-related parameters:** use single values or linear functions to
  describe deviation across the current range (1–16 A). For discretised
  current steps, classify the number of distinct states and revise
  manually. For current limiting, a scatter plot of (ΔI, Δt) helps; a
  linear fit (`a·x + b`) gives a first estimate, where `b` is the limit
  achievable within a single jump and `a` the steady-state current ramp.
  This parameter is difficult to detect automatically and is often
  revised manually. Minimum/maximum power, periphery, and standby are
  straightforward, but outliers must be excluded.
- **Time-related parameters:** determine the timestep of the set-point
  change, the timestep of the first reaction, and the steady state
  (calculated backward from the end of a sequence to the second
  occurrence, to exclude overshoot behaviour).
- Some parameters require manual revision. We recommend simulating the
  measured application test time series and comparing simulation to
  measurement directly.

---

## 7. Data Availability

The driving profile is used with permission from Fraunhofer ISE, extracted
from the [SynPro tool](https://synpro-lastprofile.de/). 

The [suplus power](https://zenodo.org/records/21672008) is derived from [PerMod](https://solar.htw-berlin.de/permod/)
based on the Oldenburg weather data [Kalisch et al 2014](https://doi.pangaea.de/10.1594/PANGAEA.847830) 
and the HTW Berlin load profiles [Tjaden et al 2015](https://doi.org/10.13140/RG.2.1.3713.1606/1) 
[Data download](https://solar.htw-berlin.de/elektrische-lastprofile-fuer-wohngebaeude/)

---

## 8. Development Notes & Provenance

The simulation model, its underlying methodology, and all parameter sets
were developed in MATLAB by Joseph Bergner and Nico Orth as part of the
*Wallbox-Inspektion* research project. The Python implementation in this
repository is a translation of that MATLAB model (`wbsim_7_81.m`), carried
out with the assistance of an AI coding assistant (Claude, Anthropic). All
translated code was reviewed, and its numerical output was validated
against the original MATLAB reference implementation by the authors to
confirm equivalence (see Section 9, once available).

## 9. License

This repository is licensed under the [MIT License](https://opensource.org/license/mit/).
See the `LICENSE` file for the full text.
