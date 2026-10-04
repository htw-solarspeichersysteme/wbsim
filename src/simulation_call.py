# src/simulation_call.py
#
# Entry point for a single simulation run with default parameters,
# including the performance indicator (PI).
#
# Run from the project root:
#   python -m src.simulation_call
#   python src/simulation_call.py
#
# Python path note:
#   sys.path.insert() below is the Python equivalent of MATLAB's addpath():
#   it tells Python where to look for modules so all imports work regardless
#   of which directory you launch the script from.
#%% =============================================================================
# 0. Imports
# =============================================================================
import sys
from pathlib import Path

# --- Locate project root and register module search paths ---
# _PROJECT_ROOT is the wb_simulation/ folder (parent of src/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "data" / "config"))

# --- Module imports (all absolute after path setup above) ---
import numpy as np
from types import SimpleNamespace
import pickle
from utils import load_Pd, load_drive_profile, decode_inputs
from wbsim import wbsim_7_81
from metrics import performance_indicator, CostParameters
from s_WB_default import s_WB_default
from s_EV_default import s_EV_default

# =============================================================================
# 1. Paths
# =============================================================================
PD_CSV       = _PROJECT_ROOT / "data" / "raw" / "Pd.csv"
META_JSON    = _PROJECT_ROOT / "data" / "raw" / "meta.json"
PROFILE_CSV  = _PROJECT_ROOT / "data" / "raw" / "driving_profile.csv"
OUTPUT_DIR   = _PROJECT_ROOT / "data" / "processed"

# =============================================================================
# 2. Load surplus power time series
# =============================================================================
Pd, dt, dtime = load_Pd(PD_CSV, META_JSON)

# dt is derived from Pd.csv (sampling interval in hours).
# The wallbox parameter set must use the same time step.
s_WB_default.dt = dt

# =============================================================================
# 3. Load driving profile
# =============================================================================
lp = load_drive_profile(
    csv_path=PROFILE_CSV,
    dtime=dtime,
)

# =============================================================================
# 4. Simulation settings
# =============================================================================
# Bundle WB and EV parameter sets into a single namespace (corresponds to
# struct s with fields s.WB and s.EV in MATLAB).
s = SimpleNamespace(WB=s_WB_default, EV=s_EV_default)

pl           = 0     # forecast: 0 = disabled, ForecastSettings instance = enabled
random_draws = None  # random plug-in: None = disabled, 1 = new seed, array = reuse
cost         = CostParameters(c_g2ac=0.32, c_pv2g=0.08)   # [EUR/kWh]

# =============================================================================
# 5. Decode inputs
# =============================================================================
# decode_inputs() unpacks all parameter structs and pre-computes lookup tables.
# It returns two flat dicts (para, ts) that the @jit loop can consume without
# needing Python objects, which keeps the simulation loop fast.
para, ts, pl_dict, random_draws_arr = decode_inputs(Pd, s, lp, pl, random_draws)

# %% =============================================================================
# 6. Run simulation (real parameter set)
# =============================================================================
results = wbsim_7_81(para, ts, pl_dict, random_draws_arr)

P_wb  = results["P_wb"]    # wallbox AC power [W]
P_bat = results["P_bat"]   # battery DC power [W]
P_obc = results["P_obc"]   # OBC loss power [W]
E_b   = results["E_b"]     # battery state of energy [kWh]

# =============================================================================
# 7. Performance indicator (runs the ideal reference simulation internally)
# =============================================================================
# The random draws of the real run are passed on, so that the ideal run sees
# identical plug-in behaviour. To evaluate many wallbox parameter sets against
# the same ideal reference, pass P_wb_ideal=pi["P_wb_ideal"] in later calls.
pi = performance_indicator(P_wb, Pd, s, lp, pl, random_draws_arr, cost)
P_wb_ideal = pi["P_wb_ideal"]   # wallbox AC power, ideal [W]

# =============================================================================
# 8. Postcalculations: energy flows
# =============================================================================
# Note: MATLAB max(0, x) acts element-wise -> np.maximum (NOT np.max, which
# would reduce the whole array to one scalar).
Pdp    = np.maximum(0, Pd)            # positive surplus only [W]
Pg2wb  = np.maximum(P_wb - Pdp, 0)    # grid -> wallbox       [W]
Ppv2wb = np.minimum(Pdp, P_wb)        # PV   -> wallbox       [W]

# =============================================================================
# 8b. Quick summary
# =============================================================================
print(f"Simulation complete  –  {len(Pd):,} timesteps  ({len(Pd)/3600:.1f} h)")
print(f"  Wallbox : Energy (total) ={np.sum(P_wb)*dt/1000:.1f} kWh Power mean={P_wb.mean():.1f} W")
print(f"            Energy PV={np.sum(Ppv2wb)*dt/1000:.1f} kWh   Energy Grid={np.sum(Pg2wb)*dt/1000:.1f} kWh")
print(f"  Ideal   : Energy (total) ={np.sum(P_wb_ideal)*dt/1000:.1f} kWh")
print(f"  EV demand (reference)    ={pi['E_EV_kWh']:.1f} kWh  ->  C_g2wb_all = {pi['C_g2wb_all']:.2f} EUR")
print(f"  Grid cost   ideal / real = {pi['C_g2wb_ideal']:.2f} / {pi['C_g2wb_real']:.2f} EUR")
print(f"  Feed-in rev ideal / real = {pi['C_pv2g_ideal']:.2f} / {pi['C_pv2g_real']:.2f} EUR")
print(f"  PI = {pi['PI']*100:.2f} %")

#%% =============================================================================
# 9. Save results (optional)
# =============================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_DIR / "P_wb.npy",       P_wb)
np.save(OUTPUT_DIR / "P_wb_ideal.npy", P_wb_ideal)
np.save(OUTPUT_DIR / "P_bat.npy",      P_bat)
np.save(OUTPUT_DIR / "P_obc.npy",      P_obc)
np.save(OUTPUT_DIR / "E_b.npy",        E_b)
np.save(OUTPUT_DIR / "Ppv2wb.npy",     Ppv2wb)
np.save(OUTPUT_DIR / "Pg2wb.npy",      Pg2wb)
np.save(OUTPUT_DIR / "Pdp.npy",        Pdp)
np.save(OUTPUT_DIR / "Pd.npy",         Pd)

results["P_wb_ideal"] = P_wb_ideal
results["PI"]         = pi["PI"]
results["costs"]      = {k: v for k, v in pi.items() if k != "P_wb_ideal"}
results["Pd"]         = Pd
results["Pdp"]        = Pdp
results["Ppv2wb"]     = Ppv2wb
results["Pg2wb"]      = Pg2wb
results["dtime"]      = dtime

with open(OUTPUT_DIR / "sim_results.pkl", "wb") as f:
    pickle.dump(results, f)
print(f"  Results saved to {OUTPUT_DIR}/")
