# src/simulation_call.py
#
# Entry point for a single simulation run with default parameters.
#
# Run from the project root:
#   python -m src.simulation_call
#   python src/simulation_call.py
#
# Python path note:
#   sys.path.insert() below registers additional directories to import
#   from, so all imports work regardless of which directory you launch
#   the script from.
#
# main() / __name__ guard:
#   Wrapping the whole run in main() means this file can also be imported
#   (e.g. `from simulation_call import main`) without immediately executing
#   a full simulation run -- useful for validation scripts that need to
#   trigger a run and inspect its outputs programmatically.
#   `if __name__ == "__main__":` means "only run this when the file itself
#   is executed directly, not when it's imported by something else."

import sys
from pathlib import Path
from dataclasses import replace

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
from s_WB_default import s_WB_default
from s_EV_default import s_EV_default

# =============================================================================
# Paths
# =============================================================================
PD_CSV      = _PROJECT_ROOT / "data" / "raw" / "Pd.csv"
META_JSON   = _PROJECT_ROOT / "data" / "raw" / "meta.json"
PROFILE_CSV = _PROJECT_ROOT / "data" / "raw" / "driving_profile.csv"
OUTPUT_DIR  = _PROJECT_ROOT / "data" / "processed"


def main():
    # =========================================================================
    # 1. Load surplus power time series
    # =========================================================================
    Pd, dt, dtime = load_Pd(PD_CSV, META_JSON)

    # dt is derived from Pd.csv (sampling interval in hours). The wallbox
    # parameter set must use the same time step. `replace()` returns a new
    # dataclass instance instead of mutating the shared default in place –
    # important as soon as this function runs more than once (e.g. a
    # parameter sweep), since mutating the imported default would leak
    # into every subsequent call.
    s_WB = replace(s_WB_default, dt=dt)

    # =========================================================================
    # 2. Load driving profile
    # =========================================================================
    lp = load_drive_profile(
        csv_path=PROFILE_CSV,
        dtime=dtime,
    )

    # =========================================================================
    # 3. Simulation settings
    # =========================================================================
    # Bundle WB and EV parameter sets into a single namespace.
    s = SimpleNamespace(WB=s_WB, EV=s_EV_default)

    pl     = 0     # forecast: 0 = disabled, ForecastSettings instance = enabled
    random_draws = None  # random plug-in: None = disabled, 1 = new seed, array = reuse

    # =========================================================================
    # 4. Decode inputs
    # =========================================================================
    # decode_inputs() unpacks all parameter structs and pre-computes lookup
    # tables. It returns two flat dicts (para, ts) that the @jit loop can
    # consume without needing Python objects.
    para, ts, pl_dict, random_draws = decode_inputs(Pd, s, lp, pl, random_draws)

    # =========================================================================
    # 5. Run simulation
    # =========================================================================
    results = wbsim_7_81(para, ts, pl_dict, random_draws)

    P_wb  = results["P_wb"]    # wallbox AC power [W]
    P_bat = results["P_bat"]   # battery DC power [W]
    P_obc = results["P_obc"]   # OBC loss power [W]
    E_b   = results["E_b"]     # battery state of energy [kWh]

    # =========================================================================
    # 6. Postcalculations
    # =========================================================================
    Pdp    = np.maximum(0, Pd)
    Pg2wb  = np.maximum(P_wb - Pdp, 0)
    Ppv2wb = np.minimum(Pdp, P_wb)

    # =========================================================================
    # 7. Quick summary
    # =========================================================================
    print(f"Simulation complete  –  {len(Pd):,} timesteps  ({len(Pd)/3600:.1f} h)")
    print(f"  Wallbox : Energy (total) ={np.sum(P_wb)*dt/1000:.1f} kWh Power mean={P_wb.mean():.1f} W")
    print(f"            Energy PV={np.sum(Ppv2wb)*dt/1000:.1f} kWh   Energy Grid={np.sum(Pg2wb)*dt/1000:.1f} kWh")

    # =========================================================================
    # 8. Save results
    # =========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "P_wb.npy",   P_wb)
    np.save(OUTPUT_DIR / "P_bat.npy",  P_bat)
    np.save(OUTPUT_DIR / "P_obc.npy",  P_obc)
    np.save(OUTPUT_DIR / "E_b.npy",    E_b)
    np.save(OUTPUT_DIR / "Ppv2wb.npy", Ppv2wb)
    np.save(OUTPUT_DIR / "Pg2wb.npy",  Pg2wb)
    np.save(OUTPUT_DIR / "Pdp.npy",    Pdp)
    np.save(OUTPUT_DIR / "Pd.npy",     Pd)

    results["Pd"]     = Pd
    results["Pdp"]    = Pdp
    results["Ppv2wb"] = Ppv2wb
    results["Pg2wb"]  = Pg2wb
    results["dtime"]  = dtime

    with open(OUTPUT_DIR / "sim_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"  Results saved to {OUTPUT_DIR}/")

    return results


if __name__ == "__main__":
    main()
