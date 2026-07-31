# src/graph.py
#
# Loads the output of a simulation run (simulation_call.py) and plots it.
#
# Run from the project root, after simulation_call.py has produced
# data/processed/sim_results.pkl:
#   python -m src.graph
#   python src/graph.py
#
#%%


import sys
from pathlib import Path

# Find this file's own folder (src/) first -- always correct regardless
# of install-path weirdness, since utils.py (imported right below) is a
# sibling file. See utils.py's find_project_root() docstring for the
# harder problem this doesn't solve (locating data/raw, data/processed).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import find_project_root

# --- Locate project root and register module search paths ---
_PROJECT_ROOT = find_project_root()
print(f"[graph.py] project root: {_PROJECT_ROOT}")
sys.path.insert(0, str(_PROJECT_ROOT / "data" / "processed"))

import pickle

try:
    import plotly.io as pio
    pio.renderers.default = "vscode"
    from plotly_resampler import FigureWidgetResampler
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError(
        "graph.py needs the plotting extras, which are intentionally NOT "
        "part of requirements.txt (they're a heavy stack -- plotly, "
        "plotly-resampler, Jupyter widgets -- only needed if you actually "
        "want to plot). Install them with:\n"
        "    pip install -r requirements-plotting.txt\n"
        "or, if using pyproject.toml: pip install \".[plotting]\""
    ) from e

#%% Load Data
with open(_PROJECT_ROOT / "data/processed/sim_results.pkl", "rb") as f: #"rb"=read binary
    results=pickle.load(f)
    
P_wb  = results["P_wb"]    # wallbox AC power [W]
P_bat = results["P_bat"]   # battery DC power [W]
P_obc = results["P_obc"]   # OBC loss power [W]
E_b   = results["E_b"]     # battery state of energy [kWh]
Pd   = results["Pd"]     # Surplus power [W]
Pdp   = results["Pdp"]     # Positive Surplus power [W]
Ppv2wb   = results["Ppv2wb"]     # PV to wallbox powerflow [W]
Pg2wb   = results["Pg2wb"]     # Grid to wallbox powerflow [W]
dtime   = results["dtime"]     # npy-Datetime [ns]


#%% Example Plot

dtime_list = dtime.astype('datetime64[us]').tolist()

fig = FigureWidgetResampler(go.Figure(),default_n_shown_samples=5000)
fig.add_trace(go.Scattergl(name="P_wb", mode="lines"), hf_x=dtime_list, hf_y=P_wb)
fig.add_trace(go.Scattergl(name="P_dp", mode="lines"), hf_x=dtime_list, hf_y=Pdp)
fig.update_layout(xaxis_title="Time", yaxis_title="Power in W", height=500)
fig
