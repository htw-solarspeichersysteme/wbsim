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

# --- Locate project root and register module search paths ---
# _PROJECT_ROOT is the wb_simulation/ folder (parent of src/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "data" / "processed"))

import pickle
import plotly.io as pio
pio.renderers.default = "vscode"
from plotly_resampler import FigureWidgetResampler
import plotly.graph_objects as go

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
