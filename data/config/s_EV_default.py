# data/config/s_EV_default.py
# Default EV parameter set (lossless / ideal, as described in the manuscript).
# Imports EV from dataclass_definitions.py – no local class definition.

from dataclass_definitions import EV

s_EV_default = EV(
    # --- Battery ---
    E_BAT=70.0,
    eta_BAT=100.0,    # ideal: no loss
    min_soe=30.0,     # 30% → 21 kWh absolute

    # --- Charging power limits [W] ---
    P_WB2EV_min_in=6*230,
    P_WB2EV_max_in=16*230*3,
    Pobc_max=16*230*3,

    # --- OBC loss coefficients: all zero (ideal, lossless) ---
    WB2EV_a_in=0.0,
    WB2EV_b_in=0.0,
    WB2EV_c_in=0.0,
)