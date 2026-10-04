# src/metrics.py
#
# Performance indicator (PI) of a solar-optimised EV charger, Eqs. (1)-(3)
# of the manuscript.
#
# Public API:
#   CostParameters          – grid price and feed-in tariff
#   simulate_ideal()        – run the ideal (lossless) reference simulation
#   pi_from_series()        – pure PI calculation from two power time series
#   performance_indicator() – ideal run (optional) + costs + PI in one call
#
# Reusing the ideal reference:
#   With a lossless EV (default: all OBC loss coefficients zero), the ideal
#   run depends only on Pd, the driving profile, the EV parameters, the
#   forecast settings and the random draws -- not on the wallbox parameters.
#   It can therefore be computed once and passed as P_wb_ideal to evaluate
#   any number of wallbox parameter sets. With OBC losses this no longer
#   holds, because the panic-charge power depends on P_WB2EV_max.

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from utils import decode_inputs
from wbsim import wbsim_7_81


@dataclass
class CostParameters:
    c_g2ac: float = 0.32   # grid consumption price [EUR/kWh]
    c_pv2g: float = 0.08   # feed-in tariff [EUR/kWh]


def simulate_ideal(Pd, s, lp, pl=0, random_draws=None):
    """
    Run the ideal reference simulation for parameter set s.

    A copy of s.WB with ideal=1 is used; s itself is not modified.
    random_draws must be the array returned by the real run, so that both
    runs see identical plug-in behaviour.

    Returns
    -------
    P_wb_ideal : np.ndarray, wallbox AC power of the ideal system [W]
    """
    s_ideal = SimpleNamespace(WB=replace(s.WB, ideal=1), EV=s.EV)
    para, ts, pl_dict, rd = decode_inputs(Pd, s_ideal, lp, pl, random_draws)
    return wbsim_7_81(para, ts, pl_dict, rd)["P_wb"]


def pi_from_series(P_wb_real, P_wb_ideal, Pd, E_EV_kWh, dt, cost=None):
    """
    Pure PI calculation according to Eqs. (1)-(3).

    Parameters
    ----------
    P_wb_real  : np.ndarray, wallbox AC power of the real system [W]
    P_wb_ideal : np.ndarray, wallbox AC power of the ideal system [W]
    Pd         : np.ndarray, surplus power [W]
    E_EV_kWh   : float, total EV energy demand in the simulation window [kWh]
    dt         : float, time step [h]
    cost       : CostParameters or None (None -> default prices)

    Returns
    -------
    dict with PI [-] and all cost terms [EUR]
    """
    cost = cost or CostParameters()
    Pdp = np.maximum(0, Pd)

    def terms(P_wb):
        # Feed-in revenue and grid cost of one run [EUR]
        C_pv2g = np.sum(np.maximum(0, Pdp - P_wb)) * dt / 1000 * cost.c_pv2g
        C_g2wb = np.sum(P_wb - np.minimum(Pdp, P_wb)) * dt / 1000 * cost.c_g2ac
        return C_pv2g, C_g2wb

    C_pv2g_ideal, C_g2wb_ideal = terms(P_wb_ideal)
    C_pv2g_real,  C_g2wb_real  = terms(P_wb_real)

    # Reference: total EV energy demand fully supplied from the grid
    C_g2wb_all = E_EV_kWh * cost.c_g2ac

    # Additional feed-in revenue of the real system, Eq. (2)
    dC_pv2g = C_pv2g_real - C_pv2g_ideal

    # Eq. (3)
    PI = (C_g2wb_all - C_g2wb_real + dC_pv2g) / (C_g2wb_all - C_g2wb_ideal)

    return dict(
        PI=PI,
        C_g2wb_all=C_g2wb_all,
        C_g2wb_ideal=C_g2wb_ideal, C_g2wb_real=C_g2wb_real,
        C_pv2g_ideal=C_pv2g_ideal, C_pv2g_real=C_pv2g_real,
        dC_pv2g=dC_pv2g,
    )


def performance_indicator(P_wb_real, Pd, s, lp, pl=0, random_draws=None,
                          cost=None, P_wb_ideal=None):
    """
    Ideal reference run (unless P_wb_ideal is given), costs and PI.

    Parameters
    ----------
    P_wb_real    : np.ndarray, wallbox AC power of the real system [W]
    Pd           : np.ndarray, surplus power [W]
    s            : namespace with .WB and .EV (parameter set of the real run)
    lp           : dict, output of load_drive_profile()
    pl           : forecast settings, as passed to decode_inputs()
    random_draws : array returned by the real run's decode_inputs(), or None
    cost         : CostParameters or None (None -> default prices)
    P_wb_ideal   : np.ndarray or None; if given, the ideal run is skipped

    Returns
    -------
    dict with PI [-], all cost terms [EUR], E_EV_kWh and P_wb_ideal [W]
    """
    if P_wb_ideal is None:
        P_wb_ideal = simulate_ideal(Pd, s, lp, pl, random_draws)

    E_EV_kWh = float(np.sum(lp["E_drive"]))
    out = pi_from_series(P_wb_real, P_wb_ideal, Pd, E_EV_kWh,
                         float(s.WB.dt), cost)
    out["E_EV_kWh"]   = E_EV_kWh
    out["P_wb_ideal"] = P_wb_ideal
    return out
