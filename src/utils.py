# src/utils.py
#
# All I/O and pre-processing helpers for wbsim_7_81 in one file.
#
# Public API:
#   load_Pd()             – load surplus-power time series from CSV + meta.json
#   load_drive_profile()  – load driving profile CSV → simulation time series
#   idx2timeseries()      – expand sparse per-event values into a full time series
#   decode_inputs()       – unpack Wallbox/EV structs, build lookup tables,
#                           pre-compute all arrays needed by the @jit loop

import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# =============================================================================
# find_project_root
#
# Used by simulation_call.py and graph.py to locate data/raw, data/config
# and data/processed -- centralised here since both entry-point scripts
# need the identical logic.
#
# Both scripts still need one line of their own before this is reachable:
#   sys.path.insert(0, str(Path(__file__).resolve().parent))
# That finds *this* file (utils.py, a sibling in src/) reliably -- it's
# always exactly "wherever simulation_call.py/graph.py itself is", so it
# doesn't have the ambiguity problem find_project_root() below is for.
# Only once utils.py is importable can find_project_root() be called for
# the (harder) question of where the data/ folders are.
# =============================================================================
def find_project_root() -> Path:
    """
    Determine the project root (the folder containing src/ and data/).

    Resolution order:
      1. WBSIM_PROJECT_ROOT environment variable, if set (an explicit,
         set-once override -- the standard way to configure this kind
         of path in Python/Unix tooling, rather than an interactive
         prompt, since callers also need to run non-interactively,
         e.g. imported from a notebook or a batch job).
      2. The current working directory -- the documented way to run
         simulation_call.py/graph.py is
         `cd <project root> && python -m src.simulation_call`, so cwd
         should already be correct for anyone following the Quickstart.

    Either candidate is validated (must contain a src/ and a data/raw/
    subdirectory) before being trusted, so a wrong guess fails loudly
    and immediately with a concrete next step, instead of silently
    reading/writing files in the wrong place.

    Why not derive it from __file__ (e.g. `.parent.parent` in the
    calling script)? That assumes the calling script always lives
    exactly two levels below the project root -- true for a cloned
    repo, but not guaranteed after `pip install -e .`: depending on
    the setuptools version, an editable install of a flat py-modules
    layout (see pyproject.toml) can copy the .py files into the
    virtualenv's site-packages instead of linking back to the repo,
    silently redirecting every read/write. cwd-based resolution with
    validation sidesteps this regardless of how/where the calling
    script physically ended up.

    Why not search upward for a marker file like pyproject.toml? That
    filename isn't guaranteed unique -- if this project is ever nested
    inside another repo, the search could stop at the wrong ancestor
    and again point somewhere unintended, just as silently. Validating
    the concrete expected layout is more specific than trusting a
    filename to only ever appear at the real root.
    """
    env_override = os.environ.get("WBSIM_PROJECT_ROOT")
    if env_override:
        candidate = Path(env_override).resolve()
        if not _looks_like_project_root(candidate):
            raise RuntimeError(
                f"WBSIM_PROJECT_ROOT={candidate} does not look like the "
                "project root (expected a src/ and a data/raw/ "
                "subdirectory there)."
            )
        return candidate

    candidate = Path.cwd()
    if _looks_like_project_root(candidate):
        return candidate

    raise RuntimeError(
        f"Could not confirm the project root from the current working "
        f"directory ({candidate}) -- expected a src/ and a data/raw/ "
        "subdirectory there.\n"
        "Run this script from the project root "
        "(e.g. `cd <project root> && python -m src.simulation_call`), "
        "or set the WBSIM_PROJECT_ROOT environment variable explicitly."
    )


def _looks_like_project_root(p: Path) -> bool:
    """True if `p` has the expected project layout (src/, data/raw/)."""
    return (p / "src").is_dir() and (p / "data" / "raw").is_dir()


# =============================================================================
# load_Pd
# =============================================================================
def load_Pd(csv_path: str, meta_path: Optional[str] = None):
    """
    Load a surplus-power time series from CSV with optional meta.json.

    Supported formats
    -----------------
    With meta.json (encoding field):
      "implicit_index" – CSV contains only power values (one per line, no header).
                         Timestamps are reconstructed from start_timestamp and
                         sampling_interval in meta.json.
      "delta"          – CSV has two columns [dt_seconds, Pd_W]; timestamps are
                         reconstructed from start_timestamp + cumulative dt.

    Without meta.json (classic CSV):
      Unix timestamps  – columns "Time" (int64 POSIX seconds) and "Pd_W".
      Human-readable   – any column whose name contains "time" (case-insensitive)
                         parsed by pandas, plus "Pd_W".

    Parameters
    ----------
    csv_path  : path to Pd.csv
    meta_path : path to meta.json (optional)

    Returns
    -------
    Pd    : np.ndarray float64, surplus power [W]
    dt    : float, time step [hours]
    dtime : np.ndarray datetime64[ns], time axis
    """
    csv_path  = Path(csv_path)
    meta_path = Path(meta_path) if meta_path else None

    # ------------------------------------------------------------------
    # 1. Load meta.json if available
    # ------------------------------------------------------------------
    meta = None
    if meta_path and meta_path.exists():
        try:
            text = meta_path.read_text(encoding="utf-8").lstrip("\ufeff")  # strip BOM
            meta = json.loads(text)
        except Exception as e:
            print(f"[WARN] meta.json could not be loaded: {e}")
            meta = None

    # ------------------------------------------------------------------
    # 2. Compact formats (require meta.json)
    # ------------------------------------------------------------------
    if meta:
        encoding = meta.get("encoding", None)

        # --- implicit_index: one power value per line, no timestamps ---
        if encoding == "implicit_index":
            start_ts   = int(meta["start_timestamp"])
            dt_seconds = int(meta.get("sampling_interval", 1))
            header_skip = _count_header_lines(csv_path)
            Pd    = np.loadtxt(csv_path, dtype=np.float64, skiprows=header_skip)
            dtime = pd.to_datetime(
                start_ts + np.arange(len(Pd)) * dt_seconds, unit="s"
            ).values.astype("datetime64[ns]")
            return Pd, dt_seconds / 3600.0, dtime

        # --- delta: two columns [dt_seconds, Pd_W] ---
        if encoding == "delta":
            start_ts   = int(meta["start_timestamp"])
            df         = pd.read_csv(csv_path, header=None, names=["dt", "Pd_W"])
            Pd         = df["Pd_W"].values.astype(np.float64)
            ts         = start_ts + np.cumsum(df["dt"].values.astype(np.int64))
            dtime      = pd.to_datetime(ts, unit="s").values.astype("datetime64[ns]")
            dt_seconds = int(df["dt"].iloc[0])
            return Pd, dt_seconds / 3600.0, dtime

        # --- generic metadata (start_timestamp + dt) ---
        if "start_timestamp" in meta and "dt" in meta:
            start_ts   = int(meta["start_timestamp"])
            dt_seconds = float(meta["dt"])
            df         = pd.read_csv(csv_path)
            Pd         = df["Pd_W"].values.astype(np.float64)
            dtime      = pd.to_datetime(
                start_ts + np.arange(len(Pd)) * dt_seconds, unit="s"
            ).values.astype("datetime64[ns]")
            return Pd, dt_seconds / 3600.0, dtime

        print("[WARN] meta.json present but encoding unknown – falling back to classic CSV.")

    # ------------------------------------------------------------------
    # 3. Classic CSV formats (timestamps in the file itself)
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)

    # Unix timestamps in column "Time"
    if "Time" in df.columns:
        unix       = df["Time"].astype(np.int64).values
        Pd         = df["Pd_W"].astype(np.float64).values
        dtime      = pd.to_datetime(unix, unit="s").values.astype("datetime64[ns]")
        dt_seconds = int(unix[1] - unix[0])
        return Pd, dt_seconds / 3600.0, dtime

    # Human-readable timestamp column (any column name containing "time")
    time_cols = [c for c in df.columns if "time" in c.lower()]
    if time_cols:
        col        = time_cols[0]
        Pd         = df["Pd_W"].astype(np.float64).values
        dtime      = pd.to_datetime(df[col]).values.astype("datetime64[ns]")
        dt_seconds = (dtime[1] - dtime[0]) / np.timedelta64(1, "s")
        return Pd, dt_seconds / 3600.0, dtime

    # ------------------------------------------------------------------
    # 4. Nothing matched
    # ------------------------------------------------------------------
    raise ValueError(
        f"Could not parse '{csv_path.name}': no recognised timestamp column found "
        "and no valid meta.json provided. "
        "Expected: meta.json with encoding='implicit_index'/'delta', "
        "or a CSV with columns 'Time'/'Pd_W' or a datetime column + 'Pd_W'."
    )


# =============================================================================
# idx2timeseries
# =============================================================================
def idx2timeseries(idx: np.ndarray, val: np.ndarray, length_t: int) -> np.ndarray:
    """
    Convert a vector of event indices into a time series (look-ahead fill).

    Fill pattern:
      - positions [0 : idx[0]-1)  ← val[0]       (before first event)
      - positions [idx[0]-1 : idx[1])  ← val[1]   (between event 0 and 1)
      - ...
      - positions [idx[-2]-1 : idx[-1])  ← val[-1]
      - positions [idx[-1]-1 : length_t)  ← 0      (after last event)

    The value at position idx[i]-1 (Python) reflects the NEXT event's value.
    This makes the array a look-ahead: at any time t, the array holds the
    value of the next upcoming event.

    Parameters
    ----------
    idx      : 1-D array of event indices, 1-based
    val      : 1-D array of values, same length as idx
    length_t : total length of the output time series

    Returns
    -------
    x_t : np.ndarray of shape (length_t,), float64
    """
    idx = np.asarray(idx, dtype=np.int64)
    val = np.asarray(val, dtype=np.float64)
    x_t = np.zeros(length_t, dtype=np.float64)

    if len(idx) == 0:
        return x_t

    # Value before the first event
    x_t[0 : idx[0] - 1] = val[0]

    # Value between each pair of consecutive events
    for i in range(1, len(idx)):
        start = int(idx[i - 1]) - 1    # 1-based → 0-based start (inclusive)
        end   = int(idx[i])             # 1-based end = 0-based exclusive end
        x_t[start:end] = val[i]

    return x_t


# =============================================================================
# load_drive_profile
# =============================================================================
def load_drive_profile(
    csv_path: str,
    dtime: np.ndarray,
) -> dict:
    """
    Load a driving profile CSV and build all time-series arrays for wbsim_7_81.

    Note: the E_BAT/min_soe-dependent parts (is_long_trip and
    soe_target_first_departure) are computed later, in decode_inputs() --
    this function never sees the EV parameters, only the raw trip data.

    Parameters
    ----------
    csv_path : path to CSV with columns [departure, arrival, SOE]
    dtime    : simulation time axis (datetime64[ns], length t)

    Returns
    -------
    dict with keys:
        Pd-aligned time series (length t):
          location_profile        – int32, 1=home / 4=driving
          E_drive           – float64, energy subtracted at arrival [kWh]

          SOE_drive_day_kWh – float64, look-ahead-filled daily total energy
                               demand, WITHOUT min_soe added
          idx_first_departure_y  – float64, next first-departure index (1-based, look-ahead)
          idx_next_departure_y  – float64, next departure index (1-based, look-ahead)
          E_next_trip      – float64, energy consumed in next arriving trip [kWh]

        Per-trip metadata (for reference / debugging):
          idx_departure       – int64, departure indices 0-based
          idx_arrival       – int64, arrival indices 0-based
          soe_drive_kWh     – float64, energy per trip [kWh]
          idx_first_departure    – int64, first departure per day, 0-based
    """
    t  = len(dtime)
    df = pd.read_csv(csv_path)

    departure_raw = _robust_parse_datetime(df["departure"])
    arrival_raw = _robust_parse_datetime(df["arrival"])
    SOE         = _robust_parse_float(df["SOE"])

    # ------------------------------------------------------------------
    # Map timestamps to time-axis indices (0-based)
    # ------------------------------------------------------------------
    idx_departure_0 = np.searchsorted(dtime, departure_raw).astype(np.int64)
    idx_arrival_0 = np.searchsorted(dtime, arrival_raw).astype(np.int64)

    # Fix A: keep only trips whose DEPARTURE falls within the simulation window.
    # Trips outside the window would be clipped to the last index, accumulating
    # all their energy at one timestep and corrupting E_drive.
    valid         = idx_departure_0 < t
    idx_departure_0 = idx_departure_0[valid]
    idx_arrival_0 = np.clip(idx_arrival_0[valid], 0, t - 1)
    departure_raw   = departure_raw[valid]
    SOE           = SOE[valid]
    n_trips       = len(SOE)

    # 1-based indices for idx2timeseries
    idx_departure_1 = idx_departure_0 + 1
    idx_arrival_1 = idx_arrival_0 + 1

    # ------------------------------------------------------------------
    # First departure per calendar day
    # ------------------------------------------------------------------
    dates            = pd.to_datetime(departure_raw).normalize()
    _, first_in_day  = np.unique(dates.view(np.int64), return_index=True)
    idx_first_departure_0 = idx_departure_0[first_in_day]
    idx_first_departure_1 = idx_first_departure_0 + 1

    # ------------------------------------------------------------------
    # Daily total SOE (short, per unique day)
    # ------------------------------------------------------------------
    day_of_trip       = dates.values
    unique_days       = np.unique(day_of_trip)
    daily_totals      = np.array(
        [SOE[day_of_trip == d].sum() for d in unique_days], dtype=np.float64
    )

    # ------------------------------------------------------------------
    # Time-series arrays via idx2timeseries
    # ------------------------------------------------------------------
    # Next first-departure index at every timestep
    idx_first_departure_y = idx2timeseries(idx_first_departure_1, idx_first_departure_1.astype(float), t)

    # Daily total energy demand, sparse assignment directly AT idx_first_departure
    # positions -- WITHOUT min_soe (added later, downstream, in decode_inputs,
    # once the EV parameters are known).
    # Deliberately NOT an idx2timeseries look-ahead fill: that would put the
    # NEXT day's total at the idx_first_departure position instead of the
    # day's own total.
    SOE_drive_day_kWh = np.zeros(t, dtype=np.float64)
    SOE_drive_day_kWh[idx_first_departure_0] = daily_totals

    # Energy consumed in next arriving trip [kWh]
    E_next_trip = idx2timeseries(idx_arrival_1, SOE.astype(float), t)

    # Next departure index at every timestep
    idx_next_departure_y = idx2timeseries(idx_departure_1, idx_departure_1.astype(float), t)

    # ------------------------------------------------------------------
    # location_profile: 1 = home, 4 = driving
    # Single source of truth, derived directly from idx_departure/idx_arrival,
    # so the location series and the trip indices can never drift apart.
    # ------------------------------------------------------------------
    location_profile = np.ones(t, dtype=np.int32)
    for i in range(n_trips):
        a, b = int(idx_departure_0[i]), int(idx_arrival_0[i])
        if a < b:
            location_profile[a:b] = 4

    # ------------------------------------------------------------------
    # E_drive: energy subtracted at arrival
    # ------------------------------------------------------------------
    E_drive = np.zeros(t, dtype=np.float64)
    for i in range(n_trips):
        E_drive[int(idx_arrival_0[i])] += SOE[i]

    return {
        # Simulation time series (length t)
        "location_profile":         location_profile,
        "E_drive":            E_drive,
        "SOE_drive_day_kWh":  SOE_drive_day_kWh,
        "idx_first_departure_y":   idx_first_departure_y,
        "idx_next_departure_y":   idx_next_departure_y,
        "E_next_trip":       E_next_trip,
        # Per-trip metadata
        "idx_departure":        idx_departure_0,
        "idx_arrival":        idx_arrival_0,
        "soe_drive_kWh":      SOE.astype(np.float64),
        "idx_first_departure":     idx_first_departure_0,
    }


# =============================================================================
# decode_inputs
# =============================================================================
def decode_inputs(Pd, s, lp, pl, random_draws):
    """
    Prepares all inputs for the wbsim_7_81 simulation loop.

    Parameters
    ----------
    Pd      : np.ndarray  – surplus power [W], length t
    s       : namespace/object with attributes .WB (Wallbox) and .EV (EV)
    lp      : dict        – output of load_drive_profile()
    pl      : ForecastSettings instance, 0 or None  (0/None = no forecast)
    random_draws  : np.ndarray, 0 or None
                0     → no random plug-in logic
                1     → generate new random array (reproducible seed)
                array → use directly

    Returns
    -------
    para, ts, pl_dict, random_draws
    """
    V_1ph = 230.0
    V_3ph = 690.0          # 230 * 3
    t     = len(Pd)

    # =========================================================================
    # Experimental parameters – cooldown timers
    # =========================================================================
    t_cooldown_start = _float_or(s.WB.t_cooldown_start, 0.0)
    t_cooldown_ps    = _float_or(s.WB.t_cooldown_ps,    0.0)

    # =========================================================================
    # Phase-switch thresholds
    # =========================================================================
    n_phase_min_orig = int(s.WB.n_phase_min)
    n_phase_max_orig = int(s.WB.n_phase_max)

    if n_phase_max_orig != n_phase_min_orig:
        _Imin_raw = (float(s.WB.P_WB2EV_min) / 230.0 - float(s.WB.b or 0.0)) \
                    / (1.0 + float(s.WB.a or 0.0))
        P_ps_3to1_low   = float(s.WB.P_ps_3to1_low)   if _is_set(s.WB.P_ps_3to1_low)   \
                          else _Imin_raw * 230.0 * n_phase_max_orig * 0.99
        P_ps_1to3_upper = float(s.WB.P_ps_1to3_upper) if _is_set(s.WB.P_ps_1to3_upper) \
                          else _Imin_raw * 230.0 * n_phase_max_orig * 1.01
    else:
        P_ps_3to1_low   = np.nan
        P_ps_1to3_upper = np.nan

    # =========================================================================
    # Random variable parsing
    # =========================================================================
    if random_draws is None or (np.isscalar(random_draws) and random_draws == 0):
        random_draws = None
    elif np.isscalar(random_draws) and random_draws == 1:
        rng        = np.random.default_rng(42)
        random_draws = rng.standard_normal(t)
    else:
        random_draws = np.asarray(random_draws, dtype=np.float64)

    # =========================================================================
    # Forecast settings parsing
    # =========================================================================
    if pl is None or (np.isscalar(pl) and pl == 0):
        use_forecast = 0
    else:
        use_forecast = 1

    if use_forecast == 1:
        t_hor      = int(getattr(pl, 't_hor', 48))
        _sunny     = getattr(pl, 'sunny',  None)
        _cloudy    = getattr(pl, 'cloudy', None)
        f_pvdrive  = float(getattr(_sunny,  'f_pvdrive',  2.0)) if _sunny  else 2.0
        f_pvdod    = float(getattr(_sunny,  'f_pvdod',    1.2)) if _sunny  else 1.2
        soc_max    = float(getattr(_sunny,  'soc_max',    0.6)) if _sunny  else 0.6
        f_red_pnc1 = float(getattr(_sunny,  'f_red_pnc1', 0.8)) if _sunny  else 0.8
        f_drivepv  = float(getattr(_cloudy, 'f_drivepv',  2.0)) if _cloudy else 2.0
    else:
        t_hor = 48; f_pvdrive = 2.0; f_pvdod = 1.2
        soc_max = 0.6; f_red_pnc1 = 0.8; f_drivepv = 2.0

    pl_dict = {
        "use_forecast": use_forecast, "t_hor": t_hor,
        "f_pvdrive": f_pvdrive, "f_pvdod": f_pvdod,
        "soc_max": soc_max, "f_red_pnc1": f_red_pnc1,
        "f_drivepv": f_drivepv,
    }

    # =========================================================================
    # Extract parameters from s.WB and s.EV
    # =========================================================================
    dt    = float(s.WB.dt)
    steps_per_minute = 1.0 / 60.0 / dt

    P_WB2EV_min    = float(s.WB.P_WB2EV_min)
    P_WB2EV_max    = float(s.WB.P_WB2EV_max)
    t_hold_off_min = float(s.WB.t_hold_off_min)

    t_wait2charge_min = float(s.WB.t_wait2charge) \
        if _is_set(s.WB.t_wait2charge) else t_hold_off_min
    t_hold_phaseswitch_min = float(s.WB.t_hold_phaseswitch) \
        if _is_set(s.WB.t_hold_phaseswitch) else t_hold_off_min
    t_stop_phaseswitch = float(s.WB.t_stop_phaseswitch)

    P_Standby_WB, P_deepStandby_WB, P_Standby_WB_discon, P_deepStandby_WB_discon = \
        _compute_standby_powers(s.WB)

    t_deepstandby        = float(s.WB.t_in_deep_standby)
    t_deepstandby_discon = float(s.WB.t_in_deep_standby_discon) \
        if _is_set(s.WB.t_in_deep_standby_discon) else t_deepstandby

    n_phase_min = n_phase_min_orig
    n_phase_max = n_phase_max_orig

    E_bat_ev   = float(s.EV.E_BAT)
    eta_BAT    = float(s.EV.eta_BAT)
    min_soe    = float(s.EV.min_soe) / 100.0 * E_bat_ev
    WB2EV_a_in = float(s.EV.WB2EV_a_in)
    WB2EV_b_in = float(s.EV.WB2EV_b_in)
    WB2EV_c_in = float(s.EV.WB2EV_c_in)
    P_EV_min   = float(s.EV.P_WB2EV_min_in)
    P_EV_max   = float(s.EV.P_WB2EV_max_in)
    Pobc_max   = float(s.EV.Pobc_max)

    p_wb_norm   = min(P_EV_max, P_WB2EV_max) / Pobc_max
    P_obc_panic = WB2EV_a_in * p_wb_norm**2 + WB2EV_b_in * p_wb_norm + WB2EV_c_in

    t_DEAD = float(s.WB.t_DEAD)

    t_CONSTANTpos, t_CONSTANTneg = _compute_time_constants(s.WB, t_DEAD)
    use_pt1_dynamics     = bool(t_CONSTANTpos > 1.0 or t_CONSTANTneg > 1.0)
    pt1_factor_pos = 1.0 - np.exp(-(dt * 3600.0) / t_CONSTANTpos) if t_CONSTANTpos > 0.0 else 0.0
    pt1_factor_neg = 1.0 - np.exp(-(dt * 3600.0) / t_CONSTANTneg) if t_CONSTANTneg > 0.0 else 0.0

    I_ratelimit_pos, I_ratelimit_neg, I_no_ratelimit_pos, I_no_ratelimit_neg = \
        _compute_ratelimits(s.WB)

    P_ratelimit_pos_init = 22000.0
    P_ratelimit_neg_init = 22000.0

    t_release = float(s.WB.t_release) \
        if _is_set(s.WB.t_release) else 5.0 / 60.0

    # =========================================================================
    # Voltages, I_step, power limits
    # =========================================================================
    # I_step_input: controls mode selection (discrete lookup vs. continuous).
    # I_step: used as the step size in np.arange – must be > 0.
    # These are two separate variables serving different purposes.
    I_step_input = float(s.WB.I_step)
    if I_step_input == 1e-4:
        I_step_input = 0.0
    I_step = max(float(s.WB.I_step), 1e-4)

    if s.WB.I_step == 0:
        Pd_ch_min0 = max(P_WB2EV_min, P_EV_min)
        Pd_ch_max0 = min(P_WB2EV_max, P_EV_max)
    else:
        Pd_ch_min0 = (max(P_WB2EV_min, P_EV_min) / V_1ph) * V_1ph \
                     if n_phase_min == 1 else \
                     (max(P_WB2EV_min, P_EV_min) / V_3ph) * V_3ph
        Pd_ch_max0 = (min(P_WB2EV_max, P_EV_max) / V_1ph) * V_1ph \
                     if n_phase_max == 1 else \
                     (min(P_WB2EV_max, P_EV_max) / V_3ph) * V_3ph

    if n_phase_min == 1:
        Imin = float(s.WB.Imin) if _is_set(s.WB.Imin) else Pd_ch_min0 / V_1ph
    else:
        Imin = float(s.WB.Imin) if _is_set(s.WB.Imin) else Pd_ch_min0 / V_3ph

    if n_phase_max == 3:
        Imax = float(s.WB.Imax) if _is_set(s.WB.Imax) else Pd_ch_max0 / V_3ph
        Vmax = V_3ph
    else:
        Imax = float(s.WB.Imax) if _is_set(s.WB.Imax) else Pd_ch_max0 / V_1ph
        Vmax = V_1ph

    Pd_ch_min1ph = Pd_ch_min0

    # Imin correction for control deviation.
    # Applied unconditionally; manufacturer-specific branches not published.
    Imin = (Imin - float(s.WB.b or 0.0)) / (1.0 + float(s.WB.a or 0.0))

    # =========================================================================
    # Ideal system override
    # =========================================================================
    if getattr(s.WB, 'ideal', 0) == 1:
        n_phase_max  = 3
        n_phase_min  = 1
        P_ps_1to3_upper = 6.0   * 230.0 * n_phase_max_orig
        P_ps_3to1_low   = 5.999 * 230.0 * n_phase_max_orig
        t_cooldown_ps    = 0.0;  t_cooldown_start = 0.0
        Imax = 16.0 * n_phase_max;  Imin = 6.0
        Pd_ch_max1ph = 16.0 * V_1ph * n_phase_max
        Pd_ch_max0   = 16.0 * V_1ph * n_phase_max
        Pd_ch_min0   = 6.0  * V_1ph * n_phase_min
        Pd_ch_min1ph = Pd_ch_min0
        n_phase_max  = n_phase_min
        t_CONSTANTpos = 0.0;  t_CONSTANTneg = 0.0
        t_DEAD = 0.0;  I_step = 1e-3;  I_step_input = 1e-3
        t_deepstandby = 0.0;  t_deepstandby_discon = 0.0
        t_wait2charge_min = 0.0;  t_hold_phaseswitch_min = 0.0
        t_stop_phaseswitch = 0.0;  t_hold_off_min = 0.0
        t_release = 1.0 / 60.0
        use_pt1_dynamics = False;  pt1_factor_pos = 0.0;  pt1_factor_neg = 0.0
        P_Standby_WB = 0.0;  P_deepStandby_WB = 0.0
        P_Standby_WB_discon = 0.0;  P_deepStandby_WB_discon = 0.0
        I_ratelimit_pos = float(Imax);  I_ratelimit_neg = float(Imax)
        I_no_ratelimit_pos = float(Imax);  I_no_ratelimit_neg = float(Imax)

    if getattr(s.WB, 'ideal', 0) != 1:
        Pd_ch_max1ph = Imax * V_1ph

    Pd_ch_min3ph = max(Pd_ch_min0, Imin * V_3ph)
    Pd_ch_max3ph = Imax * V_3ph

    # =========================================================================
    # I_discret and lookup tables
    # =========================================================================
    I_range = np.concatenate([np.arange(Imin, Imax + I_step, I_step), [Imax]])
    I_range = np.unique(I_range)

    icp = s.WB.icp_deviation_fn
    if callable(icp) and getattr(s.WB, 'ideal', 0) != 1:
        deviation_lookup = np.array([icp(i) for i in I_range], dtype=np.float64)
    else:
        deviation_lookup = np.full(len(I_range),
                                  float(s.WB.P_deviation_fallback or 0.0) / Vmax,
                                  dtype=np.float64)

    I_discret = I_range + deviation_lookup
    I_discret = I_discret[I_discret < Imax]
    I_discret = np.append(I_discret, Imax)
    I_discret = np.unique(I_discret)
    I_discret = np.append(I_discret, Imax)

    _sb_prefix = np.array([0.0, P_deepStandby_WB_discon, P_Standby_WB_discon,
                            P_deepStandby_WB, P_Standby_WB], dtype=np.float64)
    P_discretV1ph = np.concatenate([_sb_prefix, I_discret * V_1ph])
    P_discretV3ph = np.concatenate([_sb_prefix, I_discret * V_3ph])

    P_vals = np.arange(0, int(np.ceil(Pd_ch_max0)) + 1001, dtype=np.int64)
    if I_step_input > 0.0:
        idx_lookup_1ph = np.array(
            [int(np.argmin(np.abs(P_discretV1ph - pv))) for pv in P_vals], dtype=np.int64)
        idx_lookup_3ph = np.array(
            [int(np.argmin(np.abs(P_discretV3ph - pv))) for pv in P_vals], dtype=np.int64)
    else:
        idx_lookup_1ph = np.zeros(len(P_vals), dtype=np.int64)
        idx_lookup_3ph = np.zeros(len(P_vals), dtype=np.int64)

    # =========================================================================
    # / 1.3 / 2  Drive profile time series
    # =========================================================================
    E_drive           = np.asarray(lp.get("E_drive",           np.zeros(t)),         dtype=np.float64)
    location_profile        = np.asarray(lp.get("location_profile",        np.ones(t, np.int32)), dtype=np.int32)
    idx_first_departure_y  = np.asarray(lp.get("idx_first_departure_y",  np.arange(1, t+1, dtype=float)), dtype=np.float64)
    idx_next_departure_y  = np.asarray(lp.get("idx_next_departure_y",  np.arange(1, t+1, dtype=float)), dtype=np.float64)
    E_next_trip      = np.asarray(lp.get("E_next_trip",      np.zeros(t)),         dtype=np.float64)
    SOE_drive_day_kWh = np.asarray(lp.get("SOE_drive_day_kWh", np.zeros(t)),         dtype=np.float64)
    idx_arrival       = np.asarray(lp.get("idx_arrival",       np.array([], dtype=np.int64)), dtype=np.int64)
    soe_drive_kWh     = np.asarray(lp.get("soe_drive_kWh",     np.array([])),        dtype=np.float64)
    idx_first_departure    = np.asarray(lp.get("idx_first_departure",    np.array([], dtype=np.int64)), dtype=np.int64)

    # is_long_trip – long trips (SOE > 0.9 * E_bat_ev), flagged at ARRIVAL.
    # Needs E_bat_ev, which is only known here (not inside load_drive_profile).
    is_long_trip = np.zeros(t, dtype=bool)
    if len(soe_drive_kWh) > 0:
        is_long_trip[idx_arrival[soe_drive_kWh > 0.9 * E_bat_ev]] = True

    # soe_target_first_departure – extract the short per-day array from the
    # full-length SOE_drive_day_kWh (by indexing at idx_first_departure),
    # add min_soe (only known here), then re-expand to full length.
    if len(idx_first_departure) > 0:
        daily_totals_at_first_departure = SOE_drive_day_kWh[idx_first_departure]
        soe_target_first_departure = idx2timeseries(
            idx_first_departure + 1,                       # 1-based for idx2timeseries
            daily_totals_at_first_departure + min_soe,
            t
        )
    else:
        soe_target_first_departure = np.zeros(t)

    target_soe = np.maximum(
        np.minimum(E_bat_ev, soe_target_first_departure),
        np.minimum(E_bat_ev, E_next_trip + min_soe)
    )

    t_vec       = np.arange(1, t + 1, dtype=np.float64)
    time_to_departure_first = idx_first_departure_y - t_vec
    time_to_departure_next = idx_next_departure_y - t_vec - 1.0
    is_first_departure_target       = (target_soe == soe_target_first_departure)
    time_to_departure        = np.zeros((t, 3), dtype=np.float64)
    time_to_departure[:, 2]  = (time_to_departure_first * is_first_departure_target + time_to_departure_next * (~is_first_departure_target)) * dt
    time_to_departure[:, 1]  = np.maximum(0.0, time_to_departure[:, 2] - t_stop_phaseswitch * dt)
    time_to_departure[:, 0]  = np.maximum(0.0, time_to_departure[:, 2] - np.ceil(0.5 * t_hold_off_min) * dt)

    # Dead-time shift of Pd
    t_DEAD_steps = round(t_DEAD)
    if t_DEAD_steps > 0:
        Pd_shifted = np.concatenate([
            np.zeros(t_DEAD_steps, dtype=np.float64),
            Pd[:t - t_DEAD_steps].astype(np.float64)
        ])
    else:
        Pd_shifted = Pd.astype(np.float64)

    # =========================================================================
    # Build output dicts
    # =========================================================================
    para = dict(
        dt=dt, steps_per_minute=steps_per_minute,
        t_DEAD=t_DEAD, use_pt1_dynamics=use_pt1_dynamics, pt1_factor_pos=pt1_factor_pos, pt1_factor_neg=pt1_factor_neg,
        t_hold_off_min=t_hold_off_min,
        t_wait2charge_min=t_wait2charge_min,
        t_hold_phaseswitch_min=t_hold_phaseswitch_min,
        t_stop_phaseswitch=t_stop_phaseswitch,
        t_deepstandby=t_deepstandby,
        t_deepstandby_discon=t_deepstandby_discon,
        t_release=t_release,
        t_cooldown_start=t_cooldown_start,
        t_cooldown_ps=t_cooldown_ps,
        n_phase_min=n_phase_min, n_phase_max=n_phase_max,
        Pd_ch_min0=Pd_ch_min0, Pd_ch_max0=Pd_ch_max0,
        Pd_ch_min1ph=Pd_ch_min1ph, Pd_ch_max1ph=Pd_ch_max1ph,
        Pd_ch_min3ph=Pd_ch_min3ph, Pd_ch_max3ph=Pd_ch_max3ph,
        P_ps_3to1_low  =P_ps_3to1_low   if not np.isnan(P_ps_3to1_low)   else -1.0,
        P_ps_1to3_upper=P_ps_1to3_upper if not np.isnan(P_ps_1to3_upper) else -1.0,
        V_1ph=V_1ph, V_3ph=V_3ph, Imin=Imin, Imax=Imax,
        I_step=I_step, I_step_input=I_step_input,
        P_ratelimit_pos_init=P_ratelimit_pos_init,
        P_ratelimit_neg_init=P_ratelimit_neg_init,
        I_ratelimit_pos=I_ratelimit_pos, I_ratelimit_neg=I_ratelimit_neg,
        I_no_ratelimit_pos=I_no_ratelimit_pos, I_no_ratelimit_neg=I_no_ratelimit_neg,
        P_Standby_WB=P_Standby_WB, P_deepStandby_WB=P_deepStandby_WB,
        P_Standby_WB_discon=P_Standby_WB_discon,
        P_deepStandby_WB_discon=P_deepStandby_WB_discon,
        E_bat_ev=E_bat_ev, eta_BAT=eta_BAT, min_soe=min_soe,
        WB2EV_a_in=WB2EV_a_in, WB2EV_b_in=WB2EV_b_in, WB2EV_c_in=WB2EV_c_in,
        Pobc_max=Pobc_max, P_obc_panic=P_obc_panic,
        I_discret    =I_discret.astype(np.float64),
        P_discretV1ph=P_discretV1ph.astype(np.float64),
        P_discretV3ph=P_discretV3ph.astype(np.float64),
        P_vals       =P_vals.astype(np.int64),
        idx_lookup_1ph=idx_lookup_1ph,
        idx_lookup_3ph=idx_lookup_3ph,
        deviation_lookup=deviation_lookup.astype(np.float64),
        I_min_deviation_lookup  =float(Imin),
        use_forecast=use_forecast,
    )

    ts = dict(
        Pd               =Pd_shifted,
        location_profile       =location_profile,
        E_drive          =E_drive,
        is_long_trip         =is_long_trip,
        target_soe       =target_soe,
        soe_target_first_departure=soe_target_first_departure,
        E_next_trip     =E_next_trip,
        idx_first_departure_y =idx_first_departure_y,
        idx_next_departure_y =idx_next_departure_y,
        time_to_departure             =time_to_departure,
    )

    return para, ts, pl_dict, random_draws


# =============================================================================
# Private helpers
# =============================================================================

def _count_header_lines(path: Path) -> int:
    """Count non-numeric leading lines (for implicit_index CSV loading)."""
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "":
            count += 1
            continue
        try:
            float(stripped.split()[0])
            break
        except ValueError:
            count += 1
    return count


def _is_set(val) -> bool:
    """True if val is neither None nor NaN."""
    if val is None:
        return False
    try:
        return not np.isnan(float(val))
    except (TypeError, ValueError):
        return True


def _float_or(val, default: float) -> float:
    """Return float(val) if set, otherwise default."""
    return float(val) if _is_set(val) else default


_MONTH_ABBR = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
}


def _robust_parse_datetime(series: pd.Series) -> np.ndarray:
    """
    Parse a datetime column using the same mechanism as load_Pd()'s "classic"
    timestamp branch (pd.to_datetime()), with one safety step first: English
    month abbreviations (Jan, Feb, ..., Dec) in 'DD-Mon-YYYY HH:MM:SS'-style
    strings are replaced with zero-padded numbers before parsing.

    Reason: pandas'/strptime's '%b' matching is locale-dependent. On a
    non-English-locale machine (e.g. German Windows), "Mar" does not match
    the locale's own month abbreviation ("Mär") and pd.to_datetime() raises
    a ValueError -- even though the string is a perfectly valid date. Doing
    the month-name substitution ourselves sidesteps locale entirely.
    """
    s = series.astype(str)

    def _replace_month(text: str) -> str:
        m = re.match(r'^(\d{1,2})-([A-Za-z]{3})-(\d{4})(.*)$', text.strip())
        if m:
            day, mon_abbr, year, rest = m.groups()
            mon_num = _MONTH_ABBR.get(mon_abbr.lower())
            if mon_num:
                return f"{day}-{mon_num}-{year}{rest}"
        return text

    s = s.map(_replace_month)
    # dayfirst=True: after month-name substitution, strings are numeric
    # ("DD-MM-YYYY ..."). Without dayfirst, pandas' auto-inference can lock
    # onto US month-first order from early ambiguous rows (e.g. "05-10-2014")
    # and then crash on an unambiguous later row (e.g. "20-12-2014", where
    # 20 can't be a month).
    return pd.to_datetime(s, dayfirst=True).values.astype("datetime64[ns]")


def _robust_parse_float(series: pd.Series) -> np.ndarray:
    """Parse a float column, handling German comma-decimal notation."""
    cleaned = series.astype(str).str.replace(",", ".", regex=False).str.strip()
    result  = pd.to_numeric(cleaned, errors="coerce")
    if result.isna().any():
        raise ValueError("SOE column contains non-numeric values.")
    return result.values.astype(np.float64)


def _compute_standby_powers(wb):
    """Compute the four standby power levels (connected/disconnected x normal/deep)."""
    P_SYS_AC  = _float_or(wb.P_SYS_AC, 0.0)
    P_PERI_AC = wb.P_PERI_AC
    if not _is_set(P_PERI_AC):
        P_sb       = P_SYS_AC
        P_deep     = _float_or(wb.P_SYS_AC_DEEP,              P_sb)
        P_discon   = _float_or(wb.P_SYS_AC_DISCONNECTED,      P_sb)
        P_deep_dis = _float_or(wb.P_SYS_AC_DEEP_DISCONNECTED, 0.5 * P_discon)
    else:
        P_peri     = float(P_PERI_AC)
        P_sb       = max(P_SYS_AC, P_peri)
        P_deep     = max(_float_or(wb.P_SYS_AC_DEEP,              P_sb), P_peri)
        P_discon   = max(_float_or(wb.P_SYS_AC_DISCONNECTED,      P_sb), P_peri)
        P_deep_dis = max(_float_or(wb.P_SYS_AC_DEEP_DISCONNECTED,
                                    max(P_peri, 0.5 * P_discon)),         P_peri)
    return P_sb, P_deep, P_discon, P_deep_dis


def _compute_time_constants(wb, t_DEAD: float):
    """Compute the PT1 settling time constants for positive/negative steps."""
    t_SETTLING = wb.t_SETTLING
    if _is_set(t_SETTLING):
        t_C = (float(t_SETTLING) - t_DEAD) / 3.0
        return t_C, t_C
    else:
        t_Cpos = max(0.0, (_float_or(wb.t_SETTLINGpos, 0.0) - t_DEAD) / 3.0)
        t_Cneg = max(0.0, (_float_or(wb.t_SETTLINGneg, 0.0) - t_DEAD) / 3.0)
        return t_Cpos, t_Cneg


def _compute_ratelimits(wb):
    """Read the current rate-limit parameters (ramp rate + jump-after-start)."""
    pos_raw = wb.I_ratelimit_pos
    neg_raw = wb.I_ratelimit_neg
    if _is_set(pos_raw) and _is_set(neg_raw):
        pos_arr = np.atleast_1d(np.array(pos_raw, dtype=float))
        neg_arr = np.atleast_1d(np.array(neg_raw, dtype=float))
        I_rl_pos    = float(pos_arr[0])
        I_rl_neg    = float(abs(neg_arr[0]))
        I_no_rl_pos = float(pos_arr[1])      if len(pos_arr) >= 2 else I_rl_pos
        I_no_rl_neg = float(abs(neg_arr[1])) if len(neg_arr) >= 2 else I_rl_neg
        return I_rl_pos, I_rl_neg, I_no_rl_pos, I_no_rl_neg
    return 32.0, 32.0, 32.0, 32.0