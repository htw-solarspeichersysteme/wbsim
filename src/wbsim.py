# src/wbsim.py
#
# wbsim_7_81()  – main simulation function
#
# Architecture:
#   wbsim_7_81()   – Python wrapper (not @jit): unpacks the para/ts dicts,
#                    calls _sim_loop(), runs postprocessing.
#   _sim_loop()    – @jit(nopython=True): the actual timestep loop.
#                    Receives only plain floats/ints/bool and numpy arrays
#                    so Numba can fully compile it.

import math
import numpy as np
from numba import jit
from typing import NamedTuple


# =============================================================================
# Parameter groups passed into _sim_loop()
#
# Numba supports typing.NamedTuple natively in nopython mode (attribute
# access compiles to a direct, typed field lookup -- there is no dict-like
# overhead). Grouping the ~55 scalar/lookup parameters this way keeps the
# function signature stable even if a field is added or reordered later:
# a mistake in construction order below would raise immediately (wrong
# field name), whereas a mistake in a 60-argument positional call could
# silently swap two same-typed values instead. All fields here are
# read-only for the duration of the loop (never reassigned inside
# _sim_loop) -- exactly what an immutable NamedTuple is for.
# =============================================================================
class TimeParams(NamedTuple):
    dt: float
    steps_per_minute: float
    t_DEAD: float
    use_pt1_dynamics: bool
    pt1_factor_pos: float
    pt1_factor_neg: float
    t_hold_off_min: float
    t_wait2charge_min: float
    t_hold_phaseswitch_min: float
    t_stop_phaseswitch: float
    t_deepstandby: float
    t_deepstandby_discon: float
    t_release: float
    t_cooldown_start: float
    t_cooldown_ps: float


class PowerParams(NamedTuple):
    n_phase_min: int
    n_phase_max: int
    Pd_ch_min0: float
    Pd_ch_max0: float
    Pd_ch_min1ph: float
    Pd_ch_max1ph: float
    Pd_ch_min3ph: float
    Pd_ch_max3ph: float
    P_ps_3to1_low: float
    P_ps_1to3_upper: float
    V_1ph: float
    V_3ph: float
    Imin: float
    Imax: float
    I_step: float
    I_step_input: float
    I_ratelimit_pos: float
    I_ratelimit_neg: float
    I_no_ratelimit_pos: float
    I_no_ratelimit_neg: float
    P_Standby_WB: float
    P_deepStandby_WB: float
    P_Standby_WB_discon: float
    P_deepStandby_WB_discon: float
    E_bat_ev: float
    eta_BAT: float
    min_soe: float
    WB2EV_a_in: float
    WB2EV_b_in: float
    WB2EV_c_in: float
    Pobc_max: float
    P_obc_panic: float


class LookupTables(NamedTuple):
    P_discretV1ph: np.ndarray
    P_discretV3ph: np.ndarray
    P_vals: np.ndarray
    idx_lookup_1ph: np.ndarray
    idx_lookup_3ph: np.ndarray
    deviation_lookup: np.ndarray
    I_min_deviation_lookup: float


# =============================================================================
# Public entry point
# =============================================================================
def wbsim_7_81(para: dict, ts: dict, pl_dict: dict, random_draws) -> dict:
    """
    Run the EV charging simulation.

    Parameters
    ----------
    para     : dict from decode_inputs() – scalars + lookup arrays
    ts       : dict from decode_inputs() – time-series arrays (length t)
    pl_dict  : dict from decode_inputs() – forecast settings
    random_draws   : np.ndarray (float64, length t) or None

    Returns
    -------
    dict with keys:
        P_wb      – wallbox AC power [W]
        P_bat     – battery DC power [W]
        P_obc     – OBC loss power [W]
        E_b       – battery state of energy [kWh]
        E_b_ext   – externally charged energy correction [kWh]
    """
    # ------------------------------------------------------------------
    # Unpack para
    # ------------------------------------------------------------------
    dt               = float(para["dt"])
    steps_per_minute            = float(para["steps_per_minute"])
    t_DEAD           = float(para["t_DEAD"])
    use_pt1_dynamics              = bool(para["use_pt1_dynamics"])
    pt1_factor_pos          = float(para["pt1_factor_pos"])
    pt1_factor_neg          = float(para["pt1_factor_neg"])
    t_hold_off_min   = float(para["t_hold_off_min"])
    t_wait2charge_min    = float(para["t_wait2charge_min"])
    t_hold_phaseswitch_min = float(para["t_hold_phaseswitch_min"])
    t_stop_phaseswitch   = float(para["t_stop_phaseswitch"])
    t_deepstandby        = float(para["t_deepstandby"])
    t_deepstandby_discon = float(para["t_deepstandby_discon"])
    t_release           = float(para["t_release"])
    t_cooldown_start     = float(para["t_cooldown_start"])
    t_cooldown_ps        = float(para["t_cooldown_ps"])
    n_phase_min      = int(para["n_phase_min"])
    n_phase_max      = int(para["n_phase_max"])
    Pd_ch_min0       = float(para["Pd_ch_min0"])
    Pd_ch_max0       = float(para["Pd_ch_max0"])
    Pd_ch_min1ph     = float(para["Pd_ch_min1ph"])
    Pd_ch_max1ph     = float(para["Pd_ch_max1ph"])
    Pd_ch_min3ph     = float(para["Pd_ch_min3ph"])
    Pd_ch_max3ph     = float(para["Pd_ch_max3ph"])
    P_ps_3to1_low    = float(para["P_ps_3to1_low"])
    P_ps_1to3_upper  = float(para["P_ps_1to3_upper"])
    V_1ph            = float(para["V_1ph"])
    V_3ph            = float(para["V_3ph"])
    Imin             = float(para["Imin"])
    Imax             = float(para["Imax"])
    I_step           = float(para["I_step"])
    I_step_input     = float(para["I_step_input"])
    I_ratelimit_pos      = float(para["I_ratelimit_pos"])
    I_ratelimit_neg      = float(para["I_ratelimit_neg"])
    I_no_ratelimit_pos   = float(para["I_no_ratelimit_pos"])
    I_no_ratelimit_neg   = float(para["I_no_ratelimit_neg"])
    P_Standby_WB          = float(para["P_Standby_WB"])
    P_deepStandby_WB      = float(para["P_deepStandby_WB"])
    P_Standby_WB_discon   = float(para["P_Standby_WB_discon"])
    P_deepStandby_WB_discon = float(para["P_deepStandby_WB_discon"])
    E_bat_ev         = float(para["E_bat_ev"])
    eta_BAT          = float(para["eta_BAT"])
    min_soe          = float(para["min_soe"])
    WB2EV_a_in       = float(para["WB2EV_a_in"])
    WB2EV_b_in       = float(para["WB2EV_b_in"])
    WB2EV_c_in       = float(para["WB2EV_c_in"])
    Pobc_max         = float(para["Pobc_max"])
    P_obc_panic      = float(para["P_obc_panic"])
    use_forecast          = int(pl_dict["use_forecast"])
    # lookup arrays
    P_discretV1ph  = para["P_discretV1ph"]
    P_discretV3ph  = para["P_discretV3ph"]
    P_vals         = para["P_vals"]
    idx_lookup_1ph = para["idx_lookup_1ph"]
    idx_lookup_3ph = para["idx_lookup_3ph"]
    # steady-state control deviation (Fix 2: replaces placeholder _x_statabw)
    deviation_lookup = para["deviation_lookup"]
    I_min_deviation_lookup   = float(para["I_min_deviation_lookup"])

    # ------------------------------------------------------------------
    # Unpack ts
    # ------------------------------------------------------------------
    Pd                    = ts["Pd"].astype(np.float64)
    location_profile            = ts["location_profile"].astype(np.int32)
    E_drive               = ts["E_drive"].astype(np.float64)
    is_long_trip              = ts["is_long_trip"].astype(np.bool_)
    target_soe            = ts["target_soe"].astype(np.float64)
    soe_target_first_departure = ts["soe_target_first_departure"].astype(np.float64)
    time_to_departure                  = ts["time_to_departure"].astype(np.float64)

    # ------------------------------------------------------------------
    # random_draws (None → zero array, use_random_connect=False)
    # ------------------------------------------------------------------
    use_random_connect = random_draws is not None
    if use_random_connect:
        random_draws = np.asarray(random_draws, dtype=np.float64)
    else:
        random_draws = np.zeros(len(Pd), dtype=np.float64)

    # ------------------------------------------------------------------
    # Group scalar/lookup parameters into the three NamedTuples that
    # _sim_loop() expects (see class definitions above).
    # ------------------------------------------------------------------
    time_params = TimeParams(
        dt=dt, steps_per_minute=steps_per_minute, t_DEAD=t_DEAD, use_pt1_dynamics=use_pt1_dynamics,
        pt1_factor_pos=pt1_factor_pos, pt1_factor_neg=pt1_factor_neg,
        t_hold_off_min=t_hold_off_min,
        t_wait2charge_min=t_wait2charge_min,
        t_hold_phaseswitch_min=t_hold_phaseswitch_min,
        t_stop_phaseswitch=t_stop_phaseswitch,
        t_deepstandby=t_deepstandby,
        t_deepstandby_discon=t_deepstandby_discon,
        t_release=t_release,
        t_cooldown_start=t_cooldown_start,
        t_cooldown_ps=t_cooldown_ps,
    )
    power_params = PowerParams(
        n_phase_min=n_phase_min, n_phase_max=n_phase_max,
        Pd_ch_min0=Pd_ch_min0, Pd_ch_max0=Pd_ch_max0,
        Pd_ch_min1ph=Pd_ch_min1ph, Pd_ch_max1ph=Pd_ch_max1ph,
        Pd_ch_min3ph=Pd_ch_min3ph, Pd_ch_max3ph=Pd_ch_max3ph,
        P_ps_3to1_low=P_ps_3to1_low, P_ps_1to3_upper=P_ps_1to3_upper,
        V_1ph=V_1ph, V_3ph=V_3ph, Imin=Imin, Imax=Imax,
        I_step=I_step, I_step_input=I_step_input,
        I_ratelimit_pos=I_ratelimit_pos, I_ratelimit_neg=I_ratelimit_neg,
        I_no_ratelimit_pos=I_no_ratelimit_pos,
        I_no_ratelimit_neg=I_no_ratelimit_neg,
        P_Standby_WB=P_Standby_WB, P_deepStandby_WB=P_deepStandby_WB,
        P_Standby_WB_discon=P_Standby_WB_discon,
        P_deepStandby_WB_discon=P_deepStandby_WB_discon,
        E_bat_ev=E_bat_ev, eta_BAT=eta_BAT, min_soe=min_soe,
        WB2EV_a_in=WB2EV_a_in, WB2EV_b_in=WB2EV_b_in, WB2EV_c_in=WB2EV_c_in,
        Pobc_max=Pobc_max, P_obc_panic=P_obc_panic,
    )
    lookup_tables = LookupTables(
        P_discretV1ph=P_discretV1ph, P_discretV3ph=P_discretV3ph,
        P_vals=P_vals, idx_lookup_1ph=idx_lookup_1ph,
        idx_lookup_3ph=idx_lookup_3ph,
        deviation_lookup=deviation_lookup, I_min_deviation_lookup=I_min_deviation_lookup,
    )

    # ------------------------------------------------------------------
    # Run compiled loop
    # ------------------------------------------------------------------
    P_wb, P_bat, P_obc, E_b, E_b_ext = _sim_loop(
        Pd, location_profile, E_drive, is_long_trip,
        target_soe, soe_target_first_departure, time_to_departure,
        random_draws, use_random_connect, use_forecast,
        time_params, power_params, lookup_tables,
    )

    # ------------------------------------------------------------------
    # Postprocessing
    # Correct energy balance at end of simulation year.
    # ------------------------------------------------------------------
    P_wb, P_bat, P_obc, E_b, E_b_ext = _postprocessing(
        P_wb, P_bat, P_obc, E_b, E_b_ext,
        E_drive, location_profile,
        min_soe, E_bat_ev,
        Pd_ch_max0, P_obc_panic, P_Standby_WB,
        dt,
    )

    return {
        "P_wb":   P_wb,
        "P_bat":  P_bat,
        "P_obc":  P_obc,
        "E_b":    E_b,
        "E_b_ext": E_b_ext,
    }


# =============================================================================
# Compiled simulation loop
# =============================================================================
@jit(nopython=True)
def _sim_loop(
    # Time-series arrays (indexed per timestep)
    Pd, location_profile, E_drive, is_long_trip,
    target_soe, soe_target_first_departure, time_to_departure,
    random_draws, use_random_connect, use_forecast,
    # Grouped scalar/lookup parameters (NamedTuples, read-only for the
    # duration of the loop -- see TimeParams / PowerParams / LookupTables
    # near the top of this file)
    time_params, power_params, lookup_tables,
):
    """
    Compiled timestep loop: evaluates status, derives the power set-point,
    and translates it into the wallbox's actual control behaviour, once
    per second, for the whole simulated period.
    All arguments are plain Python scalars or NumPy arrays – no dicts.
    """
    n = len(Pd)

    # ------------------------------------------------------------------
    # Output arrays
    # ------------------------------------------------------------------
    P_wb        = np.zeros(n)
    P_bat       = np.zeros(n)
    P_obc       = np.zeros(n)
    E_b         = np.zeros(n)
    E_b_ext     = np.zeros(n)
    P_wb_control = np.zeros(n)

    # ------------------------------------------------------------------
    # State variables
    # ------------------------------------------------------------------
    status_active          = 1
    status_waiting         = 1
    status_charging        = 0
    status_n_phase         = 0
    status_isBatteryfull   = 0
    status_panic           = 0
    time_to_departure_col   = 0    # time_to_departure column index: 0=start-offset, 1=phaseswitch-offset, 2=unshifted
    status_doing_something = 0
    status_connected       = True
    status_first_jump_done = True
    status_phaseswitch     = 0    # 1 = a phase switch has been decided and is pending/in
                                   # progress; read by Fix C (deepstandby guard) and Fix D
                                   # (phase-switch-wait routing), 2026-07. Set to 1 when a
                                   # charge session ends specifically for a phase switch,
                                   # reset to 0 when it ends because Pd dropped below the
                                   # minimum (a genuine full stop).

    counter_startcharge1ph = 0.0
    counter_startcharge3ph = 0.0
    counter_phaseswitch    = 0.0
    counter_standby        = 0.0
    counter_endcharge      = 0.0
    counter_release       = 0.0
    counter_cooldown_ps    = time_params.t_cooldown_ps    * time_params.steps_per_minute
    counter_cooldown_start = time_params.t_cooldown_start * time_params.steps_per_minute

    # ------------------------------------------------------------------
    # normcdf pre-calculation for random connect probability
    # ------------------------------------------------------------------
    x_soe = np.arange(1, 10001, dtype=np.float64)
    p1_nc_precalc = 0.5 * (1.0 + _erf_approx((x_soe / 10000.0 - 0.45) / (0.05 * 1.41421356)))
    p2_nc_precalc = 0.5 * (1.0 + _erf_approx((x_soe / 10000.0 - 0.20) / (0.035 * 1.41421356)))

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    # tstart depends on time_params.t_DEAD: the first round(time_params.t_DEAD) steps of the
    # dead-time-shifted Pd are zero-padded (see decode_inputs), so the loop
    # must not start before that.
    tstart = max(1, round(time_params.t_DEAD))
    E_b0   = power_params.min_soe            # initial battery SOE = the configured minimum
    E_b[0] = E_b0
    P_setpoint  = 0.0
    P_setpoint_released = 0.0                # safe default before first use

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for t in range(tstart, n):

        # ==============================================================
        # AT HOME==1)
        # ==============================================================
        if location_profile[t] == 1:

            # ----------------------------------------------------------
            # Arrival: subtract trip energy
            # ----------------------------------------------------------
            if E_drive[t] > 0.0:
                E_b0 -= E_drive[t]
                status_isBatteryfull = 0

                if is_long_trip[t] or E_b0 <= 0.0:
                    # Long trip or SOE went negative: vehicle returns with 25%
                    #
                    E_b_ext[t] = 0.25 * power_params.E_bat_ev + abs(E_b0)
                    E_b0 = 0.25 * power_params.E_bat_ev

                status_connected = True

                # Random plug-in logic
                if use_random_connect:
                    i_soe = max(1, min(10000, int(round(E_b0 / power_params.E_bat_ev * 10000.0))))
                    p1_nc = p1_nc_precalc[i_soe - 1]
                    i_soe2 = max(1, min(10000, int(round((E_b0 - soe_target_first_departure[t]) / power_params.E_bat_ev * 10000.0))))
                    p2_nc = p2_nc_precalc[i_soe2 - 1]
                    status_connected = (1.0 - p1_nc * p2_nc) >= random_draws[t]

                    # Forecast logic – simplified:
                    # Full forecast requires slicing Pd/location_profile in the loop,
                    # which is legal in numba but kept as a placeholder here.
                    # use_forecast==1 branch: target_soe adjustment omitted in this
                    # release; activate by setting use_forecast=0 (default).

                if counter_standby > 0 and status_connected:
                    counter_standby = 0.0

            # ----------------------------------------------------------
            # Not connected
            # ----------------------------------------------------------
            if not status_connected:
                P_wb[t] = power_params.P_Standby_WB_discon
                E_b[t]  = E_b0
                counter_standby += 1.0
                if counter_standby >= time_params.t_deepstandby_discon * time_params.steps_per_minute:
                    P_wb[t] = power_params.P_deepStandby_WB_discon
                continue

            # Cooldown counters  (Fix A, 2026-07)
            # Moved here from inside the "ACTIVE" branch: cooldown must
            # count every timestep while connected, including while the
            # wallbox is inactive/off. The whole point of "cooldown" is to
            # measure elapsed time since switch-off; counting only while
            # active meant the timer never advanced during the very
            # standby period it is supposed to measure.
            counter_cooldown_ps    += 1.0
            counter_cooldown_start += 1.0

            # ----------------------------------------------------------
            # Inactive or battery full
            # ----------------------------------------------------------
            if status_active == 0 or status_isBatteryfull == 1:
                P_wb[t] = power_params.P_Standby_WB
                E_b[t]  = E_b0
                counter_standby += 1.0
                if counter_standby >= time_params.t_deepstandby * time_params.steps_per_minute:
                    P_wb[t] = power_params.P_deepStandby_WB

                status_active  = 0
                status_n_phase = 0

                # check_panic
                if ((power_params.Pd_ch_max0 - power_params.P_obc_panic) / 1000.0 * time_to_departure[t, time_to_departure_col]) + E_b0 \
                        <= target_soe[t] or status_panic == 1:
                    status_active          = 1
                    status_panic           = 1
                    status_doing_something = 1
                    if status_n_phase < power_params.n_phase_max or status_charging == 0:
                        status_waiting  = 1
                        status_charging = 0
                        if counter_startcharge3ph <= 0.5 * time_params.t_wait2charge_min * time_params.steps_per_minute:
                            counter_startcharge3ph = 0.5 * time_params.t_wait2charge_min * time_params.steps_per_minute
                    else:
                        status_charging = 1
                        status_waiting  = 0
                else:
                    status_panic = 0

                E_b[t] = E_b0

                if Pd[t] > power_params.Pd_ch_min0 and status_isBatteryfull == 0:
                    status_active   = 1
                    counter_standby = 0.0

                continue

            # ----------------------------------------------------------
            # ACTIVE
            # ----------------------------------------------------------
            status_active = 1

            # check_panic
            if ((power_params.Pd_ch_max0 - power_params.P_obc_panic) / 1000.0 * time_to_departure[t, time_to_departure_col]) + E_b0 \
                    <= target_soe[t] or status_panic == 1:
                status_active          = 1
                status_panic           = 1
                if status_n_phase < power_params.n_phase_max or status_charging == 0:
                    status_waiting  = 1
                    status_charging = 0
                    if counter_startcharge3ph <= 0.5 * time_params.t_wait2charge_min * time_params.steps_per_minute:
                        counter_startcharge3ph = 0.5 * time_params.t_wait2charge_min * time_params.steps_per_minute
                else:
                    status_charging = 1
                    status_waiting  = 0
            else:
                status_panic = 0

            # ===========================================================
            # WAITING logic
            # ===========================================================
            if status_waiting == 1:
                P_wb[t] = power_params.P_Standby_WB
                P_setpoint   = 0.0

                # --- Prepare 3-phase charging ---
                if power_params.n_phase_max == 3 and (Pd[t] > power_params.Pd_ch_min3ph or status_panic):
                    status_doing_something = 1

                    if status_n_phase == 1 or status_phaseswitch == 1:
                        # Phase switch 1→3: wait for time_params.t_stop_phaseswitch
                        # Fix D (2026-07): status_phaseswitch==1 covers the
                        # case where status_n_phase was reset to 0 in the
                        # meantime (e.g. by the "neither condition met"
                        # branch below on a brief power dip) while a phase
                        # switch is still pending - without it the wallbox
                        # would incorrectly take the fast "normal start"
                        # path instead of waiting out time_params.t_stop_phaseswitch.
                        counter_phaseswitch    += 1.0
                        counter_startcharge1ph += 1.0
                        if counter_phaseswitch > time_params.t_stop_phaseswitch * time_params.steps_per_minute:
                            counter_phaseswitch    = 0.0
                            counter_startcharge1ph = 0.0
                            counter_startcharge3ph = 0.0
                            counter_endcharge      = 0.0
                            status_n_phase    = 3
                            time_to_departure_col   = 2    # time_to_departure col: unshifted (3-phase active)
                            status_waiting    = 0
                            status_charging   = 1
                            P_setpoint             = 0.0
                            counter_release  = 0.0
                            P_wb[t]           = 0.0
                            counter_cooldown_ps = 0.0

                    else:
                        # Normal start: wait for time_params.t_wait2charge_min
                        if counter_cooldown_start >= time_params.t_cooldown_start * time_params.steps_per_minute:
                            counter_startcharge1ph += 1.0
                            counter_startcharge3ph += 1.0
                            if counter_startcharge3ph > time_params.t_wait2charge_min * time_params.steps_per_minute:
                                counter_phaseswitch    = 0.0
                                counter_startcharge1ph = 0.0
                                counter_startcharge3ph = 0.0
                                counter_endcharge      = 0.0
                                status_n_phase    = 3
                                time_to_departure_col   = 2
                                status_waiting    = 0
                                status_charging   = 1
                                P_setpoint             = 0.0
                                counter_release  = 0.0
                                P_wb[t]           = 0.0
                                counter_cooldown_ps = 0.0

                # --- Prepare 1-phase charging ---
                elif (Pd[t] >= power_params.Pd_ch_min1ph and power_params.n_phase_min == 1) or \
                     (status_panic and power_params.n_phase_max == 1):
                    status_doing_something = 1

                    if status_n_phase == 3 or status_phaseswitch == 1:
                        # Phase switch 3→1: wait for time_params.t_stop_phaseswitch
                        # Fix D (2026-07): see analogous comment in the
                        # 1→3 branch above.
                        counter_phaseswitch    += 1.0
                        counter_startcharge3ph += 1.0
                        if counter_phaseswitch > time_params.t_stop_phaseswitch * time_params.steps_per_minute:
                            counter_phaseswitch    = 0.0
                            counter_startcharge1ph = 0.0
                            counter_startcharge3ph = 0.0
                            counter_endcharge      = 0.0
                            status_n_phase    = 1
                            time_to_departure_col   = 1    # time_to_departure col: phaseswitch-offset (1-phase active)
                            status_waiting    = 0
                            status_charging   = 1
                            P_setpoint             = 0.0
                            counter_release  = 0.0
                            P_wb[t]           = 0.0
                            counter_cooldown_ps = 0.0

                    else:
                        # Normal 1-phase start
                        if counter_cooldown_start >= time_params.t_cooldown_start * time_params.steps_per_minute:
                            counter_startcharge1ph += 1.0
                            counter_startcharge3ph += 1.0
                            if counter_startcharge1ph > time_params.t_wait2charge_min * time_params.steps_per_minute:
                                counter_phaseswitch    = 0.0
                                counter_startcharge1ph = 0.0
                                counter_startcharge3ph = 0.0
                                counter_endcharge      = 0.0
                                counter_cooldown_ps    = 0.0
                                status_n_phase    = 1
                                time_to_departure_col   = 1
                                status_waiting    = 0
                                status_charging   = 1
                                P_setpoint             = 0.0
                                counter_release  = 0.0
                                P_wb[t]           = 0.0

                # --- Neither condition met ---
                else:
                    if status_doing_something == 1:
                        counter_phaseswitch    = 0.0
                        counter_startcharge1ph = 0.0
                        counter_startcharge3ph = 0.0
                        counter_endcharge      = 0.0
                        status_n_phase         = 0
                        time_to_departure_col        = 0
                        status_waiting         = 1
                        status_charging        = 0
                        status_doing_something = 0
                        P_setpoint                  = 0.0
                        counter_release       = 0.0

            # ===========================================================
            # CHARGING logic + setpoint
            # ===========================================================
            if status_charging and status_n_phase != 0:
                status_doing_something = 1

                # ---------------------------------------------------
                # PANIC MODE
                # ---------------------------------------------------
                if status_panic == 1:
                    P_setpoint = power_params.Pd_ch_max0
                    if power_params.n_phase_max == 1:
                        P_ratelimit_pos    = 16.0 * power_params.V_1ph
                        P_no_ratelimit_pos = 16.0 * power_params.V_1ph
                        P_discret  = lookup_tables.P_discretV1ph
                        idx_lookup = lookup_tables.idx_lookup_1ph
                    else:
                        P_ratelimit_pos    = 16.0 * power_params.V_3ph
                        P_no_ratelimit_pos = 16.0 * power_params.V_3ph
                        P_discret  = lookup_tables.P_discretV3ph
                        idx_lookup = lookup_tables.idx_lookup_3ph
                    # neg ratelimit: not overridden in panic
                    P_ratelimit_neg    = power_params.I_ratelimit_neg    * (power_params.V_3ph if power_params.n_phase_max == 3 else power_params.V_1ph)
                    P_no_ratelimit_neg = power_params.I_no_ratelimit_neg * (power_params.V_3ph if power_params.n_phase_max == 3 else power_params.V_1ph)

                else:
                    # -----------------------------------------------
                    # SUNSHINE MODE – 1-phase
                    # -----------------------------------------------
                    if status_n_phase == 1:
                        # Setpoint discretisation
                        Idset = min(power_params.Imax, max(power_params.Imin, Pd[t] / power_params.V_1ph - (Pd[t] / power_params.V_1ph % power_params.I_step)))
                        P_setpoint = min(power_params.Pd_ch_max1ph, max(power_params.Pd_ch_min0, Idset * power_params.V_1ph))

                        # Steady-state deviation (not time-variant)
                        # lookup_tables.deviation_lookup[i] = ICP(power_params.Imin + i * power_params.I_step) in A
                        # Standby power is subtracted here (fix reported by Joseph):
                        # power_params.P_Standby_WB is added back onto P_wb further down in the
                        # Battery block, but standby draw does
                        # not appear in the measured control-deviation data used to
                        # derive lookup_tables.deviation_lookup — without subtracting it here it
                        # would effectively be double-counted, undercharging the EV
                        # by a few watts.
                        if Idset < power_params.Imax:
                            i_sw = min(int(round((Idset - lookup_tables.I_min_deviation_lookup) / power_params.I_step)),
                                       len(lookup_tables.deviation_lookup) - 1)
                            i_sw = max(i_sw, 0)
                            P_deviation = lookup_tables.deviation_lookup[i_sw] * power_params.V_1ph - power_params.P_Standby_WB
                            P_setpoint = max(P_setpoint + P_deviation, power_params.Pd_ch_min0)

                        # Rate limits
                        P_ratelimit_pos    = power_params.I_ratelimit_pos    * power_params.V_1ph
                        P_ratelimit_neg    = power_params.I_ratelimit_neg    * power_params.V_1ph
                        P_no_ratelimit_neg = power_params.I_no_ratelimit_neg * power_params.V_1ph
                        if P_wb[t - 1] < 100.0:
                            P_no_ratelimit_pos = 16.0 * power_params.V_1ph
                        else:
                            P_no_ratelimit_pos = power_params.I_no_ratelimit_pos * power_params.V_1ph

                        P_discret  = lookup_tables.P_discretV1ph
                        idx_lookup = lookup_tables.idx_lookup_1ph

                        # End-charge / phase-switch decision
                        if Pd[t] < power_params.Pd_ch_min0 or \
                                (Pd[t] >= power_params.P_ps_1to3_upper and power_params.n_phase_max == 3):
                            counter_endcharge += 1.0
                            if counter_endcharge >= time_params.t_hold_off_min * time_params.steps_per_minute \
                                    and Pd[t] < power_params.Pd_ch_min0:
                                counter_endcharge  = 0.0
                                status_waiting     = 1
                                status_charging    = 0
                                counter_release   = 0.0
                                status_phaseswitch = 0
                                counter_cooldown_start = 0.0
                            else:
                                if counter_cooldown_ps >= time_params.t_cooldown_ps * time_params.steps_per_minute:
                                    if counter_endcharge >= time_params.t_hold_phaseswitch_min * time_params.steps_per_minute \
                                            and Pd[t] >= power_params.P_ps_1to3_upper \
                                            and power_params.n_phase_max == 3:
                                        counter_endcharge = 0.0
                                        status_waiting    = 1
                                        status_charging   = 0
                                        counter_release  = 0.0
                                        status_phaseswitch = 1
                                    counter_cooldown_ps = 0.0
                        elif counter_endcharge > 0.0:
                            counter_endcharge = 0.0

                    # -----------------------------------------------
                    # SUNSHINE MODE – 3-phase
                    # -----------------------------------------------
                    else:
                        # Setpoint discretisation
                        Idset = min(power_params.Imax, max(power_params.Imin, Pd[t] / power_params.V_3ph - (Pd[t] / power_params.V_3ph % power_params.I_step)))
                        P_setpoint = min(power_params.Pd_ch_max3ph, max(power_params.Pd_ch_min3ph, Idset * power_params.V_3ph))

                        # Steady-state deviation
                        # Same standby-subtraction fix as in the 1-phase block above.
                        if Idset < power_params.Imax:
                            i_sw = min(int(round((Idset - lookup_tables.I_min_deviation_lookup) / power_params.I_step)),
                                       len(lookup_tables.deviation_lookup) - 1)
                            i_sw = max(i_sw, 0)
                            P_deviation = lookup_tables.deviation_lookup[i_sw] * power_params.V_3ph - power_params.P_Standby_WB
                            P_setpoint = max(P_setpoint + P_deviation, power_params.Pd_ch_min3ph)

                        # Rate limits
                        P_ratelimit_pos    = power_params.I_ratelimit_pos    * power_params.V_3ph
                        P_ratelimit_neg    = power_params.I_ratelimit_neg    * power_params.V_3ph
                        P_no_ratelimit_neg = power_params.I_no_ratelimit_neg * power_params.V_3ph
                        if P_wb[t - 1] < 100.0:
                            P_no_ratelimit_pos = 16.0 * power_params.V_3ph
                        else:
                            P_no_ratelimit_pos = power_params.I_no_ratelimit_pos * power_params.V_3ph

                        P_discret  = lookup_tables.P_discretV3ph
                        idx_lookup = lookup_tables.idx_lookup_3ph

                        # End-charge / phase-switch decision
                        if Pd[t] <= power_params.Pd_ch_min0 or \
                                (Pd[t] <= power_params.P_ps_3to1_low and power_params.n_phase_min == 1):
                            counter_endcharge += 1.0
                            if counter_endcharge >= time_params.t_hold_off_min * time_params.steps_per_minute \
                                    and Pd[t] < power_params.Pd_ch_min0:
                                counter_endcharge  = 0.0
                                status_waiting     = 1
                                status_charging    = 0
                                status_phaseswitch = 0
                                counter_cooldown_start = 0.0
                            elif counter_endcharge >= time_params.t_hold_phaseswitch_min * time_params.steps_per_minute \
                                    and Pd[t] <= power_params.P_ps_3to1_low \
                                    and power_params.n_phase_min == 1:
                                counter_endcharge = 0.0
                                status_waiting    = 1
                                status_charging   = 0
                                counter_release  = 0.0
                                status_phaseswitch = 1
                        elif counter_endcharge > 0.0:
                            counter_endcharge = 0.0

            # ===========================================================
            # Setpoint → control signal → discrete power
            #
            # ===========================================================
            if (P_wb[t] <= 0.0 and P_setpoint > 0.0) or \
                    (P_setpoint == 0.0 and P_wb[t - 1] > power_params.P_Standby_WB):

                # Control release
                if counter_release >= 0.0:
                    counter_release = round(-time_params.t_release * time_params.steps_per_minute) + 1.0
                    P_setpoint_released = P_setpoint
                elif P_wb[t - 1] > power_params.P_Standby_WB:
                    counter_release += 1.0
                    # P_setpoint_released keeps last value
                else:
                    counter_release = 0.0
                    P_setpoint_released = P_setpoint

                # PT1 + rate limiter  (Fix B, 2026-07)
                # Restructured: when the wallbox is switching straight on or
                # off, neither the PT1 time constant NOR the rate limiter
                # apply - previously the rate limiter still ran afterwards
                # and could clamp the intentional instant jump back down,
                # defeating the point of the "no time constant" special
                # case. The rate limiter now lives inside the "with time
                # constant" branch only. The on/off checks now compare
                # against P_setpoint_released (the value P_wb_control actually ramps
                # toward) instead of P_setpoint, matching what is really being
                # commanded this step.
                if P_setpoint_released >= P_wb_control[t - 1]:
                    # Positive step
                    # Special case: if the wallbox is
                    # effectively off (P_wb[t-1] < 100 W), skip the PT1 time
                    # constant AND the rate limiter entirely and jump
                    # straight to the setpoint. A settling-time transition
                    # or ratelimiter would otherwise introduce an
                    # unrealistic start-up delay.
                    if P_wb[t - 1] < 100.0:
                        P_wb_control[t] = P_setpoint_released
                    else:
                        if time_params.use_pt1_dynamics:
                            P_wb_control[t] = P_wb_control[t-1] + \
                                (P_setpoint_released - P_wb_control[t-1]) * time_params.pt1_factor_pos
                        else:
                            P_wb_control[t] = P_setpoint_released

                        # Rate limiter positive
                        if P_wb_control[t-1] - P_wb_control[max(0, t-2)] <= P_ratelimit_pos:
                            status_first_jump_done = False
                        if not status_first_jump_done:
                            P_wb_control[t] = min(P_wb_control[t-1] + P_no_ratelimit_pos,
                                                  P_wb_control[t])
                            status_first_jump_done = True
                        else:
                            P_wb_control[t] = min(P_wb_control[t] - P_wb_control[t-1],
                                                  P_ratelimit_pos) + P_wb_control[t-1]

                else:
                    # Negative step
                    # Special case: if the setpoint is
                    # exactly zero and the wallbox was actively charging,
                    # skip the PT1 time constant AND the rate limiter
                    # entirely and jump straight to zero.
                    if P_setpoint_released == 0.0 and P_wb[t - 1] > power_params.P_Standby_WB:
                        P_wb_control[t] = P_setpoint_released
                    else:
                        if time_params.use_pt1_dynamics:
                            P_wb_control[t] = P_wb_control[t-1] + \
                                (P_setpoint_released - P_wb_control[t-1]) * time_params.pt1_factor_neg
                        else:
                            P_wb_control[t] = P_setpoint_released

                        # Rate limiter negative
                        if P_wb_control[t-1] - P_wb_control[max(0, t-2)] >= -P_ratelimit_neg:
                            status_first_jump_done = False
                        if not status_first_jump_done:
                            P_wb_control[t] = max(P_wb_control[t-1] - P_no_ratelimit_neg,
                                                  P_wb_control[t])
                            status_first_jump_done = True
                        else:
                            P_wb_control[t] = max(P_wb_control[t] - P_wb_control[t-1],
                                                  -P_ratelimit_neg) + P_wb_control[t-1]

                # Discretise
                if power_params.I_step_input == 0.0:
                    P_wb[t] = P_wb_control[t]
                else:
                    idx_p   = min(int(round(P_wb_control[t])), len(lookup_tables.P_vals) - 1)
                    idx_p   = max(idx_p, 0)
                    i_disc  = idx_lookup[idx_p]
                    P_wb[t] = P_discret[i_disc]

            else:
                P_wb[t] = power_params.P_Standby_WB
                status_first_jump_done = False

            # ===========================================================
            # Battery
            # ===========================================================
            if P_wb[t] > power_params.P_Standby_WB:
                counter_standby = 0.0
                p_wb_norm = P_wb[t] / power_params.Pobc_max
                P_obc[t]  = power_params.WB2EV_a_in * p_wb_norm**2 + power_params.WB2EV_b_in * p_wb_norm + power_params.WB2EV_c_in

                # Cap at 100 % SOE
                P_bat_max = (power_params.E_bat_ev - E_b0) * 1000.0 / time_params.dt / (power_params.eta_BAT / 100.0)
                if P_bat_max < P_wb[t] - P_obc[t]:
                    P_bat[t] = P_bat_max
                    P_wb[t]  = P_obc[t] + P_bat[t]
                    status_isBatteryfull = 1
                else:
                    P_bat[t] = P_wb[t] - P_obc[t]
                    status_isBatteryfull = 0

                P_wb[t] += power_params.P_Standby_WB
            else:
                P_wb[t] = power_params.P_Standby_WB
                counter_standby += 1.0
                # Fix:
                # without this, status_active never drops back to 0 while
                # the EV remains at home and idle — deepstandby had no
                # effect until the vehicle actually departs.
                # Fix C (2026-07): only go inactive if no phase switch is
                # currently pending - otherwise the wallbox would drop
                # into deepstandby mid-switch and never complete it.
                if counter_standby >= time_params.t_deepstandby * time_params.steps_per_minute and status_phaseswitch == 0:
                    status_active = 0

            E_b[t] = E_b0 + P_bat[t] * (power_params.eta_BAT / 100.0) * time_params.dt / 1000.0
            E_b0   = E_b[t]

        # ==============================================================
        # ON THE ROAD
        # ==============================================================
        else:
            # Reset state when departing
            if location_profile[t - 1] == 1:
                counter_phaseswitch    = 0.0
                counter_startcharge1ph = 0.0
                counter_startcharge3ph = 0.0
                counter_endcharge      = 0.0
                status_n_phase         = 0
                time_to_departure_col        = 0
                status_waiting         = 1
                status_charging        = 0
                status_panic           = 0
                status_doing_something = 1
                status_active          = 0
                status_phaseswitch     = 0
                status_connected       = False
                P_setpoint                  = 0.0
                counter_release       = 0.0
                counter_cooldown_start = time_params.t_cooldown_start

            P_wb[t] = power_params.P_Standby_WB_discon
            E_b[t]  = E_b0
            counter_standby += 1.0
            if counter_standby >= time_params.t_deepstandby_discon * time_params.steps_per_minute:
                P_wb[t] = power_params.P_deepStandby_WB_discon

    return P_wb, P_bat, P_obc, E_b, E_b_ext


# =============================================================================
# Postprocessing
# Correct energy balance at end of simulation year.
# =============================================================================
def _postprocessing(P_wb, P_bat, P_obc, E_b, E_b_ext,
                    E_drive, location_profile,
                    min_soe, E_bat_ev,
                    Pd_ch_max0, P_obc_panic, P_Standby_WB,
                    dt):
    """
    End-of-year energy balance correction: forces the final battery state
    of energy back to exactly min_soe, either by trimming the last
    charging event (if the year ended over-charged) or by adding a final
    panic-charge (if it ended under-charged).
    """
    DELTA = E_b[-1] - min_soe

    if DELTA > 0.0:
        # --- Over-charged: trim charging from the end ---
        reversecumEbat = np.flipud(np.cumsum(np.flipud(P_bat)) * dt / 1000.0)
        hits = np.where(reversecumEbat >= DELTA)[0]
        if len(hits) > 0:
            idx = hits[-1]

            P_bat[idx] = (reversecumEbat[idx] - DELTA) / dt * 1000.0
            P_bat[idx + 1:] = 0.0

            P_wb[idx]     = P_bat[idx] + P_obc[idx]
            P_wb[idx + 1:] = P_Standby_WB

            # Guard added during Python review: E_b[idx - 1] with idx == 0
            # would silently wrap around to E_b[-1] (Python negative
            # indexing) instead of raising an error.
            # Not triggered by any case seen so far (idx == 0 would require
            # essentially the entire year to be reversed in a single step),
            # but defensive here since a silent wrap-around would corrupt
            # the whole array without raising anything.
            prev_E_b = E_b[idx - 1] if idx > 0 else min_soe
            E_b[idx:] = prev_E_b + P_bat[idx] * dt / 1000.0

            # Subtract remaining trips after idx
            mask = np.zeros(len(E_drive), dtype=bool)
            mask[idx + 1:] = E_drive[idx + 1:] > 0.0
            drive_idx = np.where(mask)[0]
            for i in drive_idx:
                E_b[i:] -= E_drive[i]

    elif DELTA < 0.0:
        # --- Under-charged: add panic charging at the end ---
        # The -1 converts the length-based offset to a 0-based Python
        # index: without it, the idx: slice below would start one
        # timestep later than intended, silently dropping the first
        # corrected timestep.
        idx = len(P_wb) - 1 - int(np.floor(abs(DELTA) * 1000.0 / (Pd_ch_max0 - P_obc_panic) / dt))
        idx = max(0, idx)

        if np.any(location_profile[idx:] != 1):
            # Vehicle was away during the needed window → charge externally
            idx_ext = len(E_drive) - 1 - np.argmax((E_drive > 0.0)[::-1])
            E_b_ext[idx_ext] += abs(DELTA)
            E_b[idx_ext:]    += abs(DELTA)
        else:
            # Panic-charge at home
            P_wb[idx:]  = Pd_ch_max0
            P_obc[idx:] = P_obc_panic
            P_bat[idx:] = Pd_ch_max0 - P_obc_panic
            # Same idx==0 guard as above.
            prev_E_b = E_b[idx - 1] if idx > 0 else min_soe
            # Last step adjusted to hit exactly min_soe
            P_bat[-1] = (min_soe - (prev_E_b + np.sum(P_bat[idx:-1]) * dt / 1000.0)) \
                        / dt * 1000.0
            P_wb[-1]  = P_bat[-1] + P_obc_panic
            E_b[idx:] = prev_E_b + np.cumsum(P_bat[idx:]) * dt / 1000.0

    return P_wb, P_bat, P_obc, E_b, E_b_ext


# =============================================================================
# Helper: error function for normcdf  (numba-safe)
# Uses math.erf, which numba compiles natively in nopython mode -- this is
# exact rather than an approximation (a prior Abramowitz & Stegun
# approximation had max error 1.5e-7, which only matters for the random
# plug-in path, off by default: use_random_connect).
# =============================================================================
@jit(nopython=True)
def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Vectorised erf, compatible with numba nopython mode."""
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = math.erf(x[i])
    return out