# data/config/s_WB_default.py
# Default wallbox parameter set.
# Imports Wallbox from dataclass_definitions.py – no local class definition.

from dataclass_definitions import Wallbox

s_WB_default = Wallbox(
    # --- Time step: 1 second ---
    dt=1.0 / 3600.0,

    # --- Charging power limits [W] ---
    P_WB2EV_min=1439.26151,
    P_WB2EV_max=11441.760845,

    # --- Phase configuration ---
    n_phase_min=1,
    n_phase_max=3,

    # --- Current step size [A] ---
    I_step=1.0,

    # --- Time hystereses [minutes] ---
    t_hold_off_min=0.678333333333333,
    t_stop_phaseswitch=0.375,
    t_in_deep_standby=28.98,
    t_wait2charge=0.131666666666667,
    t_hold_phaseswitch=1.0,
    t_in_deep_standby_discon=9.36,   # value confirmed correct by lab measurement

    # --- Dead time and settling time [seconds] ---
    t_DEAD=5.0,
    t_SETTLING=0.0,       # synchronous; 0.0 → no PT1 dynamics (use_pt1_dynamics = False)
    t_SETTLINGpos=None,
    t_SETTLINGneg=None,

    # --- Steady-state control deviation: ICP(I) = a * I + b [A] ---
    a=8.433135604143055e-04,
    b=0.069120998936293,
    # icp_deviation_fn is set automatically in __post_init__ from a and b
    P_deviation_fallback=27.381920607321490,

    # --- Standby power levels [W] ---
    P_SYS_AC=6.602042016,
    P_PERI_AC=1.6322753946,
    P_SYS_AC_DEEP=6.602042016,
    P_SYS_AC_DISCONNECTED=6.580387671,
    P_SYS_AC_DEEP_DISCONNECTED=6.575299501,

    # --- Rate limiter: [normal_ramp, jump_after_start] [A] ---
    I_ratelimit_pos=[2.0, 8.0],
    I_ratelimit_neg=[-4.0, -10.0],

    # --- Control release interval [min] ---
    t_release=0.083333333333333,

    # --- Phase switch thresholds [W] ---
    P_ps_3to1_low=3794.822226868040,
    P_ps_1to3_upper=4266.493049290078,

    # --- Current overrides: not used for this parameter set ---
    Imin=None,
    Imax=None,

    # --- Ideal system flag ---
    ideal=0,

    # --- Cooldown [min] ---
    t_cooldown_start=0.0,
    t_cooldown_ps=0.0,
)