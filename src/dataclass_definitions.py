# src/dataclass_definitions.py
# Single source of truth for all data structures.
# s_WB_default.py and s_EV_default.py import from here and only create instances.

from dataclasses import dataclass, field
from typing import Optional, Callable, List
import numpy as np


# =============================================================================
# Wallbox / charging-station parameters.
# Identification metadata (serial numbers, manufacturer, etc.) is
# intentionally omitted, as it is not used anywhere in the simulation logic.
# =============================================================================
@dataclass
class Wallbox:
    """
    Wallbox / EVSE parameters.
    All fields have a default value → no isfield() checks needed.
    None means "not set" (i.e. use the computed/derived default instead).
    """

    # --- Time step ---
    dt: float = 1.0 / 3600.0       # Simulation step in hours (1 s = 1/3600 h)

    # --- Core charging power limits [W] ---
    P_WB2EV_min: float = 1400.0    # Minimum AC output power
    P_WB2EV_max: float = 11000.0   # Maximum AC output power

    # --- Phase configuration ---
    n_phase_min: int = 1            # Minimum number of phases
    n_phase_max: int = 3            # Maximum number of phases

    # --- Current step size [A] ---
    # 0 or 1e-4 → continuous mode; >0 → discrete steps
    I_step: float = 1.0

    # --- Time hystereses [minutes] ---
    t_hold_off_min:        float         = 1.0   # Minimum hold time before a session may end (hysteresis)
    t_stop_phaseswitch:  float         = 1.0   # Off-time during phase switch
    t_in_deep_standby:   float         = 1.0   # Time until deep standby [min]
    t_wait2charge:       Optional[float] = None # Wait before charge start; None → t_hold_off_min
    t_hold_phaseswitch:  Optional[float] = None # Hold time before phase switch; None → t_hold_off_min
    t_in_deep_standby_discon: Optional[float] = None  # Deep standby disconnected; None → t_in_deep_standby

    # --- Dead time and settling time [seconds] ---
    t_DEAD:        float           = 5.0   # Dead time [s]
    t_SETTLING:    Optional[float] = None  # Synchronous settling time; None → use asymmetric
    t_SETTLINGpos: Optional[float] = None  # Asymmetric positive settling [s]
    t_SETTLINGneg: Optional[float] = None  # Asymmetric negative settling [s]

    # --- Steady-state control deviation ---
    # Linear function: ICP(I) = a * I + b  [A]
    a: Optional[float] = None
    b: Optional[float] = None
    icp_deviation_fn: Optional[Callable[[float], float]] = field(
        default=None, repr=False
    )
    P_deviation_fallback: Optional[float] = 0.0  # Scalar fallback [W]

    # --- Standby power levels [W] ---
    # If P_PERI_AC is set, all standby levels must be >= P_PERI_AC
    P_SYS_AC:                   Optional[float] = None   # Standby, connected
    P_PERI_AC:                  Optional[float] = None   # Peripheral power (always on)
    P_SYS_AC_DEEP:              Optional[float] = None   # Deep standby, connected; None → P_SYS_AC
    P_SYS_AC_DISCONNECTED:      Optional[float] = None   # Standby, disconnected; None → P_SYS_AC
    P_SYS_AC_DEEP_DISCONNECTED: Optional[float] = None   # Deep standby, disconnected; None → 0.5 * P_SYS_AC_DISCONNECTED

    # --- Rate limiter [A] ---
    # List with 1 or 2 entries: [normal_ramp, jump_after_start]
    I_ratelimit_pos: Optional[List[float]] = None   # Positive ramp; None → 32 A
    I_ratelimit_neg: Optional[List[float]] = None   # Negative ramp (magnitude); None → 32 A

    # --- Control release interval [min] ---
    t_release: Optional[float] = None   # None → 5/60 min (IEC 61851-1 Table A.6)

    # --- Phase switch thresholds [W] ---
    # None → computed from Imin
    P_ps_3to1_low:   Optional[float] = None   # 3→1ph switch threshold
    P_ps_1to3_upper: Optional[float] = None   # 1→3ph switch threshold

    # --- Current overrides [A] (rarely needed) ---
    Imin: Optional[float] = None   # Override computed Imin
    Imax: Optional[float] = None   # Override computed Imax

    # --- Ideal system flag ---
    ideal: int = 0   # 1 = lossless ideal system, disables all dynamics

    # --- Cooldown timers [min] (experimental, Kostal) ---
    t_cooldown_start: float = 0.0   # Cooldown after start
    t_cooldown_ps:    float = 0.0   # Cooldown after phase switch

    def __post_init__(self):
        """Derive icp_deviation_fn from a and b if not explicitly set."""
        if self.icp_deviation_fn is None:
            if self.a is not None and self.b is not None:
                a, b = self.a, self.b
                self.icp_deviation_fn = lambda x: a * x + b


# =============================================================================
# EV  (electric vehicle) parameters
# =============================================================================
@dataclass
class EV:
    """
    Electric vehicle parameters.
    WB2EV_a_in/b_in/c_in are the coefficients of the on-board-charger loss
    model (flattened here instead of a nested sub-struct).
    """

    # --- Battery capacity ---
    E_BAT:    float = 70.0    # Usable capacity [kWh]
    eta_BAT:  float = 100.0   # Battery efficiency [%]
    min_soe:  float = 30.0    # Minimum state of energy [%]

    # --- Charging power limits [W] ---
    P_WB2EV_min_in: float = 1400.0    # Minimum AC input power
    P_WB2EV_max_in: float = 11000.0   # Maximum AC input power
    Pobc_max:       float = 11000.0   # Maximum OBC power [W]

    # --- OBC loss coefficients (quadratic model) ---
    # P_obc = a_in*(P_wb/Pobc_max)^2 + b_in*(P_wb/Pobc_max) + c_in  [W]
    WB2EV_a_in: float = 0.0
    WB2EV_b_in: float = 0.0
    WB2EV_c_in: float = 0.0

    # --- Ideal system flag ---
    ideal: int = 0   # 1 = lossless (not evaluated directly -- only Wallbox.ideal is)


# =============================================================================
# Forecast settings (optional; disabled by default -- see use_forecast below)
# =============================================================================
@dataclass
class SunnySettings:
    """Forecast tuning parameters for sunny-day scenarios."""
    f_pvdrive:  float = 2.0    # Ratio E_pv / E_drive
    f_pvdod:    float = 1.2    # Ratio E_pv / depth-of-discharge
    soc_max:    float = 0.6    # Max SOC for extra plug-in probability
    f_red_pnc1: float = 0.8    # Reduction of p_not_connected on sunny days

@dataclass
class CloudySettings:
    """Forecast tuning parameters for cloudy-day scenarios."""
    f_drivepv: float = 2.0    # Ratio E_drive / E_pv

@dataclass
class ForecastSettings:
    """Top-level forecast settings."""
    use_forecast: int           = 0    # 0 = no forecast, 1 = with forecast
    t_hor:   int           = 48   # Planning horizon [h]
    sunny:   SunnySettings  = field(default_factory=SunnySettings)
    cloudy:  CloudySettings = field(default_factory=CloudySettings)


# =============================================================================
# DriveProfile  (raw per-trip data returned by load_drive_profile())
#
# Only fields actually required by the simulation are included here; the
# full per-timestep location/energy series derived from this data are
# computed separately (see location_profile, E_drive, etc. in utils.py).
# =============================================================================
@dataclass
class DriveProfile:
    """
    Driving profile input data, one entry per recorded trip:
      - idx_departure, idx_arrival : departure/arrival indices into the time axis (0-based)
      - soe_drive_kWh             : energy consumed per trip [kWh]
      - soe_drive_day_kWh         : total energy consumed on each trip's day [kWh]
      - idx_first_departure            : index of first departure of each calendar day (0-based)
    All index arrays are 0-based.
    """
    # Per-trip data (length = number of trips in the profile)
    idx_departure:       np.ndarray   # Departure indices, 0-based
    idx_arrival:       np.ndarray   # Arrival indices, 0-based
    soe_drive_kWh:     np.ndarray   # Energy per trip [kWh]
    soe_drive_day_kWh: np.ndarray   # Total daily energy [kWh]
    idx_first_departure:    np.ndarray   # First-trip index per day