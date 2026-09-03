##Test Loading of MAVEN & AWSoM data##
import pandas as pd
import numpy as np

def load_and_filter_satellite(file_path, start_time, end_time, file_type='auto'):
    """
    Load satellite data and filter by time range.

    Parameters:
    -----------
    file_path : str
        Path to the data file
    start_time : str
        Start time in format 'YYYY-MM-DD HH:MM:SS'
    end_time : str
        End time in format 'YYYY-MM-DD HH:MM:SS'
    file_type : str
        'drivers' for the first file format, 'awsom' for the second, or 'auto' for automatic detection

    Returns:
    --------
    pandas.DataFrame
        Filtered data with Time column and all other columns
    """

    # Convert start and end times to datetime
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    # Auto-detect file type based on content
    if file_type == 'auto':
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('Satellite data'):
                file_type = 'awsom'
            else:
                file_type = 'drivers'
        print(f"Auto-detected file type: {file_type}")

    # Load based on file type
    if file_type == 'drivers':
        # Format: First column has date+time, then 10 data columns
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            header=None,
            names=['datetime_str', 'col1', 'col2', 'col3', 'col4', 'col5',
                   'col6', 'col7', 'col8', 'col9', 'col10']
        )

        # Parse the datetime string (format: YYYY-MM-DD/HH:MM:SS)
        df['Time'] = pd.to_datetime(df['datetime_str'], format='%Y-%m-%d/%H:%M:%S')

        # Drop the original datetime string column
        df = df.drop('datetime_str', axis=1)

        # Rename columns to match expected names 
        df.columns = ['X', 'Y', 'Z', 'rho', 'ux', 'uy', 'uz', 'bx', 'by', 'bz', 'Time']
        # Reorder to put Time first
        cols = ['Time'] + [col for col in df.columns if col != 'Time']
        df = df[cols]

    elif file_type == 'awsom':
        # AWSOM format with column headers on line 1, data starting at line 2
        # First, read the column names from line 1
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Get the header line (line 1, index 1)
            header_line = lines[1].strip()
            col_names = header_line.split()

        # Read the data, skipping first 2 lines
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            skiprows=2,
            header=None,
            names=col_names
        )

        # Create datetime from separate columns
        df['Time'] = pd.to_datetime(
            df['year'].astype(str).str.zfill(4) + '-' +
            df['mo'].astype(str).str.zfill(2) + '-' +
            df['dy'].astype(str).str.zfill(2) + ' ' +
            df['hr'].astype(str).str.zfill(2) + ':' +
            df['mn'].astype(str).str.zfill(2) + ':' +
            df['sc'].astype(str).str.zfill(2),
            format='%Y-%m-%d %H:%M:%S'
        )

    # Filter by time range
    mask = (df['Time'] >= start_dt) & (df['Time'] <= end_dt)
    df_filtered = df.loc[mask].reset_index(drop=True)

    print(f"Loaded {len(df_filtered)} records from {file_path}")
    print(f"Time range: {df_filtered['Time'].min()} to {df_filtered['Time'].max()}")

    return df_filtered
    
file_path_maven = "/content/drivers_merge_l2.txt"
file_path_awsom = "/content/trj_mars_n00005000.sat (CR2235)"

# For CR2235
start = "2020-09-07 17:40:00"
end = "2020-10-07 00:20:00"

# Load and filter the AWSOM data for CR2235
awsom_data = load_and_filter_satellite(
    "/content/trj_mars_n00005000.sat (CR2235)",
    "2020-09-07 17:40:00",
    "2020-10-07 00:20:00",
    file_type='awsom'  # or let it auto-detect
)

print(f"Number of records: {len(awsom_data)}")
print(awsom_data[['Time', 'X', 'Y', 'Z', 'ux', 'uy', 'uz', 'bx', 'by', 'bz']].head())

## Plotting of 4 variables: B, T, V, n against time for CR2235 & MAVEN (not interpolated) ##

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_filter_maven(file_path, start_time, end_time):
    """
    Load MAVEN data file and filter by time range.
    """
    # Read the MAVEN data file
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=[
            "UT", "n_proton", "n_alpha", "|v_proton|", "vx", "vy", "vz",
            "T_proton", "Bx", "By", "Bz"
        ],
        engine="python"
    )

    # Convert UT string to datetime (format: YYYY-MM-DD/HH:MM:SS)
    df["Time"] = pd.to_datetime(df["UT"], format="%Y-%m-%d/%H:%M:%S")

    # Filter by time range
    mask = (df["Time"] >= start_time) & (df["Time"] <= end_time)
    df_filtered = df.loc[mask].reset_index(drop=True)

    return df_filtered

file_path_maven = "/content/drivers_merge_l2.txt"
file_path_awsom = "/content/trj_mars_n00005000.sat (CR2235)"

start_time_awsom = "2020-09-07 17:40:00"
end_time_awsom   = "2020-10-05 00:20:00"
start_time_maven = "2020-09-07 17:13:44"
end_time_maven   = "2020-10-07 00:11:08"

# Load data
awsom_data = load_and_filter_satellite(file_path_awsom, start_time_awsom, end_time_awsom)
maven_data = load_and_filter_maven(file_path_maven, start_time_maven, end_time_maven)

# Convert AWSoM B field components from Gauss to nT
bx_nT_awsom = awsom_data['bx'] * 1e5
by_nT_awsom = awsom_data['by'] * 1e5
bz_nT_awsom = awsom_data['bz'] * 1e5

# Compute total magnetic field magnitude in nT for AWSoM
B_total_awsom = np.sqrt(bx_nT_awsom**2 + by_nT_awsom**2 + bz_nT_awsom**2)

# Find B with wave energy density parameters included for AWSoM
I = awsom_data['I01'] + awsom_data['I02']
# 1erg/cm**3 = 0.1J/m**3
mu_0 = 1.2566e-6
db_squared = (I * 0.1 * mu_0)*1e18
B_wave_awsom = np.sqrt(B_total_awsom**2 + db_squared)

# Calculate velocity magnitude for AWSoM
V_total_awsom = np.sqrt(awsom_data['ux']**2 + awsom_data['uy']**2 + awsom_data['uz']**2)

# Pressures for AWSoM
P_awsom_nPa = awsom_data['p'] * 1e-8   # proton pressure
Pe_awsom_nPa = awsom_data['pe'] * 1e-8 # electron pressure

# Constants
mp = 1.67e-24  # proton mass in grams
k = 1.3807e-23 # Boltzmann constant

# Calculate number density from rho for AWSoM
n_awsom = awsom_data['rho'] / mp

# Ion temperature (scaled) for AWSoM
ion_temp_awsom = ((awsom_data['p'] * mp) / (awsom_data['rho'] * k) * 1e-7) / 1e4

# Calculate magnetic field magnitude for MAVEN
B_total_maven = np.sqrt(maven_data['Bx']**2 + maven_data['By']**2 + maven_data['Bz']**2)

# Convert MAVEN proton temperature from eV to K
# 1 eV = 11604.525 K
T_proton_maven_K = (maven_data['T_proton'] * 11604.525) / 1e4

# Create stacked plots (4 rows) with both datasets
fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Density
axs[0].plot(awsom_data['Time'], n_awsom, color='r', label='AWSoM', linewidth=2)
axs[0].plot(maven_data['Time'], maven_data['n_proton'], color='b', label='MAVEN', linewidth=2)
axs[0].set_ylabel("n [/cm³]")
axs[0].set_ylim(0, 25)
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Temperature
axs[1].plot(awsom_data['Time'], ion_temp_awsom, color='r', label='AWSoM', linewidth=2)
axs[1].plot(maven_data['Time'], T_proton_maven_K, color='b', label='MAVEN', linewidth=2)
axs[1].set_ylabel("T [K]")
axs[1].set_ylim(0, 35)
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Velocity
axs[2].plot(awsom_data['Time'], V_total_awsom, color='r', label='AWSoM', linewidth=2)
axs[2].plot(maven_data['Time'], maven_data['|v_proton|'], color='b', label='MAVEN', linewidth=2)
axs[2].set_ylabel("|V| / 1e4 [km/s]")
axs[2].set_ylim(0, 900)
axs[2].legend()
axs[2].grid(True, alpha=0.3)

# Magnetic field
axs[3].plot(awsom_data['Time'], B_total_awsom, color='r', label="AWSoM |B|", linewidth=2)
axs[3].plot(awsom_data['Time'], B_wave_awsom, 'k--', label="AWSoM |B| (with wave energy)", linewidth=2)
axs[3].plot(maven_data['Time'], B_total_maven, color='b', label="MAVEN |B|", linewidth=2)
axs[3].set_ylabel("|B| [nT]")
axs[3].set_xlabel("Time")
axs[3].set_ylim(0, 10)
axs[3].legend()
axs[3].grid(True, alpha=0.3)

# Overall title
axs[0].set_title("AWSoM Solar Wind Prediction vs MAVEN Observations")

plt.tight_layout()
plt.show()

# Print some statistics
print("AWSoM Data Statistics:")
print(f"Time range: {awsom_data['Time'].min()} to {awsom_data['Time'].max()}")
print(f"Number of data points: {len(awsom_data)}")
print(f"Density range: {n_awsom.min():.2f} - {n_awsom.max():.2f} /cm³")
print(f"Temperature range: {ion_temp_awsom.min():.2f} - {ion_temp_awsom.max():.2f} K")
print(f"Velocity range: {V_total_awsom.min():.2f} - {V_total_awsom.max():.2f} km/s")
print(f"B field range: {B_total_awsom.min():.2f} - {B_total_awsom.max():.2f} nT")

print("\nMAVEN Data Statistics:")
print(f"Time range: {maven_data['Time'].min()} to {maven_data['Time'].max()}")
print(f"Number of data points: {len(maven_data)}")
print(f"Density range: {maven_data['n_proton'].min():.2f} - {maven_data['n_proton'].max():.2f} /cm³")
print(f"Temperature range: {T_proton_maven_K.min():.2f} - {T_proton_maven_K.max():.2f} K")
print(f"Velocity range: {maven_data['|v_proton|'].min():.2f} - {maven_data['|v_proton|'].max():.2f} km/s")
print(f"B field range: {B_total_maven.min():.2f} - {B_total_maven.max():.2f} nT")

##Full Final Interpolation with AWSoM Comparison##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.signal import savgol_filter, butter, filtfilt

# =====================================================
# Interpolate & Smooth MAVEN → AWSoM Time Grid
# =====================================================
def interpolate_and_smooth_maven_to_awsom(maven_df, awsom_times,
                                          interp_method="pchip",
                                          smooth="butter",
                                          window_minutes=60,
                                          extra_B_smoothing_hours=5):
    """
    Interpolate MAVEN data onto AWSoM time grid and apply smoothing.
    Includes a second smoothing pass for |B| (same as original script).
    """
    # Convert time to seconds since start
    def to_seconds_since_start(times, start_time):
        delta = times - start_time
        return delta.dt.total_seconds().values if hasattr(delta, "dt") else delta.total_seconds()

    maven_seconds = to_seconds_since_start(maven_df["Time"], maven_df["Time"].iloc[0])
    awsom_seconds = to_seconds_since_start(awsom_times, maven_df["Time"].iloc[0])

    ## Interpolation 
    if interp_method == "pchip":
        interp_func = lambda x, y: PchipInterpolator(x, y)
    else:
        interp_func = lambda x, y: interp1d(x, y, kind="linear", fill_value="extrapolate")

    cols_to_interp = ["n_proton", "T_proton", "vx", "vy", "vz", "Bx", "By", "Bz"]
    interp_data = {}
    for col in cols_to_interp:
        f = interp_func(maven_seconds, maven_df[col].values)
        interp_data[col] = f(awsom_seconds)

    # Compute |V| and |B| 
    vx, vy, vz = interp_data["vx"], interp_data["vy"], interp_data["vz"]
    bx, by, bz = interp_data["Bx"], interp_data["By"], interp_data["Bz"]
    V_total = np.sqrt(vx**2 + vy**2 + vz**2)
    B_total = np.sqrt(bx**2 + by**2 + bz**2)

    # Smoothing (same logic as original) 
    def apply_smoothing(y, dt_seconds):
        if smooth is None:
            return y
        elif smooth == "savgol":
            window = int(window_minutes * 60 / dt_seconds)
            if window % 2 == 0:
                window += 1
            window = max(5, min(window, len(y) - 1))
            return savgol_filter(y, window, polyorder=2)
        elif smooth == "butter":
            fs = 1 / dt_seconds
            cutoff = 1 / (window_minutes * 60)
            b, a = butter(2, cutoff / (0.5 * fs), btype="low", analog=False)
            return filtfilt(b, a, y)
        else:
            return y

    dt_seconds = np.median(np.diff(awsom_seconds))
    print(dt_seconds)

    # Apply main smoothing to all interpolated data
    for key in interp_data:
        interp_data[key] = apply_smoothing(interp_data[key], dt_seconds)
    V_total = apply_smoothing(V_total, dt_seconds)
    B_total = apply_smoothing(B_total, dt_seconds)

    # Extra B-field smoothing
    window_pts = int(extra_B_smoothing_hours * 3600 / dt_seconds)
    if window_pts % 2 == 0:
        window_pts += 1
    window_pts = max(5, min(window_pts, len(B_total) - 1))
    B_total_extra_smoothed = savgol_filter(B_total, window_pts, polyorder=2)

    # Build output DataFrame
    maven_interp = pd.DataFrame(interp_data)
    maven_interp["|V|"] = V_total
    maven_interp["|B|"] = B_total_extra_smoothed  # use extra-smoothed B
    maven_interp["Time"] = awsom_times.values

    return maven_interp


# =====================================================
# Main Execution

awsom_times = awsom_data["Time"]

# Interpolate MAVEN onto AWSoM grid
maven_interp = interpolate_and_smooth_maven_to_awsom(
    maven_data,
    awsom_times,
    interp_method="pchip",
    smooth="butter",
    window_minutes=60,          # main smoothing window
    extra_B_smoothing_hours=5   # secondary smoothing for |B|
)

# =====================================================
# Prepare AWSoM Variables
# =====================================================
bx_nT_awsom = awsom_data['bx'] * 1e5
by_nT_awsom = awsom_data['by'] * 1e5
bz_nT_awsom = awsom_data['bz'] * 1e5
B_total_awsom = np.sqrt(bx_nT_awsom**2 + by_nT_awsom**2 + bz_nT_awsom**2)

V_total_awsom = np.sqrt(awsom_data['ux']**2 + awsom_data['uy']**2 + awsom_data['uz']**2)
n_awsom = awsom_data['rho'] / 1.67e-24

k = 1.3807e-23
mp = 1.67e-24
ion_temp_awsom = ((awsom_data['p'] * mp) / (awsom_data['rho'] * k) * 1e-7) / 1e4
T_proton_maven_K_interp = (maven_interp['T_proton'] * 11604.525) / 1e4
T_proton_maven_K_raw = (maven_data['T_proton'] * 11604.525) / 1e4

# Find B with wave energy density parameters included for AWSoM
I = awsom_data['I01'] + awsom_data['I02']
# 1erg/cm**3 = 0.1J/m**3
mu_0 = 1.2566e-6
db_squared = (I * 0.1 * mu_0)*1e18
B_wave_awsom = np.sqrt(B_total_awsom**2 + db_squared)

# =====================================================
# 6️⃣ Plot AWSOM vs MAVEN (Interpolated & Smoothed) 
# =====================================================
fig, axs = plt.subplots(4, 1, figsize=(20, 18), sharex=True)

# ---- Density ----
axs[0].plot(awsom_times, n_awsom, 'r', label='AWSoM', linewidth=2.5)
axs[0].plot(maven_interp["Time"], maven_interp["n_proton"], 'b', label='MAVEN (Interp+Smooth)', linewidth=2.5)
axs[0].plot(maven_data['Time'], maven_data['n_proton'], color='g', label='MAVEN (Raw)', linewidth=2.5)
axs[0].set_ylabel("n [cm$^{-3}$]", fontsize=35, fontweight='bold')
axs[0].set_ylim(0, 22)
axs[0].legend(fontsize=24, loc='best')
axs[0].grid(True, alpha=0.3)
axs[0].tick_params(axis='y', labelsize=22)

# ---- Temperature ----
axs[1].plot(awsom_times, ion_temp_awsom, 'r', label='AWSoM', linewidth=2.5)
axs[1].plot(maven_interp["Time"], T_proton_maven_K_interp, 'b', label='MAVEN (Interp+Smooth)', linewidth=2.5)
axs[1].plot(maven_data['Time'], T_proton_maven_K_raw, color='g', label='MAVEN (Raw)', linewidth=2.5)
axs[1].set_ylabel("T [K]", fontsize=35, fontweight='bold')
axs[1].set_ylim(0, 25)
axs[1].legend(fontsize=24, loc='best')
axs[1].grid(True, alpha=0.3)
axs[1].tick_params(axis='y', labelsize=22)

# ---- Velocity ----
axs[2].plot(awsom_times, V_total_awsom, 'r', label='AWSoM', linewidth=2.5)
axs[2].plot(maven_interp["Time"], maven_interp["|V|"], 'b', label='MAVEN (Interp+Smooth)', linewidth=2.5)
axs[2].plot(maven_data['Time'], maven_data['|v_proton|'], color='g', label='MAVEN (Raw)', linewidth=2.5)
axs[2].set_ylabel("|V| [km/s]", fontsize=35, fontweight='bold')
axs[2].set_ylim(0, 800)
axs[2].legend(fontsize=22, loc='best')
axs[2].grid(True, alpha=0.3)
axs[2].tick_params(axis='y', labelsize=22)

# ---- Magnetic Field ----
axs[3].plot(awsom_times, B_total_awsom, 'r', linewidth=2.5)
axs[3].plot(awsom_data['Time'], B_wave_awsom, 'k--', lw=2.5, label="AWSoM |B| (with wave energy density parameters)")
axs[3].plot(maven_interp["Time"], maven_interp["|B|"], 'b', linewidth=2.5)
axs[3].plot(maven_data['Time'], B_total_maven, color='g', linewidth=2.5)
axs[3].set_ylabel("|B| [nT]", fontsize=35, fontweight='bold')
axs[3].set_xlabel("Time", fontsize=35, fontweight='bold')
axs[3].set_ylim(0, 6)
axs[3].legend(fontsize=22, loc='upper left')
axs[3].grid(True, alpha=0.3)
axs[3].tick_params(axis='x', labelsize=20)
axs[3].tick_params(axis='y', labelsize=22)

# ---- Title ----
axs[0].set_title("CR2235 AWSoM vs MAVEN (Interpolated & Smoothed onto AWSoM Time Grid, with extra |B| smoothing) vs MAVEN (Raw)",
                 fontsize=38, fontweight='bold')

plt.tight_layout()
plt.savefig('awsom_vs_maven_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

## Low-Pass Filtering

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence

# ---------- helper functions ----------
def fft_power_and_freq(signal, dt_seconds):
    n = len(signal)
    x = np.asarray(signal, dtype=float)
    x = x - np.nanmean(x)
    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals)**2
    freqs_hz = np.fft.rfftfreq(n, d=dt_seconds)
    freqs_cpd = freqs_hz * 86400.0
    return freqs_cpd, power, fft_vals

def fft_lowpass_filter(signal, dt_seconds, f_cut_cpd):
  n = len(signal)
  x = np.asarray(signal, dtype=float)
  mean = np.nanmean(x)
  fft_vals = np.fft.rfft(x - mean)
  freqs_hz = np.fft.rfftfreq(n, d=dt_seconds)
  freqs_cpd = freqs_hz * 86400.0
  fft_vals_filtered = fft_vals.copy()
  fft_vals_filtered[freqs_cpd > f_cut_cpd] = 0
  return np.fft.irfft(fft_vals_filtered, n=n) + mean

#def fft_lowpass_filter_maven(signal, dt_seconds, f_cut_cpd):
#FFT low-pass filter applied to MAVEN only (de-meaned and restored)
  #  x = safe_nanfill(signal).astype(float)
   # mean = np.nanmean(x)
    #x0 = x - mean
    #n = len(x0)
    #fft_vals = np.fft.rfft(x0)
    #freqs_hz = np.fft.rfftfreq(n, d=dt_seconds)
    #freqs_cpd = freqs_hz * 86400.0
    #fft_vals_filtered = fft_vals.copy()
    #fft_vals_filtered[freqs_cpd > f_cut_cpd] = 0.0
    #filtered = np.fft.irfft(fft_vals_filtered, n=n)

# ---------- full variables dictionary ----------
#variables = {
 #   "Density": (n_awsom.values, maven_interp["n_proton"].values),
  #  "Temperature": (ion_temp_awsom.values,
   #                 (maven_interp["T_proton"].values * 11604.525) / 1e4),
    #"Velocity": (V_total_awsom.values, maven_interp["|V|"].values),
    #"Bmag": (B_total_awsom, maven_interp["|B|"].values)
#}
dt = 600.0  # ensure this is correctly computed earlier
# Compute MAVEN |B|
maven_interp["|B|"] = np.sqrt(
    maven_interp["Bx"]**2 +
    maven_interp["By"]**2 +
    maven_interp["Bz"]**2
)

variables = {
    "Density": (
        n_awsom,                    # AWSoM density array
        maven_interp["n_proton"].values
    ),

    "Temperature": (
        ion_temp_awsom,             # AWSoM ion temperature
        (maven_interp["T_proton"].values * 11604.525) / 1e4
    ),

    "Velocity": (
        V_total_awsom,              # AWSoM |V|
        maven_interp["|V|"].values
    ),

    "Bmag": (
        B_total_awsom,              # AWSoM |B|
        maven_interp["|B|"].values
    )
}

final_cutoffs = {}

plt.figure(figsize=(12, 10))

for i, (name, (awsom_sig, maven_sig)) in enumerate(variables.items(), 1):

    a = np.asarray(awsom_sig, float)
    m = np.asarray(maven_sig, float)

    # ---- compute FFT spectra ----
    freqs, p_aw, _ = fft_power_and_freq(a, dt)
    _, p_mv, _ = fft_power_and_freq(m, dt)

    p_nonzero = p_aw.copy(); p_nonzero[0] = 0
    # rel_thr = 0.15
    # mask_rel = p_nonzero >= rel_thr * p_nonzero.max()
    # if np.any(mask_rel):
    # #     f_rel_cut = float(freqs[np.where(mask_rel)[0].max()])
    # # else:
    cum = np.cumsum(p_nonzero) / np.sum(p_nonzero)
    idx = np.searchsorted(cum, 0.95)
    f_rel_cut = float(freqs[min(idx, len(freqs)-1)])

    # # ---- compute coherence ----
    # f_hz, coh = coherence(a - np.nanmean(a), m - np.nanmean(m),
    #                       fs=1.0/dt, nperseg=min(256, len(a)))
    # f_cpd = f_hz * 86400.0

    # # ---- determine coherence-based cutoff ----
    # coh_thr = 0.59
    # mask_coh = coh >= coh_thr
    # if np.any(mask_coh):
    #     f_coh_cut = float(f_cpd[np.where(mask_coh)[0].max()])
    # else:
    #     f_coh_cut = None

    # # ---- strategy 4: coherence-based cutoff, fallback to rel-power ----
    # if f_coh_cut is not None:
    #     cutoff = max(f_coh_cut, f_rel_cut)   # require at least rel-power cutoff
    # else:
    cutoff = f_rel_cut

    # ---- clamp to sensible range (can adjust if needed) ----
    cutoff = float(np.clip(cutoff, 0.01, 3.0))

    final_cutoffs[name] = cutoff

    # ---- plot spectra ----
    ax1 = plt.subplot(len(variables), 2, 2*i - 1)
    ax1.plot(freqs, p_aw, label="AWSoM")
    ax1.plot(freqs, p_mv, label="MAVEN")
    ax1.set_yscale("log")
    ax1.set_xlim(0, 10)
    ax1.axvline(cutoff, color="k", ls="--", label=f"cut={cutoff:.3f}")
    ax1.set_title(f"{name} - power spectrum")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ax2 = plt.subplot(len(variables), 2, 2*i)
    # ax2.plot(f_cpd, coh)
    # ax2.axhline(coh_thr, color="gray", ls=":")
    # ax2.set_xlim(0, 10); ax2.set_ylim(0, 1)
    # ax2.set_title(f"{name} - coherence")
    # ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# print("\n=== Final coherence-based cutoffs (Strategy 4) ===")
for k, v in final_cutoffs.items():
    print(f"{k:<12}: {v:.4f} cycles/day")


# ===========================================================
# Apply filtering using final per-variable cutoffs
# ===========================================================
filtered = {}
for name, (a_sig, m_sig) in variables.items():
    fc = final_cutoffs[name]
    filtered[name] = (
        fft_lowpass_filter(a_sig, dt, fc),
        fft_lowpass_filter(m_sig, dt, fc)
    )


# ===========================================================
# Quick diagnostic plots
# ===========================================================
for name, (a_sig, m_sig) in variables.items():
    a_f, m_f = filtered[name]

    plt.figure(figsize=(12, 3))
    plt.plot(awsom_times, a_sig, lw=2, label="AWSoM orig")
    plt.plot(awsom_times, m_sig, lw=2, label="MAVEN orig")
    plt.plot(awsom_times, m_f, lw=2, label="MAVEN filtered")
    plt.title(f"{name}: final cutoff = {final_cutoffs[name]:.3f} cpd")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Create dictionary with MAVEN filtered, AWSoM original
filtered_for_dtw = {}

for name, (a_sig, m_sig) in variables.items():
    fc = final_cutoffs[name]

    # AWSoM: ORIGINAL (unfiltered)
    awsom_data = a_sig

    # MAVEN: LOW-PASS FILTERED
    maven_filtered = fft_lowpass_filter(m_sig, dt, fc)

    filtered_for_dtw[name] = (awsom_data, maven_filtered)

"""Windows:
- Velocity -- 3 days
- Density -- 2 days
- Temperature -- 2 days
- B field -- 2 days
"""

##DTW CODE with fixed window

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# ===========================================================
# Dynamic Time Warping Functions - OPTIMIZED VERSION
# ===========================================================

def compute_dtw_cost(series1, series2, w=None, metric='euclidean'):
    """
    Compute DTW distance matrix with REDUCED window.
    """
    n = len(series1)
    m = len(series2)

    # Define distance function
    if metric == 'euclidean':
        dist_func = lambda x, y: (x - y) ** 2
    elif metric == 'manhattan':
        dist_func = lambda x, y: abs(x - y)
    else:
        dist_func = metric

    # Initialize cost matrix
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i, j] = dist_func(series1[i], series2[j])

    # Initialize DTW matrix
    DTW = np.full((n + 1, m + 1), np.inf)
    DTW[0, 0] = 0

    # ========== REDUCED WINDOW SIZE ==========
    if w is None: #fallback window if no window exists
        max_hours_window = 12
        w = int(max_hours_window * 3600 / 600)
        w = max(w, abs(n - m))
        print(f"Using reduced window: w = {w} samples ({w*600/3600:.1f} hours)")

    # Compute DTW with REDUCED window
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)

        for j in range(j_start, j_end):
            min_cost = min(DTW[i-1, j-1], DTW[i-1, j], DTW[i, j-1])
            DTW[i, j] = cost[i-1, j-1] + min_cost

    DTW = DTW[1:, 1:]
    return DTW, cost


def get_warping_path(DTW):
    """
    Extract optimal warping path with PATH SIMPLIFICATION.
    """
    n, m = DTW.shape
    path = []
    i, j = n - 1, m - 1
    path.append((i, j))

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            neighbors = [(i-1, j-1), (i-1, j), (i, j-1)]
            costs = [DTW[idx] for idx in neighbors]
            min_idx = np.argmin(costs)
            i, j = neighbors[min_idx]

        path.append((i, j))

    path.reverse()

    # ========== PATH SIMPLIFICATION ==========
    simplified_path = []
    if len(path) > 0:
        simplified_path.append(path[0])
        for k in range(1, len(path)-1):
            prev_i, prev_j = path[k-1]
            curr_i, curr_j = path[k]
            next_i, next_j = path[k+1]

            vec1 = (curr_i - prev_i, curr_j - prev_j)
            vec2 = (next_i - curr_i, next_j - curr_j)

            if vec1 != vec2 or k % 50 == 0:
                simplified_path.append(path[k])

        simplified_path.append(path[-1])

    print(f"Path simplified: {len(path)} → {len(simplified_path)} points")
    return simplified_path


def plot_dtw_matrix(DTW, path=None, title="DTW Distance Matrix"):
    """
    Visualize DTW distance matrix.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(DTW, aspect='auto', origin='lower',
                   cmap='YlOrRd', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, label='Accumulated Distance')
    cbar.set_label('Accumulated Distance', fontsize=36, fontweight='bold')
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.tick_params(width=2, length=8)

    if path is not None:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=3.2, alpha=0.7)
        if len(path) > 100:
            stride = len(path) // 50
            ax.scatter(path_x[::stride], path_y[::stride],
                       c='cyan', s=70, alpha=0.8,
                       edgecolors='black', linewidth=1.0, zorder=5)  # ← Different color for scatter


    ax.set_xlabel('MAVEN Time Index', fontsize=21)
    ax.set_ylabel('AWSoM Time Index', fontsize=21)
    ax.set_title(title, fontsize=40, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig, ax


def plot_time_series_alignment_optimized(awsom_series, maven_series, path,
                                        awsom_times=None, maven_times=None,
                                        variable_name="Variable",
                                        max_connections=250):
    # ===========================================================
    # Variable units dictionary
    # ===========================================================
    var_name = variable_name
    variable_units = {
        'Velocity': 'km/s',
        'Density': 'cm$^{-3}$',
        'Temperature': '$10^4$ K',
        'Bmag': 'nT',
        'IMF': 'nT'
    }
    unit = variable_units.get(var_name, '')
    var_label = f"{var_name} ({unit})" if unit else var_name

    """
    Plot with connections at EQUAL TIME INTERVALS.
    """
    fig, axes = plt.subplots(2, 1, figsize=(50, 20),
                            gridspec_kw={'height_ratios': [2, 1]})

    ax1 = axes[0]

    if awsom_times is not None and maven_times is not None:
        x1 = awsom_times
        x2 = maven_times
    else:
        x1 = np.arange(len(awsom_series))
        x2 = np.arange(len(maven_series))

    # Plot main series
    ax1.plot(x1, awsom_series, 'b-', linewidth=5.0, label='AWSoM Model', alpha=0.9)
    ax1.plot(x2, maven_series, 'r-', linewidth=5.0, label='MAVEN Filtered', alpha=0.9)

    # ========== EQUAL TIME INTERVAL CONNECTIONS ==========
    awsom_indices = np.array([p[0] for p in path])
    maven_indices = np.array([p[1] for p in path])

    # Get the range of time indices
    time_min = min(np.min(awsom_indices), np.min(maven_indices))
    time_max = max(np.max(awsom_indices), np.max(maven_indices))

    # Create evenly spaced time points
    time_points = np.linspace(time_min, time_max, max_connections)

    # For each time point, find the closest point on the warping path
    selected_indices = []
    for t in time_points:
        # Find path point closest to this time
        # Use average of AWSoM and MAVEN indices as the "time" along the path
        path_times = (awsom_indices + maven_indices) / 2
        closest_idx = np.argmin(np.abs(path_times - t))
        selected_indices.append(closest_idx)

    # Remove duplicates while preserving order
    selected_indices = list(dict.fromkeys(selected_indices))

    # Ensure include first and last points
    if 0 not in selected_indices:
        selected_indices.insert(0, 0)
    if len(path) - 1 not in selected_indices:
        selected_indices.append(len(path) - 1)

    # Plot connections at equal time intervals
    connections_plotted = 0
    for idx in selected_indices:
        awsom_idx, maven_idx = path[idx]

        if awsom_times is not None and maven_times is not None:
            ax1.plot([maven_times[maven_idx], awsom_times[awsom_idx]],
                    [maven_series[maven_idx], awsom_series[awsom_idx]],
                    color='green', alpha=0.7, linewidth=2.0)
        else:
            ax1.plot([maven_idx, awsom_idx],
                    [maven_series[maven_idx], awsom_series[awsom_idx]],
                    color='green', alpha=0.7, linewidth=2.0)
        connections_plotted += 1

    print(f"Plotted {connections_plotted} connection lines at equal time intervals (from {len(path)} path points)")

    ax1.set_ylabel(var_label, fontsize=75)
    ax1.set_xlabel("Time", fontsize = 75)
    ax1.set_title(f'AWSoM vs MAVEN {variable_name} 250 Equal Interval Connections',
                 fontsize=80, fontweight='bold')
    ax1.legend(loc='best', fontsize = 65)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # Lower plot: Warping function with equal time interval points
    ax2 = axes[1]
    norm_awsom = awsom_indices / max(awsom_indices)
    norm_maven = maven_indices / max(maven_indices)
    ax2.plot(norm_maven, norm_awsom, 'k-', linewidth=2.0, alpha=0.8)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1.5, label='Perfect Alignment')

    # Highlight selected points (evenly spaced in time)
    selected_awsom = [path[idx][0] for idx in selected_indices]
    selected_maven = [path[idx][1] for idx in selected_indices]
    norm_selected_awsom = np.array(selected_awsom) / max(awsom_indices)
    norm_selected_maven = np.array(selected_maven) / max(maven_indices)

    ax2.scatter(norm_selected_maven, norm_selected_awsom,
               c='orange', s=40, alpha=0.85, label=f'{connections_plotted} evenly spaced points',
               edgecolors='red', linewidth=1.0, zorder=5)

    ax2.set_xlabel('Normalized MAVEN Time', fontsize=40)
    ax2.set_ylabel('Normalized AWSoM Time', fontsize=40)
    ax2.set_title('DTW Warping Function (Orange = evenly spaced in time)', fontsize=40)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    fig.delaxes(ax2)

    plt.tight_layout()
    return fig, axes


def plot_selective_connections(awsom_series, maven_series, path,
                              awsom_times=None, maven_times=None,
                              variable_name="Variable",
                              max_connections=120):
    """
    Plot with connections at EQUAL TIME INTERVALS.
    """
    fig, axes = plt.subplots(2, 1, figsize=(50, 20),
                            gridspec_kw={'height_ratios': [2, 1]})

    ax1 = axes[0]

    if awsom_times is not None and maven_times is not None:
        x1 = awsom_times
        x2 = maven_times
    else:
        x1 = np.arange(len(awsom_series))
        x2 = np.arange(len(maven_series))

    ax1.plot(x1, awsom_series, 'b-', linewidth=5.0, label='AWSoM Model', alpha=0.9)
    ax1.plot(x2, maven_series, 'r-', linewidth=5.0, label='MAVEN Filtered', alpha=0.9)

    # ========== EQUAL TIME INTERVAL SELECTION ==========
    awsom_indices = np.array([p[0] for p in path])
    maven_indices = np.array([p[1] for p in path])

    # Get the range of time indices
    time_min = min(np.min(awsom_indices), np.min(maven_indices))
    time_max = max(np.max(awsom_indices), np.max(maven_indices))

    # Create evenly spaced time points
    time_points = np.linspace(time_min, time_max, max_connections)

    # For each time point, find the closest point on the warping path
    selected_indices = []
    for t in time_points:
        path_times = (awsom_indices + maven_indices) / 2
        closest_idx = np.argmin(np.abs(path_times - t))
        selected_indices.append(closest_idx)

    # Remove duplicates while preserving order
    selected_indices = list(dict.fromkeys(selected_indices))

    # Ensure include first and last points
    if 0 not in selected_indices:
        selected_indices.insert(0, 0)
    if len(path) - 1 not in selected_indices:
        selected_indices.append(len(path) - 1)

    # Plot connections at equal time intervals with thicker lines
    for idx in selected_indices:
        awsom_idx, maven_idx = path[idx]
        if awsom_times is not None and maven_times is not None:
            ax1.plot([maven_times[maven_idx], awsom_times[awsom_idx]],
                    [maven_series[maven_idx], awsom_series[awsom_idx]],
                    color='green', alpha=0.7, linewidth=2.5)
        else:
            ax1.plot([maven_idx, awsom_idx],
                    [maven_series[maven_idx], awsom_series[awsom_idx]],
                    color='green', alpha=0.7, linewidth=2.5)
    # ===========================================================
    # Variable units dictionary
    # ===========================================================
    var_name = variable_name
    variable_units = {
        'Velocity': 'km/s',
        'Density': 'cm$^{-3}$',
        'Temperature': '$10^4$ K',
        'Bmag': 'nT',
        'IMF': 'nT'
    }
    unit = variable_units.get(var_name, '')
    var_label = f"{var_name} ({unit})" if unit else var_name

    n_connections = len(selected_indices)
    ax1.set_ylabel(var_label, fontsize=75)
    ax1.set_title(f'AWSoM vs MAVEN {variable_name} 250 Equal Interval Connections',
                 fontsize=80, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=50)
    ax1.legend(loc='best', fontsize = 65)
    ax1.grid(True, alpha=0.3)

    # Lower plot
    ax2 = axes[1]
    norm_awsom = awsom_indices / max(awsom_indices)
    norm_maven = maven_indices / max(maven_indices)

    ax2.plot(norm_maven, norm_awsom, 'k-', linewidth=1.5, alpha=0.7)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1.0)

    # Highlight selected points
    selected_awsom = [path[idx][0] for idx in selected_indices]
    selected_maven = [path[idx][1] for idx in selected_indices]
    norm_selected_awsom = np.array(selected_awsom) / max(awsom_indices)
    norm_selected_maven = np.array(selected_maven) / max(maven_indices)

    ax2.scatter(norm_selected_maven, norm_selected_awsom,
               c='orange', s=50, alpha=0.8, zorder=5, edgecolors='red', linewidth=1.0)

    # ax2.set_xlabel('Normalized MAVEN Time', fontsize=40)
    # ax2.set_ylabel('Normalized AWSoM Time', fontsize=40)
    ax2.set_xlabel('MAVEN Time', fontsize=40)
    ax2.set_ylabel('AWSoM Time', fontsize=40)
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.set_title(f'Warping Function ({n_connections} evenly spaced in time)', fontsize=40)
    ax2.grid(True, alpha=0.3)
    fig.delaxes(ax2)

    plt.tight_layout()
    return fig, axes


def dtw_statistics(awsom_series, maven_series, path, dt=1.0):
    """Compute DTW-based statistics."""
    awsom_idx = np.array([p[0] for p in path])
    maven_idx = np.array([p[1] for p in path])

    awsom_aligned = awsom_series[awsom_idx]
    maven_aligned = maven_series[maven_idx]

    correlation = np.corrcoef(awsom_aligned, maven_aligned)[0, 1]
    rmse = np.sqrt(np.mean((awsom_aligned - maven_aligned) ** 2))
    mae = np.mean(np.abs(awsom_aligned - maven_aligned))
    dtw_distance = np.sqrt(np.sum((awsom_aligned - maven_aligned) ** 2))

    path_length = len(path)
    compression_ratio = path_length / max(len(awsom_series), len(maven_series))

    time_distortion = np.std(awsom_idx / len(awsom_series) -
                            maven_idx / len(maven_series))

    if dt > 0:
        time_diffs = (awsom_idx - maven_idx) * dt
        avg_lag = np.mean(time_diffs)
        lag_std = np.std(time_diffs)
    else:
        avg_lag = lag_std = np.nan

    # awsom_to_maven = {}
    # for awsom_i, maven_j in path:
    #     if awsom_i not in awsom_to_maven:
    #         awsom_to_maven[awsom_i] = []
    #     awsom_to_maven[awsom_i].append(maven_j)

    # mappings_per_point = [len(v) for v in awsom_to_maven.values()]
    # avg_mappings = np.mean(mappings_per_point)
    # max_mappings = np.max(mappings_per_point)

    stats = {
        'dtw_distance': dtw_distance,
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae,
        'path_length': path_length,
        'compression_ratio': compression_ratio,
        'time_distortion': time_distortion,
        'avg_time_lag': avg_lag,
        'lag_std': lag_std,
        # 'avg_mappings_per_point': avg_mappings,
        # 'max_mappings_per_point': max_mappings,
        # 'overmapping_ratio': avg_mappings - 1.0,
        'n_awsom': len(awsom_series),
        'n_maven': len(maven_series)
    }

    return stats


# ===========================================================
# SSF and MSE SKILL SCORE CALCULATIONS
# ===========================================================
def create_mean_reference(observed_series):
    """
    Create mean reference model (constant line at the mean value).
    """
    mean_value = np.mean(observed_series)
    reference_series = np.full_like(observed_series, mean_value)
    return reference_series


def calculate_mse_skill_score(observed, model, reference):
    """
    Calculate MSE-based Skill Score from the paper (Equation 3).
    Skill Score = MSE(obs, model) / MSE(obs, reference)
    """
    mse_model = np.mean((observed - model) ** 2)
    mse_ref = np.mean((observed - reference) ** 2)

    if mse_ref == 0:
        return 0 if mse_model == 0 else float('inf')

    skill_score = mse_model / mse_ref
    return skill_score


def get_dtw_score(series1, series2, fixed_window, metric='euclidean'):
    """
    Extract DTW score (final accumulated cost) from compute_dtw_cost.
    """
    DTW, _ = compute_dtw_cost(series1, series2, fixed_window, metric=metric)
    dtw_score = DTW[-1, -1]
    return dtw_score


def calculate_ssf(observed, model, reference, fixed_window, metric='euclidean'):
    """
    Calculate Sequence Similarity Factor from the paper (Equation 2).
    SSF = DTW(O, M) / DTW(O, R)
    """
    dtw_obs_model = get_dtw_score(observed, model, fixed_window, metric=metric)
    dtw_obs_ref = get_dtw_score(observed, reference, fixed_window, metric=metric)

    if dtw_obs_ref == 0:
        return 0 if dtw_obs_model == 0 else float('inf')
    ssf = dtw_obs_model / dtw_obs_ref
    return ssf


# ===========================================================
# WINDOW OPTIMIZATION USING SECOND DERIVATIVE
# ===========================================================

# def find_optimal_window_second_derivative(awsom_norm, maven_norm, mean_ref,
#                                           window_hours_range=(80, 240, 40),
#                                           smoothing_window=7,
#                                           variable_name="Variable",
#                                           plot=True):
#     """
#     Find the optimal DTW window using second derivative analysis.
#     """
#     min_hours, max_hours, step = window_hours_range
#     window_sizes = np.arange(min_hours, max_hours + step, step)
#     ssf_values = []
#     dtw_scores = []

#     print(f"\n{'='*70}")
#     print(f"Finding optimal DTW window using SECOND DERIVATIVE method")
#     print(f"Variable: {variable_name}")
#     print(f"Testing windows from {min_hours} to {max_hours} hours (step={step}h)")
#     print(f"{'='*70}")

#     for hours in window_sizes:
#         dt = 600.0 #10 mins
#         w_samples = int(hours * 3600 / dt)

#         ssf = calculate_ssf(maven_norm, awsom_norm, mean_ref, w=w_samples)
#         ssf_values.append(ssf)

#         dtw_score = get_dtw_score(maven_norm, awsom_norm, w=w_samples)
#         dtw_scores.append(dtw_score)

#         if hours % 10 == 0 or hours == window_sizes[-1]:
#             print(f"Window: {hours:3.0f}h → SSF = {ssf:.6f}, DTW = {dtw_score:.2f}")

#     window_sizes = np.array(window_sizes)
#     ssf_values = np.array(ssf_values)
#     dtw_scores = np.array(dtw_scores)

#     if smoothing_window % 2 == 0:
#         smoothing_window += 1

#     if len(ssf_values) >= smoothing_window:
#         ssf_smooth = savgol_filter(ssf_values, smoothing_window, 3)
#     else:
#         ssf_smooth = ssf_values

#     first_derivative = np.gradient(ssf_smooth, window_sizes)
#     second_derivative = np.gradient(first_derivative, window_sizes)

#     crossing_points = []
#     for i in range(1, len(second_derivative)):
#         if ssf_smooth[i-1] == ssf_smooth[i]:
#             crossing_points.append(i-1)

#     # window_std = 5
#     # rolling_std = []
#     # for i in range(window_std - 1, len(ssf_smooth)):
#     #     std = np.std(ssf_smooth[i-window_std+1:i+1])
#     #     rolling_std.append(std)

#     # std_threshold = 0.001
#     # std_convergence_idx = None
#     # for i, std in enumerate(rolling_std):
#     #     if std < std_threshold:
#     # #         std_convergence_idx = i + window_std - 1
#     # #         break

#     # optimal_idx = None
#     # method = "Maximum window"

#     if len(crossing_points) > 0:
#         optimal_idx = crossing_points[0]
#         method = "Manual convergence"
#     #     method = "Second derivative zero-crossing"
#     # elif std_convergence_idx is not None:
#     #     optimal_idx = std_convergence_idx
#     #     method = "Standard deviation convergence"
#     # else:
#     #     improvement_rate = np.abs(np.gradient(ssf_smooth, window_sizes))
#     #     improvement_rate_normalized = improvement_rate / np.max(improvement_rate) if np.max(improvement_rate) > 0 else improvement_rate

#     #     for i in range(1, len(improvement_rate_normalized)):
#     #         if improvement_rate_normalized[i] < 0.01:
#     #             optimal_idx = i
#     #             method = "Elbow method (1% improvement threshold)"
#     #             break

#     elif len(crossing_points) == 0:
#         optimal_idx = len(window_sizes) - 1
#         method = "Maximum window (no convergence detected)"

#     optimal_window = window_sizes[optimal_idx]
#     optimal_ssf = ssf_values[optimal_idx]

#     print(f"\n{'='*70}")
#     print(f"Optimal window found: {optimal_window:.0f} hours")
#     print(f"Method used: {method}")
#     print(f"Converged SSF: {optimal_ssf:.6f}")
#     print(f"SSF improved from {ssf_values[0]:.6f} to {optimal_ssf:.6f}")
#     print(f"Improvement: {(ssf_values[0] - optimal_ssf)/ssf_values[0]*100:.2f}%")
#     print(f"{'='*70}")

#     if plot:
#         fig, axes = plt.subplots(3, 2, figsize=(15, 15))

#         ax1 = axes[0, 0]
#         ax1.plot(window_sizes, ssf_values, 'b-', alpha=0.5, label='Raw SSF')
#         ax1.plot(window_sizes, ssf_smooth, 'r-', linewidth=2, label='Smoothed SSF')
#         ax1.axvline(x=optimal_window, color='g', linestyle='--',
#                    label=f'Optimal: {optimal_window:.0f}h', alpha=0.8)
#         ax1.set_xlabel('Window Size (hours)', fontsize=12)
#         ax1.set_ylabel('SSF (lower is better)', fontsize=12)
#         ax1.set_title(f'SSF vs DTW Window: {variable_name}', fontsize=14, fontweight='bold')
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)

#         ax2 = axes[0, 1]
#         ax2.plot(window_sizes, dtw_scores, 'r-o', linewidth=2, markersize=4)
#         ax2.axvline(x=optimal_window, color='g', linestyle='--',
#                    label=f'Optimal: {optimal_window:.0f}h', alpha=0.8)
#         ax2.set_xlabel('Window Size (hours)', fontsize=12)
#         ax2.set_ylabel('DTW Score', fontsize=12)
#         ax2.set_title(f'Raw DTW Score vs Window', fontsize=14, fontweight='bold')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)

#         ax3 = axes[1, 0]
#         ax3.plot(window_sizes, first_derivative, 'b-', linewidth=2)
#         ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
#         ax3.axvline(x=optimal_window, color='g', linestyle='--',
#                    label=f'Optimal: {optimal_window:.0f}h', alpha=0.8)
#         ax3.set_xlabel('Window Size (hours)', fontsize=12)
#         ax3.set_ylabel('First Derivative (dSSF/dh)', fontsize=12)
#         ax3.set_title('First Derivative: Rate of SSF Change', fontsize=14, fontweight='bold')
#         ax3.legend()
#         ax3.grid(True, alpha=0.3)

#         ax4 = axes[1, 1]
#         ax4.plot(window_sizes, second_derivative, 'r-', linewidth=2)
#         ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
#         ax4.axvline(x=optimal_window, color='g', linestyle='--',
#                    label=f'Optimal: {optimal_window:.0f}h', alpha=0.8)

#         for idx in crossing_points:
#             ax4.plot(window_sizes[idx], second_derivative[idx], 'ro', markersize=10)
#             ax4.text(window_sizes[idx], second_derivative[idx]*1.2,
#                     f'Crossing: {window_sizes[idx]:.0f}h', fontsize=9, ha='center')

#         ax4.set_xlabel('Window Size (hours)', fontsize=12)
#         ax4.set_ylabel('Second Derivative (d²SSF/dh²)', fontsize=12)
#         ax4.set_title('Second Derivative: Curvature (Zero-crossing = convergence)', fontsize=14, fontweight='bold')
#         ax4.legend()
#         ax4.grid(True, alpha=0.3)

#         # ax5 = axes[2, 0]
#         # if len(rolling_std) > 0:
#         #     std_window_sizes = window_sizes[window_std-1:]
#         #     ax5.plot(std_window_sizes, rolling_std, 'purple', linewidth=2)
#         #     ax5.axhline(y=std_threshold, color='orange', linestyle='--',
#         #                label=f'Threshold: {std_threshold:.3f}', alpha=0.8)
#         #     ax5.axvline(x=optimal_window, color='g', linestyle='--',
#         #                label=f'Optimal: {optimal_window:.0f}h', alpha=0.8)
#         #     ax5.set_xlabel('Window Size (hours)', fontsize=12)
#         #     ax5.set_ylabel('Rolling Std Dev of SSF', fontsize=12)
#         #     ax5.set_title('Convergence Detection: Rolling Standard Deviation', fontsize=14, fontweight='bold')
#         #     ax5.legend()
#         #     ax5.grid(True, alpha=0.3)
#         # else:
#         #     ax5.text(0.5, 0.5, 'Insufficient data for rolling std',
#         #             transform=ax5.transAxes, ha='center', va='center')
#         #     ax5.set_title('Convergence Detection: Rolling Standard Deviation', fontsize=14, fontweight='bold')

#         # ax6 = axes[2, 1]
#         # improvement_rate = np.abs(np.gradient(ssf_smooth, window_sizes))
#         # improvement_rate_normalized = improvement_rate / np.max(improvement_rate) if np.max(improvement_rate) > 0 else improvement_rate

#         # ax6.plot(window_sizes, improvement_rate_normalized, 'orange', linewidth=2)
#         # ax6.axhline(y=0.01, color='red', linestyle='--', label='1% threshold', alpha=0.8)
#         # ax6.axvline(x=optimal_window, color='g', linestyle='--',
#         #            label=f'Optimal: {optimal_window:.0f}h', alpha=0.8)
#         # ax6.set_xlabel('Window Size (hours)', fontsize=12)
#         # ax6.set_ylabel('Normalized Improvement Rate', fontsize=12)
#         # ax6.set_title('Improvement Rate (Elbow Detection)', fontsize=14, fontweight='bold')
#         # ax6.legend()
#         # ax6.grid(True, alpha=0.3)

#         plt.suptitle(f'Window Optimization Analysis: {variable_name}', fontsize=16, fontweight='bold')
#         plt.tight_layout()
#         plt.savefig(f'window_convergence_{variable_name}.png', dpi=150, bbox_inches='tight')
#         plt.show()

#         fig2, ax = plt.subplots(figsize=(12, 8))
#         ax.plot(window_sizes, ssf_values, 'b-', linewidth=2, label='SSF')
#         ax.plot(window_sizes, ssf_smooth, 'r--', linewidth=2, label='Smoothed SSF')
#         ax.axvline(x=optimal_window, color='g', linestyle='--', linewidth=2,
#                    label=f'Optimal window: {optimal_window:.0f}h')
#         ax.axhline(y=optimal_ssf, color='purple', linestyle=':', linewidth=2,
#                    label=f'Converged SSF: {optimal_ssf:.4f}')

#         ax.axvspan(window_sizes[0], optimal_window, alpha=0.1, color='blue', label='Rapid improvement')
#         if optimal_idx < len(window_sizes) - 1:
#             ax.axvspan(optimal_window, window_sizes[-1], alpha=0.1, color='green', label='Convergence region')

#         ax.set_xlabel('Window Size (hours)', fontsize=13)
#         ax.set_ylabel('SSF (lower is better)', fontsize=13)
#         ax.set_title(f'SSF Convergence Analysis: {variable_name}\nOptimal window = {optimal_window:.0f} hours',
#                     fontsize=14, fontweight='bold')
#         ax.legend(loc='best')
#         ax.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig(f'window_convergence_summary_{variable_name}.png', dpi=150, bbox_inches='tight')
#         plt.show()

#     return optimal_window, ssf_values, window_sizes


def compute_all_metrics(awsom_norm, maven_norm, fixed_window, variable_name="Variable"):
    """
    Compute all metrics using NORMALIZED data.
    """
    print(f"\n{'='*60}")
    print(f"Calculating metrics for: {variable_name}")
    print(f"{'='*60}")

    mean_ref_norm = create_mean_reference(maven_norm)
    mse_skill = calculate_mse_skill_score(maven_norm, awsom_norm, mean_ref_norm)
    ssf = calculate_ssf(maven_norm, awsom_norm, mean_ref_norm, fixed_window)

    DTW, cost = compute_dtw_cost(awsom_norm, maven_norm, fixed_window)
    path = get_warping_path(DTW)
    stats = dtw_statistics(awsom_norm, maven_norm, path, dt=600.0)

    print(f"\n{'Metric':<30} {'Value':<20}")
    print(f"{'-'*50}")
    print(f"{'MSE Skill Score':<30} {mse_skill:<20.4f}")
    print(f"{'SSF (DTW-based)':<30} {ssf:<20.4f}")
    print(f"{'DTW Score (raw)':<30} {stats['dtw_distance']:<20.4f}")
    print(f"{'Correlation':<30} {stats['correlation']:<20.4f}")
    print(f"{'RMSE':<30} {stats['rmse']:<20.4f}")
    print(f"{'MAE':<30} {stats['mae']:<20.4f}")

    print(f"\nInterpretation for {variable_name}:")
    print(f"  • MSE Skill Score = {mse_skill:.3f}: ", end="")
    if mse_skill < 1:
        print("AWSoM performs BETTER than mean model")
    elif mse_skill == 1:
        print("AWSoM performs EQUAL to mean model")
    else:
        print("AWSoM performs WORSE than mean model")

    print(f"  • SSF = {ssf:.3f}: ", end="")
    if ssf < 1:
        print("AWSoM performs BETTER than mean model (DTW perspective)")
    elif ssf == 1:
        print("AWSoM performs EQUAL to mean model (DTW perspective)")
    else:
        print("AWSoM performs WORSE than mean model (DTW perspective)")

    if (mse_skill < 1 and ssf > 1) or (mse_skill > 1 and ssf < 1):
        print(f"  • ⚠️  DISAGREEMENT detected: MSE={mse_skill:.3f} vs SSF={ssf:.3f}")
        print(f"    This indicates time-shift issues that DTW handles better than MSE")
    else:
        print(f"  • ✓ AGREEMENT: Both metrics agree on model performance")

    return {
        'variable_name': variable_name,
        'mse_skill_score': mse_skill,
        'ssf_score': ssf,
        'dtw_stats': stats,
        'dtw_matrix': DTW,
        'cost_matrix': cost,
        'path': path,
        'mean_reference_normalized': mean_ref_norm
    }


def plot_metrics_comparison(all_results):
    """
    Create comparison plot between MSE and SSF metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    variables = [r['variable_name'] for r in all_results]
    mse_scores = [r['mse_skill_score'] for r in all_results]
    ssf_scores = [r['ssf_score'] for r in all_results]

    ax.scatter(mse_scores, ssf_scores, s=120, alpha=0.7, c='blue', edgecolors='black', zorder=3)

    max_val = max(max(mse_scores), max(ssf_scores), 2)
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='Perfect agreement', zorder=1)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5, zorder=1)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5, zorder=1)

    ax.text(0.3, 1.8, 'MSE: GOOD\nSSF: BAD', fontsize=10, alpha=0.7, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    ax.text(1.8, 1.8, 'Both: BAD', fontsize=10, alpha=0.7, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
    ax.text(0.3, 0.5, 'Both: GOOD', fontsize=10, alpha=0.7, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))
    ax.text(1.8, 0.5, 'MSE: BAD\nSSF: GOOD', fontsize=10, alpha=0.7, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

    for i, var in enumerate(variables):
        ax.annotate(var, (mse_scores[i], ssf_scores[i]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, alpha=0.8, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_xlabel('MSE Skill Score (lower is better)', fontsize=40, fontweight='bold')
    ax.set_ylabel('SSF - DTW-based (lower is better)', fontsize=40, fontweight='bold')
    ax.set_title('Comparison: MSE Skill Score vs SSF (Mean Reference Model)\nPoints below diagonal = DTW more optimistic',
                fontsize=40, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.1, max_val)
    ax.set_ylim(-0.1, max_val)

    for i, (mse, ssf) in enumerate(zip(mse_scores, ssf_scores)):
        if (mse < 1 and ssf > 1) or (mse > 1 and ssf < 1):
            ax.plot(mse, ssf, 'ro', markersize=15, markeredgecolor='red',
                   markerfacecolor='none', markeredgewidth=2, zorder=2)

    plt.tight_layout()
    plt.savefig('ssf_vs_mse_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_reference_comparison(maven_original, awsom_original, times, variable_name="Variable"):
    """
    Visualize the original series with mean reference model.
    """
    mean_value = np.mean(maven_original)
    mean_ref_original = np.full_like(maven_original, mean_value)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    ax1 = axes[0]
    ax1.plot(times, maven_original, 'b-', linewidth=1.5, label='MAVEN Observations', alpha=0.8)
    ax1.plot(times, awsom_original, 'r-', linewidth=1.5, label='AWSoM Model', alpha=0.8)
    ax1.set_ylabel(f'{variable_name}', fontsize=40)
    ax1.set_title(f'AWSoM vs MAVEN - {variable_name} (Original Data)', fontsize=40, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=25)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(times, maven_original, 'b-', linewidth=1.5, label='MAVEN Observations', alpha=0.8)
    ax2.plot(times, mean_ref_original, 'g--', linewidth=2,
             label=f'Mean Reference Model (μ = {mean_value:.3f})', alpha=0.8)
    ax2.set_xlabel('Time Index', fontsize=40)
    ax2.set_ylabel(f'{variable_name}', fontsize=40)
    ax2.set_title(f'MAVEN vs Mean Reference Model (Original Data)', fontsize=40, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=25)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'reference_comparison_{variable_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

# ===========================================================
#    FIXED: Overmapping Analysis (Counts MAVEN per AWSoM)
# ===========================================================

# def plot_overmapping_corrected(path, ax, max_points=100, title="Overmapping Analysis"):
#     """
#     Correctly plot the number of MAVEN points mapped to each AWSoM point.
#     """
#     # Build AWSoM → MAVEN mapping dictionary
#     awsom_to_maven = {}
#     for awsom_i, maven_j in path:
#         if awsom_i not in awsom_to_maven:
#             awsom_to_maven[awsom_i] = []
#         awsom_to_maven[awsom_i].append(maven_j)

#     # Count unique MAVEN points per AWSoM point
#     awsom_indices = []
#     maven_counts = []

#     for awsom_i, maven_list in awsom_to_maven.items():
#         # Count UNIQUE MAVEN indices mapped to this AWSoM point
#         unique_maven_count = len(set(maven_list))
#         awsom_indices.append(awsom_i)
#         maven_counts.append(unique_maven_count)

#     # Sort by AWSoM index for plotting
#     sorted_pairs = sorted(zip(awsom_indices, maven_counts))
#     awsom_indices_sorted = [p[0] for p in sorted_pairs]
#     maven_counts_sorted = [p[1] for p in sorted_pairs]

#     # Plot only first max_points
#     if len(awsom_indices_sorted) > max_points:
#         awsom_indices_sorted = awsom_indices_sorted[:max_points]
#         maven_counts_sorted = maven_counts_sorted[:max_points]

#     # Create bar chart
#     ax.bar(awsom_indices_sorted, maven_counts_sorted, alpha=0.7, width=1.0, color='steelblue')
#     ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ideal (1:1)')
#     ax.set_xlabel('AWSoM Index', fontsize=12)
#     ax.set_ylabel('Number of MAVEN Mappings', fontsize=12)
#     ax.set_title(title, fontsize=14, fontweight='bold')
#     ax.legend()
#     ax.grid(True, alpha=0.3)

#     return awsom_indices_sorted, maven_counts_sorted


# def plot_warping_path_with_overmapping(path, ax, title="Warping Path with Overmapping"):
#     """
#     Plot the warping path with colors indicating overmapping.
#     """
#     # Build AWSoM → MAVEN mapping
#     awsom_to_maven = {}
#     for awsom_i, maven_j in path:
#         if awsom_i not in awsom_to_maven:
#             awsom_to_maven[awsom_i] = []
#         awsom_to_maven[awsom_i].append(maven_j)

#     # Create color mapping based on number of MAVEN mappings
#     colors = []
#     for awsom_i, maven_j in path:
#         n_mappings = len(set(awsom_to_maven[awsom_i]))
#         if n_mappings == 1:
#             colors.append('blue')      # 1:1 mapping
#         elif n_mappings == 2:
#             colors.append('orange')    # Slight overmapping
#         else:
#             colors.append('red')       # Severe overmapping

#     # Plot path points with colors
#     path_x = [p[1] for p in path]
#     path_y = [p[0] for p in path]

#     scatter = ax.scatter(path_x, path_y, c=colors, s=10, alpha=0.7)
#     ax.plot(path_x, path_y, 'k-', alpha=0.3, linewidth=0.5)

#     ax.set_xlabel('MAVEN Index', fontsize=12)
#     ax.set_ylabel('AWSoM Index', fontsize=12)
#     ax.set_title(title, fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)

#     # Create legend
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor='blue', alpha=0.7, label='1 MAVEN mapping (1:1)'),
#         Patch(facecolor='orange', alpha=0.7, label='2 MAVEN mappings'),
#         Patch(facecolor='red', alpha=0.7, label='3+ MAVEN mappings')
#     ]
#     ax.legend(handles=legend_elements, loc='upper left')

#     return scatter

# ===========================================================
# OPTIMIZED Main DTW Analysis WITH METRICS
# ===========================================================
def analyze_awsom_maven_dtw(filtered_data, awsom_times, fixed_window, variable_names=None,
                                     max_connections=120):
    """
    OPTIMIZED DTW analysis with SSF and MSE Skill Score metrics.
    Uses EQUAL INTERVAL connections for all plots.
    """
    if variable_names is None:
        variable_names = list(filtered_data.keys())

    print(f"\n{'='*70}")
    # print("Starting OPTIMIZED DTW analysis with SSF and MSE Skill Score")
    print(f"Variables to analyze: {variable_names}")
    # if max_window_hours is None:
    #     print("Window: AUTO-OPTIMIZED (second derivative method, 80-240 hours)")
    # else:
    print(f"Selective Connections: {max_connections} (EQUAL INTERVALS)")
    print(f"Clean Plot Connections: 250 (EQUAL INTERVALS)")
    print(f"{'='*70}")

    all_results = []
    window_result = {}

    for var_name in variable_names:
        print(f"\n{'='*60}")
        print(f"Analyzing: {var_name}")
        print(f"{'='*60}")

        awsom_filt, maven_filt = filtered_data[var_name]

        min_len = min(len(awsom_filt), len(maven_filt))
        awsom_filt = awsom_filt[:min_len]
        maven_filt = maven_filt[:min_len]
        times = awsom_times[:min_len]

        awsom_original = awsom_filt.copy()
        maven_original = maven_filt.copy()

        awsom_norm = (awsom_filt - np.mean(awsom_filt)) / np.std(awsom_filt)
        maven_norm = (maven_filt - np.mean(maven_filt)) / np.std(maven_filt)

        mean_ref = create_mean_reference(maven_norm)

        # if max_window_hours is None:
        #     optimal_window, ssf_values, window_sizes = find_optimal_window_second_derivative(
        #         awsom_norm, maven_norm, mean_ref,
        #         window_hours_range=(80, 240, 40),
        #         smoothing_window=7,
        #         variable_name=var_name,
        #         plot=True
        #     )
        if fixed_window == int(96 * 3600 / dt) or int(72 * 3600 / dt):
            ssf_value = calculate_ssf(maven_norm, awsom_norm, mean_ref, fixed_window)
            print(f"ssf_value: {ssf_value}")
            # w_samples = int(optimal_window * 3600 / 600)
            # window_results[var_name] = {
            #     'optimal_window': optimal_window,
            #     'ssf_values': ssf_values,
            #     'window_sizes': window_sizes
            # }
        # else:
        #     w_samples = int(max_window_hours * 3600 / 600)
        #     optimal_window = max_window_hours
        if fixed_window == int(72 * 3600 / dt):
            print(f"\nUsing DTW window: 3 days")
        else:
            print(f"\nUsing DTW window: 4 days")

        print("\nComputing SSF and MSE Skill Score...")
        results = compute_all_metrics(awsom_norm, maven_norm, fixed_window, var_name)

        results['awsom_original'] = awsom_original
        results['maven_original'] = maven_original
        results['times'] = times
        # results['optimal_window'] = optimal_window

        all_results.append(results)

        print("\nGenerating optimized plots...")

        # ===========================================================
        # Variable units dictionary
        # ===========================================================
        variable_units = {
            'Velocity': 'km/s',
            'Density': 'cm$^{-3}$',
            'Temperature': '$10^4$ K',
            'Bmag': 'nT',
            'IMF': 'nT'
        }
        unit = variable_units.get(var_name, '')
        var_label = f"{var_name} ({unit})" if unit else var_name

        # Plot 1: DTW matrix with SSF score
        if fixed_window == int(72 * 3600 / dt):
            fig1, ax1 = plot_dtw_matrix(results['dtw_matrix'], results['path'],
                                      title=f"DTW: {var_name}, window = 3 days")
            ax1.set_xlabel("MAVEN Time Index", fontsize=40, fontweight='bold')
            ax1.set_ylabel("AWSoM Time Index", fontsize=40, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=30)
            plt.savefig(f'dtw_matrix_{var_name}_w = {fixed_window}.png', dpi=150, bbox_inches='tight')
            plt.show()
        else:
            fig1, ax1 = plot_dtw_matrix(results['dtw_matrix'], results['path'],
                                      title=f"DTW: {var_name}, window = 4 days")
            ax1.set_xlabel("MAVEN Time Index", fontsize=40, fontweight='bold')
            ax1.set_ylabel("AWSoM Time Index", fontsize=40, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=30)
            plt.savefig(f'dtw_matrix_{var_name}_w = {fixed_window}.png', dpi=150, bbox_inches='tight')
            plt.show()

        # Plot 2: Selective connections (EQUAL INTERVALS)
        if fixed_window == int(72 * 3600 / dt):
            fig2, ax2 = plot_selective_connections(
                awsom_original, maven_original, results['path'],
                awsom_times=times, maven_times=times,
                variable_name=f"{var_name}",
                max_connections=max_connections)
            ax2[0].set_xlabel("Time", fontsize=95, fontweight='bold')
            ax2[0].set_ylabel(var_label, fontsize=95, fontweight='bold')
            ax2[0].set_title(f"DTW: {var_name}, 250 equal interval connections", fontsize=95, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=60)
            plt.savefig(f'dtw_alignment_{var_name}_selective_w = 3 days.png', dpi=150, bbox_inches='tight')
            plt.show()
        else:
            fig2, ax2 = plot_selective_connections(
                awsom_original, maven_original, results['path'],
                awsom_times=times, maven_times=times,
                variable_name=f"{var_name}",
                max_connections=max_connections)
            ax2[0].set_xlabel("Time", fontsize=95, fontweight='bold')
            ax2[0].set_ylabel(var_label, fontsize=95, fontweight='bold')
            ax2[0].set_title(f"DTW: {var_name}, 250 equal interval connections", fontsize=95, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=60)
            plt.savefig(f'dtw_alignment_{var_name}_selective_w = 4 days.png', dpi=150, bbox_inches='tight')
            plt.show()

        # Plot 3: Clean alignment with EQUAL INTERVALS (250 connections)
        if fixed_window == int(72 * 3600 / dt):
            fig3, ax3 = plot_time_series_alignment_optimized(
                awsom_original, maven_original, results['path'],
                awsom_times=times, maven_times=times,
                variable_name=f"{var_name} (250 equal intervals)",
                max_connections=250)
            ax3[0].set_xlabel("Time", fontsize=95, fontweight='bold')
            ax3[0].set_ylabel(var_label, fontsize=95, fontweight='bold')
            ax3[0].set_title(f"DTW: {var_name}, 250 equal interval connections", fontsize=95, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=60)
            plt.savefig(f'dtw_alignment_{var_name}_clean_equal_w = 3 days.png', dpi=150, bbox_inches='tight')
            plt.show()
        else:
            fig3, ax3 = plot_time_series_alignment_optimized(
                awsom_original, maven_original, results['path'],
                awsom_times=times, maven_times=times,
                variable_name=f"{var_name} (250 equal intervals)",
                max_connections=250)
            ax3[0].set_xlabel("Time", fontsize=95, fontweight='bold')
            ax3[0].set_ylabel(var_label, fontsize=95, fontweight='bold')
            ax3[0].set_title(f"DTW: {var_name}, 250 equal interval connections", fontsize=95, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=60)
            plt.savefig(f'dtw_alignment_{var_name}_clean_equal_w = 4 days.png', dpi=150, bbox_inches='tight')
            plt.show()

        # Plot 4: Overmapping analysis
        # fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

        # ax4a = axes4[0]
        # awsom_idx = np.array([p[0] for p in results['path']])
        # unique_awsom, counts = np.unique(awsom_idx, return_counts=True)
        # ax4a.bar(unique_awsom[:100], counts[:100], alpha=0.7, width=1.0)
        # ax4a.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ideal (1:1)')
        # ax4a.set_xlabel('AWSoM Index')
        # ax4a.set_ylabel('Number of MAVEN mappings')
        # ax4a.set_title(f'Overmapping Analysis: {var_name}')
        # ax4a.legend()
        # ax4a.grid(True, alpha=0.3)

        # ax4b = axes4[1]
        # maven_idx = np.array([p[1] for p in results['path']])
        # norm_awsom = awsom_idx / max(awsom_idx) if max(awsom_idx) > 0 else awsom_idx
        # norm_maven = maven_idx / max(maven_idx) if max(maven_idx) > 0 else maven_idx

        # from matplotlib.cm import ScalarMappable
        # colors = counts[np.searchsorted(unique_awsom, awsom_idx)]
        # scatter = ax4b.scatter(norm_maven, norm_awsom, c=colors,
        #                       cmap='viridis', s=10, alpha=0.6)
        # plt.colorbar(scatter, ax=ax4b, label='Mapping density')

        # ax4b.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1.5)
        # ax4b.set_xlabel('Normalized MAVEN Time')
        # ax4b.set_ylabel('Normalized AWSoM Time')
        # ax4b.set_title(f'Warping with Overmapping Highlight (SSF={results["ssf_score"]:.3f})')
        # ax4b.grid(True, alpha=0.3)

        # plt.tight_layout()
        # plt.savefig(f'dtw_overmapping_{var_name}_w{optimal_window:.0f}h.png', dpi=150, bbox_inches='tight')
        # plt.show()

        # # Plot 4: Overmapping analysis (CORRECTED)
        # fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))

        # # Left: Bar chart of MAVEN mappings per AWSoM point
        # ax4a = axes4[0]
        # awsom_indices, maven_counts = plot_overmapping_corrected(
        #     results['path'],
        #     ax4a,
        #     max_points=100,
        #     title=f'Overmapping Analysis: {var_name}'
        # # )

        # # Add statistics text box
        # stats_text = (
        #     f"Total AWSoM points: {len(set([p[0] for p in results['path']]))}\n"
        #     f"Avg MAVEN mappings per AWSoM: {np.mean(maven_counts):.2f}\n"
        #     f"Max MAVEN mappings: {np.max(maven_counts)}\n"
        #     f"Overmapping ratio: {np.mean(maven_counts) - 1.0:.3f}"
        # )
        # ax4a.text(0.98, 0.95, stats_text, transform=ax4a.transAxes,
        #           verticalalignment='top', horizontalalignment='right',
        #           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        #           fontsize=10)

    #     # Right: Warping path with overmapping colors
    #     ax4b = axes4[1]
    #     plot_warping_path_with_overmapping(
    #         results['path'],
    #         ax4b,
    #         title='Warping Path (Color = Overmapping Level)'
    #     )

    #     plt.tight_layout()
    #     plt.savefig(f'dtw_overmapping_{var_name}_corrected.png', dpi=150, bbox_inches='tight')
    #     plt.show()

    #     plot_reference_comparison(maven_original, awsom_original, times, var_name)

    # if len(all_results) > 1:
    #     plot_metrics_comparison(all_results)

    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Variable':<20} {'MSE Skill':<12} {'SSF':<12} {'Agreement':<15}")
    print(f"{'-'*74}")
    for r in all_results:
        mse = r['mse_skill_score']
        ssf = r['ssf_score']
        # window = r['optimal_window']
        agreement = "✓ AGREE" if ((mse < 1 and ssf < 1) or (mse > 1 and ssf > 1)) else "DISAGREE"
        print(f"{r['variable_name']:<20} {mse:<12.4f} {ssf:<12.4f} {agreement:<15}")

    return all_results


# ===========================================================
#    EXECUTION WITH EQUAL INTERVAL CONNECTIONS
# ===========================================================
three_day_window = int(72*3600 / dt)
four_day_window = int(96*3600 / dt)
print("Starting OPTIMIZED DTW analysis with EQUAL INTERVAL connections...")
print("="*70)

print("\nAvailable variables in filtered data:")
print(list(filtered.keys()))

# variable_names_three_days = ['Temperature']
variable_names_four_days = ['Velocity', 'Temperature', 'Density', 'Bmag']
variable_names_to_analyze = variable_names_four_days

# results_with_metrics_three_days = analyze_awsom_maven_dtw(
#     filtered_data=filtered_for_dtw,
#     awsom_times=awsom_times,
#     variable_names=variable_names_three_days,
#     fixed_window=three_day_window,
#     max_connections=120     # For selective connections
# )

results_with_metrics_four_days = analyze_awsom_maven_dtw(
    filtered_data=filtered_for_dtw,
    awsom_times=awsom_times,
    variable_names=variable_names_four_days,
    fixed_window=four_day_window,
    max_connections=120
)

print("\n" + "="*70)
print("ANALYSIS COMPLETE - EQUAL INTERVAL CONNECTIONS")
print("="*70)
print("\nChanges applied:")
print("  ✅ ALL connections now at EQUAL INTERVALS (not importance-based)")
print("  ✅ Selective plot: 120 equally spaced connections")
print("  ✅ Clean plot: 250 equally spaced connections")
print("  ✅ Consistent line thickness (2.0-2.5)")
print("\nVariables analyzed:")
for var in variable_names_to_analyze:
    print(f"  ✓ {var}")
print("\nFiles saved:")
print("- window_convergence_*.png")
print("- window_convergence_summary_*.png")
print("- dtw_matrix_*_w*.png")
print("- dtw_alignment_*_selective_w*.png (120 EQUAL intervals)")
print("- dtw_alignment_*_clean_equal_w*.png (250 EQUAL intervals)")
print("- dtw_overmapping_*_w*.png")
print("- reference_comparison_*.png")
print("- ssf_vs_mse_comparison.png")
print("="*70)

# ===========================================================
# HISTOGRAM PLOTTING FOR DTW TIME AND AMPLITUDE DIFFERENCES
# (Section 4.2 of the paper) 
# ===========================================================

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def plot_dtw_histograms(awsom_series, maven_series, path,
                        dt=600.0,  # 10-minute cadence in seconds
                        variable_name="Variable",
                        units="km/s",
                        max_time_diff_days=10,
                        min_time_diff_days=-10,
                        max_amp_diff=10,
                        min_amp_diff=-10,
                        bins=100,
                        save=True):
    """
    Plot histograms of time and amplitude differences between DTW-aligned points.
    ALL FONTS ARE 4X LARGER FOR POSTER READABILITY.
    """
    # Extract indices from warping path
    awsom_idx = np.array([p[0] for p in path])
    maven_idx = np.array([p[1] for p in path])

    # Get aligned values
    awsom_aligned = awsom_series[awsom_idx]
    maven_aligned = maven_series[maven_idx]

    # ========== 1. Calculate Time Differences ==========
    time_diffs_seconds = (awsom_idx - maven_idx) * dt
    time_diffs_days = time_diffs_seconds / (3600.0 * 24)

    # ========== 2. Calculate Amplitude Differences ==========
    amp_diffs = awsom_aligned - maven_aligned

    # ========== 3. Create the Figure with LARGER SIZE ==========
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # ---- Histogram 1: Time Differences ----
    ax1 = axes[0]

    # Determine text box position for time differences
    if variable_name in ['Density', 'Bmag']:
        text_x_time = 0.02
        ha_time = 'left'
    else:
        text_x_time = 0.98
        ha_time = 'right'

    bin_edges_time = np.linspace(min_time_diff_days, max_time_diff_days, bins)
    n, bins_time, patches = ax1.hist(time_diffs_days, bins=bin_edges_time,
                                     color='steelblue', edgecolor='black',
                                     alpha=0.7, linewidth=0.5)

    # Vertical lines
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=3, alpha=0.7, label='Perfect alignment')
    mean_time = np.mean(time_diffs_days)
    ax1.axvline(x=mean_time, color='blue', linestyle='-', linewidth=3, alpha=0.7,
                label=f'Mean: {mean_time:.2f} days')
    median_time = np.median(time_diffs_days)
    ax1.axvline(x=median_time, color='green', linestyle='-', linewidth=3, alpha=0.7,
                label=f'Median: {median_time:.2f} days')

    # Statistics text box - Time differences
    stats_text = (f"Std: {np.std(time_diffs_days):.2f} days\n"
                  f"Max: {np.max(np.abs(time_diffs_days)):.2f} days")
    ax1.text(text_x_time, 0.5, stats_text, transform=ax1.transAxes,
             verticalalignment='center', horizontalalignment=ha_time,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=2),
             fontsize=20)

    # Labels and title 
    ax1.set_xlabel(f'Time Difference (Days)\n(positive = AWSoM arrives later)',
                   fontsize=32, fontweight='bold')
    ax1.set_ylabel('Number of Matched Points', fontsize=30, fontweight='bold')
    ax1.set_title(f'Time Differences: {variable_name}',
                  fontsize=32, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=28)
    ax1.legend(loc='best', fontsize=20)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim(min_time_diff_days, max_time_diff_days)

    # ---- Histogram 2: Amplitude Differences ----
    ax2 = axes[1]
    if variable_name == 'Velocity':
        text_x_amp = 0.98
        text_y_amp = 0.95
        va_amp = 'top'
        ha_amp = 'right'
    elif variable_name in ['Density', 'Bmag']:
        text_x_amp = 0.02
        text_y_amp = 0.5
        va_amp = 'center'
        ha_amp = 'left'
    else:  # Temperature and others
        text_x_amp = 0.98
        text_y_amp = 0.5
        va_amp = 'center'
        ha_amp = 'right'

    bin_edges_amp = np.linspace(min_amp_diff, max_amp_diff, bins)
    n, bins_amp, patches = ax2.hist(amp_diffs, bins=bin_edges_amp,
                                     color='coral', edgecolor='black',
                                     alpha=0.7, linewidth=0.5)

    # Vertical lines
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=3, alpha=0.7, label='Perfect agreement')
    mean_amp = np.mean(amp_diffs)
    ax2.axvline(x=mean_amp, color='blue', linestyle='-', linewidth=3, alpha=0.7,
                label=f'Mean: {mean_amp:.2f} {units}')
    median_amp = np.median(amp_diffs)
    ax2.axvline(x=median_amp, color='green', linestyle='-', linewidth=3, alpha=0.7,
                label=f'Median: {median_amp:.2f} {units}')

    # Statistics text box - Amplitude differences
    stats_text = (f"Std: {np.std(amp_diffs):.2f} {units}\n"
                  f"RMSE: {np.sqrt(np.mean(amp_diffs**2)):.2f} {units}")
    ax2.text(text_x_amp, text_y_amp, stats_text, transform=ax2.transAxes,
             verticalalignment=va_amp, horizontalalignment=ha_amp,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=2),
             fontsize=20)

    # Labels and title 
    ax2.set_xlabel(f'Amplitude Difference ({units})\n(positive = AWSoM higher than MAVEN)',
                   fontsize=32, fontweight='bold')
    ax2.set_ylabel('Number of Matched Points', fontsize=30, fontweight='bold')
    ax2.set_title(f'Amplitude Differences: {variable_name}',
                  fontsize=32, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=28)
    ax2.legend(loc='best', fontsize=20)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xlim(min_amp_diff, max_amp_diff)

    # Overall title 
    fig.suptitle(f'DTW Alignment Statistics: {variable_name}\n(n = {len(path)} matched points)',
                 fontsize=48, fontweight='bold')

    plt.tight_layout()

    if save:
        plt.savefig(f'dtw_histograms_{variable_name}.png', dpi=150, bbox_inches='tight')

    plt.show()

    # ========== 4. Print Summary Statistics ==========
    print(f"\n{'='*60}")
    print(f"DTW Alignment Statistics: {variable_name}")
    print(f"{'='*60}")
    print(f"Number of matched points: {len(path)}")
    print(f"\nTime Differences:")
    print(f"  Mean: {mean_time:.2f} days")
    print(f" Median: {median_time:.2f} days")
    print(f"  Std:  {np.std(time_diffs_days):.2f} days")
    print(f"  Min:  {np.min(time_diffs_days):.2f} days")
    print(f"  Max:  {np.max(time_diffs_days):.2f} days")
    print(f"\nAmplitude Differences ({units}):")
    print(f"  Mean: {mean_amp:.2f} {units}")
    print(f" Median: {median_amp:.2f} {units}")
    print(f"  Std:  {np.std(amp_diffs):.2f} {units}")
    print(f"  Min:  {np.min(amp_diffs):.2f} {units}")
    print(f"  Max:  {np.max(amp_diffs):.2f} {units}")
    print(f"  RMSE: {np.sqrt(np.mean(amp_diffs**2)):.2f} {units}")

    return {'time_diffs': time_diffs_days, 'amp_diffs': amp_diffs}


def plot_all_histograms(all_results, variable_names=None,
                        dt=600.0, max_time_diff_days=10,
                        min_time_diff_days=-10, max_amp_diff=10,
                        min_amp_diff=-5, bins=100):
    """
    Plot histograms for multiple variables with 4X LARGER fonts.
    """
    if variable_names is None:
        variable_names = [r['variable_name'] for r in all_results]

    for var_name in variable_names:
        result = None
        for r in all_results:
            if r['variable_name'] == var_name:
                result = r
                break

        if result is None:
            print(f"Warning: No results found for {var_name}")
            continue

        awsom_original = result['awsom_original']
        maven_original = result['maven_original']
        path = result['path']

        units_map = {
            'Velocity': 'km/s',
            'Density': 'cm$^{-3}$',
            'Temperature': 'K',
            'Bmag': 'nT',
            'B': 'nT',
            'MagneticField': 'nT'
        }
        units = units_map.get(var_name, 'units')

        plot_dtw_histograms(
            awsom_original, maven_original, path,
            dt=dt,
            variable_name=var_name,
            units=units,
            max_time_diff_days=max_time_diff_days,
            min_time_diff_days=min_time_diff_days,
            max_amp_diff=max_amp_diff,
            min_amp_diff=min_amp_diff,
            bins=bins,
            save=True
        )


# ===========================================================
# EXECUTION
# ===========================================================

# If already have results, generate histograms 
results_with_metrics = results_with_metrics_four_days

# Velocity (Amplitude: Top Right, Time: Center Right)
plot_all_histograms(
    results_with_metrics,
    variable_names=['Velocity'],
    dt=600.0,
    max_time_diff_days=4,
    min_time_diff_days=-5,
    max_amp_diff=250,
    min_amp_diff=-10,
    bins=100
)

# Density (Amplitude: Center Left, Time: Center Left)
plot_all_histograms(
    results_with_metrics,
    variable_names=['Density'],
    dt=600.0,
    max_time_diff_days=2,
    min_time_diff_days=-5,
    max_amp_diff=2,
    min_amp_diff=-3,
    bins=100
)

# Temperature (Amplitude: Center Right, Time: Center Right)
plot_all_histograms(
    results_with_metrics,
    variable_names=['Temperature'],
    dt=600.0,
    max_time_diff_days=4,
    min_time_diff_days=-5,
    max_amp_diff=0,
    min_amp_diff=-6,
    bins=100
)

# Bmag (Amplitude: Center Left, Time: Center Left)
plot_all_histograms(
    results_with_metrics,
    variable_names=['Bmag'],
    dt=600.0,
    max_time_diff_days=5,
    min_time_diff_days=-4,
    max_amp_diff=0,
    min_amp_diff=-3,
    bins=100
)
