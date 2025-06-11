import os
import json
import numpy as np
import matplotlib.pyplot as plt  # Fixed import
from utils_visualization import read_sto_file, apply_low_pass
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks
from cma_toolbox import *

# ========== Session parameters ==========
date = "20250603_004908_assisted_converged"
subj = "S001"
task = "sit-to-stand_4"
body_weight = 75 * 9.81  # N
max_robot_force = 50 # N # maximum force that robot can provide

# ========== Paths ==========
sim_path = f'./sim_result/{subj}_{task}_{date}'
fig_title = f'{subj}_{task}_{date}'

# ========== Helper Functions ==========
def add_time_percentage(df):
    """Add a time percentage [0-100%] column."""
    total_time = df['time'].max() - df['time'].min()
    df['time_perc'] = ((df['time'] - df['time'].min()) / total_time) * 100
    return df

def apply_low_pass_filter(data, cutoff, order, fs):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist       # normalized cutoff
    b, a = butter(N=order, Wn=normal_cutoff, btype='low', analog=False)
    data_filtered = filtfilt(b, a, data)  # zero-phase filtering
    return data_filtered

# ========== Load Simulation Results ==========
df_sim_sol = None
sim_params = None

for cur_file in os.listdir(sim_path):
    cur_file_path = os.path.join(sim_path, cur_file)
    if cur_file.endswith('.sto') and 'grf' not in cur_file and df_sim_sol is None:
        df_sim_sol = read_sto_file(cur_file_path)
    elif cur_file.endswith('.json') and sim_params is None:
        with open(cur_file_path, 'r') as file:
            sim_params = json.load(file)

if df_sim_sol is None:
    raise FileNotFoundError("No simulation solution (.sto) file found in path!")
if sim_params is None:
    raise FileNotFoundError("No parameter (.json) file found in path!")

# ========== Post-processing ==========
df_sim_sol = add_time_percentage(df_sim_sol)
raw_perc = df_sim_sol['time_perc'].values
assist_x_force_raw = df_sim_sol['/forceset/assistive_force_x'].values * sim_params['opt_force_robot_assistance']
assist_y_force_raw = df_sim_sol['/forceset/assistive_force_y'].values * sim_params['opt_force_robot_assistance']

# Step 1: Interpolate 200Hz
# Generate new time array (from min to max, spaced at 0.005s intervals = 200Hz)
num_samples = 200
new_perc = np.linspace(raw_perc[0], raw_perc[-1], num=num_samples)

# Use cubic interpolation for smoothness (you may use 'linear' if you prefer)
fx_interp = interp1d(raw_perc, assist_x_force_raw, kind='cubic')
fy_interp = interp1d(raw_perc, assist_y_force_raw, kind='cubic')

assist_x_force_intp = fx_interp(new_perc)
assist_y_force_intp = fy_interp(new_perc)

# Step 2: Low pass filter the interpolated force profile
# ========== Low-pass filter specs ==========
lpf_order = 4
lpf_cutoff = 6  # Hz

fs = 200  # sampling freq in Hz (because that's what we interpolated to)
assist_x_force_lpf = apply_low_pass_filter(assist_x_force_intp, lpf_cutoff, lpf_order, fs)
assist_y_force_lpf = apply_low_pass_filter(assist_y_force_intp, lpf_cutoff, lpf_order, fs)

# Step 3: Non_negative constraints
# Clip the forces to the robot's limits
assist_x_force_clipped = np.clip(assist_x_force_lpf, 0, np.inf)
assist_y_force_clipped = np.clip(assist_y_force_lpf, 0, np.inf)

# plotting
figsize = (10,4)
fontsize = 10
fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)

# X direction
axs[0].plot(new_perc, assist_x_force_intp, '--', alpha=1, color='grey', label='intp (200Hz)')
axs[0].plot(new_perc, assist_x_force_lpf, '-', alpha=1, color='red', label='LPF (6Hz)')
axs[0].plot(new_perc, assist_x_force_clipped, '-', alpha=0.5, color='k', label='clipped')
axs[0].set_title(f'Sim Assistive Force X', fontsize=fontsize)
axs[0].set_xlabel('Task completion (%)', fontsize=fontsize)
axs[0].set_ylabel('Force (N)', fontsize=fontsize)
axs[0].legend(frameon=False, loc='upper right')
axs[0].tick_params(labelsize=fontsize)

axs[1].plot(new_perc, assist_y_force_intp, '--', alpha=1, color='grey', label='intp (200Hz)')
axs[1].plot(new_perc, assist_y_force_lpf, '-', alpha=1, color='red', label='LPF (6Hz)')
axs[1].plot(new_perc, assist_y_force_clipped, '-', alpha=0.5, color='k', label='clipped')
axs[1].set_title(f'Sim Assistive Force Y', fontsize=fontsize)
axs[1].set_xlabel('Task completion (%)', fontsize=fontsize)
axs[1].set_ylabel('Force (N)', fontsize=fontsize)
axs[1].legend(frameon=False, loc='upper right')
axs[1].tick_params(labelsize=fontsize)

# Remove top and right spines
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
# exit()

def parameterize_force_profile_x(x_data, y_data):

    from scipy.interpolate import CubicHermiteSpline
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    # ======= 2. Hermite spline RMSE cost function (with zero tangents) ======
    # Hermite spline RMSE cost function with safety masking
    def hermite_cost(params, x_data, y_data):
        t_points = [params[0], params[2], params[4]] # time
        f_points = [params[1], params[3], params[5]] # force
        
        # ----- Constraints ----- #
        # Ensure strictly increasing knots # t3>t2>t1
        if not np.all(np.diff(t_points) > 0):
            return 1e12

        # Ensure the interval between t1 and t3 is at least 20%
        if params[4] - params[0] < 20:
            return 1e12

        m_points = np.zeros(3)
        spline = CubicHermiteSpline(t_points, f_points, m_points, extrapolate=True)
        y_fit = spline(x_data)
        
        # Mask: zero outside [t1, t3]
        t1, t3 = t_points[0], t_points[-1]
        y_fit = np.where((x_data < t1) | (x_data > t3), 0, y_fit)
        
        # Enforce non-negativity
        rmse = np.sqrt(np.mean((y_data - y_fit) ** 2))
        return rmse

    # ======= 3. Initial guess and bounds =========
    init_params = [0, 0, 33, max(y_data), 66, 0]  # [t1, f1, t2, f2, t3, f3, t4, f4]
    bounds = [
        (0, 0),    # t1
        (0, 0),    # f1
        (0, 80),    # t2
        (0, max(y_data)), # f2
        (0, 100),    # t3
        (0, 0),    # f3
    ]

    # ======= 4. Optimization =========
    result = minimize(
        hermite_cost,
        init_params,
        args=(x_data, y_data),
        bounds=bounds,
        method='L-BFGS-B',  # Good for box-constrained problems
    )

    print("Optimal parameters:", result.x)
    print("Minimum RMSE:", result.fun)

    # ======= 5. Plotting the result =========
    t_points_opt = [result.x[0], result.x[2], result.x[4]]
    f_points_opt = [result.x[1], result.x[3], result.x[5]]
    spline_opt = CubicHermiteSpline(t_points_opt, f_points_opt, np.zeros(3), extrapolate=True)
    y_fit = spline_opt(x_data)

    # Clipping for knot range
    t1 = t_points_opt[0]
    t4 = t_points_opt[-1]
    y_fit = np.where((x_data < t1) | (x_data > t4), 0, y_fit)
    # Now also enforce non-negativity
    y_fit = np.clip(y_fit, 0, np.inf)
    rmse = np.round(np.sqrt(np.mean((y_data - y_fit) ** 2)),2)

    return t_points_opt, f_points_opt, y_fit, rmse

def parameterize_force_profile(x_data, y_data):

    from scipy.interpolate import CubicHermiteSpline
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    # ======= 2. Hermite spline RMSE cost function (with zero tangents) ======
    # Hermite spline RMSE cost function with safety masking
    def hermite_cost(params, x_data, y_data):
        t_points = [params[0], params[2], params[4]] # time
        f_points = [params[1], params[3], params[5]] # force
        
        # ----- Constraints ----- #
        # Ensure strictly increasing knots # t3>t2>t1
        if not np.all(np.diff(t_points) > 0):
            return 1e12

        # Ensure the interval between t1 and t3 is at least 20%
        if params[4] - params[0] < 20:
            return 1e12

        m_points = np.zeros(3)
        spline = CubicHermiteSpline(t_points, f_points, m_points, extrapolate=True)
        y_fit = spline(x_data)
        
        # Mask: zero outside [t1, t3]
        t1, t3 = t_points[0], t_points[-1]
        y_fit = np.where((x_data < t1) | (x_data > t3), 0, y_fit)
        
        # Enforce non-negativity
        rmse = np.sqrt(np.mean((y_data - y_fit) ** 2))
        return rmse

    # ======= 3. Initial guess and bounds =========
    init_params = [0, 0, 33, max(y_data), 66, 0]  # [t1, f1, t2, f2, t3, f3, t4, f4]
    bounds = [
        (0, 50),    # t1
        (0, 0),    # f1
        (0, 80),    # t2
        (0, max(y_data)), # f2
        (0, 100),    # t3
        (0, 0),    # f3
    ]

    # ======= 4. Optimization =========
    result = minimize(
        hermite_cost,
        init_params,
        args=(x_data, y_data),
        bounds=bounds,
        method='L-BFGS-B',  # Good for box-constrained problems
    )

    print("Optimal parameters:", result.x)
    print("Minimum RMSE:", result.fun)

    # ======= 5. Plotting the result =========
    t_points_opt = [result.x[0], result.x[2], result.x[4]]
    f_points_opt = [result.x[1], result.x[3], result.x[5]]
    spline_opt = CubicHermiteSpline(t_points_opt, f_points_opt, np.zeros(3), extrapolate=True)
    y_fit = spline_opt(x_data)

    # Clipping for knot range
    t1 = t_points_opt[0]
    t4 = t_points_opt[-1]
    y_fit = np.where((x_data < t1) | (x_data > t4), 0, y_fit)
    # Now also enforce non-negativity
    y_fit = np.clip(y_fit, 0, np.inf)
    rmse = np.round(np.sqrt(np.mean((y_data - y_fit) ** 2)),2)

    return t_points_opt, f_points_opt, y_fit, rmse    

fx_t_points, fx_f_points, fx_fit, rmse_x =  parameterize_force_profile(x_data= new_perc, y_data=assist_x_force_clipped)
fy_t_points, fy_f_points, fy_fit, rmse_y =  parameterize_force_profile(x_data= new_perc, y_data=assist_y_force_clipped)

figsize = (10,4)
fontsize = 10
fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)

# X direction
axs[0].plot(new_perc, assist_x_force_clipped, '-', alpha=0.5, color='k', label='clipped')
axs[0].plot(new_perc, fx_fit, '-', alpha=1, color='red', label='optimized')
axs[0].scatter(fx_t_points, fx_f_points, color='red', zorder=5, label='points')

axs[0].set_title(f'Sim Assistive Force X_RSME: {rmse_x}N', fontsize=fontsize)
axs[0].set_xlabel('Task completion (%)', fontsize=fontsize)
axs[0].set_ylabel('Force (N)', fontsize=fontsize)
axs[0].legend(frameon=False, loc='upper right')
axs[0].tick_params(labelsize=fontsize)

axs[1].plot(new_perc, assist_y_force_clipped, '-', alpha=0.5, color='k', label='clipped')
axs[1].plot(new_perc, fy_fit, '-', alpha=1, color='red', label='optimized')
axs[1].scatter(fy_t_points, fy_f_points, color='red', zorder=5, label='points')

axs[1].set_title(f'Sim Assistive Force Y_RSME: {rmse_y}N', fontsize=fontsize)
axs[1].set_xlabel('Task completion (%)', fontsize=fontsize)
axs[1].set_ylabel('Force (N)', fontsize=fontsize)
axs[1].legend(frameon=False, loc='upper right')
axs[1].tick_params(labelsize=fontsize)

# Remove top and right spines
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
plt.suptitle(fig_title)
plt.tight_layout()
plt.show()

print("=============================================================================\n")
print(f'Force x parameters: t1: {fx_t_points[0]:.2f}%, t2: {fx_t_points[1]:.2f}%, f2: {fx_f_points[1]/100: .2f}, t3: {fx_t_points[-1]:.2f}%')
print(f'Force y parameters: t1: {fy_t_points[0]:.2f}%, t2: {fy_t_points[1]:.2f}%, f2: {fy_f_points[1]/100: .2f}, t3: {fy_t_points[-1]:.2f}%')

exit()
