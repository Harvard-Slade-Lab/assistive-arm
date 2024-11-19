import os
import numpy as np
import pandas as pd
import yaml
import bayes_opt
from chspy import CubicHermiteSpline
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from scipy.optimize import NonlinearConstraint
import matplotlib.pyplot as plt


# Define the path for saving/loading optimization progress
SAVE_PATH = "/Users/nathanirniger/Desktop/Bayes/bayesian_optimization_progress.json"

# Track scores for real-time visualization
score_history = []

def cubic_hermite_spline(points):
    spline = CubicHermiteSpline(n=1)
    for t, value, derivative in points:
        spline.add((t, [value], [derivative]))
    return spline


def validate_constraints(force1_end_time, force2_start_time, force2_peak_time, force2_end_time):
    if force2_end_time <= force2_start_time:
        return False
    if force1_end_time <= force2_peak_time:
        return False
    if force2_peak_time <= force2_start_time:
        return False
    return True


def objective(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time):

    # Constraint: Ensure valid parameter relationships
    if not validate_constraints(force1_end_time, force2_start_time, force2_peak_time, force2_end_time):
        return -1  # Penalty for violating constraints

    base_profile = get_profile(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time)
    # score = get_score(base_profile)
    score = easy_score(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time)

    # Log scores for visualization
    score_history.append(score)
    # Log results
    if score > -0.09:
        save_optimization_results(base_profile, score, force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time)

    return score


def save_optimization_results(base_profile, score, force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time):
    try:
        os.makedirs("/Users/nathanirniger/Desktop/profiles/optim/plots", exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")

    # Save the profile as CSV
    profile_path = f"/Users/nathanirniger/Desktop/profiles/optim/score_{score}_profile.csv"
    base_profile.to_csv(profile_path, index=False)

    # Plot the profile and save the figure
    target_profile = pd.read_csv("/Users/nathanirniger/Desktop/profiles/reference_profile_2.csv")
    plt.figure(figsize=(10, 6))
    plt.plot(base_profile["force_X"], label="Optimized Profile X Force")
    plt.plot(base_profile["force_Y"], label="Optimized Profile Y Force")
    plt.plot(target_profile["force_X"], label="Target Profile X Force", linestyle="--")
    plt.plot(target_profile["force_Y"], label="Target Profile Y Force", linestyle="--")
    plt.legend()
    plt.title(f"Score: {score}")
    plt.savefig(f"/Users/nathanirniger/Desktop/profiles/optim/plots/score_{score}_profile_t11_{force1_end_time}_f1_{force1_peak_force}_t21_{force2_start_time}_t22_{force2_peak_time}_f21_{force2_peak_force}_t23_{force2_end_time}.png")
    plt.close()


def get_profile(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time):
    max_force = 65  # Maximum force for the profile
    max_time = 360  # Maximum time for the profile

    force1_end_time = force1_end_time * max_time
    force1_peak_force = force1_peak_force * max_force

    force2_start_time = force2_start_time * max_time
    force2_peak_time = force2_peak_time * max_time
    force2_peak_force = force2_peak_force * max_force
    force2_end_time = force2_end_time * max_time

    # Load calibration data
    calibration_path = "/Users/nathanirniger/Desktop/profiles/device_height_calibration.yaml"
    with open(calibration_path, 'r') as file:
        data = yaml.safe_load(file)

    length = len(data['theta_2_values'])
    base_profile = pd.DataFrame({"force_x": np.zeros(length), "force_y": np.zeros(length), "theta_2": data['theta_2_values']})

    # X Force Profile
    grf_x = cubic_hermite_spline([(0, 0, 0), (force1_end_time / 2, force1_peak_force, 0), (force1_end_time, 0, 0)])
    curve_x = [grf_x.get_state(i)[0] for i in range(int(np.round(force1_end_time)))]
    padded_curve_x = np.concatenate([curve_x, np.zeros(length - len(curve_x))])
    
    # Y Force Profile
    grf_y = cubic_hermite_spline([(0, 0, 0), (force2_peak_time - force2_start_time, force2_peak_force, 0), (force2_end_time - force2_start_time, 0, 0)])
    curve_y = [grf_y.get_state(i)[0] for i in range(int(np.round(force2_end_time - force2_start_time)))]
    padded_curve_y = np.concatenate([np.zeros(int(np.round(force2_start_time))), curve_y, np.zeros(length - len(curve_y) - int(np.round(force2_start_time)))])

    base_profile["force_X"] = padded_curve_x
    base_profile["force_Y"] = padded_curve_y

    return base_profile


def plot_scores():
    plt.figure(figsize=(10, 6))
    plt.plot(score_history, label="Score History")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Optimization Score Trend")
    plt.legend()
    plt.show()


def easy_score(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time):
    force1_end_time_target = 0.8
    force1_peak_force_target = 0.5

    force2_start_time_target = 0.2
    force2_peak_time_target = 0.5
    force2_peak_force_target = 0.8
    force2_end_time_target = 0.7

    # score = np.abs(force1_end_time - force1_end_time_target) + np.abs(force1_peak_force - force1_peak_force_target) + np.abs(force2_start_time - force2_start_time_target) + np.abs(force2_peak_time - force2_peak_time_target) + np.abs(force2_peak_force - force2_peak_force_target) + np.abs(force2_end_time - force2_end_time_target)
    score = (force1_end_time - force1_end_time_target)**2 + (force1_peak_force - force1_peak_force_target)**2 + (force2_start_time - force2_start_time_target)**2 + (force2_peak_time - force2_peak_time_target)**2 + (force2_peak_force - force2_peak_force_target)**2 + (force2_end_time - force2_end_time_target)**2
    return -score

# Define parameter bounds
pbounds = {
    "force1_end_time": (0.0, 1.0),        # End time for force1
    "force1_peak_force": (0.0, 1.0),      # Ratio for force1 peak time
    "force2_start_time": (0.0, 1.0),      # Start time for force2
    "force2_peak_time": (0.0, 1.0),       # Peak time for force2
    "force2_peak_force": (0.0, 1.0),      # Peak force for force2
    "force2_end_time": (0.0, 1.0)         # End time for force2
}

def load_optimizer(kappa):
    # constraint = NonlinearConstraint(constraint_func, lb=-1, ub=0)
    
    acquisition = bayes_opt.acquisition.UpperConfidenceBound(kappa=kappa)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,                   # Parameter bounds
        acquisition_function=acquisition,  # Acquisition function: Upper Confidence Bound
        # constraint=constraint,             # Constraint
        random_state=0,                    # Random seed accepts integer value and is used for reproducibility.
        verbose=2                          # Verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )
    
    if os.path.exists(SAVE_PATH):
        load_logs(optimizer, logs=[SAVE_PATH])
        print("Loaded optimization progress from file.")
    else:
        print("No saved progress found. Starting a new optimization.")
        
    return optimizer

# Save progress using a JSONLogger
def save_progress(optimizer):
    logger = JSONLogger(path=SAVE_PATH)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)


def explorate(optimizer, init_points=10, n_iter=0):
    optimizer.maximize(
        init_points=init_points,  # Number of random initial points
        n_iter=n_iter,            # Number of optimization iterations
    )

def optimize(optimizer, init_points=0, n_iter=1):
    optimizer.maximize(
        init_points=init_points,  # Number of random initial points
        n_iter=n_iter,            # Number of optimization iterations
    )

# Main function to coordinate the optimization
def main():
    optimizer = load_optimizer(kappa=2.5)
    save_progress(optimizer)

    # Run initial exploration
    explorate(optimizer)

    # Run optimization
    for i in range(100):
        optimize(optimizer)
    
    # Output the best parameters and score
    print("Best parameters found:", optimizer.max["params"])
    print("Best score:", optimizer.max["target"])

    # Plot the optimization score trend
    plot_scores()

# Run the main function
if __name__ == "__main__":
    main()











# def constraint_func(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time):
#     # Check each constraint and return 1 if violated
#     if force2_end_time <= force2_start_time:
#         return 1  # Violation of force2_end_time > force2_start_time
#     elif force1_end_time <= force2_peak_time:
#         return 1  # Violation of force1_end_time > force2_peak_time
#     elif force2_peak_time <= force2_start_time:
#         return 1  # Violation of force2_peak_time > force2_start_time
#     else:
#         return -1  # All constraints satisfied


