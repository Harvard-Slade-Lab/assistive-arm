



# def save_optimization_results(base_profile, score, force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time):
#     try:
#         os.makedirs("/Users/nathanirniger/Desktop/profiles/optim/plots", exist_ok=True)
#     except Exception as e:
#         print(f"Error creating directories: {e}")

#     # Save the profile as CSV
#     profile_path = f"/Users/nathanirniger/Desktop/profiles/optim/score_{score}_profile.csv"
#     base_profile.to_csv(profile_path, index=False)

#     # Plot the profile and save the figure
#     target_profile = pd.read_csv("/Users/nathanirniger/Desktop/profiles/reference_profile_3.csv")
#     plt.figure(figsize=(10, 6))
#     plt.plot(base_profile["force_X"], label="Optimized Profile X Force")
#     plt.plot(base_profile["force_Y"], label="Optimized Profile Y Force")
#     plt.plot(target_profile["force_X"], label="Target Profile X Force", linestyle="--")
#     plt.plot(target_profile["force_Y"], label="Target Profile Y Force", linestyle="--")
#     plt.legend()
#     plt.title(f"Score: {score}")
#     plt.savefig(f"/Users/nathanirniger/Desktop/profiles/optim/plots/score_{score}_profile_t11_{force1_end_time}_f1_{force1_peak_force}_t21_{force2_start_time}_t22_{force2_peak_time}_f21_{force2_peak_force}_t23_{force2_end_time}.png")
#     plt.close()

# def plot_scores():
#     plt.figure(figsize=(10, 6))
#     plt.plot(score_history, label="Score History")
#     plt.xlabel("Iteration")
#     plt.ylabel("Score")
#     plt.title("Optimization Score Trend")
#     plt.legend()
#     plt.show()




import numpy as np
import pandas as pd
import yaml
import os
from chspy import CubicHermiteSpline
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from scipy.optimize import NonlinearConstraint
import bayes_opt.acquisition


class ForceProfileOptimizer:
    def __init__(self, calibration_path, save_path, kappa, max_force=65, max_time=360):
        self.calibration_path = calibration_path
        self.save_path = save_path
        self.max_force = max_force
        self.max_time = max_time
        self.score_history = []
        self.kappa = kappa
        self.pbounds = {
            "force1_end_time_p": (0.0, 1.0),        # End time for force1
            "force1_peak_force_p": (0.0, 1.0),      # Ratio for force1 peak time
            "force2_start_time_p": (0.0, 1.0),      # Start time for force2
            "force2_peak_time_p": (0.0, 1.0),       # Peak time for force2
            "force2_peak_force_p": (0.0, 1.0),      # Peak force for force2
            "force2_end_time_p": (0.0, 1.0)         # End time for force2
        }
        self.optimizer = None
        
        self.load_optimizer()

    @staticmethod
    def cubic_hermite_spline(points):
        spline = CubicHermiteSpline(n=1)
        for t, value, derivative in points:
            spline.add((t, [value], [derivative]))
        return spline

    @staticmethod
    def validate_constraints(force2_start_time, force2_peak_time, force2_end_time):
        return (
            force2_end_time > force2_start_time and
            force2_end_time > force2_peak_time and
            force2_peak_time > force2_start_time
        )

    def get_profile(self, force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time):
        with open(self.calibration_path, 'r') as file:
            data = yaml.safe_load(file)

        length = len(data['theta_2_values'])
        base_profile = pd.DataFrame({"force_X": np.zeros(length), "force_Y": np.zeros(length), "theta_2": data['theta_2_values']})

        # X Force Profile
        grf_x = self.cubic_hermite_spline([(0, 0, 0), (force1_end_time / 2, force1_peak_force, 0), (force1_end_time, 0, 0)])
        curve_x = [grf_x.get_state(i)[0] for i in range(int(np.round(force1_end_time)))]
        padded_curve_x = np.concatenate([curve_x, np.zeros(length - len(curve_x))])

        # Y Force Profile
        grf_y = self.cubic_hermite_spline([(0, 0, 0), (force2_peak_time - force2_start_time, force2_peak_force, 0), (force2_end_time - force2_start_time, 0, 0)])
        curve_y = [grf_y.get_state(i)[0] for i in range(int(np.round(force2_end_time - force2_start_time)))]
        padded_curve_y = np.concatenate([np.zeros(int(np.round(force2_start_time))), curve_y, np.zeros(length - len(curve_y) - int(np.round(force2_start_time)))])

        base_profile["force_X"] = padded_curve_x
        base_profile["force_Y"] = padded_curve_y

        return base_profile

    def objective(self, force1_end_time_p, force1_peak_force_p, force2_start_time_p, force2_peak_time_p, force2_peak_force_p, force2_end_time_p):
        force1_end_time = force1_end_time_p * self.max_time
        force1_peak_force = force1_peak_force_p * self.max_force

        # Standard with check later on and returning -1 if constraints are violated
        # force2_start_time = force2_start_time_p * self.max_time 
        # force2_peak_time = force2_peak_time_p * self.max_time
        # force2_peak_force = force2_peak_force_p * self.max_force
        # force2_end_time = force2_end_time_p * self.max_time

        # Constrain the time values to be within the range of 0.25 to 0.75 of the max time
        # force2_start_time = force2_start_time_p * self.max_time * 0.25
        # force2_peak_time = force2_peak_time_p * self.max_time * 0.5 + 0.25 * self.max_time
        # force2_peak_force = force2_peak_force_p * self.max_force
        # force2_end_time = force2_end_time_p * self.max_time * 0.25 + 0.75 * self.max_time

        # Dynamic constraints
        force2_peak_time = force2_peak_time_p * self.max_time * 0.8 + 0.1 * self.max_time
        force2_start_time = force2_peak_time * force2_start_time_p
        force2_end_time = force2_peak_time + force2_end_time_p * (self.max_time - force2_peak_time)
        force2_peak_force = force2_peak_force_p * self.max_force

        if not self.validate_constraints(force2_start_time, force2_peak_time, force2_end_time):
            return -1  # Penalize invalid constraints

        base_profile = self.get_profile(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time)
        # score = easy_score(force1_end_time_p, force1_peak_force_p, force2_start_time_p, force2_peak_time_p, force2_peak_force_p, force2_end_time_p)

        self.score_history.append(score)
        # if score > -0.05:
        #     save_optimization_results(base_profile, score, force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time)

        return score

    def load_optimizer(self):
        acquisition = bayes_opt.acquisition.UpperConfidenceBound(kappa=self.kappa)
        optimizer = BayesianOptimization(
            f=self.objective,
            pbounds=self.pbounds,
            acquisition_function=acquisition,
            random_state=0,
            verbose=2
        )
        if os.path.exists(self.save_path):
            load_logs(optimizer, logs=[self.save_path])
        self.optimizer = optimizer

    def save_progress(self, optimizer):
        logger = JSONLogger(path=self.save_path)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    def explorate(self, optimizer, init_points=10, n_iter=0):
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

    def optimize(self, optimizer, init_points=0, n_iter=1):
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
