import os
import numpy as np
import pandas as pd
import yaml
from chspy import CubicHermiteSpline
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
import matplotlib.pyplot as plt

class BayesianForceOptimizer:
    SAVE_PATH = "/Users/nathanirniger/Desktop/Bayes/bayesian_optimization_progress.json"
    CALIBRATION_PATH = "/Users/nathanirniger/Desktop/profiles/device_height_calibration.yaml"
    TARGET_PROFILE_PATH = "/Users/nathanirniger/Desktop/profiles/reference_profile.csv"
    PROFILE_SAVE_DIR = "/Users/nathanirniger/Desktop/profiles/optim/"
    PROFILE_PLOT_DIR = os.path.join(PROFILE_SAVE_DIR, "plots/")
    
    def __init__(self, kappa=2.5):
        self.kappa = kappa
        self.pbounds = {
            "force1_end_time": (0.0, 360.0),
            "force1_peak_force": (0.0, 62.0),
            "force2_start_time": (0.0, 360.0),
            "force2_peak_time": (0.0, 360.0),
            "force2_peak_force": (0.0, 62.0),
            "force2_end_time": (0.0, 360.0),
        }
        self.optimizer = self._load_optimizer()

    def _load_optimizer(self):
        optimizer = BayesianOptimization(
            f=self._objective,
            pbounds=self.pbounds,
            random_state=0,
            verbose=2
        )
        if os.path.exists(self.SAVE_PATH):
            load_logs(optimizer, logs=[self.SAVE_PATH])
            print("Loaded optimization progress from file.")
        else:
            print("No saved progress found. Starting a new optimization.")
        return optimizer

    def save_progress(self):
        logger = JSONLogger(path=self.SAVE_PATH)
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    @staticmethod
    def _cubic_hermite_spline(points):
        spline = CubicHermiteSpline(n=1)
        for t, value, derivative in points:
            spline.add((t, [value], [derivative]))
        return spline

    def _generate_profile(self, force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time):
        with open(self.CALIBRATION_PATH, 'r') as file:
            data = yaml.safe_load(file)

        length = len(data['theta_2_values'])
        base_profile = pd.DataFrame({"force_X": np.zeros(length), "force_Y": np.zeros(length), "theta_2": data['theta_2_values']})

        # Force X
        spline_x = self._cubic_hermite_spline([(0, 0, 0), (force1_end_time / 2, force1_peak_force, 0), (force1_end_time, 0, 0)])
        curve_x = [spline_x.get_state(i)[0] for i in range(int(np.round(force1_end_time)))]
        base_profile["force_X"] = np.concatenate([curve_x, np.zeros(length - len(curve_x))])

        # Force Y
        spline_y = self._cubic_hermite_spline([(force2_start_time, 0, 0),(force2_peak_time, force2_peak_force, 0),(force2_end_time, 0, 0)])
        curve_y = [spline_y.get_state(i)[0] for i in range(int(force2_end_time - force2_start_time))]
        base_profile["force_Y"] = np.concatenate([np.zeros(int(force2_start_time)), curve_y, np.zeros(length - int(force2_end_time))])

        return base_profile

    def _compute_score(self, base_profile):
        target_profile = pd.read_csv(self.TARGET_PROFILE_PATH)
        force_similarity = np.mean(np.abs(base_profile["force_x"] - target_profile["force_X"])) + \
                           np.mean(np.abs(base_profile["force_y"] - target_profile["force_Y"]))
        return -force_similarity

    def _objective(self, force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time):
        if any([force2_end_time <= force2_start_time, force1_end_time <= force2_peak_time, force2_peak_time <= force2_start_time]):
            return -1e6

        base_profile = self._generate_profile(
            force1_end_time, force1_peak_force, force2_start_time,
            force2_peak_time, force2_peak_force, force2_end_time
        )
        score = self._compute_score(base_profile)
        
        if score > self.optimizer.max["target"]:
            file_name = f"score_{score}_profile_t11_{force1_end_time}_f1_{force1_peak_force}_t21_{force2_start_time}_t22_{force2_peak_time}_f21_{force2_peak_force}_t23_{force2_end_time}"
            base_profile.to_csv(os.path.join(self.PROFILE_SAVE_DIR, f"{file_name}.csv"), index=False)
            plt.plot(base_profile["force_x"], label="Optimized Profile X Force")
            plt.plot(base_profile["force_y"], label="Optimized Profile Y Force")
            plt.legend()
            plt.savefig(os.path.join(self.PROFILE_PLOT_DIR, f"{file_name}.png"))
            plt.clf()

        return score

    def explore(self, init_points=10):
        self.optimizer.maximize(init_points=init_points, n_iter=0)

    def optimize(self, n_iter=1):
        self.optimizer.maximize(init_points=0, n_iter=n_iter)

    def run(self, exploration_steps=10, optimization_steps=100):
        self.save_progress()
        self.explore(init_points=exploration_steps)
        for _ in range(optimization_steps):
            self.optimize(n_iter=1)

        print("Best parameters found:", self.optimizer.max["params"])
        print("Best score:", self.optimizer.max["target"])


if __name__ == "__main__":
    optimizer = BayesianForceOptimizer()
    optimizer.run()
