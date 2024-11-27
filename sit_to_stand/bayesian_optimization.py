import numpy as np
import pandas as pd
import yaml
import os
import time
import bayes_opt.acquisition

from chspy import CubicHermiteSpline
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from sts_control import apply_simulation_profile



class ForceProfileOptimizer:
    def __init__(self, motor_1, motor_2, kappa, freq, session_manager, trigger_mode, socket_server, max_force=65, max_time=360, minimum_width_p=0.1):   
        self.motor_1 = motor_1
        self.motor_2 = motor_2
        self.session_manager = session_manager
        self.trigger_mode = trigger_mode
        self.socket_server = socket_server

        self.max_force = max_force
        self.max_time = max_time
        self.minimum_width_p = minimum_width_p
        self.minimum_distance = self.max_time * self.minimum_width_p / 2 # Minimum distance between force2_start_time and force2_peak_time / orce2_peak_time and force2_end_time

        self.score_history = []

        self.optimizer_path = session_manager.session_dir / "optimizer_logs.json"

        self.kappa = kappa
        self.freq = freq
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

        # Set up logging directories
        self.profile_dir = session_manager.session_dir / "profiles"
        if not os.path.exists(self.profile_dir):
            os.makedirs(self.profile_dir)
        self.remote_profile_dir = session_manager.session_remote_dir / "profiles"
        if not os.path.exists(self.remote_profile_dir):
            os.system(f"ssh macbook 'mkdir -p {self.remote_profile_dir}'")

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

        length = len(self.session_manager.theta_2_scaled)
        base_profile = pd.DataFrame({"force_X": np.zeros(length), "force_Y": np.zeros(length), "theta_2": self.session_manager.theta_2_scaled})

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
        force1_end_time = self.minimum_width_p * self.max_time + force1_end_time_p * self.max_time * (1 - self.minimum_width_p)
        force1_peak_force = force1_peak_force_p * self.max_force * 2/3

        # Standard with check later on and returning -1 if constraints are violated
        # force2_start_time = force2_start_time_p * self.max_time 
        # force2_peak_time = force2_peak_time_p * self.max_time
        # force2_peak_force = force2_peak_force_p * self.max_force
        # force2_end_time = force2_end_time_p * self.max_time

        # Constrain the time values to be within the range of 0.25 to 0.75 of the self.max time
        # force2_start_time = force2_start_time_p * self.max_time * 0.25
        # force2_peak_time = force2_peak_time_p * self.max_time * 0.5 + 0.25 * self.max_time
        # force2_peak_force = force2_peak_force_p * self.max_force
        # force2_end_time = force2_end_time_p * self.max_time * 0.25 + 0.75 * self.max_time

        # Dynamic constraints
        force2_peak_time = force2_peak_time_p * self.max_time * 0.8 + 0.1 * self.max_time # 0.1 to 0.9
        force2_start_time = (force2_peak_time - self.minimum_distance) * force2_start_time_p # 0 to 0.05 of peak time
        force2_end_time = force2_peak_time + self.minimum_distance + force2_end_time_p * (self.max_time - force2_peak_time - self.minimum_distance) # 0.05 of peak to max time
        force2_peak_force = force2_peak_force_p * self.max_force

        # Just a sanity check
        if not self.validate_constraints(force2_start_time, force2_peak_time, force2_end_time):
            print("violated constraints")
        
        base_profile = self.get_profile(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time)

        profile_name = f"t11_{force1_end_time}_f11_{force1_peak_force}_t21_{force2_start_time}_t22_{force2_peak_time}_t23_{force2_end_time}_f21_{force2_peak_force}"

        # Save the profile as CSV
        profile_path = self.profile_dir / f"profile_{profile_name}.csv"
        base_profile.to_csv(profile_path, index=False)

        # Send to remote host
        os.system(f"scp {profile_path} macbook:{self.remote_profile_dir}")

        for i in range(5):
            print("\nReady to apply profile, iteration: ", i+1)

            apply_simulation_profile(
                motor_1=self.motor_1,
                motor_2=self.motor_2,
                freq=self.freq,
                session_manager=self.session_manager,
                profile = base_profile,
                profile_name = profile_name,
                mode=self.trigger_mode,
                server=self.socket_server
            )

            # Check if the most recent (correct) score was received
            # Wait until self.socket_server.score_receival_time is smaller than 1 second
            # while self.socket_server.score_receival_time is None or (time.time() - self.socket_server.score_receival_time) > 1:
            #     time.sleep(0.1)

            # Check if the score's tag is the same as the most recent profile
            while not self.socket_server.profile_name == self.socket_server.score_tag:
                time.sleep(0.1)

            score = self.socket_server.score
            self.score_history.append(score)
            print(f"Score: {score}")
        return score

    def load_optimizer(self):
        acquisition = bayes_opt.acquisition.UpperConfidenceBound(kappa=self.kappa)
        self.optimizer = BayesianOptimization(
            f=self.objective,
            pbounds=self.pbounds,
            acquisition_function=acquisition,
            random_state=0,
            verbose=2
        )          
        if os.path.exists(self.optimizer_path):
            load_logs(self.optimizer, logs=[self.optimizer_path])
            if self.optimizer.space:
                # Refit the GP model
                self.optimizer._gp.fit(self.optimizer.space.params, self.optimizer.space.target)
                print(f"Loaded optimization progress from file. {len(self.optimizer.space)} evaluations loaded.")
            else:
                print("No evaluations found in the log. Starting a new optimization.")
        else:
            print("No saved progress found. Starting a new optimization.")
        
        # It is valid to inlude this here
        self.save_progress()
            

    def save_progress(self):
        logger = JSONLogger(path=self.optimizer_path)
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    def log_to_remote(self):
        os.system(f"scp {self.optimizer_path} macbook:{self.session_manager.session_remote_dir}")

    def informed_optimization(self):
        # Points for profile
        force1_end_time = 150.0
        force1_peak_force = 20.0

        force2_start_time = 70.0
        force2_peak_time = 160.0
        force2_end_time = 250.0
        force2_peak_force = 60.0

        # Revert to the original values (for the dynamic constraints)
        force1_end_time_p = (force1_end_time - self.minimum_width_p * self.max_time) / (self.max_time * (1 - self.minimum_width_p))
        force1_peak_force_p = force1_peak_force / (2/3 * self.max_force)

        force2_start_time_p = force2_start_time / (force2_peak_time - self.minimum_distance)
        force2_peak_time_p = (force2_peak_time - 0.1 * self.max_time) / (0.8 * self.max_time)
        force2_end_time_p = (force2_end_time - force2_peak_time - self.minimum_distance) / (self.max_time - force2_peak_time - self.minimum_distance)
        force2_peak_force_p = force2_peak_force / self.max_force

        # Define the initial points
        initial_points = {
            "force1_end_time_p": force1_end_time_p,
            "force1_peak_force_p": force1_peak_force_p,
            "force2_start_time_p": force2_start_time_p,
            "force2_peak_time_p": force2_peak_time_p,
            "force2_peak_force_p": force2_peak_force_p,
            "force2_end_time_p": force2_end_time_p
        }

        # Very unlikely but might happen
        if initial_points not in self.optimizer.space.params:
            self.optimizer.probe(params=initial_points, lazy=True)
        else:
            print("Informed points already in optimizer space.")

    def explorate(self, init_points=1, n_iter=0):
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)

    def optimize(self, init_points=0, n_iter=1):
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)






# Sim proflie information
# profile_reference_profile_force_Y_fmax_63.79933415313509_start_80_max_idx_106_end_254

# profile_simulation_profile_Camille_force_Y_fmax_63.79933415313509_start_64_max_idx_122_end_265

# profile_simulation_profile_Camille_xy_force_Y_fmax_63.79933415313509_start_64_max_idx_122_end_265

# profile_simulation_profile_Camille_scalex_scaley_fitted_force_X_fmax_17.403562682388543_end_140
# profile_simulation_profile_Camille_scalex_scaley_fitted_force_Y_fmax_85.56774268113895_start_84_max_idx_102_end_260

# profile_simulation_profile_Camille_scalex_scaley_force_X_fmax_17.403562682388543_end_162
# profile_simulation_profile_Camille_scalex_scaley_force_Y_fmax_85.56774268113895_start_70_max_idx_116_end_270

# profile_peak_time_57_peak_force_62_scaled_force_Y_fmax_61.92935701912673_start_21_max_idx_137_end_251

# profile_simulation_profile_Camille_y_force_Y_fmax_63.79933415313509_start_64_max_idx_122_end_265

# profile_simulation_profile_Camille_y_fitted_force_Y_fmax_63.79933415313509_start_80_max_idx_106_end_254

# profile_peak_time_57_peak_force_62_scaled_fitted_force_Y_fmax_61.92935701912673_start_40_max_idx_118_end_238

# profile_simulation_profile_Camille_fitted_force_Y_fmax_63.79933415313509_start_80_max_idx_106_end_254

# profile_simulation_profile_Camille_xy_fitted_force_Y_fmax_63.79933415313509_start_80_max_idx_106_end_254
