import numpy as np
import pandas as pd
import yaml
import os
import time
import logging
import bayes_opt.acquisition

from chspy import CubicHermiteSpline
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from sts_control import apply_simulation_profile



class ForceProfileOptimizer:
    def __init__(self, motor_1, motor_2, kappa, freq, iterations, session_manager, trigger_mode, socket_server, imu_reader, max_force=55, scale_factor_x=2/3, max_time=360, minimum_width_p=0.2):   
        self.motor_1 = motor_1
        self.motor_2 = motor_2
        self.session_manager = session_manager
        self.trigger_mode = trigger_mode
        self.socket_server = socket_server
        self.imu_reader = imu_reader

        self.iterations = iterations
        self.max_force = max_force
        self.scale_factor_x = scale_factor_x
        # self.max_time = max_time
        self.max_time = len(self.session_manager.roll_angles)
        self.minimum_width_p = minimum_width_p
        self.minimum_distance = self.max_time * self.minimum_width_p / 2 # Minimum distance between force2_start_time and force2_peak_time / force2_peak_time and force2_end_time

        self.score_history = []

        self.optimizer_path = session_manager.session_dir / "optimizer_logs.json"

        self.path_to_delete = session_manager.session_dir / "profile_to_delete.csv"

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
    
    def get_logger(self, profile_name, profile_path):
        logger = logging.getLogger(profile_name)
        handler = logging.FileHandler(f"{profile_path.with_suffix('.log')}")
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def get_profile(self, force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time):
        
        length = len(self.session_manager.roll_angles)
        base_profile = pd.DataFrame({"force_X": np.zeros(length), "force_Y": np.zeros(length)})
        base_profile.index = self.session_manager.roll_angles.index
        base_profile = pd.concat([self.session_manager.roll_angles, base_profile], axis=1)

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
        # Catch zero force cases
        if force1_peak_force_p == 0 and force2_peak_force_p == 0:
            # Those cases are the unassisted case and hence should get a score of 0
            return 0
        
        # X-force profile
        force1_end_time = self.minimum_width_p * self.max_time + force1_end_time_p * self.max_time * (1 - self.minimum_width_p)
        force1_peak_force = force1_peak_force_p * self.max_force * self.scale_factor_x

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
        force2_peak_time = force2_peak_time_p * self.max_time * 0.7 + 0.15 * self.max_time # 0.15 to 0.85
        force2_start_time = (force2_peak_time - self.minimum_distance) * force2_start_time_p # minimum_distance off peak time
        force2_end_time = force2_peak_time + self.minimum_distance + force2_end_time_p * (self.max_time - force2_peak_time - self.minimum_distance) # minimum_distance off peak to max time
        force2_peak_force = force2_peak_force_p * self.max_force

        # Just a sanity check
        if not self.validate_constraints(force2_start_time, force2_peak_time, force2_end_time):
            print("violated constraints")
        
        base_profile = self.get_profile(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time, force2_peak_force, force2_end_time)

        profile_name = f"t11_{int(np.round(force1_end_time))}_f11_{int(np.round(force1_peak_force))}_t21_{int(np.round(force2_start_time))}_t22_{int(np.round(force2_peak_time))}_t23_{int(np.round(force2_end_time))}_f21_{int(np.round(force2_peak_force))}_Profile_{self.socket_server.profile_name}"

        # Configure a new logger for each call
        profile_path = self.profile_dir / f"profile_{profile_name}.csv"
        logger = self.get_logger(profile_name, profile_path)

        # Log the initial inputs and calculated values
        logger.info(f"Inputs: force1_end_time_p={force1_end_time_p}, force1_peak_force_p={force1_peak_force_p}, force2_start_time_p={force2_start_time_p}, force2_peak_time_p={force2_peak_time_p}, force2_peak_force_p={force2_peak_force_p}, force2_end_time_p={force2_end_time_p}")
        logger.info(f"Calculated Values: force1_end_time={force1_end_time}, force1_peak_force={force1_peak_force}, force2_start_time={force2_start_time}, force2_peak_time={force2_peak_time}, force2_peak_force={force2_peak_force}, force2_end_time={force2_end_time}")

        # Save the profile as CSV
        profile_path = self.profile_dir / f"profile_{profile_name}.csv"
        base_profile.to_csv(profile_path, index=False)

        # Send to remote host
        try:
            os.system(f"scp {profile_path} macbook:{self.remote_profile_dir}")
        except:
            print("Could not send profile to remote host.")

        scores = []
        i = 1
        while i <= self.iterations:
            # TODO find a better way to exit, if the mode flag is set
            if self.socket_server.mode_flag or self.socket_server.kill_flag:
                raise Exception("Exiting optimization, flag triggered")
            print("\nReady to apply profile, iteration: ", i)

            current_profile_name = self.socket_server.profile_name
            profile_name = f"t11_{int(np.round(force1_end_time))}_f11_{int(np.round(force1_peak_force))}_t21_{int(np.round(force2_start_time))}_t22_{int(np.round(force2_peak_time))}_t23_{int(np.round(force2_end_time))}_f21_{int(np.round(force2_peak_force))}_Profile_{current_profile_name}"

            print(f"motor1 type (70-10): {self.motor_1.type}, motor2 type (60-6): {self.motor_2.type}")

            apply_simulation_profile(
                motor_1=self.motor_1,
                motor_2=self.motor_2,
                freq=self.freq,
                session_manager=self.session_manager,
                profile = base_profile,
                profile_name = profile_name,
                mode=self.trigger_mode,
                socket_server=self.socket_server,
                imu_reader=self.imu_reader
            )

            # Check if the most recent (correct) score was received
            # Wait until self.socket_server.score_receival_time is smaller than 1 second
            # while self.socket_server.score_receival_time is None or (time.time() - self.socket_server.score_receival_time) > 1:
            #     time.sleep(0.1)

            # If we jumped out of apply_simulation_profile due to a stop, return to the main menu
            # Chose to neglect the full iteration, could also still use the score if e.g the 4 previous iterations were okay
            if self.socket_server.kill_flag or self.socket_server.mode_flag:
                # Exit the optimization
                raise Exception("Exiting optimization, flag triggered")

            start_time = time.time()
            local_repeat_flag = False
            # Check if the score's tag is the same as the most recent profile -> tested (can also be set by repeat command)
            while not current_profile_name == self.socket_server.score_tag:
                time.sleep(0.1)
                # If the score is not received in 5 seconds, break
                if time.time() - start_time > 5:
                    print("Score not received in 5 seconds, repeat iteration.")
                    local_repeat_flag = True
                    break

            # Ask the user if the iteration should be repeated
            if not local_repeat_flag:
                repeat = input("Accept iteration? (y/n): ")
                if repeat == "y":
                    print("Continuing optimization...")
                else:
                    local_repeat_flag = True
                    # Save the current profile name so it can be deleted later on
                    with open(self.path_to_delete, "a") as f:
                        f.write(current_profile_name + "\n")

            # Repeat if the flag is set (last iteration can also be repeated, as flag is set True if the score can not be calculated)
            if self.socket_server.repeat_flag or local_repeat_flag:
                print("Iteration has to be repeated.")
                self.socket_server.repeat_flag = False
                local_repeat_flag = False
            else:
                i += 1
                score = self.socket_server.score
                self.score_history.append(score)
                scores.append(score)
                logger.info(f"Profile Name: {profile_name}, Score: {score}")
                print(f"Score: {score}")

        # Get mean score over last iterations
        mean_score = np.mean(scores)
        
        return mean_score

    def load_optimizer(self):
        acquisition = bayes_opt.acquisition.UpperConfidenceBound(kappa=self.kappa)
        self.optimizer = BayesianOptimization(
            f=self.objective,
            pbounds=self.pbounds,
            acquisition_function=acquisition,
            random_state=0,
            verbose=2
        )          
        
        # Load existing logs
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

        # Attach a logger (append mode)
        self.logger = JSONLogger(path=self.optimizer_path, reset=False)
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, self.logger)


    def log_to_remote(self):
        try:
            os.system(f"scp {self.optimizer_path} macbook:{self.session_manager.session_remote_dir}")
        except:
            print("Could not send optimizer logs to remote host.")

    def informed_optimization(self):
        # Points for profile (similar to camille's)
        force1_end_time = 150.0/360.0 * self.max_time
        force1_peak_force = 20.0

        force2_start_time = 70.0/360.0 * self.max_time
        force2_peak_time = 160.0/360.0 * self.max_time
        force2_end_time = 250.0/360.0 * self.max_time
        force2_peak_force = 55.0

        # Revert to the original values (for the dynamic constraints)
        force1_end_time_p = (force1_end_time - self.minimum_width_p * self.max_time) / (self.max_time * (1 - self.minimum_width_p))
        force1_peak_force_p = force1_peak_force / (self.scale_factor_x * self.max_force)

        force2_start_time_p = force2_start_time / (force2_peak_time - self.minimum_distance)
        force2_peak_time_p = (force2_peak_time - 0.15 * self.max_time) / (0.7 * self.max_time)
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

        # Very unlikely but usefull if optimizer was loaded (as they would already be in the space)
        if initial_points not in self.optimizer.space.params:
            self.optimizer.probe(params=initial_points, lazy=True)
        else:
            print("Informed points already in optimizer space.")


        # Added profile
        force1_end_time_p = 1.0
        force1_peak_force_p = 1.0

        force2_start_time_p = 0.2
        force2_peak_time_p = 0.65
        force2_end_time_p = 1.0
        force2_peak_force_p = 0.7

        # Define the initial points
        initial_points = {
            "force1_end_time_p": force1_end_time_p,
            "force1_peak_force_p": force1_peak_force_p,
            "force2_start_time_p": force2_start_time_p,
            "force2_peak_time_p": force2_peak_time_p,
            "force2_peak_force_p": force2_peak_force_p,
            "force2_end_time_p": force2_end_time_p
        }

        # Very unlikely but usefull if optimizer was loaded (as they would already be in the space)
        if initial_points not in self.optimizer.space.params:
            self.optimizer.probe(params=initial_points, lazy=True)
        else:
            print("Informed points already in optimizer space.")


        # # Add profiles with extreme times and zero force
        # initial_points = {
        #     "force1_end_time_p": 0.0,
        #     "force1_peak_force_p": 0.0,
        #     "force2_start_time_p": 0.0,
        #     "force2_peak_time_p": 0.0,
        #     "force2_peak_force_p": 0.0,
        #     "force2_end_time_p": 0.0
        # }
        # if initial_points not in self.optimizer.space.params:
        #     self.optimizer.probe(params=initial_points, lazy=True)
        # else:
        #     print("Zero points already in optimizer space.")

        # initial_points = {
        #     "force1_end_time_p": 1.0,
        #     "force1_peak_force_p": 0.0,
        #     "force2_start_time_p": 1.0,
        #     "force2_peak_time_p": 1.0,
        #     "force2_peak_force_p": 0.0,
        #     "force2_end_time_p": 1.0
        # }
        # if initial_points not in self.optimizer.space.params:
        #     self.optimizer.probe(params=initial_points, lazy=True)
        # else:
        #     print("One points already in optimizer space.")

    def explorate(self, init_points=1, n_iter=0):
        if not self.socket_server.kill_flag or self.socket_server.mode_flag:
            try:
                self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
            except Exception as e:
                print(e)


    def optimize(self, init_points=0, n_iter=1):
        if not self.socket_server.kill_flag or self.socket_server.mode_flag:
            try:
                self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
            except Exception as e:
                print(e)
