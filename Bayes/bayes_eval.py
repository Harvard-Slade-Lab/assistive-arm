import pandas as pd
import os
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
from chspy import CubicHermiteSpline


# def plot_saved_profiles():
#     optimized_profiles = "/Users/nathanirniger/Desktop/profiles/optim"
#     target_profile = pd.read_csv("/Users/nathanirniger/Desktop/profiles/reference_profile.csv")

#     for profile in os.listdir(optimized_profiles):
#         if profile.endswith(".csv"):
#             optimized_profile = pd.read_csv(os.path.join(optimized_profiles, profile))
            
#             # plot the optimized profile x force and y force and the target profile
#             plt.plot(optimized_profile["force_x"], label="Optimized Profile X Force")
#             plt.plot(optimized_profile["force_y"], label="Optimized Profile Y Force")
#             plt.plot(target_profile["force_X"], label="Target Profile X Force")
#             plt.plot(target_profile["force_Y"], label="Target Profile Y Force")
#             plt.legend()
#             plt.savefig(f"/Users/nathanirniger/Desktop/profiles/optim/plots/{profile.removesuffix(".csv")}.png")


def cubic_hermite_spline(points):
    spline = CubicHermiteSpline(n=1)
    for t, value, derivative in points:
        spline.add((t, [value], [derivative]))
    return spline


def read_progress():
    # Load calubration data
    calibration_path = "/Users/nathanirniger/Desktop/profiles/device_height_calibration.yaml"
    with open(calibration_path, 'r') as file:
        data = yaml.safe_load(file)

    length = len(data['theta_2_values'])

    # Path to your JSON Lines file
    file_path = '/Users/nathanirniger/Desktop/Bayes/bayesian_optimization_progress.json'
    # List to store the loaded data
    data = []
    # Read the JSON Lines file
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ignore empty lines
                data.append(json.loads(line))
    df = pd.json_normalize(data)

    return df, length


def get_profile(df, length):
    # best score
    max_target_row = df.loc[df['target'].idxmax()]

    t11 = max_target_row['params.force1_end_time']
    f1 = max_target_row['params.force1_peak_force']
    grf_x = cubic_hermite_spline([(0, 0, 0), (t11 / 2, f1, 0), (t11, 0, 0)])
    curve_x = [grf_x.get_state(i)[0] for i in range(int(np.round(t11)))]
    # Pad the curve with zeros to match the full length of `result`
    padded_curve_x = np.concatenate([curve_x, np.zeros(length - int(np.round(t11)))])
    
    t21 = max_target_row['params.force2_start_time']
    t22 = max_target_row['params.force2_peak_time']
    f21 = max_target_row['params.force2_peak_force']
    t23 = max_target_row['params.force2_end_time']
    grf_y = cubic_hermite_spline([(0, 0, 0), (t22, f21, 0), (t23, 0, 0), (t23 - t21, 0, 0)])
    curve_y = [grf_y.get_state(i)[0] for i in range(int(np.round(t23)) - int(np.round(t21)))]
    # Pad the curve with zeros to match the full length of `result`
    padded_curve_y = np.concatenate([np.zeros(int(np.round(t21))), curve_y, np.zeros(length - int(np.round(t23)))])

    plt.plot(padded_curve_x, label="Optimized Profile X Force")
    plt.plot(padded_curve_y, label="Optimized Profile Y Force")
    plt.legend()
    plt.show()

def plot_scores(df):
    # replace all scores < -100 with -100
    df['target'] = df['target'].apply(lambda x: -100 if x < -100 else x)
    plt.plot(df['target'])
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.show()


def define_new_reference_profile():
    force1_end_time_target = 0.8
    force1_peak_force_target = 0.5

    force2_start_time_target = 0.2
    force2_peak_time_target = 0.5
    force2_peak_force_target = 0.8
    force2_end_time_target = 0.7

    t_max = 360
    max_force = 65

    t11 = force1_end_time_target * t_max
    f1 = force1_peak_force_target * max_force

    t21 = force2_start_time_target * t_max
    t22 = force2_peak_time_target * t_max
    f21 = force2_peak_force_target * max_force
    t23 = force2_end_time_target * t_max

    grf_x = cubic_hermite_spline([(0, 0, 0), (t11 / 2, f1, 0), (t11, 0, 0)])
    curve_x = [grf_x.get_state(i)[0] for i in range(int(np.round(t11)))]
    # Pad the curve with zeros to match the full length of `result`
    padded_curve_x = np.concatenate([curve_x, np.zeros(360 - int(np.round(t11)))])
                                    
    grf_y = cubic_hermite_spline([(0, 0, 0), (t22 - t21, f21, 0), (t23 - t21, 0, 0)])
    curve_y = [grf_y.get_state(i)[0] for i in range(int(np.round(t23)) - int(np.round(t21)))]
    # Pad the curve with zeros to match the full length of `result`
    padded_curve_y = np.concatenate([np.zeros(int(np.round(t21))), curve_y, np.zeros(360 - int(np.round(t23)))])

    length = 360
    base_profile = pd.DataFrame({"force_X": np.zeros(length), "force_X": np.zeros(length)})
                                    
    base_profile["force_X"] = padded_curve_x
    base_profile["force_Y"] = padded_curve_y

    plt.plot(base_profile["force_X"], label="Optimized Profile X Force")
    plt.plot(base_profile["force_Y"], label="Optimized Profile Y Force")
    plt.legend()
    plt.show()

    base_profile.to_csv("/Users/nathanirniger/Desktop/profiles/reference_profile_3.csv", index=False)


def main():
    define_new_reference_profile()
    # df, length = read_progress()
    # plot_scores(df)


if __name__ == "__main__":
    main()