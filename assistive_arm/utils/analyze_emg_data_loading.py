import yaml
from pathlib import Path
import pandas as pd
import jsonlines
import os
import json
import re
from copy import deepcopy
from itertools import groupby
from bisect import bisect_left
from assistive_arm.utils.data_preprocessing import read_headers
from assistive_arm.utils.emg_processing import filter_emg



def setup_datastructure(selected_subjects = None, selected_sessions = None):
    # Setup the directories and datastructures

    subject_dir = Path(f"../subject_logs/")

    subjects = sorted([subject for subject in subject_dir.iterdir() if subject.is_dir()])
    if selected_subjects is not None:
        subjects = [subject for subject in subjects if subject.name in selected_subjects]

    subject_data = {}

    subject_dirs = {}

    for subject in subjects:
        subject_name = subject.name
        subject_data[subject_name] = {}
        subject_dirs[subject_name] = {}


        sessions = [session for session in subject.iterdir() if session.is_dir()]
        if selected_sessions is not None:
            sessions = [session for session in sessions if session.name in selected_sessions]

        for session in sessions:
            if session.is_dir():
                session_dir = session
                subject_data[subject_name][session.name] = {}
                subject_dirs[subject_name][session.name] = {}

                with open("../motor_config.yaml", "r") as f:
                    motor_config = yaml.load(f, Loader=yaml.FullLoader)
                
                motor_dir = session_dir / "Motor"
                emg_raw_dir = session_dir / "EMG/Raw"
                emg_sts_dir = session_dir / "EMG/STS"
                log_dir = session_dir / "Log"
                mvic_dir = session_dir / "MVIC"
                plot_dir = session_dir / "plots"
                plot_dir.mkdir(exist_ok=True)

                angle_calibration_path = motor_dir / "device_height_calibration.yaml"

                subject_dirs[subject_name][session.name]["emg_raw_dir"] = emg_raw_dir
                subject_dirs[subject_name][session.name]["emg_sts_dir"] = emg_sts_dir
                subject_dirs[subject_name][session.name]["plot_dir"] = plot_dir
                subject_dirs[subject_name][session.name]["motor_dir"] = motor_dir
                subject_dirs[subject_name][session.name]["log_dir"] = log_dir
                subject_dirs[subject_name][session.name]["angle_calibration_path"] = angle_calibration_path


                # Read EMG file to extract configuration
                # Get a random emg file containing the string "EMG" in the file name
                emg_file = next(emg_raw_dir.glob("*EMG*.csv"))

                emg_config = dict()

                for i, header in enumerate(read_headers(emg_file, 1, delimiter="\t")):
                    print(i, header)
                    if i == 0:
                        emg_config["CHANNEL_NAMES"] = header
                        # TODO get real emg frequency
                        emg_config["EMG_FREQUENCY"] = float(2148.259)
                        emg_config["IMU_FREQUENCY"] = float(518.519)
                        emg_config["OR_FREQUENCY"] = float(222.222)

                subject_data[subject_name][session.name]["emg_config"] = emg_config

    return subject_data, subject_dirs, subjects, motor_config


def get_muscle_mapping(emg_config):
    # The mapping converts the col name from the csv. output file to a muscle name
    mapping = {}
    for channel in emg_config["CHANNEL_NAMES"][0].split(','):
        info = channel.split(" ")[1:2]
        if info[0] == "IMU":
            mapping[channel] = "IMU"

        elif info[0] == "OR":
            mapping[channel] = "OR"
            
        else:
            muscle, side = info[0].split("_")

            if side == "L":
                side_prefix = "LEFT"
            else:
                side_prefix = "RIGHT"

            mapping[channel] = "_".join([muscle, side_prefix])

    return mapping


def write_emg_config_yaml(subject_data, subject_dirs, mapping):
    #  Write EMG-config yaml files
    for subject in subject_data:
        for session in subject_data[subject]:
            subject_dir = Path(f"../subject_logs/")

            session_dir = subject_dir / subject / session

            subject_data[subject][session]["emg_config"]["MAPPING"] = mapping

            emg_config = subject_data[subject][session]["emg_config"]

            if not (session_dir / "emg_config.yaml").exists():
                print("Writing EMG config...")
                with open(session_dir / "emg_config.yaml", "w") as f:
                    yaml.dump(emg_config, f)
            else:
                print("EMG config already exists")
                # Open existing config
                with open(session_dir / "emg_config.yaml", "r") as f:
                    emg_config = yaml.load(f, Loader=yaml.FullLoader)

            with open(subject_dirs[subject][session]["angle_calibration_path"] , 'r') as f:
                angle_calibration = yaml.load(f, Loader=yaml.FullLoader)

            subject_data[subject][session]["angle_calibration"] = angle_calibration

    return subject_data

def group_paths_by_iteration(paths):
    groups = []
    current_group = []
    previous_iteration = None

    for path in paths:
        # Extract the filename
        filename = Path(path).name
        
        # Extract the iteration number using regex
        match = re.search(r'_iteration_(\d+)_', filename)
        if match:
            current_iteration = int(match.group(1))
            
            # If iteration resets to 1 or no previous iteration, start a new group
            if previous_iteration is not None and current_iteration <= previous_iteration:
                groups.append(current_group)
                current_group = []

            # Add path to the current group
            current_group.append(path)
            previous_iteration = current_iteration
    
    # Add the last group
    if current_group:
        groups.append(current_group)

    return groups


def group_paths_by_profile(paths):
    # Extract the prefix (everything before "_Profile") from the filename
    def extract_prefix(path):
        if "Profile" in Path(path).name:
            filename = Path(path).name
            return filename.split("_Profile")[0]
        elif "Tag" in Path(path).name:
            filename = Path(path).name
            return filename.split("_Tag")[0]
        
    # Group paths by their extracted prefix
    paths.sort(key=extract_prefix)  # Ensure paths are sorted by prefix
    grouped = groupby(paths, key=extract_prefix)
    
    # Return groups as a list of lists
    return [list(group) for _, group in grouped]


def load_motor_data_hilo(subject_data, subject_dirs, subjects, validation):
    # LOAD MOTOR DATA
    session_dict = {}
    session_dict["MVIC"] = {"Unfiltered": [], "Filtered": []}

    data_dict = {"EMG": {"Unfiltered": [], "Filtered": []}, "MOTOR_DATA": [], "LOG_INFO": [], "FORCE_PROFILE": [], "PROFILE_TAGS": []}

    session_dict["UNPOWERED"] = {"ALL_TAGS": [], "FIRST_TAGS": []}
    session_dict["ASSISTED"] = {"ALL_TAGS": [], "FIRST_TAGS": []}

    # Initialize RAW data
    session_dict["API_DATA"] = {"UNPOWERED": {}, "ASSISTED": {}}

    name_tag_mapping = {}

    # Skip first x s to avoid initial noise
    skip_first = 0.0 #s

    for subject in subjects:
        for session in iter(subject_data[subject.name].keys()):

            session_data = deepcopy(session_dict)
            motor_dir = subject_dirs[subject.name][session]["motor_dir"]

            profile_to_delete = set(subject_data[subject.name][session]["profile_to_delete"])

            # Use list to filter out unwanted files
            sorted_files = sorted(motor_dir.iterdir(), key=lambda f: f.stat().st_ctime)

            for file in sorted_files:
                if file.is_file():
                    # Check if the file name contains any of the iterations to delete
                    for iteration in profile_to_delete:
                        if iteration in file.stem:
                            sorted_files.remove(file)
                            if "Profile" in file.stem:
                                tag = re.search(r"Profile_(\d+)_", file.stem).group(1)
                                print(f"Removing Profile {tag} from session {session}")
                            elif "Tag" in file.stem:
                                tag = re.search(r"Tag_(\d+)_", file.stem).group(1)
                                print(f"Removing Tag {tag} from session {session}")


            # Lists to hold unpowered and assisted files
            unpowered_files = []
            assisted_files = []

            # Sort unpowered and assisted files and safe all tags accordingly
            for file_path in sorted_files:
                if file_path.suffix == ".csv":
                    # File size should be larger than 500 bytes to avoid empty files
                    if os.stat(file_path).st_size > 500:
                        if "unpowered" in file_path.stem:
                            unpowered_files.append(file_path)
                            tag = re.search(r"tag_(\d+)_", file_path.stem).group(1)
                            session_data["UNPOWERED"]["ALL_TAGS"].append(tag)
                        else:
                            assisted_files.append(file_path)
                            if "Profile" in file_path.stem:
                                tag = re.search(r"Profile_(\d+)_", file_path.stem).group(1)
                            elif "Tag" in file_path.stem:
                                tag = re.search(r"Tag_(\d+)_", file_path.stem).group(1)
                            session_data["ASSISTED"]["ALL_TAGS"].append(tag)
                    else:
                        print(f"File {file_path} is empty")

            # Sort unpowered files according to the number after ..tag_number_... (chronological)
            unpowered_files = sorted(unpowered_files, key=lambda f: int(re.search(r"tag_(\d+)_", f.stem).group(1)))

            # Go through all the unpowered files and add consecutive ones to the same profile (["UNPOWERED"][profile], where profile is iteration_{unpowered_iteration})
            # Extract all the files from iteration 1 to iteration n (before iteration number goes back to 1 or no more files are in the list)
            unpowered_groups = group_paths_by_iteration(unpowered_files)
            for group in unpowered_groups:
                unpowered_first_file = group[0]
                df = pd.read_csv(unpowered_first_file, index_col="time").loc[skip_first:]
                first_tag = re.search(r"tag_(\d+)_", unpowered_first_file.stem).group(1)
                if first_tag not in session_data["UNPOWERED"].keys():
                    session_data["UNPOWERED"][first_tag] = deepcopy(data_dict)
                session_data["UNPOWERED"]["FIRST_TAGS"].append(first_tag)
                session_data["UNPOWERED"][first_tag]["PROFILE_TAGS"].append(tag)
                session_data["UNPOWERED"][first_tag]["MOTOR_DATA"].append(df)

                # Go throught he rest of the files in the group
                for file_path in group[1:]:
                    df = pd.read_csv(file_path, index_col="time").loc[skip_first:]
                    session_data["UNPOWERED"][first_tag]["MOTOR_DATA"].append(df)
                    tag = re.search(r"tag_(\d+)_", file_path.stem).group(1)
                    session_data["UNPOWERED"][first_tag]["PROFILE_TAGS"].append(tag)


            # Sort assisted files according to the number after ..Profile_number_... (chronological)
            if "Profile" in assisted_files[0].stem:
                assisted_files = sorted(assisted_files, key=lambda f: int(re.search(r"Profile_(\d+)_", f.stem).group(1)))
            elif "Tag" in assisted_files[0].stem:
                assisted_files = sorted(assisted_files, key=lambda f: int(re.search(r"Tag_(\d+)_", f.stem).group(1)))
            # Go through assisted files and group them according to the force profile
            assisted_groups = group_paths_by_profile(assisted_files)
            for group in assisted_groups:
                assisted_first_file = group[0]
                parts = re.split('_', assisted_first_file.stem)

                df = pd.read_csv(assisted_first_file, index_col="time").loc[skip_first:]
                if "Profile" in assisted_first_file.stem:
                    first_tag = re.search(r"Profile_(\d+)_", assisted_first_file.stem).group(1)
                elif "Tag" in assisted_first_file.stem:
                    first_tag = re.search(r"Tag_(\d+)_", assisted_first_file.stem).group(1)
                if first_tag not in session_data["ASSISTED"].keys():
                    session_data["ASSISTED"][first_tag] = deepcopy(data_dict)
                session_data["ASSISTED"]["FIRST_TAGS"].append(first_tag)
                session_data["ASSISTED"][first_tag]["PROFILE_TAGS"].append(first_tag)
                session_data["ASSISTED"][first_tag]["MOTOR_DATA"].append(df)
                if not validation:
                    name_tag_mapping[first_tag] = f"Assisted X-force(t11_{parts[1]}, f11_{parts[3]}), Y-force(t21_{parts[5]}, t22_{parts[7]}, t23_{parts[9]}, f21_{parts[11]})"
                else:
                    name_tag_mapping[first_tag] = f"{assisted_first_file.stem}"

                # Go throught the rest of the files in the group
                for file_path in group[1:]:
                    df = pd.read_csv(file_path, index_col="time").loc[skip_first:]
                    session_data["ASSISTED"][first_tag]["MOTOR_DATA"].append(df)
                    if "Profile" in file_path.stem:
                        tag = re.search(r"Profile_(\d+)_", file_path.stem).group(1)
                    elif "Tag" in file_path.stem:
                        tag = re.search(r"Tag_(\d+)_", file_path.stem).group(1)
                    session_data["ASSISTED"][first_tag]["PROFILE_TAGS"].append(tag)

            subject_data[subject.name][session]["session_data"] = session_data
            subject_data[subject.name][session]["name_tag_mapping"] = name_tag_mapping

    return subject_data, subject_dirs, subjects, data_dict


def load_profile_data_hilo(subject_data, subject_dirs, subjects):
    for subject in subjects:
        for session in iter(subject_data[subject.name].keys()):
            profile_dir = subject_dirs[subject.name][session]["motor_dir"] / "profiles"

            session_data = subject_data[subject.name][session]["session_data"]

            sorted_files = sorted(profile_dir.iterdir(), key=lambda f: f.stat().st_ctime)

            for file_path in sorted_files:
                if file_path.suffix == ".csv":
                    if "t11" in file_path.stem:
                        df = pd.read_csv(file_path)

                        tag = re.search(r"Profile_([^_]+)", file_path.stem).group(1)

                        # Check if the profile exists in the session data, filters out profiles that were created but never run
                        # This is also acts as a sanity test to make sure that sorting in the motor data loader was correct
                        if tag in session_data["ASSISTED"].keys():
                            session_data["ASSISTED"][tag]["FORCE_PROFILE"] = df
                        else: 
                            print(f"Profile {file_path.stem} not found in session {session}")

            subject_data[subject.name][session]["session_data"] = session_data

    return subject_data, subject_dirs, subjects


def load_raw_emg_imu_data_hilo(subject_data, subject_dirs, subjects):
    # LOAD EMG DATA
    for subject in subjects:
        for session in subject.iterdir():
            if not session.is_dir():
                continue
            emg_config = subject_data[subject.name][session.name]["emg_config"]

            sampling_freq =  emg_config["EMG_FREQUENCY"]
            sampling_interval = 1 / sampling_freq  # Time interval between samples

            IMU_sampling_freq = emg_config["IMU_FREQUENCY"]
            IMU_sampling_interval = 1 / IMU_sampling_freq  # Time interval between samples

            OR_sampling_freq = emg_config["OR_FREQUENCY"]
            OR_sampling_interval = 1 / OR_sampling_freq  # Time interval between samples

            raw_dir = subject_dirs[subject.name][session.name]["emg_raw_dir"]

            session_data = subject_data[subject.name][session.name]["session_data"]

            # Initialize RAW data
            data_dict = {"EMG": [], "IMU": [], "OR": [], "LOG": []}
            
            relevant_cols = list(emg_config["MAPPING"].keys())
            
            for file_path in sorted(raw_dir.iterdir()):
                if file_path.suffix == ".csv":
                    tag = re.search(r"Profile_([^_]+)", file_path.stem).group(1)
                    if "EMG" in file_path.stem:
                        df = pd.read_csv(file_path, sep=",", usecols=relevant_cols, engine='python')
                        df.rename(columns=emg_config["MAPPING"], inplace=True)
                        df.sort_index(axis=1, inplace=True)

                        # Drop IMU column
                        df.drop(columns=["IMU"], inplace=True)
                        # Drop OR column
                        df.drop(columns=["OR"], inplace=True)
                        # Drop nan rows
                        df = df.dropna()
                        # Create the "TIME" column
                        num_samples = len(df)
                        time_column = pd.Series([i * sampling_interval for i in range(num_samples)], name="TIME")
                        df.set_index(time_column, inplace=True)  # Set time as index

                        if "Assisted" in file_path.stem:
                            # There is only one emg file per tag
                            if tag not in session_data["API_DATA"]["ASSISTED"]:
                                session_data["API_DATA"]["ASSISTED"][tag] = deepcopy(data_dict)
                            session_data["API_DATA"]["ASSISTED"][tag]["EMG"] = df

                        # elif profile in session_data["UNPOWERED"].keys():
                        elif "Unpowered" in file_path.stem:
                            if tag not in session_data["API_DATA"]["UNPOWERED"]:
                                session_data["API_DATA"]["UNPOWERED"][tag] = deepcopy(data_dict)
                            session_data["API_DATA"]["UNPOWERED"][tag]["EMG"] = df

                        # Would work, but as the raw data is not always needed, there is an extra function for MVIC
                        # elif "MVIC" in file_path.stem:
                        #     # Initialize with the new data
                        #     session_data["MVIC"] = df
                    
                    if "ACC" in file_path.stem:
                        # Ignore the columns from orientation data
                        df_acc = pd.read_csv(file_path, delimiter=',', usecols=[0, 1, 2])
                        # Rename the columns based on the file format
                        df_acc.columns = ['ACC IMU X', 'ACC IMU Y', 'ACC IMU Z']
                        df_acc = df_acc.apply(pd.to_numeric, errors='coerce')
                        # Remove columns with nan values
                        df_acc = df_acc.dropna()

                        # Replace "ACC" with "GYRO" in the file name
                        file_path = file_path.with_name(file_path.name.replace("ACC", "GYRO"))

                        # Read the GYRO data
                        df_gyro = pd.read_csv(file_path, delimiter=',', usecols=[0, 1, 2])
                        # Rename the columns based on the file format
                        df_gyro.columns = ['GYRO IMU X', 'GYRO IMU Y', 'GYRO IMU Z']
                        # Remove columns with nan values
                        df_gyro = df_gyro.dropna()

                        # Merge the ACC and GYRO data
                        df = pd.concat([df_acc, df_gyro], axis=1) 

                        # Create the "TIME" column
                        num_samples = len(df)
                        time_column = pd.Series([i * IMU_sampling_interval for i in range(num_samples)], name="TIME")
                        df.set_index(time_column, inplace=True)
                        
                        if "Assisted" in file_path.stem:
                            if tag not in session_data["API_DATA"]["ASSISTED"]:
                                session_data["API_DATA"]["ASSISTED"][tag] = deepcopy(data_dict)
                            session_data["API_DATA"]["ASSISTED"][tag]["IMU"] = df
                        elif "Unpowered" in file_path.stem:
                            if tag not in session_data["API_DATA"]["UNPOWERED"]:
                                session_data["API_DATA"]["UNPOWERED"][tag] = deepcopy(data_dict)
                            session_data["API_DATA"]["UNPOWERED"][tag]["IMU"] = df
                        else:
                            continue

                    if "OR" in file_path.stem:
                        df = pd.read_csv(file_path, delimiter=',')
                        # Rename the columns based on the file format
                        df.columns = ['OR X', 'OR Y', 'OR Z', 'OR W']
                        df = df.apply(pd.to_numeric, errors='coerce')
                        # Remove columns with nan values
                        df = df.dropna()

                        # Create the "TIME" column
                        num_samples = len(df)
                        time_column = pd.Series([i * OR_sampling_interval for i in range(num_samples)], name="TIME")
                        df.set_index(time_column, inplace=True)
                        
                        match = re.search(r'Profile_(.*?)_Trial', file_path.stem)
                        if match:
                            profile = match.group(1)
                        
                        if "Assisted" in file_path.stem:
                            if tag not in session_data["API_DATA"]["ASSISTED"]:
                                session_data["API_DATA"]["ASSISTED"][tag] = deepcopy(data_dict)
                            session_data["API_DATA"]["ASSISTED"][tag]["OR"] = df
                        elif "Unpowered" in file_path.stem:
                            if tag not in session_data["API_DATA"]["UNPOWERED"]:
                                session_data["API_DATA"]["UNPOWERED"][tag] = deepcopy(data_dict)
                            session_data["API_DATA"]["UNPOWERED"][tag]["OR"] = df
                        else:
                            continue

            subject_data[subject.name][session.name]["session_data"] = session_data

    return subject_data, subject_dirs, subjects

def load_mvic_data_hilo(subject_data, subject_dirs, subjects):
    # LOAD MVIC EMG DATA

    for subject in subjects:
        for session in subject.iterdir():
            if not session.is_dir():
                continue
            raw_dir = subject_dirs[subject.name][session.name]["emg_raw_dir"]
            session_data = subject_data[subject.name][session.name]["session_data"]
            emg_config = subject_data[subject.name][session.name]["emg_config"]

            sampling_freq =  emg_config["EMG_FREQUENCY"]
            sampling_interval = 1 / sampling_freq  # Time interval between samples
            
            relevant_cols = list(emg_config["MAPPING"].keys())

            for file_path in sorted(raw_dir.iterdir()):
                if file_path.suffix == ".csv":
                    if "EMG" in file_path.stem:
                        if "MVIC" in file_path.stem:
                            df = pd.read_csv(file_path, sep=",", usecols=relevant_cols, engine='python')
                            df.rename(columns=emg_config["MAPPING"], inplace=True)
                            df.sort_index(axis=1, inplace=True)

                            # Drop IMU column
                            df.drop(columns=["IMU"], inplace=True)
                            # Drop OR column
                            df.drop(columns=["OR"], inplace=True)
                            # Drop nan rows
                            df = df.dropna()
                        
                            # Create the "TIME" column
                            num_samples = len(df)
                            time_column = pd.Series([i * sampling_interval for i in range(num_samples)], name="TIME")
                            df.set_index(time_column, inplace=True)  # Set time as index            

                            # Initialize with the new data
                            session_data["MVIC"]["Unfiltered"] = df

                            filtered_df, _ = filter_emg(df, sfreq=sampling_freq)
                            session_data["MVIC"]["Filtered"] = filtered_df
                        
            subject_data[subject.name][session.name]["session_data"] = session_data

    return subject_data, subject_dirs, subjects


def find_closest_past_tag(tags, current_tag):
    # Convert to integers for comparison
    tags = [int(tag) for tag in tags]
    # Sort the tags
    tags.sort()
    current_tag = int(current_tag)
    
    # Find the insertion point using binary search
    index = bisect_left(tags, current_tag)
    
    # Check the closest past tag
    if index > 0:
        return str(tags[index - 1])  # The closest past tag
    return None

def load_segmented_emg_data_hilo(subject_data, subject_dirs, subjects):
    # LOAD STS EMG DATA

    for subject in subjects:
        for session in subject.iterdir():
            if not session.is_dir():
                continue
            processed_dir = subject_dirs[subject.name][session.name]["emg_sts_dir"]

            session_data = subject_data[subject.name][session.name]["session_data"]
            emg_config = subject_data[subject.name][session.name]["emg_config"]

            sampling_freq =  emg_config["EMG_FREQUENCY"]
            sampling_interval = 1 / sampling_freq  # Time interval between samples

            IMU_sampling_freq =  emg_config["IMU_FREQUENCY"]
            IMU_sampling_interval = 1 / IMU_sampling_freq  # Time interval between samples

            relevant_cols = list(emg_config["MAPPING"].keys())

            # Drop the IMU and OR columns
            relevant_cols.remove("EMG IMU")
            relevant_cols.remove("EMG OR")

            # Remove the "EMG " prefix
            relevant_cols = [col.replace("EMG ", "") for col in relevant_cols]
            
            for file_path in sorted(processed_dir.iterdir()):
                if file_path.suffix == ".csv":
                    tag = re.search(r"Profile_([^_]+)", file_path.stem).group(1)

                    if "all_Sensors" in file_path.stem:
                        df = pd.read_csv(file_path, sep=",", usecols=relevant_cols, engine='python')
                        df.rename(columns=emg_config["MAPPING"], inplace=True)
                        df.sort_index(axis=1, inplace=True)

                        # Create the "TIME" column
                        num_samples = len(df)
                        time_column = pd.Series([i * sampling_interval for i in range(num_samples)], name="TIME")
                        df.set_index(time_column, inplace=True)  # Set time as index

                        filtered_df, _ = filter_emg(df, sfreq=sampling_freq)
                        
                        if "Assisted" in file_path.stem:  
                            if tag in session_data["ASSISTED"]["ALL_TAGS"]:
                                if tag in session_data["ASSISTED"]["FIRST_TAGS"]:
                                    session_data["ASSISTED"][tag]["EMG"]["Unfiltered"].append(df)
                                    session_data["ASSISTED"][tag]["EMG"]["Filtered"].append(filtered_df)
                                else:
                                    # closest_first_tag = find_closest_past_tag(session_data["ASSISTED"]["FIRST_TAGS"], tag)
                                    # if closest_first_tag is not None:
                                    #     session_data["ASSISTED"][closest_first_tag]["EMG"]["Unfiltered"].append(df)
                                    #     session_data["ASSISTED"][closest_first_tag]["EMG"]["Filtered"].append(filtered_df)
                                    # else:
                                    #     print(f"Finding closest first tag failed") 
                                    for first_tag in session_data["ASSISTED"]["FIRST_TAGS"]:
                                        if tag in session_data["ASSISTED"][first_tag]["PROFILE_TAGS"]:
                                            session_data["ASSISTED"][first_tag]["EMG"]["Unfiltered"].append(df)
                                            session_data["ASSISTED"][first_tag]["EMG"]["Filtered"].append(filtered_df)
                                            continue
                                        
                            else:
                                print(f"File {file_path.stem} not found in session data")
                        
                        if "Unpowered" in file_path.stem:
                            if tag in session_data["UNPOWERED"]["ALL_TAGS"]:
                                if tag in session_data["UNPOWERED"]["FIRST_TAGS"]:
                                    session_data["UNPOWERED"][tag]["EMG"]["Unfiltered"].append(df)
                                    session_data["UNPOWERED"][tag]["EMG"]["Filtered"].append(filtered_df)
                                else:
                                    # closest_first_tag = find_closest_past_tag(session_data["UNPOWERED"]["FIRST_TAGS"], tag)
                                    # if closest_first_tag is not None:
                                    #     session_data["UNPOWERED"][closest_first_tag]["EMG"]["Unfiltered"].append(df)
                                    #     session_data["UNPOWERED"][closest_first_tag]["EMG"]["Filtered"].append(filtered_df)
                                    # else:
                                    #     print(f"Finding closest first tag failed")   
                                    for first_tag in session_data["UNPOWERED"]["FIRST_TAGS"]:
                                        if tag in session_data["UNPOWERED"][first_tag]["PROFILE_TAGS"]:
                                            session_data["UNPOWERED"][first_tag]["EMG"]["Unfiltered"].append(df)
                                            session_data["UNPOWERED"][first_tag]["EMG"]["Filtered"].append(filtered_df)
                                            continue
                            else:
                                print(f"File {file_path.stem} not found in session data")

                    # In case IMU or OR data is also saved
                    # if "Sensor_IMU" in file_path.stem:
                    #     df_acc = pd.read_csv(file_path, delimiter=',', usecols=[0, 1, 2])
                    #     # Rename the columns based on the file format
                    #     df_acc.columns = ['ACC IMU X', 'ACC IMU Y', 'ACC IMU Z']
                    #     df_acc = df_acc.apply(pd.to_numeric, errors='coerce')
                    #     # Remove columns with nan values
                    #     df_acc = df_acc.dropna()

                    #     # Replace "ACC" with "GYRO" in the file name
                    #     file_path = file_path.with_name(file_path.name.replace("ACC", "GYRO"))

                    #     # Read the GYRO data
                    #     df_gyro = pd.read_csv(file_path, delimiter=',', usecols=[0, 1, 2])
                    #     # Rename the columns based on the file format
                    #     df_gyro.columns = ['GYRO IMU X', 'GYRO IMU Y', 'GYRO IMU Z']
                    #     # Remove columns with nan values
                    #     df_gyro = df_gyro.dropna()

                    #     # Merge the ACC and GYRO data
                    #     df = pd.concat([df_acc, df_gyro], axis=1) 

                    #     # Create the "TIME" column
                    #     num_samples = len(df)
                    #     time_column = pd.Series([i * IMU_sampling_interval for i in range(num_samples)], name="TIME")
                    #     df.set_index(time_column, inplace=True)
                        
                    #     match = re.search(r'Profile_(.*?)_Trial', file_path.stem)
                    #     if match:
                    #         profile = match.group(1)
                        
                    #     if profile in session_data["UNPOWERED"].keys():
                    #         session_data["UNPOWERED"][profile]["IMU"].append(df)

                    #     elif "MVIC" in file_path.stem:
                    #         # Nothing
                    #         continue

                    #     elif profile in session_data["ASSISTED"].keys():
                    #         session_data["ASSISTED"][profile]["IMU"].append(df)

            subject_data[subject.name][session.name]["session_data"] = session_data

    return subject_data, subject_dirs, subjects


def load_log_data_hilo(subject_data, subject_dirs, subjects):
    # LOAD LOG DATA
    for subject in subjects:
        for session in subject.iterdir():
            if not session.is_dir():
                continue
            log_dir = subject_dirs[subject.name][session.name]["log_dir"]
            session_data = subject_data[subject.name][session.name]["session_data"]

            complete_log_data = pd.DataFrame()
            
            for file_path in sorted(log_dir.iterdir()):
                if file_path.suffix == ".csv":
                    # File size should be larger than 5 bytes to avoid empty files
                    if os.stat(file_path).st_size > 5:

                        df = pd.read_csv(file_path)

                        # Get the first tag from the file name
                        # Only works, if other API daa was loaded (forgot to add the what tag for logging in the emg script for)
                        if session_data["API_DATA"]["ASSISTED"] != {}:
                            first_tag = re.search(r"Profile_(\d+)_", file_path.stem).group(1)
                            if first_tag in session_data["ASSISTED"]["ALL_TAGS"]:
                                session_data["API_DATA"]["ASSISTED"][first_tag]["LOG_INFO"].append(df)
                            elif first_tag in session_data["UNPOWERED"]["ALL_TAGS"]:
                                session_data["API_DATA"]["UNPOWERED"][first_tag]["LOG_INFO"].append(df)
                            else:
                                print(f"Log tag {first_tag} did not match any tag in the session data")

                        for i, info in df.iterrows():
                            tag = str(int(info.iloc[0]))
                            duration = info.iloc[2] - info.iloc[1]
                        
                            if tag in session_data["ASSISTED"]["ALL_TAGS"]:
                                if tag in session_data["ASSISTED"]["FIRST_TAGS"]:
                                    session_data["ASSISTED"][tag]["LOG_INFO"].append(info)
                                else:
                                    # closest_first_tag = find_closest_past_tag(session_data["ASSISTED"]["FIRST_TAGS"], tag)
                                    # if closest_first_tag is not None:
                                    #     session_data["ASSISTED"][closest_first_tag]["LOG_INFO"].append(info)
                                    # else:
                                    #     print(f"Tag {tag} not found in session data, duration was {duration}")
                                    for first_tag in session_data["ASSISTED"]["FIRST_TAGS"]:
                                        if tag in session_data["ASSISTED"][first_tag]["PROFILE_TAGS"]:
                                            session_data["ASSISTED"][first_tag]["LOG_INFO"].append(info)
                                            continue

                            elif tag in session_data["UNPOWERED"]["ALL_TAGS"]:
                                if tag in session_data["UNPOWERED"]["FIRST_TAGS"]:
                                    session_data["UNPOWERED"][tag]["LOG_INFO"].append(info)
                                else:
                                    # closest_first_tag = find_closest_past_tag(session_data["UNPOWERED"]["FIRST_TAGS"], tag)
                                    # if closest_first_tag is not None:
                                    #     session_data["UNPOWERED"][closest_first_tag]["LOG_INFO"].append(info)
                                    # else:
                                    #     print(f"Tag {tag} not found in session data, duration was {duration}")
                                    for first_tag in session_data["UNPOWERED"]["FIRST_TAGS"]:
                                        if tag in session_data["UNPOWERED"][first_tag]["PROFILE_TAGS"]:
                                            session_data["UNPOWERED"][first_tag]["LOG_INFO"].append(info)
                                            continue
               
            subject_data[subject.name][session.name]["session_data"] = session_data
    
    return subject_data, subject_dirs, subjects


def load_optimizer_data_hilo(subject_data, subject_dirs, subjects):
    # LOAD OPTIMIZER DATA
    for subject in subjects:
        for session in subject.iterdir():
            if not session.is_dir():
                continue
            optimizer_dir = subject_dirs[subject.name][session.name]["motor_dir"]
            session_data = subject_data[subject.name][session.name]["session_data"]
            
            for file_path in sorted(optimizer_dir.iterdir()):
                if file_path.suffix == ".json":
                    if "optimizer_logs" in file_path.stem:

                        # load json file
                        with jsonlines.open(file_path) as reader:
                            data = [record for record in reader]

                        df = pd.json_normalize(data)

                        # Remove duplicates
                        df = df[df['datetime.delta'] >= 1]

                        # Convert datetime
                        # df["profile"] = pd.to_datetime(df['datetime.datetime']).dt.strftime('%Y%m%d%H%M%S')

                        session_data["OPTIMIZER"] = df

            subject_data[subject.name][session.name]["session_data"] = session_data
    
    return subject_data, subject_dirs, subjects
