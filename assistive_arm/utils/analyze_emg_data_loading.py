import yaml
from pathlib import Path
import pandas as pd
import jsonlines
import os
import json
import re
from copy import deepcopy
from assistive_arm.utils.data_preprocessing import read_headers



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
                emg_dir = session_dir / "EMG"
                log_dir = session_dir / "Log"
                mvic_dir = session_dir / "MVIC"
                plot_dir = session_dir / "plots"
                plot_dir.mkdir(exist_ok=True)

                angle_calibration_path = motor_dir / "device_height_calibration.yaml"

                subject_dirs[subject_name][session.name]["emg_dir"] = emg_dir
                subject_dirs[subject_name][session.name]["plot_dir"] = plot_dir
                subject_dirs[subject_name][session.name]["motor_dir"] = motor_dir
                subject_dirs[subject_name][session.name]["log_dir"] = log_dir
                subject_dirs[subject_name][session.name]["mvic_dir"] = mvic_dir
                subject_dirs[subject_name][session.name]["angle_calibration_path"] = angle_calibration_path


                # Read EMG file to extract configuration
                # Extract one header
                emg_raw = emg_dir / "Raw" 

                # Get a random emg file containing the string "EMG" in the file name
                emg_file = next(emg_raw.glob("*EMG*.csv"))

                emg_config = dict()

                for i, header in enumerate(read_headers(emg_file, 1, delimiter="\t")):
                    print(i, header)
                    if i == 0:
                        emg_config["CHANNEL_NAMES"] = header
                        # TODO get real emg frequency
                        emg_config["FREQUENCY"] = float(2148.259)
                        emg_config["IMU_FREQUENCY"] = float(518.519)

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


def load_motor_data(subject_data, subject_dirs, subjects):
    # LOAD MOTOR DATA
    session_dict = {}
    session_dict["MVIC"] = {}
    session_dict["MVIC"]["LEFT"] = None
    session_dict["MVIC"]["RIGHT"] = None

    data_dict = {"EMG": [],
                "MOTOR_DATA": [],
                "IMU": [],
                "LOG": []
                }

    session_dict["UNPOWERED"] = {}
    session_dict["ASSISTED"] = {}
    session_dict["UNPOWERED"]["BEFORE"] = deepcopy(data_dict)
    session_dict["UNPOWERED"]["AFTER"] = deepcopy(data_dict)

    # Load all session data
    profile_to_num = {}
    profile_infos = {}

    # Skip first 0.5s to avoid initial noise
    skip_first = 0.0 #s
    remove_unpowered_after = False

    # Counter for the number of profiles
    # profile_counter = 0

    # Handle all motor data
    for subject in subjects:
        for session in subject.iterdir():
            if not session.is_dir():
                continue
            session_data = deepcopy(session_dict)
            motor_dir = subject_dirs[subject.name][session.name]["motor_dir"]
            for file_path in sorted(motor_dir.iterdir()):
                if file_path.suffix == ".csv":
                    if "unassisted" in file_path.stem:
                        df = pd.read_csv(file_path, index_col="time").loc[skip_first:]
                        
                        # match = re.search(r'unpowered_device_(\d+)', file_path.stem)
                        # if match:
                        #     num = match.group(1)
                        if "before" in file_path.stem:
                            session_data["UNPOWERED"]["BEFORE"]["MOTOR_DATA"].append(df)
                            # if not isinstance(session_data["UNPOWERED"]["BEFORE"]["MOTOR_DATA"], dict):
                            #     session_data["UNPOWERED"]["BEFORE"]["MOTOR_DATA"] = {}
                            # session_data["UNPOWERED"]["BEFORE"]["MOTOR_DATA"][num] = df
                        else:
                            session_data["UNPOWERED"]["AFTER"]["MOTOR_DATA"].append(df)
                            # if not isinstance(session_data["UNPOWERED"]["AFTER"]["MOTOR_DATA"], dict):
                            #     session_data["UNPOWERED"]["AFTER"]["MOTOR_DATA"] = {}
                            # session_data["UNPOWERED"]["AFTER"]["MOTOR_DATA"][num] = df



                    elif "scaled" in file_path.stem:
                        df = pd.read_csv(file_path, index_col="time").loc[skip_first:]

                        match = re.search(r'scaled_(\d+)_([^_]+)_', file_path.stem)
                        if match:
                            num = match.group(1)

                        match = re.search(r'scaled_\d+_(.*)_\d+$', file_path.stem)
                        if match:
                            profile = match.group(1)

                        if profile not in session_data["ASSISTED"].keys():
                            session_data["ASSISTED"][profile] = deepcopy(data_dict)
                        #     session_data["ASSISTED"][profile]["MOTOR_DATA"] = {}
                        # session_data["ASSISTED"][profile]["MOTOR_DATA"][num] = df

                        session_data["ASSISTED"][profile]["MOTOR_DATA"].append(df)

            subject_data[subject.name][session.name]["session_data"] = session_data

    return subject_data, subject_dirs, subjects, data_dict



def load_emg_data(subject_data, subject_dirs, subjects):

    # LOAD EMG DATA
    sampling_freq =  float(2148.259)  # Frequency in Hz
    sampling_interval = 1 / sampling_freq  # Time interval between samples

    IMU_sampling_freq = 518.519  # Frequency in Hz
    IMU_sampling_interval = 1 / IMU_sampling_freq  # Time interval between samples


    for subject in subjects:
        for session in subject.iterdir():
            if not session.is_dir():
                continue
            emg_dir = subject_dirs[subject.name][session.name]["emg_dir"]
            raw_dir = emg_dir / "Raw"

            session_data = subject_data[subject.name][session.name]["session_data"]
            emg_config = subject_data[subject.name][session.name]["emg_config"]
            
            relevant_cols = list(emg_config["MAPPING"].keys())
            
            # Handle second EMG data
            for file_path in sorted(raw_dir.iterdir()):
                if file_path.suffix == ".csv":
                    if "EMG" in file_path.stem:
                        if "unassisted" in file_path.stem:
                            # n = int(file_path.stem.split("_")[1])
            
                            df = pd.read_csv(file_path, sep=",", usecols=relevant_cols, engine='python')                    
                            df.rename(columns=emg_config["MAPPING"], inplace=True)
                            df.sort_index(axis=1, inplace=True)

                            # Create the "TIME" column
                            num_samples = len(df)
                            time_column = pd.Series([i * sampling_interval for i in range(num_samples)], name="TIME")
                            # df.insert(0, "TIME", time_column)  # Insert at the first column
                            df.set_index(time_column, inplace=True)  # Set time as index

                            num = re.search(r'\d+$', file_path.stem).group()

                            # Add fake TA and SO data
                            df["TA_LEFT"] = df["TA_RIGHT"]
                            df["SO_LEFT"] = df["SO_RIGHT"]

                            if "before" in file_path.stem:
                                session_data["UNPOWERED"]["BEFORE"]["EMG"].append(df)
                                # if not isinstance(session_data["UNPOWERED"]["BEFORE"]["EMG"], dict):
                                #     session_data["UNPOWERED"]["BEFORE"]["EMG"] = {}
                                # session_data["UNPOWERED"]["BEFORE"]["EMG"][num] = df
                            else:
                                session_data["UNPOWERED"]["AFTER"]["EMG"].append(df)
                                # if not isinstance(session_data["UNPOWERED"]["AFTER"]["EMG"], dict):
                                #     session_data["UNPOWERED"]["AFTER"]["EMG"] = {}
                                # session_data["UNPOWERED"]["AFTER"]["EMG"][num] = df

                                
                        elif "MVIC" in file_path.stem:
                            df = pd.read_csv(file_path, sep=",", usecols=relevant_cols, engine='python')
                            df.rename(columns=emg_config["MAPPING"], inplace=True)
                            df.sort_index(axis=1, inplace=True)
                            
                            # Create the "TIME" column
                            num_samples = len(df)
                            time_column = pd.Series([i * sampling_interval for i in range(num_samples)], name="TIME")
                            df.set_index(time_column, inplace=True)  # Set time as index            

                            if "BF_L" in file_path.name:  # If it's the BF file
                                side = "LEFT"
                                bf_col = [col for col in df.columns if "BF_" + side in col]  # Adjust this if your naming convention is different
                                df = df[bf_col]
                            elif "RF_L" in file_path.name:
                                side = "LEFT"
                                rf_vm_cols = [col for col in df.columns if any(sub in col for sub in ["RF_" + side, "VM_" + side])]
                                df = df[rf_vm_cols]
                            elif "G_L" in file_path.name:
                                side = "LEFT"
                                g_col = [col for col in df.columns if "G_" + side in col]
                                df = df[g_col]

                            elif "BF_R" in file_path.name:  # If it's the BF file
                                side = "RIGHT"
                                bf_col = [col for col in df.columns if "BF_" + side in col]  # Adjust this if your naming convention is different
                                df = df[bf_col]
                            elif "RF_R" in file_path.name:
                                side = "RIGHT"
                                rf_vm_cols = [col for col in df.columns if any(sub in col for sub in ["RF_" + side, "VM_" + side])]
                                df = df[rf_vm_cols]
                            elif "G_R" in file_path.name:
                                side = "RIGHT"
                                g_col = [col for col in df.columns if "G_" + side in col]
                                df = df[g_col]

                            # Aditional muscles
                            elif "SO_R" in file_path.name:
                                side = "RIGHT"
                                so_col = [col for col in df.columns if "SO_" + side in col]
                                df = df[so_col]
                                # continue
                            elif "TA_R" in file_path.name:
                                side = "RIGHT"
                                ta_col = [col for col in df.columns if "TA_" + side in col]
                                df = df[ta_col]
                                # continue

                            if session_data["MVIC"][side] is not None:
                                # Merge with existing data
                                session_data["MVIC"][side] = pd.merge(session_data["MVIC"][side], df, left_index=True, right_index=True, how='outer')
                            else:
                                # Initialize with the new data
                                session_data["MVIC"][side] = df


                        else:
                            df = pd.read_csv(file_path, sep=",", usecols=relevant_cols, engine='python')
                            df.rename(columns=emg_config["MAPPING"], inplace=True)
                            df.sort_index(axis=1, inplace=True)
                            data_type = "EMG"
                            # print(file_path.stem)

                            # Create the "TIME" column
                            num_samples = len(df)
                            time_column = pd.Series([i * sampling_interval for i in range(num_samples)], name="TIME")
                            df.set_index(time_column, inplace=True)  # Set time as index

                            num = re.search(r'\d+$', file_path.stem).group()
                            match = re.search(r'Profile_(.*?)_Trial', file_path.stem)

                            if match:
                                profile = match.group(1)

                            # Add fake TA and SO data
                            df["TA_LEFT"] = df["TA_RIGHT"]
                            df["SO_LEFT"] = df["SO_RIGHT"]
                            
                            # if not isinstance(session_data["ASSISTED"][profile]["EMG"], dict):
                            #     session_data["ASSISTED"][profile]["EMG"] = {} 
                            # session_data["ASSISTED"][profile]["EMG"][num] = df

                            session_data["ASSISTED"][profile]["EMG"].append(df)
                    
                    if "ACC" in file_path.stem:
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

                        num = re.search(r'\d+$', file_path.stem).group()

                        if "unassisted" in file_path.stem:
                            if "before" in file_path.stem:
                                # if not isinstance(session_data["UNPOWERED"]["BEFORE"]["IMU"], dict):
                                #     session_data["UNPOWERED"]["BEFORE"]["IMU"] = {}
                                # session_data["UNPOWERED"]["BEFORE"]["IMU"][num] = df
                                session_data["UNPOWERED"]["BEFORE"]["IMU"].append(df)
                            else:
                                # if not isinstance(session_data["UNPOWERED"]["AFTER"]["IMU"], dict):
                                #     session_data["UNPOWERED"]["AFTER"]["IMU"] = {}
                                # session_data["UNPOWERED"]["AFTER"]["IMU"][num] = df
                                session_data["UNPOWERED"]["AFTER"]["IMU"].append(df)

                        elif "MVIC" in file_path.stem:
                            # Nothing
                            continue

                        else:
                            match = re.search(r'Profile_(.*?)_Trial', file_path.stem)
                            if match:
                                profile = match.group(1)
                            # if not isinstance(session_data["ASSISTED"][profile]["IMU"], dict):
                            #     session_data["ASSISTED"][profile]["IMU"] = {} 
                            # session_data["ASSISTED"][profile]["IMU"][num] = df
                            session_data["ASSISTED"][profile]["IMU"].append(df)

            subject_data[subject.name][session.name]["session_data"] = session_data

    return subject_data, subject_dirs, subjects


def load_imu_data(subject_data, subject_dirs, subjects):
    # LOAD LOG DATA
    for subject in subjects:
        for session in subject.iterdir():
            if not session.is_dir():
                continue
            log_dir = subject_dirs[subject.name][session.name]["log_dir"]
            session_data = subject_data[subject.name][session.name]["session_data"]
            
            # Handle second EMG data
            for file_path in sorted(log_dir.iterdir()):
                if file_path.suffix == ".csv":
                    # File size should be larger than 5 bytes to avoid empty files
                    if os.stat(file_path).st_size > 5:
                        if "unassisted" in file_path.stem:
                            df = pd.read_csv(file_path)
                            
                            if "before" in file_path.stem:
                                session_data["UNPOWERED"]["BEFORE"]["LOG"].append(df)
                            else:
                                remove_unpowered_after = False
                                session_data["UNPOWERED"]["AFTER"]["LOG"].append(df)

                        else:
                            df = pd.read_csv(file_path)
                            match = re.search(r'Profile_(.*?)_Trial', file_path.stem)
                            if match:
                                profile = match.group(1)
                                session_data["ASSISTED"][profile]["LOG"].append(df)


            subject_data[subject.name][session.name]["session_data"] = session_data
    
    return subject_data, subject_dirs, subjects



def load_motor_data_hilo(subject_data, subject_dirs, subjects):
    # LOAD MOTOR DATA
    session_dict = {}
    session_dict["MVIC"] = {}
    session_dict["MVIC"]["LEFT"] = None
    session_dict["MVIC"]["RIGHT"] = None

    data_dict = {"EMG": [], "EMG_RAW": [], "MOTOR_DATA": [], "IMU": [], "IMU_RAW": [], "OR": [], "LOG": [], "PROFILE": []}

    session_dict["UNPOWERED"] = {}
    session_dict["ASSISTED"] = {}

    name_tag_mapping = {}

    # Skip first 0.5s to avoid initial noise
    skip_first = 0.0 #s

    for subject in subjects:
        for session in iter(subject_data[subject.name].keys()):

            session_data = deepcopy(session_dict)
            motor_dir = subject_dirs[subject.name][session]["motor_dir"]

            sorted_files = sorted(motor_dir.iterdir(), key=lambda f: f.stat().st_ctime)
            unpowered_iteration = 0

            for file_path in sorted_files:
                if file_path.suffix == ".csv":
                    if "unpowered" in file_path.stem:
                        df = pd.read_csv(file_path, index_col="time").loc[skip_first:]

                        parts = re.split('_', file_path.stem)
                        # profile = parts[-2]
                        # TODO fix, once better identifiers added in collection
                        profile = f"{parts[-2]}_{parts[-1]}"
                        
                        if profile not in session_data["UNPOWERED"].keys():
                            session_data["UNPOWERED"][profile] = deepcopy(data_dict)
                            name_tag_mapping[profile] = f"Unpowered_iteration_{unpowered_iteration}"
                            unpowered_iteration += 1
                        
                        session_data["UNPOWERED"][profile]["MOTOR_DATA"].append(df)

                    elif "t11" in file_path.stem:
                        df = pd.read_csv(file_path, skiprows = 6, index_col="time").loc[skip_first:]

                        parts = re.split('_', file_path.stem)
                        profile = parts[-2]

                        if profile not in session_data["ASSISTED"].keys():
                            session_data["ASSISTED"][profile] = deepcopy(data_dict)

                        name_tag_mapping[profile] = f"Assisted X-force(t11_{parts[1]}, f11_{parts[3]}), Y-force(t21_{parts[5]}, t22_{parts[7]}, t23_{parts[9]}, f21_{parts[11]})"

                        session_data["ASSISTED"][profile]["MOTOR_DATA"].append(df)

            subject_data[subject.name][session]["session_data"] = session_data
            subject_data[subject.name][session]["name_tag_mapping"] = name_tag_mapping

    return subject_data, subject_dirs, subjects, data_dict


def load_profile_data_hilo(subject_data, subject_dirs, subjects):

    # Skip first 0.5s to avoid initial noise
    skip_first = 0.0 #s

    for subject in subjects:
        for session in iter(subject_data[subject.name].keys()):
            profile_dir = subject_dirs[subject.name][session]["motor_dir"] / "profiles"

            session_data = subject_data[subject.name][session]["session_data"]

            sorted_files = sorted(profile_dir.iterdir(), key=lambda f: f.stat().st_ctime)
            unpowered_iteration = 0

            for file_path in sorted_files:
                if file_path.suffix == ".csv":
                    if "t11" in file_path.stem:
                        df = pd.read_csv(file_path)

                        match = re.search(r"Profile_([^_]+)", file_path.stem)
                        if match:
                            profile = match.group(1)

                        if profile in session_data["ASSISTED"].keys():
                            session_data["ASSISTED"][profile]["PROFILE"] = df
                        else: 
                            print(f"Profile {profile} not found in session {session}")

            subject_data[subject.name][session]["session_data"] = session_data

    return subject_data, subject_dirs, subjects


def load_raw_emg_imu_data_hilo(subject_data, subject_dirs, subjects):
    # LOAD EMG DATA
    sampling_freq =  float(2148.259)  # Frequency in Hz
    sampling_interval = 1 / sampling_freq  # Time interval between samples

    IMU_sampling_freq = 518.519  # Frequency in Hz
    IMU_sampling_interval = 1 / IMU_sampling_freq  # Time interval between samples

    OR_sampling_freq = 222.222  # Frequency in Hz
    OR_sampling_interval = 1 / OR_sampling_freq  # Time interval between samples

    for subject in subjects:
        for session in subject.iterdir():
            if not session.is_dir():
                continue
            emg_dir = subject_dirs[subject.name][session.name]["emg_dir"]
            raw_dir = emg_dir / "Raw"

            session_data = subject_data[subject.name][session.name]["session_data"]
            emg_config = subject_data[subject.name][session.name]["emg_config"]
            
            relevant_cols = list(emg_config["MAPPING"].keys())
            
            for file_path in sorted(raw_dir.iterdir()):
                if file_path.suffix == ".csv":
                    if "EMG" in file_path.stem:
                        match = re.search(r"Profile_([^_]+)", file_path.stem)
                        if match:
                            profile = match.group(1)

                        if profile in session_data["ASSISTED"].keys():
                            df = pd.read_csv(file_path, sep=",", usecols=relevant_cols, engine='python')
                            df.rename(columns=emg_config["MAPPING"], inplace=True)
                            df.sort_index(axis=1, inplace=True)

                            match = re.search(r'Profile_(.*?)_Trial', file_path.stem)
                            if match:
                                profile = match.group(1)

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
                            session_data["ASSISTED"][profile]["EMG_RAW"].append(df)

                                
                        elif "MVIC" in file_path.stem:
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
                            session_data["MVIC"] = df

                        # elif profile in session_data["UNPOWERED"].keys():
                        # TODO fix, once better identifiers added in collection
                        else:
                            # n = int(file_path.stem.split("_")[1])
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

                            # TODO remove this
                            if profile not in session_data["UNPOWERED"].keys():
                                session_data["UNPOWERED"][profile] = {"EMG": [], "EMG_RAW": [], "MOTOR_DATA": [], "IMU": [], "IMU_RAW": [], "OR": [], "LOG": [], "PROFILE": []}

                            session_data["UNPOWERED"][profile]["EMG_RAW"].append(df)
                    
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
                        
                        match = re.search(r'Profile_(.*?)_Trial', file_path.stem)
                        if match:
                            profile = match.group(1)
                        
                        if profile in session_data["UNPOWERED"].keys():
                            session_data["UNPOWERED"][profile]["IMU_RAW"].append(df)

                        elif "MVIC" in file_path.stem:
                            # Nothing
                            continue

                        elif profile in session_data["ASSISTED"].keys():
                            session_data["ASSISTED"][profile]["IMU_RAW"].append(df)

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
                        
                        if profile in session_data["UNPOWERED"].keys():
                            session_data["UNPOWERED"][profile]["OR"].append(df)

                        elif "MVIC" in file_path.stem:
                            # Nothing
                            continue

                        elif profile in session_data["ASSISTED"].keys():
                            session_data["ASSISTED"][profile]["OR"].append(df)

            subject_data[subject.name][session.name]["session_data"] = session_data

    return subject_data, subject_dirs, subjects

def load_mvic_data_hilo(subject_data, subject_dirs, subjects):
    # LOAD MVIC EMG DATA
    sampling_freq =  float(2148.259)  # Frequency in Hz
    sampling_interval = 1 / sampling_freq  # Time interval between samples

    for subject in subjects:
        for session in subject.iterdir():
            if not session.is_dir():
                continue
            mvic_dir = subject_dirs[subject.name][session.name]["mvic_dir"]
            session_data = subject_data[subject.name][session.name]["session_data"]
            emg_config = subject_data[subject.name][session.name]["emg_config"]
            
            relevant_cols = list(emg_config["MAPPING"].keys())

            for file_path in sorted(mvic_dir.iterdir()):
                if file_path.suffix == ".csv":
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

                        # Initialize with the new data
                        session_data["MVIC"] = df

            subject_data[subject.name][session.name]["session_data"] = session_data

    return subject_data, subject_dirs, subjects

def load_processed_emg_imu_data_hilo(subject_data, subject_dirs, subjects):
    # LOAD EMG DATA
    sampling_freq =  float(2148.259)  # Frequency in Hz
    sampling_interval = 1 / sampling_freq  # Time interval between samples

    IMU_sampling_freq = 518.519  # Frequency in Hz
    IMU_sampling_interval = 1 / IMU_sampling_freq  # Time interval between samples

    for subject in subjects:
        for session in subject.iterdir():
            if not session.is_dir():
                continue
            emg_dir = subject_dirs[subject.name][session.name]["emg_dir"]
            processed_dir = emg_dir / "sts"

            session_data = subject_data[subject.name][session.name]["session_data"]
            emg_config = subject_data[subject.name][session.name]["emg_config"]
            
            relevant_cols = list(emg_config["MAPPING"].keys())

            # Drop the IMU and OR columns
            relevant_cols.remove("EMG IMU")
            relevant_cols.remove("EMG OR")

            # Remove the "EMG " prefix
            relevant_cols = [col.replace("EMG ", "") for col in relevant_cols]
            
            for file_path in sorted(processed_dir.iterdir()):
                if file_path.suffix == ".csv":
                    if "all_Sensors" in file_path.stem:
                        match = re.search(r"Profile_([^_]+)", file_path.stem)
                        if match:
                            profile = match.group(1)

                        if profile in session_data["ASSISTED"].keys():
                            df = pd.read_csv(file_path, sep=",", usecols=relevant_cols, engine='python')
                            df.rename(columns=emg_config["MAPPING"], inplace=True)
                            df.sort_index(axis=1, inplace=True)

                            match = re.search(r'Profile_(.*?)_Trial', file_path.stem)
                            if match:
                                profile = match.group(1)

                            # Create the "TIME" column
                            num_samples = len(df)
                            time_column = pd.Series([i * sampling_interval for i in range(num_samples)], name="TIME")
                            df.set_index(time_column, inplace=True)  # Set time as index
                            session_data["ASSISTED"][profile]["EMG"].append(df)
                    
                        # elif profile in session_data["UNPOWERED"].keys():
                        # TODO fix, once better identifiers added in collection
                        else:
                            df = pd.read_csv(file_path, sep=",", usecols=relevant_cols, engine='python')                    
                            df.rename(columns=emg_config["MAPPING"], inplace=True)
                            df.sort_index(axis=1, inplace=True)

                            # Create the "TIME" column
                            num_samples = len(df)
                            time_column = pd.Series([i * sampling_interval for i in range(num_samples)], name="TIME")
                            df.set_index(time_column, inplace=True)  # Set time as index


                            # TODO remove this
                            if profile not in session_data["UNPOWERED"].keys():
                                session_data["UNPOWERED"][profile] = {"EMG": [], "EMG_RAW": [], "MOTOR_DATA": [], "IMU": [], "IMU_RAW": [], "OR": [], "LOG": [], "PROFILE": []}

                            session_data["UNPOWERED"][profile]["EMG"].append(df)

                    if "Sensor_IMU" in file_path.stem:
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
                        
                        match = re.search(r'Profile_(.*?)_Trial', file_path.stem)
                        if match:
                            profile = match.group(1)
                        
                        if profile in session_data["UNPOWERED"].keys():
                            session_data["UNPOWERED"][profile]["IMU"].append(df)

                        elif "MVIC" in file_path.stem:
                            # Nothing
                            continue

                        elif profile in session_data["ASSISTED"].keys():
                            session_data["ASSISTED"][profile]["IMU"].append(df)

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

                        complete_log_data = pd.concat([complete_log_data, df])
                        
                        logged_profiles = df["Tag"].unique()
                        matched_profiles = []

                        for i, tag in df.iterrows():
                            profile = str(int(tag.iloc[0]))
                            score = tag.iloc[-1]
                            info = tag
                            duration = tag.iloc[2] - tag.iloc[1]
                            
                            if profile not in matched_profiles and duration <= 10000:
                                # 0 score are unassisted trials
                                if score == 0:
                                    if profile in session_data["UNPOWERED"].keys():
                                        # Get the next 4 entries, those and the current are the log data for all 5 iterations of that profile
                                        session_data["UNPOWERED"][profile]["LOG"].append(info)
                                        matched_profiles.append(profile)
                                        # Collect the next 4 tags and append their info to the same list
                                        for offset in range(1, 5):
                                            if i + offset < len(df):
                                                next_tag = df.iloc[i + offset]
                                                next_info = next_tag 
                                                duration = next_tag.iloc[2] - next_tag.iloc[1]
                                                if duration <= 10000:
                                                    session_data["UNPOWERED"][profile]["LOG"].append(next_info)
                                                    matched_profiles.append(str(next_tag.iloc[0]))
                                else:
                                    if profile in session_data["ASSISTED"].keys():
                                        # Get the next 4 entries, those and the current are the log data for all 5 iterations of that profile
                                        session_data["ASSISTED"][profile]["LOG"].append(info)
                                        matched_profiles.append(profile)
                                        # Collect the next 4 tags and append their info to the same list
                                        for offset in range(1, 5):
                                            if i + offset < len(df):
                                                next_tag = df.iloc[i + offset]
                                                next_info = next_tag
                                                duration = next_tag.iloc[2] - next_tag.iloc[1]
                                                if duration <= 10000:
                                                    session_data["ASSISTED"][profile]["LOG"].append(next_info)
                                                    matched_profiles.append(str(next_tag.iloc[0]))

                        # Check if all profiles were found
                        if len(logged_profiles) != len(matched_profiles):
                            not_found_profiles = set(logged_profiles) - set(matched_profiles)
                            print(f"Profiles not found: {not_found_profiles}")
                            
                            # For the profiles, that were not found, check the duration of the trial
                            for profile in not_found_profiles:
                                profile_data = df[df["Tag"] == profile]
                                length = profile_data["End Time"] - profile_data["Start Time"]
                                print(f"Profile {profile} duration: {length}")
                            
            # Add the complete log data
            subject_data[subject.name][session.name]["complete_log_data"] = complete_log_data
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