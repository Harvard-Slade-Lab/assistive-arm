import yaml
from pathlib import Path
import pandas as pd
import os
import re
from copy import deepcopy
from assistive_arm.utils.data_preprocessing import read_headers



def setup_datastructure():
    # Setup the directories and datastructures

    subject_dir = Path(f"../subject_logs/")
    subjects = sorted([subject for subject in subject_dir.iterdir() if subject.is_dir()])

    subject_data = {}

    subject_dirs = {}

    for subject in subjects:
        subject_name = subject.name
        subject_data[subject_name] = {}
        subject_dirs[subject_name] = {}

        for session in subject.iterdir():
            if session.is_dir():
                session_dir = session
                subject_data[subject_name][session.name] = {}
                subject_dirs[subject_name][session.name] = {}

                with open("../motor_config.yaml", "r") as f:
                    motor_config = yaml.load(f, Loader=yaml.FullLoader)
                
                motor_dir = session_dir / "Motor"
                emg_dir = session_dir / "EMG"
                log_dir = session_dir / "Log"
                plot_dir = session_dir / "plots"
                plot_dir.mkdir(exist_ok=True)

                angle_calibration_path = motor_dir / "device_height_calibration.yaml"

                subject_dirs[subject_name][session.name]["emg_dir"] = emg_dir
                subject_dirs[subject_name][session.name]["plot_dir"] = plot_dir
                subject_dirs[subject_name][session.name]["motor_dir"] = motor_dir
                subject_dirs[subject_name][session.name]["log_dir"] = log_dir
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

                subject_data[subject_name][session.name]["emg_config"] = emg_config

    return subject_data, subject_dirs, subjects


def muscle_mapping(emg_config):
    # The mapping converts the col name from the csv. output file to a muscle name
    mapping = {}
    for channel in emg_config["CHANNEL_NAMES"][0].split(','):
        info = channel.split(" ")[1:2]
        if info[0] == "IMU":
            mapping[channel] = "IMU"
            
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
                        
                        match = re.search(r'unpowered_device_(\d+)', file_path.stem)
                        if match:
                            num = match.group(1)
                        if "before" in file_path.stem:
                            session_data["UNPOWERED"]["BEFORE"]["MOTOR_DATA"].append(df)
                            # if not isinstance(session_data["UNPOWERED"]["BEFORE"]["MOTOR_DATA"], dict):
                            #     session_data["UNPOWERED"]["BEFORE"]["MOTOR_DATA"] = {}
                            # session_data["UNPOWERED"]["BEFORE"]["MOTOR_DATA"][num] = df
                        else:
                            remove_unpowered_after = False
                            session_data["UNPOWERED"]["AFTER"]["MOTOR_DATA"].append(df)
                            # if not isinstance(session_data["UNPOWERED"]["AFTER"]["MOTOR_DATA"], dict):
                            #     session_data["UNPOWERED"]["AFTER"]["MOTOR_DATA"] = {}
                            # session_data["UNPOWERED"]["AFTER"]["MOTOR_DATA"][num] = df



                    elif "scaled" or "no_counter" in file_path.stem:
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

    return subject_data, subject_dirs, subjects



def load_emg_data(subject_data, subject_dirs, subjects):

    # LOAD EMG DATA
    sampling_freq =  emg_config["FREQUENCY"]  # Frequency in Hz
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
                            print(df)
                            df = pd.read_csv(file_path)
                            match = re.search(r'Profile_(.*?)_Trial', file_path.stem)
                            if match:
                                profile = match.group(1)
                                session_data["ASSISTED"][profile]["LOG"].append(df)


            subject_data[subject.name][session.name]["session_data"] = session_data
    
    return subject_data, subject_dirs, subjects