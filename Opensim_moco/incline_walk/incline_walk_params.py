import numpy as np  # Import NumPy for array use

PARAMS = {
    # Configuration for files
    'subject': 'S001',
    'motion': 'incline_10deg_1ms_trial_4',
    'opencap_data_path': './data/OpenCapData/',
    'osim_model_path': 'OpenSimData/Model/LaiUhlrich2022_scaled.osim',
    'moco_path': './sim_result',

    #Note
    'note': "kinematics error + muscle activation + contact dynamics + residual_minimization",

    # subject specific information
    'foot_length': 0.28,

    # model check
    'check_model': False,
    'print_optimum_force': True, 
    'print_control_weight': True,

    # Trial-specific timing
    't0': 23,
    'tf': 24.1,

    # incline deg
    'incline_deg': 10,

    # Moco solver parameters
    'mesh_interval': 0.05,
    'max_iterations': 1000,
    'convergence_tolerance': 1e-3,
    'constraint_tolerance': 1e-3,

    # ---- Kinematics error cost weights
    'tracking_weight': 1, # W_S, Global penalization
    'kinematic_state_tracking_weight': 1, # camille's thesis value (fixed)
    'tracking_weight_pelvis_txy': 1, # camille's thesis

    # ---- Control effort cost weights
    'control_effort_weight': 0.01,  # W_C, Global penalization

    # --- Control Effort Parameters --- # 
    'coord_actuator_opt_value': 250,  # Nm for coordinate actuators: lumbar, arms
    'reserve_actuator_opt_value': 20,  # N for reserve actuators: pelvis, legs, feet
    
    # Weights in control effort cost term
    # Set the weight to use for the term in the cost associated with controlName
    # To remove a control from the cost function, set its weight to 0.
    'control_effort_muscles': 1,  # Set to 1 if assisted, otherwise 20
    'reserve_feet_weight': 1,
    'reserve_knee_weight': 1,
    'reserve_hip_weight': 1, 
    'reserve_pelvis_weight': 1, # higher means more penalty (minimizing)
    'reserve_pelvis_weight_txy': 1, # higher means more penalty (minimizing)

    # Assistance hyperparameters
    'assist_with_reserve_pelvis_txy': True,
    'reserve_pelvis_tx_ty_opt_value': 20,
    'pelvis_ty_positive': False,

    # GRF hyperparameters
    'osim_GRF': True,
    'osim_GRF_type': 's8',  # This may not be used

    # assistive force (not used for now)
    'assist_with_force_ext': False,
    'force_ext_opt_value': None,  # Can be set to 700N

    # Muscles configuration
    'muscles_to_remove_v2': [
        "addlong", "addbrev", "addmagDist", "addmagIsch", "addmagProx", "addmagMid",
        "fdl", "ehl", "fhl", "gaslat", "glmin1", "glmin2", "glmin3", "perlong",
        "edl", "grac", "iliacus", "perbrev", "piri", "sart", "semimem", "semiten",
        "tfl", "tibpost", "vasint"
    ],

    # Reference marker set
    # Floor contact space
    'floor_ref_marker': ['ankle', 'calc', '5meta', 'toe'],
    'floor_offset': 0.1,  # Offset between foot markers and ground 

    # Foot contact sphere markers
    'foot_contact_marker': ['calc', '5meta', 'toe'],
    'foot_socket': ['toes', 'calcn'],  # Required by OpenSim to specify the body part when creating contact spheres

    # --- parameters for sit-to-stand --- #
    # Chair contact space
    'chair_ref_marker': ['thigh3', 'knee', 'PSIS'],
    'chair_height': 0.48,

    # --- parameters for stair walk --- #
    'stair_width': 0.26,
    'stair_height': 0.16,
    'num_of_steps': 3,

    # Contact model hyperparameters (ChatGPT recommended parameter set -- less aggressive stickiness)
    # 'stiffness': 5e5,
    # 'dissipation': 0.8,
    # 'static_friction': 0.8,
    # 'dynamic_friction': 0.6,
    # 'viscous_friction': 0.2,
    # 'transition_velocity': 0.1,

    # Contact model hyperparameters (for sit-to-stand)
    'stiffness': 1000000,
    'dissipation': 2.0,
    'static_friction': 0.8,
    'dynamic_friction': 0.8,
    'viscous_friction': 0.5,
    'transition_velocity': 0.2,

    # Hip contact spheres
    'hip_contact_spheres': {
        "s8_l": {
            "radius": 0.12,
            "location": np.array([-0.15, -0.1, -0.1]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "pelvis"
        },
        "s8_r": {
            "radius": 0.12,
            "location": np.array([-0.15, -0.1, 0.1]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "pelvis"
        }
    },

    # Foot contact spheres
    'foot_contact_spheres': {
        "s1_r": {
            "radius": 0.04,
            "location": np.array([0.0019011578840796601, -0.01, -0.00382630379623308]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_r"
        },  # Heel
        "s2_r": {
            "radius": 0.02,
            "location": np.array([0.14838639994206301, -0.01, -0.028713422052654002]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_r"
        },  # Interior side
        "s3_r": {
            "radius": 0.02,
            "location": np.array([0.13300117060705099, -0.01, 0.051636247344956601]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_r"
        },  # Exterior side
        "s4_r": {
            "radius": 0.02,
            "location": np.array([0.066234666199163503, -0.01, 0.026364160674169801]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_r"
        },  # Exterior between heel and side
        "s5_r": {
            "radius": 0.02,
            "location": np.array([0.059999999999999998, -0.01, -0.018760308461917698]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "toes_r"
        },  # Toe interior
        "s6_r": {
            "radius": 0.02,
            "location": np.array([0.044999999999999998, -0.01, 0.061856956754965199]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "toes_r"
        },  # Toe exterior
        "s1_l": {
            "radius": 0.04,
            "location": np.array([0.0019011578840796601, -0.01, 0.00382630379623308]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_l"
        },
        "s2_l": {
            "radius": 0.02,
            "location": np.array([0.14838639994206301, -0.01, 0.028713422052654002]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_l"
        },
        "s3_l": {
            "radius": 0.02,
            "location": np.array([0.13300117060705099, -0.01, -0.051636247344956601]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_l"
        },
        "s4_l": {
            "radius": 0.02,
            "location": np.array([0.066234666199163503, -0.01, -0.026364160674169801]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_l"
        },
        "s5_l": {
            "radius": 0.02,
            "location": np.array([0.059999999999999998, -0.01, 0.018760308461917698]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "toes_l"
        },
        "s6_l": {
            "radius": 0.02,
            "location": np.array([0.044999999999999998, -0.01, -0.061856956754965199]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "toes_l"
        }
    }
}