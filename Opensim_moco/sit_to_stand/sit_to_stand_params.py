import numpy as np  # Import NumPy for array use

PARAMS = {
    # === File & Model Configuration === #
    'subject': 'S002',
    'motion': 'sit_to_stand_4',
    'opencap_data_path': './data/OpenCapData/',
    'osim_model_path': 'OpenSimData/Model/LaiUhlrich2022_scaled.osim',
    'moco_path': './sim_result',
    'note': "[Assisted case] S002 using unassited converged solution",

    # === Initial guess === #
    'initial_guess': True,
    'previous_solution_path': './sim_result/S002_sit_to_stand_4_20250605_161549_unassisted_converged/S002_sit_to_stand_4_20250605_161549.sto',

    # === Model Check & Logging === #
    'check_model': False,
    'print_optimum_force': True, 
    'print_control_weight': True,

    # === Contact dynamics === #
    'add_contact_dynamics': True,

    # === Trial-Time Parameters === #
    't0': 3,     # Start time (s)
    'tf': 5,     # End time (s)

    # === Moco Solver === #
    'mesh_interval': 0.05,
    'max_iterations': 1000,
    'convergence_tolerance': 5e-3,
    'constraint_tolerance': 5e-3,

    # === State extra bound  === #
    'state_extra_bound_size': 0.25,

    # === Kinematics Tracking Weights === #
    'tracking_weight_global_state': 1,
    'tracking_weight_individual_state': 1,

    # === Control Effort Weights === #
    'control_effort_weight_global': 0.05,
    'control_effort_weight_muscles': 0.5,
    'control_effort_weight_reserve_actuator': 15,
    'control_effort_weight_coord_actuators': 0.1,

    # === Optimal Force Values === #
    'opt_force_coord_actuator': 250,    # Nm for lumbar/arms
    'opt_force_reserve_actuator': 250,  # N for pelvis/legs/feet

    # === Control Bounds === #
    'reserve_actuator_control_bound': [-1, 1],
    'coord_actuator_control_bound': [-1, 1],

    # === Robot Assistance === #
    'robot_assistance': True,
    'robot_assistance_target_body': 'pelvis',
    'opt_force_robot_assistance': 100, # maximum force that robot can provide
    'robot_assistance_control_bound': [-1, 1],
    'control_effort_weight_robot_assistance': 0.1,

    # Muscles configuration
    'muscles_to_remove': [
        "addlong", "addbrev", "addmagDist", "addmagIsch", "addmagProx", "addmagMid",
        "fdl", "ehl", "fhl", "gaslat", "glmin1", "glmin2", "glmin3", "perlong",
        "edl", "grac", "iliacus", "perbrev", "piri", "sart", "semiten",
        "tfl", "tibpost", "vasint"
    ],

    # Reference marker set
    # Floor contact space
    'floor_ref_marker': ['ankle', 'calc', '5meta', 'toe'],
    'floor_offset': 0.05,  # Offset between foot markers and ground

    # Foot contact sphere markers
    'foot_contact_marker': ['calc', '5meta', 'toe'],
    'foot_socket': ['toes', 'calcn'],  # Required by OpenSim to specify the body part when creating contact spheres

    # --- parameters for sit-to-stand --- #
    # Chair contact space
    'chair_ref_marker': ['thigh3', 'knee', 'PSIS'],
    'chair_height': 0.48,

    # Contact model hyperparameters
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
        # ---- RIGHT FOOT ----
        "s1_r": {
            "radius": 0.04,
            "location": np.array([0.0019, -0.01, -0.0038]),   # Heel
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_r"
        },
        "s2_r": {
            "radius": 0.02,
            "location": np.array([0.1484, -0.01, -0.0287]),   # Medial/metatarsal head
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_r"
        },
        "s3_r": {
            "radius": 0.02,
            "location": np.array([0.1330, -0.01, 0.0516]),    # Lateral/metatarsal head
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_r"
        },
        "s4_r": {
            "radius": 0.02,
            "location": np.array([0.0662, -0.01, 0.0264]),    # Lateral midfoot
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_r"
        },
        "s5_r": {
            "radius": 0.02,
            "location": np.array([0.0600, -0.01, -0.0188]),   # Medial toe
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "toes_r"
        },
        "s6_r": {
            "radius": 0.02,
            "location": np.array([0.0450, -0.01, 0.0619]),    # Lateral toe
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "toes_r"
        },
        # ---- LEFT FOOT ----
        "s1_l": {
            "radius": 0.04,
            "location": np.array([0.0019, -0.01, 0.0038]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_l"
        },
        "s2_l": {
            "radius": 0.02,
            "location": np.array([0.1484, -0.01, 0.0287]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_l"
        },
        "s3_l": {
            "radius": 0.02,
            "location": np.array([0.1330, -0.01, -0.0516]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_l"
        },
        "s4_l": {
            "radius": 0.02,
            "location": np.array([0.0662, -0.01, -0.0264]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "calcn_l"
        },
        "s5_l": {
            "radius": 0.02,
            "location": np.array([0.0600, -0.01, 0.0188]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "toes_l"
        },
        "s6_l": {
            "radius": 0.02,
            "location": np.array([0.0450, -0.01, -0.0619]),
            "orientation": np.array([0, 0, 0]),
            "socket_frame": "toes_l"
        }
    }
}