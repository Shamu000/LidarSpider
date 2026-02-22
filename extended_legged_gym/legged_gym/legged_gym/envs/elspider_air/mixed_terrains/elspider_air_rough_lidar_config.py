import inspect

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.elspider_air.mixed_terrains.elspider_air_rough_train_config import ElSpiderAirRoughTrainCfg, ElSpiderAirRoughTrainCfgPPO

import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from abc import ABC, abstractmethod

class ElSpiderAirRoughLidarCfg(ElSpiderAirRoughTrainCfg):
    class env(ElSpiderAirRoughTrainCfg.env):
        # Update observation space for raycast data
        num_observations = 66 + 192*3  # MID360
        # num_observations = 66 # 无雷达

    class terrain(ElSpiderAirRoughTrainCfg.terrain):
        use_terrain_obj = False  # use TerrainObj class to create terrain
        # path to the terrain file
        terrain_file = None

        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh or confined_trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 10  # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                             0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level =0  # starting curriculum state
        terrain_length = 6.
        terrain_width = 6.
        num_rows = 4  # number of terrain rows (levels)
        num_cols = 6  # number of terrain cols (types)
        difficulty_scale = 1.0  # Scale for difficulty in curriculum
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.3, 0.3, 0.2]
        # confined terrain types: [tunnel, barrier, timber_piles, confined_gap]
        confined_terrain_proportions = [0.0, 0.2, 0.3, 0.3]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class BaseConfig:
        def __init__(self) -> None:
            """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
            self.init_member_classes(self)
        
        @staticmethod
        def init_member_classes(obj):
            # iterate over all attributes names
            for key in dir(obj):
                # disregard builtin attributes
                # if key.startswith("__"):
                if key=="__class__":
                    continue
                # get the corresponding attribute object
                var =  getattr(obj, key)
                # check if it the attribute is a class
                if inspect.isclass(var):
                    if isinstance(var, Enum):
                        continue
                    # instantate the class
                    i_var = var()
                    # set the attribute to the instance instead of the type
                    setattr(obj, key, i_var)
                    # recursively init members of the attribute
                    ElSpiderAirRoughLidarCfg.BaseConfig.init_member_classes(i_var)

    class BaseSensorConfig(ABC):
        pass

    class Terrain_cfg(BaseConfig):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-1.2, -1.05,-0.9,-0.75,-0.6,-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2,1.35,1.5,1.65,1.8,1.95,2.1,2.25,2.4,2.55,2.7] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-1.2,-1.05,-0.9,-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75,0.9,1.05,1.2]
        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        max_difficulty = False
        terrain_length = 18.
        terrain_width = 18
        num_rows= 2 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 2# number of terrain cols (types)
        
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0.0, 
                        "rough stairs down": 0.0, 
                        "discrete": 0., 
                        "stepping stones": 0.05,
                        "gaps": 0.05, 
                        "smooth flat": 0,
                        "pit": 0.,
                        "wall": 0.,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.2,
                        "parkour_hurdle": 0.2,
                        "parkour_flat": 0.0,
                        "parkour_step": 0.2,
                        "parkour_gap": 0.15,
                        "demo": 0.15,}
        terrain_proportions = list(terrain_dict.values())
        flat_wall = False # if True, wall is flat
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True
        num_sub_terrains = num_rows * num_cols

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class LidarType(Enum):
        """Standardized lidar sensor types"""
        # Simple grid-based lidar
        SIMPLE_GRID = "simple_grid"
        
        # Height scanner
        HEIGHT_SCANNER = "height_scanner"
        
        # Livox sensors
        AVIA = "avia"
        HORIZON = "horizon" 
        HAP = "HAP"
        MID360 = "mid360"
        MID40 = "mid40"
        MID70 = "mid70"
        TELE = "tele"
        
        # Traditional spinning lidars (to be implemented)
        HDL64 = "hdl64"
        VLP32 = "vlp32"
        OS128 = "os128"

    @dataclass
    class LidarConfig(BaseSensorConfig):
        """Optimized LidarSensor configuration"""
        
        # Core sensor settings
        sensor_type: "ElSpiderAirRoughLidarCfg.LidarType" = field(
            default_factory=lambda: ElSpiderAirRoughLidarCfg.LidarType.MID360
        )
        num_sensors: int = 1
        dt: float = 0.02  # simulation time step
        update_frequency: float = 50.0  # sensor update rate in Hz

        # 雷达初始姿态
        sensor_offset_pos: list = field(default_factory=lambda: [0.3, 0.0, 0.35])  # [x, y, z] in meters
        sensor_offset_rpy: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [roll, pitch, yaw] in degrees
        
        # Range settings
        max_range: float = 50.0
        min_range: float = 0.2
        
        # Grid-based lidar settings (only used when sensor_type is SIMPLE_GRID)
        horizontal_line_num: int = 80
        vertical_line_num: int = 50
        horizontal_fov_deg_min: float = -180
        horizontal_fov_deg_max: float = 180
        vertical_fov_deg_min: float = -2
        vertical_fov_deg_max: float = 57
        
        # Height scanner settings (only used when sensor_type is HEIGHT_SCANNER)
        height_scanner_size: list = field(default_factory=lambda: [2.0, 2.0])  # [length, width] in meters
        height_scanner_resolution: float = 0.1  # spacing between rays in meters
        height_scanner_direction: list = field(default_factory=lambda: [0.0, 0.0, -1.0])  # ray direction (downward)
        height_scanner_ordering: str = "xy"  # grid ordering: "xy" or "yx"
        height_scanner_offset: list = field(default_factory=lambda: [0.0, 0.0])  # [x, y] offset in meters for ray start positions
        height_scanner_height_above_ground: float = 10.0  # height above ground level where rays start (in meters)
        
        # Output settings
        return_pointcloud: bool = True
        pointcloud_in_world_frame: bool = False
        segmentation_camera: bool = False
        
        # Noise settings
        enable_sensor_noise: bool = False
        random_distance_noise: float = 0.03
        random_angle_noise: float = 0.15 / 180 * np.pi
        pixel_dropout_prob: float = 0.01
        pixel_std_dev_multiplier: float = 0.01
        
        # Transform settings
        euler_frame_rot_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
        
        # Placement randomization
        randomize_placement: bool = True
        min_translation: list = field(default_factory=lambda: [0.07, -0.06, 0.01])
        max_translation: list = field(default_factory=lambda: [0.12, 0.03, 0.04])
        min_euler_rotation_deg: list = field(default_factory=lambda: [-5.0, -5.0, -5.0])
        max_euler_rotation_deg: list = field(default_factory=lambda: [5.0, 5.0, 5.0])
        
        # Nominal position (for Isaac Gym sensors)
        nominal_position: list = field(default_factory=lambda: [0.10, 0.0, 0.03])
        nominal_orientation_euler_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
        
        # Data normalization
        normalize_range: bool = False
        far_out_of_range_value: float = -1.0
        near_out_of_range_value: float = -1.0
        
        def __post_init__(self):
            """Post-initialization validation and adjustments"""
            # Convert string sensor_type to enum if needed
            if isinstance(self.sensor_type, str):
                self.sensor_type = ElSpiderAirRoughLidarCfg.LidarType(self.sensor_type)
            
            # Auto-adjust normalization settings
            if self.return_pointcloud and self.pointcloud_in_world_frame:
                self.normalize_range = False
            
            # Set out-of-range values based on normalization
            if self.normalize_range:
                self.far_out_of_range_value = self.max_range
                self.near_out_of_range_value = -self.max_range
            else:
                self.far_out_of_range_value = -1.0
                self.near_out_of_range_value = -1.0
        
        @property
        def is_simple_grid(self) -> bool:
            """Check if this is a simple grid-based lidar"""
            return self.sensor_type == ElSpiderAirRoughLidarCfg.LidarType.SIMPLE_GRID
        
        @property
        def is_height_scanner(self) -> bool:
            """Check if this is a height scanner"""
            return self.sensor_type == ElSpiderAirRoughLidarCfg.LidarType.HEIGHT_SCANNER
        
        @property
        def is_livox_sensor(self) -> bool:
            """Check if this is a Livox-type sensor"""
            return self.sensor_type in [
                ElSpiderAirRoughLidarCfg.LidarType.AVIA,
                ElSpiderAirRoughLidarCfg.LidarType.HORIZON,
                ElSpiderAirRoughLidarCfg.LidarType.HAP,
                ElSpiderAirRoughLidarCfg.LidarType.MID360,
                ElSpiderAirRoughLidarCfg.LidarType.MID40,
                ElSpiderAirRoughLidarCfg.LidarType.MID70,
                ElSpiderAirRoughLidarCfg.LidarType.TELE,
            ]
        
        @property
        def is_spinning_lidar(self) -> bool:
            """Check if this is a traditional spinning lidar"""
            return self.sensor_type in [
                ElSpiderAirRoughLidarCfg.LidarType.HDL64,
                ElSpiderAirRoughLidarCfg.LidarType.VLP32,
                ElSpiderAirRoughLidarCfg.LidarType.OS128,
            ]

    class init_state(ElSpiderAirRoughTrainCfg.init_state):
        pos = [0.0, 0.0, 0.45]  # x,y,z [m]
    
    class commands(ElSpiderAirRoughTrainCfg.commands):
        curriculum = False
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-2.0, 2.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards(ElSpiderAirRoughTrainCfg.rewards):
        base_height_target = 0.35
        max_contact_force = 500.
        only_positive_rewards = True

        # Obstacle avoidance parameters
        # safe_obstacle_dist = 0.5    # Distance considered safe (meters)
        # danger_obstacle_dist = 0.2  # Distance considered dangerous (meters)
        collision_threshold = 0.08  # Distance for collision termination (meters) - reduced from 0.15
        
        # Termination protection - disable collision termination during early training steps
        # collision_termination_after_steps = 10  # Only terminate after this many steps
        # allow_initial_contact_steps = 5  # Allow contact termination grace period

        # Multi-stage rewards
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 5.0
        # Stage0-1: plane, Stage2: curriculum
        reward_min_stage = 2  # Start from 0
        reward_max_stage = 2

        class scales():

            # Tracking rewards
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # Base penalties
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = [-5.0, -5.0, 0.0]
            base_height = [-8.0, -8.0, 0.0]
            # DOF penalties
            torques = -0.00001
            dof_vel = -0.
            dof_acc = [-5e-8, -5e-8, -5e-8]
            dof_pos_limits = -1.0
            action_rate = [-0.001, -0.001, -0.002]
            # Feet penalties
            feet_slip = [-0.0, -0.4]  # Before feet_air_time
            feet_air_time = [0.8, 1.5]
            feet_stumble = [-1.0, -1.0, -2.0]
            feet_stumble_liftup = [1.0, 1.0, 2.0]
            feet_contact_forces = [0, 0, -0.05]  # Avoid jumping
            # Misc
            termination = -1.0
            collision = -1.
            stand_still = -0.
            # Gait
            async_gait_scheduler = [-0.2, -0.2, -0.1]
            gait_2_step = [-5.0, -5.0, -2.0]

        class async_gait_scheduler:
            # Reward for the async gait scheduler
            dof_align = 0.3
            dof_nominal_pos = 0.2
            reward_foot_z_align = 0.0
            
        foot_clearance_target = 0.04 # desired foot clearance above ground [m]
        foot_height_offset = 0.0     # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01

    # class viewer:
    #     ref_env = 0
    #     pos = [30, 0, -10]  # [m]
    #     lookat = [0., 0, 0.]  # [m]

# class ElSpiderAirRoughLidarCfgPPO(ElSpiderAirRoughTrainCfgPPO):
#     class runner(ElSpiderAirRoughTrainCfgPPO.runner):
#         run_name = 'lidar512'
#         experiment_name = 'rough_elspider_air'
#         load_run = -1
#         max_iterations = 5000  # number of policy updates

#         multi_stage_rewards = True

class ElSpiderAirRoughLidarCfgPPO(LeggedRobotCfgPPO):
    """PPO training configuration for ElSpider LiDAR confined space task."""
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-3
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'elspider_air_rough_lidar' # 保存的log文件夹名称
        load_run = -1
        max_iterations = 5000
        
        # Enable multi-stage rewards
        multi_stage_rewards = True
        
        # Checkpointing
        save_interval = 100
        
        # Logging
        log_interval = 10