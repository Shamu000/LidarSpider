from time import time
import numpy as np
import os
import inspect
import random
import time
import trimesh
import warp as wp
import threading
from math import sqrt

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

import isaacgym
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym.envs import ElSpider
from legged_gym import LEGGED_GYM_ROOT_DIR
from .mixed_terrains.elspider_air_rough_lidar_config import ElSpiderAirRoughLidarCfg
from legged_gym.utils import GaitScheduler, GaitSchedulerCfg, AsyncGaitSchedulerCfg, AsyncGaitScheduler, \
    SimpleRaibertPlannerConfig, SimpleRaibertPlanner, RaibertPlanner, RaibertPlannerConfig
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math_utils import quat_apply_yaw

from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.example.isaacgym.utils.terrain.terrain import Terrain
from LidarSensor.example.isaacgym.utils.terrain.terrain_cfg import Terrain_cfg
from LidarSensor import SENSOR_ROOT_DIR,RESOURCES_DIR

KEY_W = gymapi.KEY_W
KEY_A = gymapi.KEY_A
KEY_S = gymapi.KEY_S
KEY_D = gymapi.KEY_D
KEY_Q = gymapi.KEY_Q
KEY_E = gymapi.KEY_E
KEY_UP = gymapi.KEY_UP
KEY_DOWN = gymapi.KEY_DOWN
KEY_LEFT = gymapi.KEY_LEFT
KEY_RIGHT = gymapi.KEY_RIGHT
KEY_ESCAPE = gymapi.KEY_ESCAPE
# KEY_W = ord('w')
# KEY_A = ord('a')
# KEY_S = ord('s')
# KEY_D = ord('d')
# KEY_Q = ord('q')
# KEY_E = ord('e')
# KEY_UP = 273
# KEY_DOWN = 274
# KEY_LEFT = 276
# KEY_RIGHT = 275
# KEY_ESCAPE = 27

@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw): # 欧拉角转四元数
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

# 笛卡尔坐标系到球坐标系转换函数，输入为 (x, y, z)，输出为 (r, theta, phi)
@torch.jit.script
def cart2sphere(cart): # 笛卡尔坐标系转球面坐标系
    epsilon = 1e-9
    x = cart[:, 0]
    y = cart[:, 1]
    z = cart[:, 2]
    r = torch.norm(cart, dim=1)
    theta = torch.atan2(y, x)
    phi = torch.asin(z / (r + epsilon))
    return torch.stack((r, theta, phi), dim=-1)

# extended_legged_gym/legged_gym/legged_gym/envs/base/legged_robot_config.py/250
class sim:  # 仿真参数 
    dt =  0.005
    substeps = 1 # 每步物理参数求解数
    gravity = [0., 0. ,-9.81]  # [m/s^2]
    up_axis = 1  # 0 is y, 1 is z

    class physx:
        num_threads = 10 # 在 CPU 上用于物理计算的线程数
        solver_type = 1  # 0: pgs, 1: tgs 求解器模式，1更好
        num_position_iterations = 4 
        num_velocity_iterations = 0 # TODO：数值待定
        contact_offset = 0.01  # 碰撞距离[m]
        rest_offset = 0.0   # 真正接触时物体之间的最小间距[m]
        bounce_threshold_velocity = 0.5 #0.5 [m/s] 低于该速度的碰撞被认为不弹跳
        max_depenetration_velocity = 1.0 # 穿透时最大允许的矫正速度
        max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more TODO：验证尽量减小
        default_buffer_size_multiplier = 5
        contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2) 碰撞信息收集模式


# # parse arguments extended_legged_gym/legged_gym/legged_gym/utils/helpers.py
# args = gymutil.parse_arguments(
#     description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
#     custom_parameters=[
#         {"name": "--num_envs", "type": int, "default": 16, "help": "Number of environments to create"},
#         {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions"},
#         {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions"},
#         {"name": "--headless", "type": bool, "default": False, "help": "Run in headless mode"},])

# headless = args.headless

# extended_legged_gym/legged_gym/legged_gym/utils/math_utils.py/40
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.  # 把 roll、pitch 清零
    quat_yaw = normalize(quat_yaw)  # 归一化
    return quat_apply(quat_yaw, vec) # 只应用 yaw 旋转

def euler_from_quaternion(quat_angle):  # 四元数转欧拉角
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians
    
def farthest_point_sampling(point_cloud, sample_size): # 远点采样FPS算法
    """
    Sample points using the farthest point sampling algorithm
    Args:
        point_cloud: Tensor of shape (num_envs, 1, num_points,1, 3)
        sample_size: Number of points to sample
    Returns:
        Downsampled point cloud of shape (num_envs, 1, sample_size, 3)
    """
    num_envs, _, num_points, _ = point_cloud.shape
    device = point_cloud.device
    result = []
    
    for env_idx in range(num_envs):
        points = point_cloud[env_idx, 0]  # (num_points, 3)
        
        # Initialize with a random point
        sampled_indices = torch.zeros(sample_size, dtype=torch.long, device=device)
        sampled_indices[0] = torch.randint(0, num_points, (1,), device=device) # 随机取一个点的索引作为初始点
        
        # Calculate distances
        distances = torch.norm(points - points[sampled_indices[0]], dim=1)
        
        # Iteratively select farthest points
        for i in range(1, sample_size):
            # Select the farthest point
            sampled_indices[i] = torch.argmax(distances) # 得到当前最远点索引，新增采样集合
            
            # Update distances
            if i < sample_size - 1:
                new_distances = torch.norm(points - points[sampled_indices[i]], dim=1) # 全距离相对上次最远点更新
                distances = torch.min(distances, new_distances) # 最终结果为各点到采样集合中最近点的距离
        
        # Get the sampled points
        sampled_points = points[sampled_indices] # 保留特征点
        result.append(sampled_points.unsqueeze(0))  # 新增一个维度，即一个传感器数量：[sample_size, 3]->[1, sample_size, 3]
    
    return torch.stack(result) 

def downsample_spherical_points_vectorized(sphere_points, num_theta_bins=10, num_phi_bins=10): # 球面坐标点云进行二维角度网格划分
    """
    Downsample points in spherical coordinates by binning theta and phi values.
    
    Args:
        sphere_points: Tensor of shape (num_envs, num_points, 3) where dim 2 is (r, theta, phi)
        num_theta_bins: Number of bins for theta range (-3.14, 3.14)水平视场切片数量
        num_phi_bins: Number of bins for phi range (-0.12, 0.907)垂直视场切片数量
        
    Returns:
        Downsampled points tensor of shape (num_envs, num_theta_bins*num_phi_bins, 3)
    """
    num_envs = sphere_points.shape[0]
    num_points = sphere_points.shape[1]
    device = sphere_points.device
    num_bins = num_theta_bins * num_phi_bins
    
    # Define bin ranges
    theta_min, theta_max = -3.14, 3.14
    phi_min, phi_max = -0.12, 0.907
    
    # Extract r, theta, phi for all environments
    r = sphere_points[:, :, 0]       # [num_envs, num_points]
    theta = sphere_points[:, :, 1]   # [num_envs, num_points]
    phi = sphere_points[:, :, 2]     # [num_envs, num_points]
    
    # Compute bin indices for theta and phi
    theta_bin = ((theta - theta_min) / (theta_max - theta_min) * num_theta_bins).long()
    phi_bin = ((phi - phi_min) / (phi_max - phi_min) * num_phi_bins).long()
    
    # Clamp to valid bin indices
    theta_bin = torch.clamp(theta_bin, 0, num_theta_bins - 1)
    phi_bin = torch.clamp(phi_bin, 0, num_phi_bins - 1)
    
    # Compute linear bin index (flatten 2D bin indices to 1D)
    bin_indices = theta_bin * num_phi_bins + phi_bin  # [num_envs, num_points]
    
    # Create an environment index tensor to handle multiple environments
    env_indices = torch.arange(num_envs, device=device).view(-1, 1).expand(-1, num_points)
    
    # Flatten tensors for scatter operation
    flat_bin_indices = bin_indices.reshape(-1)            # [num_envs * num_points]
    flat_env_indices = env_indices.reshape(-1)            # [num_envs * num_points]
    flat_r = r.view(-1)                               # [num_envs * num_points]
    
    # Create 2D indices for scatter operation (env_idx, bin_idx)
    scatter_indices = torch.stack([flat_env_indices, flat_bin_indices], dim=1)  # [num_envs * num_points, 2]
    
    # Prepare tensors for scatter operations
    r_sum = torch.zeros(num_envs, num_bins, device=device)
    bin_count = torch.zeros(num_envs, num_bins, device=device)
    
    # Use scatter_add_ to compute sum and count for each bin
    r_sum.scatter_add_(1, bin_indices, r)
    ones = torch.ones_like(r)
    bin_count.scatter_add_(1, bin_indices, ones)
    
    # Avoid division by zero for empty bins
    bin_count = torch.clamp(bin_count, min=1.0)
    
    # Compute average r per bin
    avg_r = r_sum / bin_count  # [num_envs, num_bins]
    
    # Create bin centers for theta and phi
    theta_centers = torch.linspace(
        theta_min + (theta_max - theta_min) / (2 * num_theta_bins),
        theta_max - (theta_max - theta_min) / (2 * num_theta_bins),
        num_theta_bins, device=device
    )
    
    phi_centers = torch.linspace(
        phi_min + (phi_max - phi_min) / (2 * num_phi_bins),
        phi_max - (phi_max - phi_min) / (2 * num_phi_bins),
        num_phi_bins, device=device
    )
    
    # Create meshgrid of bin centers
    theta_grid, phi_grid = torch.meshgrid(theta_centers, phi_centers, indexing='ij')
    theta_centers_flat = theta_grid.reshape(-1)  # [num_bins]
    phi_centers_flat = phi_grid.reshape(-1)      # [num_bins]
    
    # Create final output tensor
    downsampled = torch.zeros(num_envs, num_bins, 3, device=device)
    downsampled[:, :, 0] = avg_r                              # r values
    downsampled[:, :, 1] = theta_centers_flat.unsqueeze(0)    # theta values
    downsampled[:, :, 2] = phi_centers_flat.unsqueeze(0)      # phi values
    
    return downsampled

# 让机器人学习通用行走
class ElSpiderLidar(ElSpider): # 继承
    cfg: ElSpiderAirRoughLidarCfg # 类型注解
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless) # 调用父类构造函数
        sensor_cfg: ElSpiderAirRoughLidarCfg.LidarConfig = (
            cfg.LidarConfig if not inspect.isclass(cfg.LidarConfig) else cfg.LidarConfig()
        )
        self._init_lidar_cfg(sensor_cfg)
        self._init_lidar_sensor()

    def _init_lidar_cfg(self, sensor_cfg: ElSpiderAirRoughLidarCfg.LidarConfig):
        self.sensor_cfg = sensor_cfg
        """Initialize a minimal lidar sensor environment."""
        self.sensor_cfg = sensor_cfg # Lidar 配置对象
        self.sensor_cfg.sensor_type = self.cfg.LidarType.AVIA # mid360,horizon,HAP,mid70,mid40,tele,avia
        self.sim_time = 0 # 记录仿真时间
        self.sensor_update_time = 0 # 记录传感器更新时间
        self.state_update_time = 0 # 传感器更新时间，超过阈值后更新并清空
        self.sensor_cfg.update_frequency = 20.0 # 传感器更新频率
        self.sensor_cfg.max_range = 5.0
        self.sensor_cfg.min_range = 0.1
        self.sensor_cfg.horizontal_line_num = 36
        self.sensor_cfg.vertical_line_num = 10
        self.sensor_cfg.horizontal_fov_deg_min = -180
        self.sensor_cfg.horizontal_fov_deg_max = 180
        self.sensor_cfg.vertical_fov_deg_min = -15
        self.sensor_cfg.vertical_fov_deg_max = 15
        self.sensor_cfg.dt = 0.02 # 传感器时间步长(TODO: 与sim对应吗？)
        self.sensor_cfg.pointcloud_in_world_frame = False # 点云数据是否在世界坐标系下
        self.num_theta_bins = 12
        self.num_phi_bins = 8

    def _init_lidar_sensor(self,
                            num_obstacles=5,
                            publish_ros=True,
                            save_data=False,
                            save_interval=0.1  # 每0.1秒保存一次数据
                           ):
        self.num_obstacles = num_obstacles
        self.save_data = save_data
        self.save_interval = save_interval # 保存间隔
        self.save_time = 0
        self.last_save_time = 0
        self.sequence_number=0 # 保存数据的序号（用于文件编号）

        wp.init() # 加速计算依赖 NVIDIA Warp库
        if self.save_data:
            # 创建保存数据的目录
            self.data_dir = f"./sensor_data_{time.strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self.data_dir, exist_ok=True)
            
            # 初始化数据存储列表
            self.stored_local_pixels = []  # 存储局部点云数据
            self.stored_robot_positions = []  # 存储机器人位置
            self.stored_robot_orientations = []  # 存储机器人方向
            self.stored_terrain_heights = []  # 存储地形高度测量值
            self.stored_timestamps = []  # 存储时间戳
            
            print(f"######Data will be saved to: {self.data_dir}")
        
        self.create_ground()

        self._init_buffer() # 绑定 Isaac Gym root state，与 GPU 张量同步
        self.create_warp_env() # Warp 格式的 mesh 用于 Lidar 仿真

        self.create_warp_tensor() # GPU 张量（点云输出/距离输出/姿态）
        
        self.sensor = LidarSensor(self.warp_tensor_dict, None, self.sensor_cfg, 1, self.device)
        self.lidar_update_counter = 0
        self.lidar_update_interval = self._get_lidar_update_interval()

        # sensor_points_tensor:形状为 (num_envs, num_sensors, V, H, 3) 的点云(x, y, z)
        # sensor_dist_tensor:形状为 (num_envs, num_sensors, V, H) 的距离图(depth map)
        self.sensor_points_tensor, self.sensor_dist_tensor = self.sensor.update() # 雷达扫描

        # Initialize keyboard state dictionary
        self.key_pressed = {}
        
        # Movement and rotation speeds 运动速度初始化
        self.linear_speed = 0.0  # m/s
        self.angular_speed = 0.0  # rad/s
        self.selected_env_idx = 0  # Environment to control (default to env 0 as in your draw code)
        
        # Initialize random movement parameters
        self.enable_random_movement = False
        self.movement_update_interval = 0.2  # Generate new target every 3 seconds
        self.movement_speed = 2.0  # Movement speed scalar
        self.rotation_speed = 1.0  # Rotation speed scalar
        
            
        if self.viewer:# 订阅所有需要的键
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_W,"move_forward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_A,"move_left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_S,"move_backward")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_D,"move_right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_Q,"move_up")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_E,"move_down")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_UP,"rotate_up")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_DOWN,"rotate_down")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_LEFT,"rotate_left")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_RIGHT,"rotate_right")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, KEY_ESCAPE,"exit")
        
        # Print control instructions
        print("Keyboard controls:")
        print("  WASD: Move robot horizontally")
        print("  Q/E: Move robot up/down")
        print("  Arrow keys: Rotate robot")
        print("  ESC: Exit simulation")

    def create_ground(self):
        """Create a ground plane."""
        self.terrain_cfg = Terrain_cfg()
        self.terrain = Terrain(self.terrain_cfg, self.num_envs)
        

    def _init_buffer(self):
        """Initialize buffers including LiDAR observation buffers."""
        super()._init_buffers()
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
            
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state) # 底层 buffer 转成 PyTorch tensor 视图

        self.base_quat = self.root_states[:, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.up_axis_idx=2
        self.gravity_vec = to_torch([0., 0., -1.], device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.base_pose = self.root_states[:, 0:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.last_base_ang_vel = self.base_ang_vel.clone()
        
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec) # 全局重力投影到机体坐标系
        
        self.last_projected_gravity = self.projected_gravity.clone()
        
        
        self.height_points = self._init_height_points()
        self.measured_heights=self._get_heights()
        
        # LiDAR observation buffers
        num_lidar_obs = self.num_theta_bins * self.num_phi_bins
        self.lidar_obs_buf = torch.zeros(
            self.num_envs, num_lidar_obs, device=self.device, requires_grad=False
        )

        # Raw LiDAR data buffers
        total_rays = self.sensor_cfg.horizontal_line_num * self.sensor_cfg.vertical_line_num
        self.lidar_points_buf = torch.zeros(
            self.num_envs, total_rays, 3, device=self.device, requires_grad=False
        )
        self.lidar_dist_buf = torch.zeros(
            self.num_envs, total_rays, device=self.device, requires_grad=False
        )

        # Minimum distance to obstacles (for rewards)
        self.min_obstacle_dist = torch.ones(
            self.num_envs, device=self.device, requires_grad=False
        ) * self.sensor_cfg.max_range
        
        self.sensor_translation = torch.tensor([0.0002835, 0.00003, 0.41818], device=self.device).repeat((self.num_envs, 1)) # 重点：把传感器放置在正确位置
        rpy_offset = torch.tensor([3.14, 0., 0], device=self.device) # 传感器的固定姿态偏移
        self.sensor_offset_quat = quat_from_euler_xyz(rpy_offset[0], rpy_offset[1], rpy_offset[2]).repeat((self.num_envs, 1))

    # 创建warp格式的环境网格
    def create_warp_env(self):
        terrain_mesh = trimesh.Trimesh(vertices=self.terrain.vertices, faces=self.terrain.triangles)
        #save terrain mesh
        transform = np.zeros((3,))
        transform[0] = -self.terrain_cfg.border_size 
        transform[1] = -self.terrain_cfg.border_size
        transform[2] = 0.0
        translation = trimesh.transformations.translation_matrix(transform)
        terrain_mesh.apply_transform(translation)

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        
        two_levels_up = os.path.dirname(os.path.dirname(os.path.dirname(current_script_dir)))
        
        
        obstacle_mesh_path = os.path.join(two_levels_up, "resources", "robots","el_mini", "robot_combined.stl")
        
        # obstacle_mesh = trimesh.load(obstacle_mesh_path)
        obstacle_mesh = trimesh.load(obstacle_mesh_path, force='mesh')

        # trimesh.load may still return a Scene (e.g., OBJ/GLTF with multiple meshes)
        if isinstance(obstacle_mesh, trimesh.Scene):
            obstacle_mesh = trimesh.util.concatenate(
                tuple(
                    geom for geom in obstacle_mesh.geometry.values()
                    if isinstance(geom, trimesh.Trimesh)
                )
            )


            #obstacle_mesh = trimesh.load(self.terrain_cfg.obstacle_config.obstacle_root_path+"/human/meshes/Male.OBJ")
        transaltion = np.zeros((3,))
        transaltion[0]=self.root_states[0,0]
        transaltion[1]=self.root_states[0,1]
        transaltion[2]=self.root_states[0,2]
        # quat = self.root_states[0,3:7].numpy()
        # rotation = trimesh.transformations.quaternion_matrix(quat)
        translation = trimesh.transformations.translation_matrix(transaltion)
            
        obstacle_mesh.apply_transform(translation)

        combine_mesh = trimesh.util.concatenate([terrain_mesh, obstacle_mesh])
        #save combined mesh
        #combine_mesh.export("robot_terrain_combined.stl")
        vertices = combine_mesh.vertices
        triangles = combine_mesh.faces
        vertex_tensor = torch.tensor( 
                vertices,
                device=self.device,
                requires_grad=False,
                dtype=torch.float32,
            )
        
        #if none type in vertex_tensor
        if vertex_tensor.any() is None:
            print("vertex_tensor is None")
        vertex_vec3_array = wp.from_torch(vertex_tensor,dtype=wp.vec3)        
        faces_wp_int32_array = wp.from_numpy(triangles.flatten(), dtype=wp.int32,device=self.device)
                
        self.wp_meshes =  wp.Mesh(points=vertex_vec3_array,indices=faces_wp_int32_array)
        
        self.mesh_ids = self.mesh_ids_array = wp.array([self.wp_meshes.id], dtype=wp.uint64)

    def create_warp_tensor(self):
        self.warp_tensor_dict={}
        self.sensor_points_tensor = torch.zeros(
                (
                    self.num_envs,  #4
                    self.sensor_cfg.num_sensors, #1
                    self.sensor_cfg.vertical_line_num, #128
                    self.sensor_cfg.horizontal_line_num, #512
                    3, #3
                ),
                device=self.device,
                requires_grad=False,
            )        
        self.sensor_dist_tensor = torch.zeros(
                (
                    self.num_envs,  #4
                    self.sensor_cfg.num_sensors, #1
                    self.sensor_cfg.vertical_line_num, #128
                    self.sensor_cfg.horizontal_line_num, #512
                ),
                device=self.device,
                requires_grad=False,
            ) 
        # self.mesh_ids = self.mesh_ids_array = wp.array(self.warp_mesh_id_list, dtype=wp.uint64)
        # 定义传感器位姿（位置和朝向）
        self.sensor_pos_tensor = torch.zeros_like(self.root_states[:, 0:3])
        self.sensor_quat_tensor = torch.zeros_like(self.root_states[:, 3:7])
        
        # 传感器相对于载体的安装偏移
        offset_pos = getattr(self.cfg.LidarConfig, "lidar_offset_pos", None)
        if offset_pos is None:
            offset_pos = [0.0, 0.0, 0.35]

        offset_rpy = getattr(self.cfg.LidarConfig, "lidar_offset_rpy", None)
        if offset_rpy is None:
            offset_rpy = [0.0, 0.0, 0.0]

        self.sensor_translation = torch.tensor(offset_pos, device=self.device).repeat((self.num_envs, 1))
        rpy_offset = torch.tensor(offset_rpy, device=self.device, dtype=torch.float32)
        self.sensor_offset_quat = quat_from_euler_xyz(rpy_offset[0], rpy_offset[1], rpy_offset[2]).repeat((self.num_envs, 1))

        
        self.warp_tensor_dict["sensor_dist_tensor"] = self.sensor_dist_tensor
        self.warp_tensor_dict['device'] = self.device
        self.warp_tensor_dict['num_envs'] = self.num_envs
        self.warp_tensor_dict['num_sensors'] = self.sensor_cfg.num_sensors
        self.warp_tensor_dict['sensor_pos_tensor'] = self.sensor_pos_tensor
        self.warp_tensor_dict['sensor_quat_tensor'] = self.sensor_quat_tensor
        self.warp_tensor_dict['mesh_ids'] = self.mesh_ids

    # keyboard 控制机器人移动
    def keyboard_input(self):
        """Process keyboard input to move the robot"""
        # 在没有查看器的情况下直接返回，因为无法获取键盘输入
        if not self.viewer:
            return True
        
        # 处理自上次调用以来的所有事件
        for evt in self.gym.query_viewer_action_events(self.viewer):
            print(f"Key event: action={evt.action}, value={evt.value}")  # 调试信息
            
            # 处理按键事件 - 当值大于0时表示按下，等于0时表示释放
            if evt.action == "move_forward":
                self.key_pressed[KEY_W] = evt.value > 0
            elif evt.action == "move_backward":
                self.key_pressed[KEY_S] = evt.value > 0
            elif evt.action == "move_left":
                self.key_pressed[KEY_A] = evt.value > 0
            elif evt.action == "move_right":
                self.key_pressed[KEY_D] = evt.value > 0
            elif evt.action == "move_up":
                self.key_pressed[KEY_Q] = evt.value > 0
            elif evt.action == "move_down":
                self.key_pressed[KEY_E] = evt.value > 0
            elif evt.action == "rotate_up":
                self.key_pressed[KEY_UP] = evt.value > 0
            elif evt.action == "rotate_down":
                self.key_pressed[KEY_DOWN] = evt.value > 0
            elif evt.action == "rotate_left":
                self.key_pressed[KEY_LEFT] = evt.value > 0
            elif evt.action == "rotate_right":
                self.key_pressed[KEY_RIGHT] = evt.value > 0
            elif evt.action == "exit" and evt.value > 0:
                print("Exiting simulation")
                return False
        
        # 固定时间步长（与模拟一致）
        dt = 0.005
        
        # 获取选定机器人的当前状态
        env_idx = self.selected_env_idx
        current_pos = self.root_states[env_idx, 0:3].clone()
        current_quat = self.root_states[env_idx, 3:7].clone()
        
        # 设置速度 (每次调用时设置固定速度，而不是累加)
        self.linear_speed = 3.0  # 1 m/s
        self.angular_speed = 3.0  # 1 rad/s
        
        # 初始化速度向量 - 始终从零开始以响应当前按键状态
        linear_vel = torch.zeros(3, device=self.device)
        euler_rates = torch.zeros(3, device=self.device)
        
        # 处理按键状态 - 设置当前速度
        # 前后移动
        if self.key_pressed.get(KEY_W, False):
            linear_vel[0] = self.linear_speed
        if self.key_pressed.get(KEY_S, False):
            linear_vel[0] = -self.linear_speed
            
        # 左右移动
        if self.key_pressed.get(KEY_A, False):
            linear_vel[1] = -self.linear_speed
        if self.key_pressed.get(KEY_D, False):
            linear_vel[1] = self.linear_speed
            
        # 上下移动
        if self.key_pressed.get(KEY_Q, False):
            linear_vel[2] = self.linear_speed
        if self.key_pressed.get(KEY_E, False):
            linear_vel[2] = -self.linear_speed
            
        # 旋转控制（偏航）
        if self.key_pressed.get(KEY_LEFT, False):
            euler_rates[2] = self.angular_speed
        if self.key_pressed.get(KEY_RIGHT, False):
            euler_rates[2] = -self.angular_speed
            
        # 旋转控制（俯仰）
        if self.key_pressed.get(KEY_UP, False):
            euler_rates[1] = self.angular_speed
        if self.key_pressed.get(KEY_DOWN, False):
            euler_rates[1] = -self.angular_speed
        
        # 将局部线性速度转换为全局速度
        global_vel = quat_apply(current_quat, linear_vel)
        
        # 应用移动 - 根据当前速度和时间步长计算位移
        new_pos = current_pos + global_vel * dt
        
        # 应用旋转（将欧拉角速率转换为四元数变化）
        roll, pitch, yaw = euler_from_quaternion(current_quat.unsqueeze(0))
        
        # 更新欧拉角 - 根据当前角速度和时间步长计算角度变化
        roll = roll + euler_rates[0] * dt
        pitch = pitch + euler_rates[1] * dt
        yaw = yaw + euler_rates[2] * dt
        
        # 转换回四元数
        new_quat = quat_from_euler_xyz(roll, pitch, yaw)
        
        # 更新机器人状态
        self.root_states[env_idx, 0:3] = new_pos
        self.root_states[env_idx, 3:7] = new_quat
        
        # 应用更改到模拟
        env_ids_int32 = torch.tensor([env_idx], dtype=torch.int32, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )
        
        return True  # 继续模拟

    def collect_and_save_data(self):
        """收集当前时刻的数据并添加到存储列表"""
        current_time = self.sim_time
        
        # 1. 收集激光雷达局部点云数据 - 在激光雷达坐标系中
        local_pixels = self.sensor_points_tensor.clone()  # [num_envs, num_sensors, vertical_lines, horizontal_lines, 3]
        
        # 2. 收集机器人位置 - 世界坐标系
        robot_positions = self.root_states[:, 0:3].clone()  # [num_envs, 3]
        
        # 3. 收集机器人方向 (四元数) - 世界坐标系
        robot_orientations = self.root_states[:, 3:7].clone()  # [num_envs, 4]
        
        # 4. 收集地形高度测量值 - 世界坐标系
        terrain_heights = self.measured_heights.clone()  # [num_envs, num_height_points]
        
        # 将当前数据添加到存储列表 (保持原始张量格式)
        self.stored_local_pixels.append(local_pixels)
        self.stored_robot_positions.append(robot_positions)
        self.stored_robot_orientations.append(robot_orientations)
        self.stored_terrain_heights.append(terrain_heights)
        self.stored_timestamps.append(current_time)
        
        # 如果列表变得太大，保存并清空
        if len(self.stored_timestamps) >= 10:  # 每1000帧保存一次
            self.save_data_to_files()

    def save_data_to_files(self):
        """将收集的数据保存到文件中并清空存储列表"""
        if not self.stored_timestamps:
            return  # 如果没有数据，直接返回
        
        # 生成时间戳字符串作为文件名的一部分
        timestamp_str = f"{self.stored_timestamps[0]:.2f}_{self.stored_timestamps[-1]:.2f}"
        
        # 将存储的列表转换为张量
        # 注意：这里我们堆叠张量以创建时间序列数据
        local_pixels_tensor = torch.stack(self.stored_local_pixels)
        robot_positions_tensor = torch.stack(self.stored_robot_positions)
        robot_orientations_tensor = torch.stack(self.stored_robot_orientations)
        terrain_heights_tensor = torch.stack(self.stored_terrain_heights)
        timestamps_tensor = torch.tensor(self.stored_timestamps, device=self.device)
        
        # 创建数据字典
        data_dict = {
            'local_pixels': local_pixels_tensor,
            'robot_positions': robot_positions_tensor,
            'robot_orientations': robot_orientations_tensor, 
            'terrain_heights': terrain_heights_tensor,
            'timestamps': timestamps_tensor
        }
        
        # 使用torch.save保存字典
        torch.save(data_dict, f"{self.data_dir}/sensor_data_{timestamp_str}.pt")
        
        print(f"Saved {len(self.stored_timestamps)} frames of data with timestamp {timestamp_str}")
        
        # 清空存储列表
        self.stored_local_pixels = []
        self.stored_robot_positions = []
        self.stored_robot_orientations = []
        self.stored_terrain_heights = []
        self.stored_timestamps = []

    # 添加析构函数确保数据保存
    def __del__(self):
        """确保在对象销毁前保存所有数据"""
        if hasattr(self, 'save_data') and self.save_data and hasattr(self, 'stored_timestamps') and self.stored_timestamps:
            print("Saving remaining data before exit...")
            self.save_data_to_files()

    def _get_lidar_update_interval(self) -> int:
        if self.sensor_cfg.update_frequency <= 0:
            return 1
        return max(1, int(round(1.0 / (self.sensor_cfg.update_frequency * self.dt))))


    # 每物理步后，为后续奖励与观测提供最新数据
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        if self.sensor is None:
            return

        self._update_lidar_pose()
        if self.lidar_update_counter % self.lidar_update_interval == 0:
            self.sensor_points_tensor, self.sensor_dist_tensor = self.sensor.update()
            # Reshape data: (num_envs, num_sensors, v_lines, h_lines, 3) -> (num_envs, total_rays, 3)
            total_rays = self.sensor_cfg.horizontal_line_num * self.sensor_cfg.vertical_line_num
            self.lidar_points_buf[:] = self.sensor_points_tensor.view(self.num_envs, -1, 3)[:, :total_rays, :]
            self.lidar_dist_buf[:] = self.sensor_dist_tensor.view(self.num_envs, -1)[:, :total_rays]
        self.lidar_update_counter += 1

        # Compute minimum obstacle distance
        valid_mask = self.lidar_dist_buf < self.sensor_cfg.max_range
        self.min_obstacle_dist[:] = self.sensor_cfg.max_range
        for i in range(self.num_envs):
            valid_dists = self.lidar_dist_buf[i][valid_mask[i]]
            if valid_dists.numel() > 0:
                self.min_obstacle_dist[i] = valid_dists.min()

        sphere_points = cart2sphere(self.lidar_points_buf.view(-1, 3)).view(self.num_envs, -1, 3)
        downsampled = downsample_spherical_points_vectorized(
            sphere_points, self.num_theta_bins, self.num_phi_bins
        )
        
        # Use normalized distance as observation (0 = close, 1 = far/no hit)
        self.lidar_obs_buf[:] = downsampled[:, :, 0].clamp(0, self.sensor_cfg.max_range) / self.sensor_cfg.max_range

    def check_termination(self):
        """Check termination conditions including collision detection."""
        super().check_termination()
        
        # Get collision parameters from config
        if hasattr(self.cfg.rewards, 'collision_threshold'):
            collision_threshold = self.cfg.rewards.collision_threshold
        else:
            collision_threshold = 0.08  # Default 8cm - more permissive
        
        # Get protection steps (grace period during early training)
        # if hasattr(self.cfg.rewards, 'collision_termination_after_steps'):
        #     min_steps = self.cfg.rewards.collision_termination_after_steps
        # else:
        min_steps = 10  # Default: only check collision after 10 steps
        
        # Only terminate due to collision after protection period
        # This allows the robot to learn without being immediately terminated
        collision = self.min_obstacle_dist < collision_threshold
        collision_termination = collision & (self.episode_length_buf > min_steps)
        self.reset_buf |= collision_termination # 按位或赋值
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.sensor is not None and env_ids.numel() > 0:
            self.sensor.reset(env_ids)

    def compute_observations(self):
        """Compute observations including LiDAR data."""
        # Base observations (same as ElSpider)
        base_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                  self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  self.commands[:, 4:],  # TODO: 确保commands的第四位开始真的有数据
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)
        
        # Add height measurements if configured
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1, 1.
            ) * self.obs_scales.height_measurements
            base_obs_buf = torch.cat((base_obs_buf, heights), dim=-1)
        
        # Add LiDAR observations
        self.obs_buf = torch.cat((base_obs_buf, self.lidar_obs_buf), dim=-1)
        
        # Add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _update_lidar_pose(self) -> None:
        sensor_quat = quat_mul(self.base_quat, self.sensor_offset_quat)
        sensor_pos = self.base_pos + quat_apply(self.base_quat, self.sensor_translation)
        self.sensor_pos_tensor[:] = sensor_pos
        self.sensor_quat_tensor[:] = sensor_quat


    # ============== Reward Functions ==============
    
    def _reward_obstacle_avoidance(self):
        """Reward for maintaining safe distance from obstacles."""
        # Reward increases with distance from obstacles
        safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        
        # Compute reward based on minimum distance
        dist_reward = torch.clamp(self.min_obstacle_dist / safe_dist, 0, 1)
        return dist_reward

    def _reward_collision_penalty(self):
        """Penalty for getting too close to obstacles."""
        danger_dist = getattr(self.cfg.rewards, 'danger_obstacle_dist', 0.3)
        
        # Exponential penalty for being too close
        penalty = torch.exp(-self.min_obstacle_dist / danger_dist + 1) - 1
        penalty = torch.clamp(penalty, 0, 10)
        return -penalty

    def _reward_exploration(self):
        """Reward for exploring while avoiding obstacles."""
        # Combine forward velocity with obstacle avoidance
        forward_vel = self.base_lin_vel[:, 0]
        safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        
        # Only reward forward movement when it's safe
        safety_factor = torch.clamp(self.min_obstacle_dist / safe_dist, 0, 1)
        exploration_reward = forward_vel * safety_factor
        return torch.clamp(exploration_reward, -1, 1)

    def _draw_debug_vis(self):
        """Draw debug visualization including LiDAR points."""
        super()._draw_debug_vis()
        
        # Draw LiDAR points for first environment
        if not self.headless and hasattr(self, 'lidar_points_buf'):
            self._draw_lidar_points()

    def _draw_lidar_points(self):
        """Visualize LiDAR point cloud."""
        if not hasattr(self, 'viewer') or self.viewer is None:
            return
        
        # Only draw for selected environment
        env_idx = 0
        points = self.lidar_points_buf[env_idx]
        sensor_pos = self.sensor_pos_tensor[env_idx]
        sensor_quat = self.sensor_quat_tensor[env_idx]
        
        # Transform points to world frame
        world_points = sensor_pos + quat_apply(sensor_quat.unsqueeze(0), points)
        
        # Draw subset of points (for performance)
        step = max(1, points.shape[0] // 100)
        for i in range(0, points.shape[0], step):
            pos = world_points[i].cpu().numpy()
            sphere = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))
            pose = gymapi.Transform(gymapi.Vec3(*pos), r=None)
            gymutil.draw_lines(sphere, self.gym, self.viewer, self.envs[env_idx], pose)



def print_lidar_pos():
    """Get and print the lidar position data."""
    # Update the sensor to get the latest data
   
    
    # Print lidar position
    # The env.sensor_points_tensor contains the lidar data we want to print
    print(f"Lidar Position at {time.time():.3f}:")
    
    # Example: Print a summary of the lidar position data
    # You can customize this to print specific parts of the data that are of interes
    print("-" * 50)
    timer = threading.Timer(0.02, print_lidar_pos) # 0.02s后再次调用print_lidar_pos
    timer.daemon = True # 守护线程：主程序退出时这个定时器线程不会阻止程序结束
    timer.start() # 启动定时器线程
    
