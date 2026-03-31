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
from legged_gym.utils.gym_editor import ObstacleGen, ObstacleGenConfig
from legged_gym.utils import GaitScheduler, GaitSchedulerCfg, AsyncGaitSchedulerCfg, AsyncGaitScheduler, \
    SimpleRaibertPlannerConfig, SimpleRaibertPlanner, RaibertPlanner, RaibertPlannerConfig
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math_utils import quat_apply_yaw
from legged_gym.utils.gym_visualizer import GymVisualizer

from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.example.isaacgym.utils.terrain.terrain import Terrain
from LidarSensor.example.isaacgym.utils.terrain.terrain_cfg import Terrain_cfg
from LidarSensor import SENSOR_ROOT_DIR,RESOURCES_DIR

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

@torch.jit.script
def quat_from_euler_xyz_tensor(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert euler angles to quaternion (tensor version)"""
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
    x = cart[:, :, 0]
    y = cart[:, :, 1]
    z = cart[:, :, 2]
    r = torch.norm(cart, dim=-1) # 沿着最后一维的xyz计算
    theta = torch.atan2(y, x)
    phi = torch.asin(z / (r + epsilon))
    return torch.stack((r, theta, phi), dim=-1)

@torch.jit.script
def sphere2cart(sphere):
    r     = sphere[:, :, 0]
    theta = sphere[:, :, 1]
    phi   = sphere[:, :, 2]

    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    x = r * cos_phi * torch.cos(theta)
    y = r * cos_phi * torch.sin(theta)
    z = r * sin_phi

    return torch.stack((x, y, z), dim=-1)

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
    
def farthest_point_sampling(point_cloud, sample_size):
    """
    point_cloud: (B, 1, N, 3)
    return: (B, 1, sample_size, 3)
    """
    B, _, N, _ = point_cloud.shape
    device = point_cloud.device

    points = point_cloud[:, 0]  # (B, N, 3)

    # 初始化
    sampled_indices = torch.zeros(B, sample_size, dtype=torch.long, device=device)

    # 每个 batch 随机选一个初始点
    farthest = torch.randint(0, N, (B,), device=device)
    sampled_indices[:, 0] = farthest

    # 初始化距离
    batch_indices = torch.arange(B, device=device)
    distances = torch.norm(points - points[batch_indices, farthest].unsqueeze(1), dim=2)  # (B, N)

    for i in range(1, sample_size):
        farthest = torch.argmax(distances, dim=1)
        sampled_indices[:, i] = farthest

        if i < sample_size - 1:
            new_dist = torch.norm(points - points[batch_indices, farthest].unsqueeze(1), dim=2)
            distances = torch.minimum(distances, new_dist)

    # gather points
    sampled_points = points.gather(
        1,
        sampled_indices.unsqueeze(-1).expand(-1, -1, 3)
    )  # (B, sample_size, 3)

    return sampled_points.unsqueeze(1)


def downsample_spherical_points_vectorized(sphere_points, num_theta_bins=10, num_phi_bins=10, max_range: float = 50.0): # 球面坐标点云进行二维角度网格划分
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
    device = sphere_points.device
    num_bins = num_theta_bins * num_phi_bins
    
    theta_min, theta_max = -3.14, 3.14
    phi_min, phi_max = -0.5, 0.5  # Adjusted for typical LiDAR FOV
    
    r = sphere_points[:, :, 0]
    theta = sphere_points[:, :, 1]
    phi = sphere_points[:, :, 2]
    
    theta_bin = ((theta - theta_min) / (theta_max - theta_min) * num_theta_bins).long()
    phi_bin = ((phi - phi_min) / (phi_max - phi_min) * num_phi_bins).long()
    theta_bin = torch.clamp(theta_bin, 0, num_theta_bins - 1)
    phi_bin = torch.clamp(phi_bin, 0, num_phi_bins - 1)
    bin_indices = theta_bin * num_phi_bins + phi_bin
    
    # Preserve the nearest obstacle in each bin. Empty bins stay at max range.
    min_r = torch.full((num_envs, num_bins), max_range, device=device)
    min_r.scatter_reduce_(1, bin_indices, r, reduce="amin", include_self=True)
    
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
    theta_grid, phi_grid = torch.meshgrid(theta_centers, phi_centers, indexing='ij')
    theta_centers_flat = theta_grid.reshape(-1)
    phi_centers_flat = phi_grid.reshape(-1)
    
    downsampled = torch.zeros(num_envs, num_bins, 3, device=device)
    downsampled[:, :, 0] = min_r
    downsampled[:, :, 1] = theta_centers_flat.unsqueeze(0)
    downsampled[:, :, 2] = phi_centers_flat.unsqueeze(0)
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
        """Initialize a minimal lidar sensor environment."""
        self.sensor_cfg = sensor_cfg # Lidar 配置对象
        self.sim_time = 0 # 记录仿真时间
        self.sensor_update_time = 0 # 记录传感器更新时间
        self.state_update_time = 0 # 传感器更新时间，超过阈值后更新并清空
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
        
        # self.create_ground()
        # self.create_viewer()

        self._init_buffer() # 绑定 Isaac Gym root state，与 GPU 张量同步
        self._create_warp_mesh() # Warp 格式的 mesh 用于 Lidar 仿真

        self._create_warp_tensor_dict() # GPU 张量（点云输出/距离输出/姿态）
        
        self.sensor = LidarSensor(self.warp_tensor_dict, None, self.sensor_cfg, 1, self.device)
        # self.lidar_update_interval = self._get_lidar_update_interval()

        # sensor_points_tensor:形状为 (num_envs, num_sensors, V, H, 3) 的点云(x, y, z)
        # sensor_dist_tensor:形状为 (num_envs, num_sensors, V, H) 的距离图(depth map)
        self.sensor.capture()
        self.sensor_points_tensor, self.sensor_dist_tensor = self.sensor.update() # 雷达扫描
        

    # def create_sim(self):
    #     """Create a Genesis simulation."""
    #     # configure sim
    #     self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly

    #     self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
    #     if self.sim is None:
    #         print("*** Failed to create sim")
    #         quit()

    #     mesh_type = self.cfg.terrain.mesh_type
    #     self.terrain_cfg = ElSpiderAirRoughLidarCfg.terrain()
    #     self.terrain = Terrain(self.terrain_cfg, self.num_envs)

    #     if mesh_type == 'plane':
    #         self._create_ground_plane()
    #     elif mesh_type == 'heightfield':
    #         self._create_heightfield()
    #     elif mesh_type in ['trimesh', 'confined_trimesh']:
    #         self._create_trimesh()
    #     elif mesh_type is not None:
    #         raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh, confined_trimesh]")

    #     self._create_envs()
    #     self._setup_enhanced_lighting()

    #     # Initialize obstacle config (will be used in create_sim)
    #     self.obstacle_config = None
    #     if hasattr(self.cfg, 'obstacle_gen') and self.cfg.obstacle_gen.enable_obstacles:
    #         self.obstacle_config = ObstacleGenConfig()
    #         # Apply custom settings from config
    #         self.obstacle_config.min_stones_per_env = self.cfg.obstacle_gen.min_obstacles
    #         self.obstacle_config.max_stones_per_env = self.cfg.obstacle_gen.max_obstacles
    #         self.obstacle_config.spawn_height_range = self.cfg.obstacle_gen.spawn_height_range
    #         self.obstacle_config.spawn_radius_range = self.cfg.obstacle_gen.spawn_radius_range
    #         self.obstacle_config.density_range = self.cfg.obstacle_gen.stone_density_range
    #         self.obstacle_config.friction_range = self.cfg.obstacle_gen.stone_friction_range
    #         self.obstacle_config.restitution_range = self.cfg.obstacle_gen.stone_restitution_range
    #         self.obstacle_config.cluster_probability = self.cfg.obstacle_gen.cluster_probability
    #         self.obstacle_gen = ObstacleGen(self.gym, self.sim, self.envs, self.obstacle_config)
    #         self.obstacle_gen.generate_stones()
        
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
        # self.measured_heights=self._get_heights() # 待删：是否注释
        
        # LiDAR observation buffers
        num_lidar_obs = self.num_theta_bins * self.num_phi_bins
        self.lidar_obs_buf = torch.zeros(
            self.num_envs, num_lidar_obs*3, device=self.device, requires_grad=False
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
        
    # 创建warp格式的环境网格
    def _create_warp_mesh(self):
        """Create WARP mesh from terrain for ray casting."""
        wp.init()
        
        # Get terrain mesh vertices and triangles
        if hasattr(self, 'terrain') and self.terrain is not None:
            if hasattr(self.terrain, 'vertices') and self.terrain.vertices is not None:
                vertices = self.terrain.vertices.copy()
                triangles = self.terrain.triangles.copy()
                
                # Apply terrain offset
                if hasattr(self.cfg.terrain, 'border_size'):
                    vertices[:, 0] -= self.cfg.terrain.border_size
                    vertices[:, 1] -= self.cfg.terrain.border_size
            else:
                # Create simple ground plane if no terrain mesh
                vertices = np.array([
                    [-50, -50, 0],
                    [50, -50, 0],
                    [50, 50, 0],
                    [-50, 50, 0]
                ], dtype=np.float32)
                triangles = np.array([
                    [0, 1, 2],
                    [0, 2, 3]
                ], dtype=np.int32)
        else:
            # Create simple ground plane
            vertices = np.array([
                [-50, -50, 0],
                [50, -50, 0],
                [50, 50, 0],
                [-50, 50, 0]
            ], dtype=np.float32)
            triangles = np.array([
                [0, 1, 2],
                [0, 2, 3]
            ], dtype=np.int32)
        
        # Convert to WARP arrays
        vertex_tensor = torch.tensor(vertices, device=self.device, dtype=torch.float32)
        vertex_wp = wp.from_torch(vertex_tensor, dtype=wp.vec3)
        faces_wp = wp.from_numpy(triangles.flatten().astype(np.int32), dtype=wp.int32, device=self.device)
        
        # Create WARP mesh
        self.wp_mesh = wp.Mesh(points=vertex_wp, indices=faces_wp)
        self.mesh_ids = wp.array([self.wp_mesh.id], dtype=wp.uint64)

    def _create_warp_tensor_dict(self):
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
        # 定义传感器位姿（位置和朝向)
        self.sensor_pos_tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.sensor_quat_tensor = torch.zeros(self.num_envs, 4, device=self.device)
        
        # 传感器相对于载体的安装偏移
        offset_pos = getattr(self.cfg.LidarConfig, "sensor_offset_pos", None)
        if offset_pos is None:
            offset_pos = [0.3, 0.0, 0.35]

        if hasattr(self.cfg, 'LidarConfig') and hasattr(self.cfg.LidarConfig, 'sensor_offset_pos'):
            self.sensor_translation_local = torch.tensor(
                self.cfg.LidarConfig.sensor_offset_pos, device=self.device
            )
        else:
            self.sensor_translation_local = torch.tensor([0.3, 0.0, 0.35], device=self.device)

        if hasattr(self.cfg, 'LidarConfig') and hasattr(self.cfg.LidarConfig, 'sensor_offset_rpy'):
            roll = np.deg2rad(self.cfg.LidarConfig.sensor_offset_rpy[0])
            pitch = np.deg2rad(self.cfg.LidarConfig.sensor_offset_rpy[1])
            yaw = np.deg2rad(self.cfg.LidarConfig.sensor_offset_rpy[2])
            self.sensor_offset_quat_local = quat_from_euler_xyz_tensor(
                torch.tensor([roll], device=self.device),
                torch.tensor([pitch], device=self.device),
                torch.tensor([yaw], device=self.device)
            ).squeeze()
        else:
            self.sensor_offset_quat_local = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        self.sensor_translation = self.sensor_translation_local.repeat(self.num_envs, 1)
        self.sensor_offset_quat = self.sensor_offset_quat_local.repeat(self.num_envs, 1)

        self._update_lidar_pose()
        
        self.warp_tensor_dict["sensor_dist_tensor"] = self.sensor_dist_tensor
        self.warp_tensor_dict['device'] = self.device
        self.warp_tensor_dict['num_envs'] = self.num_envs
        self.warp_tensor_dict['num_sensors'] = self.sensor_cfg.num_sensors
        self.warp_tensor_dict['sensor_pos_tensor'] = self.sensor_pos_tensor
        self.warp_tensor_dict['sensor_quat_tensor'] = self.sensor_quat_tensor
        self.warp_tensor_dict['mesh_ids'] = self.mesh_ids

    def create_viewer(self):
        # create viewer
        if self.headless == True:
            self.viewer = None
            print("Running in headless mode")
        else:
            self.debug_viz = True
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT") # 按 Esc 关闭仿真窗口。
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync") # 焦点在仿真与显示之间切换
            
            self.vis = GymVisualizer(self.gym, self.sim, self.viewer, self.envs)


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

    # def _render_headless(self):
    #     self.gym.render_all_camera_sensors(self.sim)
    #     bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
    #     self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx, by - 1.0, bz + 1.0),
    #                                     gymapi.Vec3(bx, by, bz))
    #     #camera_hanle=self.gym.get_viewer_camera_handle(self.viewer)
    #     self.video_frame = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera,
    #                                                     gymapi.IMAGE_COLOR)
    #     self.video_frame = self.video_frame.reshape((self.camera_props.height, self.camera_props.width, 4))
    #     self.video_frames.append(self.video_frame)
    #     self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(bx-3, by- 2.5, bz + 3.5), gymapi.Vec3(bx, by, bz))
    #     if len(self.video_frames)>250:
    #         save_video(self.video_frames,f"videos/Lidar_demo.mp4",fps=50)
    #         print("save video!!")
    #         self.video_frames=[]
    #     self.sequence_number = self.sequence_number + 1
    #     rgb_image_filename = "images/rgb_image_%03d.png" % (self.sequence_number)

    #     self.gym.write_viewer_image_to_file(self.viewer,rgb_image_filename)


    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.last_base_ang_vel = self.base_ang_vel.clone()
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, :3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_acc[:] = self.base_lin_acc[:] * self.acc_ema + (1 - self.acc_ema) * \
            quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_ang_acc[:] = self.base_ang_acc[:] * self.acc_ema + (1 - self.acc_ema) * \
            quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13] - self.last_root_vel[:, 3:]) / self.dt
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.last_foot_velocities[:] = self.foot_velocities[:]

        # self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # 每物理步后，为后续奖励与观测提供最新数据
    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        if self.sensor is None:
            return

        self._update_lidar_pose()
        self.sim_time += self.dt
        self.sensor_update_time += self.dt
        self.state_update_time += self.dt
        self.save_time += self.dt

        self.sensor_points_tensor, self.sensor_dist_tensor = self.sensor.update()
        self.sensor_points_tensor = self.sensor_points_tensor.contiguous().view(self.num_envs, 1, -1, 3)
        self.sensor_dist_tensor = self.sensor_dist_tensor.contiguous().view(self.num_envs, 1, -1)

        total_rays = self.sensor_cfg.horizontal_line_num * self.sensor_cfg.vertical_line_num
        # 远点采样，用于绘制点云
        # if self.cfg.terrain.draw_lidar_points:
        #     if self.sensor_points_tensor.shape[1] > 0:
        #         self.downsampled_cloud = farthest_point_sampling(
        #             self.sensor_points_tensor, sample_size=total_rays
        #         )
        #     else:
        #         self.downsampled_cloud = torch.zeros(
        #             self.num_envs,1 , 1, 3, device=self.device, requires_grad=False
        #         )

        used_rays = min(total_rays, self.sensor_points_tensor.shape[2])

        self.lidar_points_buf.zero_()
        self.lidar_dist_buf.fill_(self.sensor_cfg.max_range)
        self.lidar_points_buf[:, :used_rays, :] = self.sensor_points_tensor.squeeze(1)[:, :used_rays, :]
        self.lidar_dist_buf[:, :used_rays] = self.sensor_dist_tensor.squeeze(1)[:, :used_rays]

        # valid_hit = (self.lidar_dist_buf > self.sensor_cfg.min_range) & (
        #     self.lidar_dist_buf < self.sensor_cfg.max_range
        # )
        # clamped_dist = torch.where(
        #     valid_hit,
        #     self.lidar_dist_buf,
        #     torch.full_like(self.lidar_dist_buf, self.sensor_cfg.max_range)
        # )
        # self.min_obstacle_dist[:] = clamped_dist.min(dim=1)[0]

        # Compute minimum obstacle distance: 用于检查是否重置
        dist_flat = self.sensor_dist_tensor.view(self.num_envs, -1)
        maxr = float(self.sensor_cfg.max_range)
        # Replace invalid distances with max_range so min will be max_range when no valid returns
        dist_flat_clean = torch.where((dist_flat > 0) & (dist_flat < maxr), dist_flat, torch.full_like(dist_flat, maxr))
        self.min_obstacle_dist[:] = torch.min(dist_flat_clean, dim=1).values


        sphere_points = cart2sphere(self.lidar_points_buf).view(self.num_envs, -1, 3)
        downsampled = downsample_spherical_points_vectorized(
            sphere_points, self.num_theta_bins, self.num_phi_bins, self.sensor_cfg.max_range
        )
        
        # 用于训练
        # Use normalized distance as observation (0 = close, 1 = far/no hit) 左侧形状和右侧相同
        downsampled[:, :, 0] = downsampled[:, :, 0].clamp(0, self.sensor_cfg.max_range) / self.sensor_cfg.max_range
        self.lidar_obs_buf[:] = torch.cat((downsampled[:, :, 0], downsampled[:, :, 1], downsampled[:, :, 2]), dim=-1).view(self.num_envs, -1)
        # self.lidar_obs_buf[:] = downsampled[:, :, 0].clamp(0, self.sensor_cfg.max_range) / self.sensor_cfg.max_range

        if self.save_data and (self.save_time) >= self.save_interval:
            self.collect_and_save_data()
            self.save_time = 0
        # Update visualization

    def check_termination(self):
        """Check termination conditions including collision detection."""
        super().check_termination()
        
        # Get collision parameters from config
        if hasattr(self.cfg.rewards, 'collision_threshold'):
            collision_threshold = self.cfg.rewards.collision_threshold
        else:
            collision_threshold = 0.08  # Default 0.08m - more permissive
        
        # Get protection steps (grace period during early training)
        if hasattr(self.cfg.rewards, 'collision_termination_after_steps'):
            min_steps = self.cfg.rewards.collision_termination_after_steps
        else:
            min_steps = 24  # Default: only check collision after 10 steps
        
        # Only terminate due to collision after protection period
        # This allows the robot to learn without being immediately terminated
        collision = self.min_obstacle_dist < collision_threshold
        collision_termination = collision & (self.episode_length_buf > min_steps)
        self.reset_buf |= collision_termination # 按位或赋值
        # print(f"Test:Collision termination: {collision_termination}")

        # print(f"Test:Reset buffer: {self.reset_buf}")

    
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

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * \
            noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:30] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[30:48] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[48:66] = 0.  # previous actions
        noise_vec[66:] = 0. # LiDAR observations already have noise from the sensor model
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _update_lidar_pose(self) -> None:
        self.sensor_pos_tensor[:] = self.root_states[:, :3] + quat_apply(
            self.root_states[:, 3:7], self.sensor_translation
        )
        
        # Compute sensor orientation in world frame
        self.sensor_quat_tensor[:] = quat_mul(
            self.root_states[:, 3:7], self.sensor_offset_quat
        )

    def update_reward_scales(self, mean_reward):
        if mean_reward > self.cfg.rewards.reward_stage_threshold and \
                self.reward_scales_stage < self.cfg.rewards.reward_max_stage:
            self.reward_scales_stage += 1
            self.reward_scales = self._get_reward_scales(self.reward_scales_stage)
            self._prepare_reward_function()
            return True
        return False

    # ============== Reward Functions ==============
    # 继承于legged_robot_rew_mixin.py
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        # print(f"base height: {base_height}, reward: {rew}")
        return rew
    
    def _reward_dof_power(self):
        # Penalize power consumption
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1) # 功率
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions - 2*self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1. # feet_indices从参数表中的关节名称找到asset中的对应关节，然后取得的索引
    #     contact_filt = torch.logical_or(contact, self.last_contacts)
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     self.feet_contact_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)  # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     self.feet_contact_time *= contact_filt
    #     return rew_airTime

    def _reward_feet_contact_stand_still(self):
        # Encourage feet contact with the ground at zero commands
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        contact_count = len(self.feet_indices) - torch.sum(1.*contacts, dim=1)
        return 1.0*contact_count * (torch.norm(self.commands[:, :3], dim=1) < 0.1)
    
    def _reward_dof_close_to_default(self):
        # Penalize dof position deviation from default
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        """
        foot_vel_xy_norm = torch.norm(self.foot_velocities[:, :, :2], dim=-1)
        # print(f"feet pos: {self.feet_pos[:, :, 2]}")
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.foot_positions[:, :, 2] -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_foot_acc(self):
        '''reward for foot acceleration'''
        foot_acc = (self.foot_velocities - self.last_foot_velocities) / self.dt
        return torch.sum(torch.square(foot_acc), dim=(1, 2))




    def _reward_obstacle_avoidance(self):
        """Reward for maintaining safe distance from obstacles."""
        # Reward increases with distance from obstacles
        safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        
        # Compute reward based on minimum distance
        dist_reward = torch.clamp(self.min_obstacle_dist / safe_dist, 0, 1)
        return dist_reward

    def _reward_collision(self):
        """Penalty for getting too close to obstacles."""
        danger_dist = getattr(self.cfg.rewards, 'danger_obstacle_dist', 0.3)
        
        # Exponential penalty for being too close
        penalty = torch.exp(-self.min_obstacle_dist / danger_dist + 1) - 1
        penalty = torch.clamp(penalty, 0, 10)
        return penalty

    def _reward_body_joint_contact(self):
        """Penalty for body/joint contacting the ground.原collision"""
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])


    # def _reward_exploration(self):
    #     """Reward for exploring while avoiding obstacles."""
    #     # Combine forward velocity with obstacle avoidance
    #     forward_vel = self.base_lin_vel[:, 0]
    #     safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        
    #     # Only reward forward movement when it's safe
    #     safety_factor = torch.clamp(self.min_obstacle_dist / safe_dist, 0, 1)
    #     exploration_reward = forward_vel * safety_factor
    #     return torch.clamp(exploration_reward, -1, 1)

    def _draw_debug_vis(self):
        """Draw debug visualization including LiDAR points."""
        super()._draw_debug_vis()
        
        # Draw LiDAR points for first environment
        # if self.cfg.terrain.draw_lidar_points and not self.headless and self.sensor_update_time > 1/self.sensor_cfg.update_frequency:
        if self.cfg.terrain.draw_lidar_points and not self.headless:
            # self.gym.clear_lines(self.viewer)
            self._draw_lidar_points()
            self.sensor_update_time=0
        
    def _draw_lidar_points(self):
        """Visualize LiDAR point cloud."""
        if not hasattr(self, 'viewer') or self.viewer is None:
            return

        # sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))

        # if self.sensor_cfg.pointcloud_in_world_frame:
        #     self.global_pixels =  self.downsampled_cloud
        #     for i in range(self.selected_env_idx,self.selected_env_idx+1):
        #         for j in range(int(self.global_pixels.shape[2])):
        #             for k in range(self.global_pixels.shape[3]):
        #                 x = self.global_pixels[i, 0,j,k,0]#+self.root_states[:1, 0]
        #                 y = self.global_pixels[i, 0,j,k,1]
        #                 z = self.global_pixels[i, 0,j,k,2]
        #                 sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        #                 gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        # else:
        #     self.local_pixels_downsampled = self.downsampled_cloud.reshape(-1, 3)
        #     self.sensor_axis= self.sensor_pos_tensor[:,:]       
        #     pixels = self.local_pixels_downsampled.view(self.num_envs,-1,3)
        #     pixels_num = pixels.shape[1]
        #     sensor_axis_shaped = self.sensor_axis.unsqueeze(1).repeat(1, pixels_num, 1).view(self.num_envs, -1, 3)
        #     sensor_quat = self.sensor_quat_tensor.unsqueeze(1).repeat(1, pixels_num, 1).view(self.num_envs, -1, 4)
        #     self.global_pixels = sensor_axis_shaped + quat_apply(sensor_quat, pixels)
            
        #     # def draw_line(p1, p2, color, gym, viewer, env):

        #     self.global_pixels.view(self.num_envs,-1, 3)
        #     for i in range(self.selected_env_idx,self.selected_env_idx+1):
        #         for j in range(0,self.global_pixels.shape[1]):
        #                 x = self.global_pixels[i, j,0]
        #                 y = self.global_pixels[i, j,1]
        #                 z = self.global_pixels[i, j,2]
        #                 sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        #                 gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

        max_envs_to_draw = int(getattr(self.cfg.viewer, 'lidar_vis_num_envs', self.num_envs))
        max_envs_to_draw = max(1, min(max_envs_to_draw, self.num_envs))
        max_envs_to_draw = 1
        max_points = int(getattr(self.cfg.viewer, 'lidar_vis_max_points', 180))
        near_geom = gymutil.WireframeSphereGeometry(0.012, 4, 4, None, color=(1, 0, 0))
        far_geom = gymutil.WireframeSphereGeometry(0.012, 4, 4, None, color=(0, 1, 0))
        near_threshold = 0.6

        for env_idx in range(max_envs_to_draw):
            points_local = self.lidar_points_buf.squeeze(1)[env_idx]
            dists = self.lidar_dist_buf.squeeze(1)[env_idx]
            valid_mask = (dists > self.sensor_cfg.min_range) & (dists < self.sensor_cfg.max_range)
            if not torch.any(valid_mask):
                continue

            points_local = points_local[valid_mask]
            dists = dists[valid_mask]

            if points_local.shape[0] > max_points:
                idx = torch.linspace(0, points_local.shape[0] - 1, max_points, device=self.device).long()
                points_local = points_local[idx]
                dists = dists[idx]

            sensor_pos = self.sensor_pos_tensor[env_idx]
            sensor_quat = self.sensor_quat_tensor[env_idx]
            sensor_quat_expand = sensor_quat.unsqueeze(0).expand(points_local.shape[0], -1)
            world_points = sensor_pos.unsqueeze(0) + quat_apply(sensor_quat_expand, points_local)

            for point_idx in range(world_points.shape[0]):
                pos = world_points[point_idx]
                geom = near_geom if dists[point_idx] < near_threshold else far_geom
                pose = gymapi.Transform(gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2])), r=None)
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[env_idx], pose)




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
    
