from legged_gym.envs.elspider_air.flat.elspider_air_flat_config import ElSpiderAirFlatCfg, ElSpiderAirFlatCfgPPO  
  
class ElSpiderAirLidarProgressiveCfg(ElSpiderAirFlatCfg):  
    """渐进式LiDAR训练配置"""  
      
    class env(ElSpiderAirFlatCfg.env):  
        # num_observations = 66 + 512  # 基础观察 + LiDAR  
        num_observations = 66
          
    class raycaster:  
        enable_raycast = True  
        use_omni_lidar = True  # 启用外部LiDAR  
        ray_pattern = "spherical2"  
        spherical2_num_points = 512  
        max_distance = 8.0  
        offset_pos = [0.0, 0.0, 0.3]  # 头部安装  
          
        # LiDAR硬件参数  
        scan_frequency = 10.0  # Hz  
        vertical_fov = 30.0  # degrees  
        lidar_noise_std = 0.01  # 低噪声（初级训练）  
          
    class rewards:  
        class scales:  
            # 基础运动奖励  
            lin_vel_tracking = 1.0  
            ang_vel_tracking = 0.5  
              
            # 避障奖励（渐进增加）  
            obstacle_avoidance = 0.5  # 初级阶段较低权重  
              
            # 足端奖励  
            feet_contact_force = 0.0  
              
    class curriculum:  
        enable_curriculum = True  
        levels = [  
            {  
                'name': 'flat_basic',  
                'lidar_noise_std': 0.01,  
                'obstacle_avoidance_weight': 0.5,  
                'max_distance': 5.0  
            },  
            {  
                'name': 'flat_noise',  
                'lidar_noise_std': 0.02,  
                'obstacle_avoidance_weight': 1.0,  
                'max_distance': 8.0  
            },  
            {  
                'name': 'confined_space',  
                'lidar_noise_std': 0.03,  
                'obstacle_avoidance_weight': 2.0,  
                'max_distance': 10.0  
            }  
        ]  
  
class ElSpiderAirLidarProgressiveCfgPPO(ElSpiderAirFlatCfgPPO):  
    class algorithm:  
        # 学习率调度  
        learning_rate = 3e-4  
        schedule = 'adaptive'  
          
        # 网络结构调整（适应LiDAR输入）  
        hidden_dim = 512  
        num_layers = 3  
          
    class runner:  
        max_iterations = 4000  
        curriculum_learning = True