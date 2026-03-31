# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs import ElSpiderAirRoughLidarCfg, ElSpiderAirRoughLidarCfgPPO


class ElSpiderAirFlatLidarCfg(ElSpiderAirRoughLidarCfg):
    class env(ElSpiderAirRoughLidarCfg.env):
        # Update observation space for raycast data
        # num_observations = 253 + 272 + 192*3  # MID360
        num_observations = 354  # MID360, 不扫描周围高度
        # num_observations = 66 # 无雷达
        episode_length_s = 10

    class terrain(ElSpiderAirRoughLidarCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False # 若为true，则要给obs_buf加187维
        draw_lidar_points = True

    class rewards(ElSpiderAirRoughLidarCfg.rewards):
        base_height_target = 0.30
        max_contact_force = 500.
        only_positive_rewards = False

        # Obstacle avoidance parameters
        safe_obstacle_dist = 0.5    # Distance considered safe (meters)
        danger_obstacle_dist = 0.3  # Distance considered dangerous (meters)
        collision_threshold = 0.4  # Distance for collision termination (meters) - reduced from 0.15
        
        # Termination protection - disable collision termination during early training steps
        # collision_termination_after_steps = 10  # Only terminate after this many steps
        # allow_initial_contact_steps = 5  # Allow contact termination grace period

        # Multi-stage rewards
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 6.0
        # Stage0-1: plane, Stage2: curriculum
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 0

        class scales(ElSpiderAirRoughLidarCfg.rewards.scales):
            # Tracking rewards
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.5
            foot_acc = [0, 0, 0]
            obstacle_avoidance = [1., 1., 1.]
            # Base penalties
            lin_vel_z = [-1.0, -1.5, -4.0]
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            base_height = [-8.0, -8.0, -8.0]
            # DOF penalties
            dof_vel = 0.
            dof_acc = -5e-8
            dof_pos_limits = -1.0
            # dof_power = -2.e-4
            action_rate = [-0.001, -0.001, -0.001]
            action_smoothness = -0
            # Feet penalties
            feet_slip = [-0.0, -0.4, -1.2]  #脚部打滑惩罚
            jump_air = 0
            feet_air_time = 2.0
            feet_stumble = -0.0 # 脚部碰到垂直面的惩罚
            # feet_stumble_liftup = [1.0, 1.0, 2.0] # 脚部碰到垂直面时向上抬起的奖励
            # feet_contact_forces = [-0.01, -0.05, -0.05]  # Avoid jumping
            body_joint_contact = [-1.0, -2.0, -3.0] # 原collision
            # Misc
            termination = -5.
            collision = -2.5
            stand_still = -0.
            # dof_vel_stand_still = -1e-4
            # dof_pos_stand_still = -2e-2
            # feet_contact_stand_still = -0.1
            # Gait
            # async_gait_scheduler = [-5., -6, -7.]
            gait_2_step = [-5.0, -5.0, -5.0]
            foot_clearance = 0.5

        class async_gait_scheduler:
            # Reward for the async gait scheduler
            dof_align = 0.5 # 关节角度对称/一致性奖励
            dof_nominal_pos = [0.1, 0.2, 0.2] # 回到参考姿态奖励
            reward_foot_z_align = [0.2, 0.05, 0.05] # 足端高度一致性奖励（平整度奖励）
            
        foot_clearance_target = 0.08 # desired foot clearance above ground [m]
        foot_height_offset = 0.0     # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01


class ElSpiderAirFlatLidarCfgPPO(ElSpiderAirRoughLidarCfgPPO):
    """PPO training configuration for ElSpider LiDAR confined space task."""

    class policy(ElSpiderAirRoughLidarCfgPPO.policy):
        init_noise_std = 1.0

    class runner(ElSpiderAirRoughLidarCfgPPO.runner):
        run_name = ''
        experiment_name = 'elspider_air_flat_lidar' # 保存的log文件夹名称
        load_run = -1
        max_iterations = 3000
        
        # Enable multi-stage rewards
        multi_stage_rewards = True
        
        # Checkpointing
        save_interval = 50
        
        # Logging
        log_interval = 10
