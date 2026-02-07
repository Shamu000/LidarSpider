import argparse  
from legged_gym.utils.task_registry import task_registry  
  
def main():  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--task', type=str, default='elspider_air_lidar_progressive')  
    parser.add_argument('--num_envs', type=int, default=2048)  
    parser.add_argument('--headless', action='store_true')  
    parser.add_argument('--curriculum_level', type=int, default=0)  
      
    args = parser.parse_args()  
      
    # 创建环境  
    env, env_cfg = task_registry.make_env(name=args.task, args=args)  
      
    # 设置课程学习级别  
    if hasattr(env_cfg, 'curriculum') and env_cfg.curriculum.enable_curriculum:  
        level = env_cfg.curriculum.levels[args.curriculum_level]  
          
        # 更新配置  
        env_cfg.raycaster.lidar_noise_std = level['lidar_noise_std']  
        env_cfg.rewards.scales.obstacle_avoidance = level['obstacle_avoidance_weight']  
        env_cfg.raycaster.max_distance = level['max_distance']  
          
        print(f"Starting curriculum level: {level['name']}")  
      
    # 开始训练  
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)  
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations)  
  
if __name__ == '__main__':  
    main()