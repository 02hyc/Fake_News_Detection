import gym
from stable_baselines3 import PPO  # 也可以选择其他算法，比如DQN、A2C等
from stable_baselines3.common.envs import DummyVecEnv
import numpy as np


class InvertedPendulumEnv(gym.Env):
    def __init__(self):
        super(InvertedPendulumEnv, self).__init__()
        # 定义环境参数
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_length = 1.0
        self.pole_mass = 0.1
        self.dt = 0.02
        self.theta_threshold_radians = 15 * 2 * np.pi / 360  # 转换为弧度

        # 定义观察空间和动作空间
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 初始化状态
        self.reset()

    def reset(self):
        # 初始化状态
        self.cart_x = 0.0
        self.theta = np.random.uniform(-0.05, 0.05)
        self.theta_dot = 0.0

        return np.array([self.cart_x, self.theta, self.theta_dot, 0.0])

    def step(self, action):
        # 执行动作
        force = np.clip(action, -1.0, 1.0)
        costheta = np.cos(self.theta)
        sintheta = np.sin(self.theta)
        temp = (force + self.pole_mass * self.pole_length * self.theta_dot ** 2 * sintheta) / (
                    self.cart_mass + self.pole_mass)
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.pole_length * (4.0 / 3.0 - self.pole_mass * costheta ** 2 / (self.cart_mass + self.pole_mass)))
        xacc = temp - self.pole_mass * self.pole_length * thetaacc * costheta / (self.cart_mass + self.pole_mass)

        # 更新状态
        self.cart_x += self.dt * self.theta_dot
        self.theta += self.dt * self.theta_dot
        self.theta_dot += self.dt * thetaacc

        # 计算奖励
        reward = 1.0 if abs(self.theta) < self.theta_threshold_radians else 0.0
        done = abs(self.theta) > self.theta_threshold_radians

        return np.array([self.cart_x, self.theta, self.theta_dot, xacc]), reward, done, {}


# 创建环境
env = InvertedPendulumEnv()
env = DummyVecEnv([lambda: env])

# 创建模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 保存模型
model.save("inverted_pendulum_model")

# 测试模型
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    if done:
        obs = env.reset()
