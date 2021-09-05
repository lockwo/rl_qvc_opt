from env import RandomCircuits
import sys
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

steps = 5000000
windows = 50
learn_delay = 10000
buffer = 100000
load = False

# Max Qubits, Max Depth, Max Optimization Steps, Learner, t

q = int(sys.argv[1])
d = int(sys.argv[2])
opt = int(sys.argv[3])
env = RandomCircuits(q, d, opt)
if sys.argv[5] == 't':
    load = True

if sys.argv[4] == "t":
    if load:
        agent = TD3.load("td3", env=env)
        print("loaded")
    else: 
        policy_kwargs = dict(net_arch=dict(pi=[1028, 512], qf=[1028, 512]))
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        agent = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, buffer_size=buffer, policy_kwargs=policy_kwargs, learning_starts=learn_delay)
elif sys.argv[4] == 's':
    if load:
        agent = SAC.load("sac", env=env)
        print("loaded")
    else:
        policy_kwargs = dict(net_arch=dict(pi=[1028, 512], qf=[1028, 512]))
        agent = SAC("MlpPolicy", env, verbose=1, buffer_size=buffer, policy_kwargs=policy_kwargs, learning_starts=learn_delay)
elif sys.argv[4] == 'p':
    if load:
        agent = PPO.load("ppo", env=env)
        print("loaded")
    else:
        policy_kwargs = dict(net_arch=[dict(pi=[1028, 512], vf=[1028, 512])])
        agent = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./', name_prefix=sys.argv[4])
agent.learn(total_timesteps=steps, log_interval=10, callback=checkpoint_callback)
