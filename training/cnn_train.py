from env import RandomCircuits
import sys
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import gym

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels Last)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512),
)

steps = 5000000
learn_delay = 10000
buffer = 100000
load = False

# Max Qubits, Max Depth, Max Optimization Steps, Learner, Load

q = int(sys.argv[1])
d = int(sys.argv[2])
opt = int(sys.argv[3])
env = RandomCircuits(q, d, opt, "block")
#print(env.observation_space.shape, env.action_space.shape)
if sys.argv[5] == 't':
    load = True

if sys.argv[4] == "t":
    if load:
        agent = TD3.load("tcnn", env=env)
        print("loaded")
    else: 
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        agent = TD3("CnnPolicy", env, action_noise=action_noise, verbose=1, buffer_size=buffer, policy_kwargs=policy_kwargs, learning_starts=learn_delay)
elif sys.argv[4] == 's':
    if load:
        agent = SAC.load("scnn", env=env)
        print("loaded")
    else:
        agent = SAC("CnnPolicy", env, verbose=1, buffer_size=buffer, policy_kwargs=policy_kwargs, learning_starts=learn_delay)
elif sys.argv[4] == 'p':
    if load:
        agent = PPO.load("pcnn", env=env)
        print("loaded")
    else:
        agent = PPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./', name_prefix=sys.argv[4]+"cnn")
agent.learn(total_timesteps=steps, log_interval=10, callback=checkpoint_callback)
