# Reinforcement Learning for Quantum Variational Circuit Optimization
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)
Code for Optimizing Quantum Variational Circuits with Deep Reinforcement Learning

To use the pretrained models, don't worry about training or testing. Download deploy folder and use the `mixed` function in `augment.py` to work with your circuit. See the file for the required inputs. Note that deployment usage requires stable_baselines3 (which requires PyTorch) and numpy. 

To recreate the results from the paper, use the testing folder. Each file runs and outputs the information as presented in the table. Note that this has dependencies on TensorFlow, TensorFlow-Quantum, sklearn, and stable_baselines3 (which requires PyTorch).

To train your own agent, go to the training folder. Specifiy the maximum sizes and training durations for the agent and run the code. 
