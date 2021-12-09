import numpy as np
from stable_baselines3 import SAC

sac_mlp_agent = SAC.load("sac_mlp")
sac_cnn_agent = SAC.load("sac_cnn")

def mlp_encoding(size, num_q, num_d, struct, weights, error, input_type):
    state = np.zeros(shape=(size, 8))
    for i in range(len(weights)):
        q = i % num_q
        state[i] = [-error, weights[i], struct[i], q, i//num_q, num_q, num_d, input_type]
    return state.flatten()

def cnn_encoding(max_q, max_d, num_q, struct, weights, error, input_type):
    state = np.zeros(shape=(max_q, max_d, 5))
    for i in range(len(weights)):
        qubit_number = i % num_q
        depth_number = i // num_q
        state[qubit_number][depth_number][struct[i]] = weights[i]
        state[:,:,3] = input_type
        state[:,:,4] = error
    return state.transpose(2, 0, 1)


'''
Inputs:
f = function that takes weights as input and returns the loss
structure = 1D list of integers that represent the structure (0 = RX, 1 = RY, 2 = RZ) starting a qubit 0 depth 0 and going down each qubit for every depth
number_of_qubits = number of qubits in the system
current_params = the current weights of the circuit
current_loss = the current loss of the model
input_type = 0 for |0> state and 1 for equal superposition state
ind = whether or not to use a single [ind]ividual example
'''
def mixed(f, structure, number_of_qubits, current_params, current_loss, input_type, ind=False):
    number_of_params = len(structure)
    mlp_enc = mlp_encoding(400, number_of_qubits, len(structure) // number_of_qubits, structure, current_params, current_loss, input_type)
    cnn_enc = cnn_encoding(20, 20, number_of_qubits, structure, current_params, current_loss, input_type)
    mlp_weights = np.array([sac_mlp_agent.predict(mlp_enc)[0][:number_of_params],])
    cnn_weights = np.array([sac_cnn_agent.predict(cnn_enc)[0][:number_of_params],])
    if ind:
        mlp_loss = f(mlp_weights[0])
        cnn_loss = f(cnn_weights[0])
        if mlp_loss < cnn_loss:
            return mlp_weights[0]
        else:
            return cnn_weights[0]
    else:
        mlp_loss = f(mlp_weights)
        cnn_loss = f(cnn_weights)
        if mlp_loss < cnn_loss:
            return mlp_weights
        else:
            return cnn_weights
