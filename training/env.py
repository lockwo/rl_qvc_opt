import tensorflow as tf
import tensorflow_quantum as tfq 
import cirq
import random
import sympy
import numpy as np
import operator
from functools import reduce 
import gym
from gym import spaces

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

class RandomCircuits(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, max_qubits, max_depth, optimization_steps, enc_type='feature') -> None:
        super(RandomCircuits, self).__init__()
        self.max_qubits = max_qubits
        self.max_depth = max_depth
        self.max_symbols = max_depth * max_qubits
        self.symbols = sympy.symbols('q0:%d'%self.max_symbols)
        self.cost_functions = [self.vqe_cost, self.first_cost, self.sum_cost]
        self.struct, self.rend, self.circuit, self.target, self.input_circuit, self.ins = self.create_circuit()
        self.steps = 0
        self.max_steps = optimization_steps
        self.num_params = None
        self.num_q = None
        self.num_d = None
        self.action_space = spaces.Box(0, 2 * np.pi, (self.max_symbols,), dtype=np.float32)
        if enc_type == 'feature':
            self.observation_space = spaces.Box(-np.inf, np.inf, (self.max_symbols * 8,), dtype=np.float32)
        elif enc_type == 'block':
            self.observation_space = spaces.Box(-np.inf, np.inf, (5, self.max_qubits, self.max_depth), dtype=np.float32)
        self.info = dict()
        self.enc_type = enc_type

    def vqe_cost(self, qubits):
        return prod([cirq.Z(qubits[i]) for i in range(len(qubits))]), np.random.uniform(-1, 1, 1)

    def first_cost(self, qubits):
        return [cirq.Z(qubits[0])], np.random.uniform(-1, 1, 1)
    
    def sum_cost(self, qubits):
        return sum([cirq.Z(qubits[i]) for i in range(len(qubits))]), np.random.uniform(-len(qubits), len(qubits), 1)

    # struct is qubit first, by depth
    def create_circuit(self):
        qubits = [cirq.GridQubit(0, i) for i in range(random.randint(2, self.max_qubits))]
        depth = random.randint(1, self.max_depth)
        circuit = cirq.Circuit()
        struct = []
        for d in range(depth):
            for i in range(len(qubits)):
                circuit += cirq.CNOT(qubits[i], qubits[(i + 1) % len(qubits)])
            for i, qubit in enumerate(qubits):
                random_n = np.random.uniform()
                if random_n > 2. / 3.:
                    circuit += cirq.rz(self.symbols[d * len(qubits) + i])(qubit)
                    struct.append(2)
                elif random_n > 1. / 3.:
                    circuit += cirq.ry(self.symbols[d * len(qubits) + i])(qubit)
                    struct.append(1)
                else:
                    circuit += cirq.rx(self.symbols[d * len(qubits) + i])(qubit)
                    struct.append(0)
        self.num_q = len(qubits)
        self.num_d = depth
        self.num_params = depth * len(qubits)
        cost_fn = random.choice(self.cost_functions)
        readout_ops, target_val = cost_fn(qubits)
        ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
        init = tf.keras.initializers.RandomUniform(0, 2 * np.pi)
        layer1 = tfq.layers.PQC(circuit, readout_ops, repetitions=1000, initializer=init)(ins)
        model = tf.keras.models.Model(inputs=ins, outputs=layer1)
        if np.random.uniform() > 0.5:
            cir_in = tfq.convert_to_tensor([cirq.Circuit()])
            cir_ins = None
        else:
            cir_in = tfq.convert_to_tensor([cirq.Circuit([cirq.H(i) for i in qubits])])
            cir_ins = "H"
        return struct, circuit, model, target_val, cir_in, cir_ins

    def get_error(self):
        return -abs(self.circuit(self.input_circuit).numpy()[0][0] - self.target[0])**2

    # Return State, Reward, Done
    def step(self, action):
        done = False
        if self.steps > self.max_steps:
            done = True
        #old = self.circuit(self.input_circuit)
        #print(action)
        self.circuit.set_weights([action[:self.num_params]])
        error = self.get_error()
        if -error < 0.001:
            done = True
        self.steps += 1
        observation = self.encoding(action[:self.num_params], error)
        return observation, error, done, self.info 

    # FLIP inspired
    # max_params X 8 array
    # each param = [current error, current value, gate type, qubit number, qubit layer, max_qubits, max_depth, input]
    # Feature = https://arxiv.org/pdf/2103.07585.pdf inspired
    # [qubit, depth, num_gate_classes (3) + inputs (1) + error (1)]
    # num_gate_classes = 3 (Rx, Ry, Rz)
    def encoding(self, weights, error):
        if self.enc_type == 'feature':
            state = np.zeros(shape=(self.max_symbols, 8))
            for i in range(len(weights)):
                q = i % self.num_q
                #print(weights[i], self.struct[i], q, i//self.max_qubits, self.max_qubits, self.max_depth, 0 if self.ins is None else 1)
                state[i] = [error, weights[i], self.struct[i], q, i//self.num_q, self.num_q, self.num_d, 0 if self.ins is None else 1]
            return state.flatten()
        elif self.enc_type == 'block':
            state = np.zeros(shape=(self.max_qubits, self.max_depth, 5))
            for i in range(len(weights)):
                qubit_number = i % self.num_q
                depth_number = i // self.num_q
                state[qubit_number][depth_number][self.struct[i]] = weights[i]
                state[:,:,3] = 0 if self.ins is None else 1
                state[:,:,4] = error
            return state.transpose(2, 0, 1)

    def reset(self):
        self.struct, self.rend, self.circuit, self.target, self.input_circuit, self.ins = self.create_circuit()
        self.steps = 0
        return self.encoding(self.circuit.trainable_variables[0].numpy(), self.get_error())

    def render(self, mode='human'):
        print(self.rend)

'''
if __name__ == "__main__":
    env = RandomCircuits(20, 20, 150, enc_type='feature')
    #print(env.observation_space, env.max_symbols)
    state = env.reset()
    iterations = 1
    for i in range(iterations):
        done = False
        env.render()
        while not done:
            state, reward, done, _ = env.step(np.random.uniform(0, np.pi, env.max_symbols))
            print(state, reward, done)
            for i in range(5):
                print("SHEET")
                print(state[:,:,i])
'''
'''
from random_circuit_env import RandomCircuits
env = RandomCircuits(10, 10, 100, 100)
from stable_baselines3.common.env_checker import check_env
check_env(env)
'''
