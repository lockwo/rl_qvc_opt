import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import networkx as nx
from stable_baselines3 import SAC

def to_dec(x):
    return int("".join(str(i) for i in x), 2) 

nodes = 5
regularity = 2
maxcut_graph = nx.random_regular_graph(n=nodes, d=regularity)

def mixing_hamiltonian(c, qubits, par):
    for i in range(len(qubits)):
        c += cirq.rx(2 * par).on(qubits[i])
    return c

def cost_hamiltonian(c, qubits, g, ps):
    for edge in g.edges():
        c += cirq.CNOT(qubits[edge[0]], qubits[edge[1]])
        c += cirq.rz(ps).on(qubits[edge[1]])
        c += cirq.CNOT(qubits[edge[0]], qubits[edge[1]])
    return c

qs = [cirq.GridQubit(0, i) for i in range(nodes)]
qaoa_circuit = cirq.Circuit()
p = 10

num_param = 2 * p 
qaoa_parameters = sympy.symbols("q0:%d"%num_param)
for i in range(p):
    qaoa_circuit = cost_hamiltonian(qaoa_circuit, qs, maxcut_graph, qaoa_parameters[2 * i])
    qaoa_circuit = mixing_hamiltonian(qaoa_circuit, qs, qaoa_parameters[2 * i + 1])

initial = cirq.Circuit()
for i in qs:
    initial.append(cirq.H(i))

c_inputs = tfq.convert_to_tensor([initial])

def cc(qubits, g):
    c = 0
    for edge in g.edges():
        c += cirq.PauliString(1/2 * cirq.Z(qubits[edge[0]]) * cirq.Z(qubits[edge[1]]))
    return c

def loss_fn(x):
    return x.numpy()

cost = cc(qs, maxcut_graph)
opt = tf.keras.optimizers.Adam(lr=0.01)
es = 150

def encoding(size, num_q, num_d, struct, weights, error):
    state = np.zeros(shape=(size, 8))
    for i in range(len(weights)):
        q = i % num_q
        #print(weights[i], self.struct[i], q, i//self.max_qubits, self.max_qubits, self.max_depth, 0 if self.ins is None else 1)
        state[i] = [-error, weights[i], struct[i], q, i//num_q, num_q, num_d, 1]
    return state.flatten()

def cnn_enc(max_q, max_d, num_q, struct, weights, error):
    state = np.zeros(shape=(max_q, max_d, 5))
    for i in range(len(weights)):
        qubit_number = i % num_q
        depth_number = i // num_q
        state[qubit_number][depth_number][struct[i]] = weights[i]
        state[:,:,3] = 1
        state[:,:,4] = error
    return state.transpose(2, 0, 1)

sac_agent = SAC.load("sac_mlp_large")
sac_cnn_agent = SAC.load("sac_cnn_20_20_150")
opter = tf.keras.optimizers.Adam(lr=0.01)

mlp_mins_train = []
cnn_mins_train = []
grad_mins_train = []
mixed_mins_train = []

rep = 3
for it in range(rep):
    print(it)
    ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    outs = tfq.layers.NoisyPQC(qaoa_circuit, cost, repetitions=1000, sample_based=True, differentiator=tfq.differentiators.ParameterShift())(ins)
    vqc = tf.keras.models.Model(inputs=ins, outputs=outs)

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    layer1 = tfq.layers.NoisyPQC(qaoa_circuit, cost, repetitions=1000, sample_based=True, differentiator=tfq.differentiators.ParameterShift())(inputs)
    sac_vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    layer1 = tfq.layers.NoisyPQC(qaoa_circuit, cost, repetitions=1000, sample_based=True, differentiator=tfq.differentiators.ParameterShift())(inputs)
    sac_cnn_vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    layer1 = tfq.layers.NoisyPQC(qaoa_circuit, cost, repetitions=1000, sample_based=True, differentiator=tfq.differentiators.ParameterShift())(inputs)
    mixed = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    layer1 = tfq.layers.NoisyPQC(qaoa_circuit, cost, repetitions=1000, sample_based=True, differentiator=tfq.differentiators.ParameterShift())(inputs)
    mixed_test = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    mixed_test.set_weights(vqc.get_weights())
    sac_vqc.set_weights(vqc.get_weights())
    sac_cnn_vqc.set_weights(vqc.get_weights())
    mixed.set_weights(vqc.get_weights())

    sac_loss = []
    sac_cnn_loss = []
    mixed_loss = []

    history = []

    for i in range(es):
        with tf.GradientTape() as tape:
            error = vqc(c_inputs)
        grads = tape.gradient(error, vqc.trainable_variables)
        opt.apply_gradients(zip(grads, vqc.trainable_variables))
        history.append(error.numpy())
        print(i, history[-1])

    for i in range(es):
        print(i, es)

        # SAC
        sac_error = loss_fn(sac_vqc(c_inputs))
        sac_enc = encoding(400, nodes, 2, ([2 for _ in range(nodes)] + [0 for _ in range(nodes)]) * p, sac_vqc.trainable_variables[0].numpy(), sac_error)
        action, _ = sac_agent.predict(sac_enc)
        sac_vqc.set_weights([action[:num_param]])

        # SAC_CNN
        sac_cnn_error = loss_fn(sac_cnn_vqc(c_inputs))
        sac_cnn_enc = cnn_enc(20, 20, nodes, ([2 for _ in range(nodes)] + [0 for _ in range(nodes)]) * p, sac_cnn_vqc.trainable_variables[0].numpy(), sac_cnn_error)
        action, _ = sac_cnn_agent.predict(sac_cnn_enc)
        sac_cnn_vqc.set_weights([action[:num_param]])

        with tf.GradientTape() as tape1:
            loss = mixed(c_inputs)
        grads = tape1.gradient(loss, mixed.trainable_variables)
        opter.apply_gradients(zip(grads, mixed.trainable_variables))

        sac_loss.append(loss_fn(sac_vqc(c_inputs)))
        sac_cnn_loss.append(loss_fn(sac_cnn_vqc(c_inputs)))

        sac_enc = encoding(400, nodes, 2, ([2 for _ in range(nodes)] + [0 for _ in range(nodes)]) * p, sac_vqc.trainable_variables[0].numpy(), loss.numpy())
        mlp_action, _ = sac_agent.predict(sac_enc)
        mixed_test.set_weights([mlp_action[:vqc.trainable_variables[0].shape[0]]])
        mlp_loss = loss_fn(mixed_test(c_inputs))
        sac_cnn_enc = cnn_enc(20, 20, nodes, ([2 for _ in range(nodes)] + [0 for _ in range(nodes)]) * p, sac_cnn_vqc.trainable_variables[0].numpy(), loss.numpy())
        cnn_action, _ = sac_cnn_agent.predict(sac_cnn_enc)
        mixed_test.set_weights([cnn_action[:vqc.trainable_variables[0].shape[0]]])
        cnn_loss = loss_fn(mixed_test(c_inputs))

        losses = [mlp_loss, cnn_loss, loss_fn(mixed(c_inputs))]
        best = losses.index(min(losses))
        if best == 0:
            mixed.set_weights([mlp_action[:vqc.trainable_variables[0].shape[0]]])
        elif best == 1:
            mixed.set_weights([cnn_action[:vqc.trainable_variables[0].shape[0]]])

        mixed_loss.append(loss_fn(mixed(c_inputs)))

    cnn_mins_train.append(min(sac_cnn_loss))
    mlp_mins_train.append(min(sac_loss))
    mixed_mins_train.append(min(mixed_loss))
    grad_mins_train.append(min(history))

print("Training")
print("SAC", np.mean(mlp_mins_train), np.std(mlp_mins_train), "SAC CNN", np.mean(cnn_mins_train), np.std(cnn_mins_train), "Gradient",\
     np.mean(grad_mins_train), np.std(grad_mins_train), "Mixed", np.mean(mixed_mins_train), np.std(mixed_mins_train))
