import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import random
from functools import reduce 
import operator
from stable_baselines3 import SAC

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def layer(circuit, qubits, parameters):
    for i in range(len(qubits)):
        circuit.append([cirq.rx(parameters[i]).on(qubits[i])])
    for i in range(len(qubits)):
        circuit.append([cirq.rz(parameters[len(qubits) + i]).on(qubits[i])])
    for i in range(len(qubits)-1):
        circuit.append([cirq.CNOT(qubits[i], qubits[i+1])])
    return circuit

def ansatz(circuit, qubits, layers, parameters):
    for i in range(layers):
        p = parameters[2 * i * len(qubits):2 * (i + 1) * len(qubits)]
        circuit = layer(circuit, qubits, p)
    return circuit

def hamiltonian(circuit, qubits, ham):
    for i in range(len(qubits)):
        if ham[i] == "x":
            circuit.append(cirq.ry(-np.pi/2).on(qubits[i]))
        elif ham[i] == "y":
            circuit.append(cirq.rx(np.pi/2).on(qubits[i]))
    return circuit

def create_vqe(qubits, layers, parameters, ham):
    circuit = ansatz(cirq.Circuit(), qubits, layers, parameters)
    circuit += hamiltonian(circuit, qubits, ham)
    return circuit

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def expcost(qubits, ham):
    return prod([cirq.Z(qubits[i]) for i in range(len(qubits)) if ham[i] != "i"])

possibilities = ["i", "x", "y", "z"]
l = 10
q = 20

hamilton = [[random.choice(possibilities) for _ in range(q)] for _ in range(l)]
h_weights = [random.uniform(-1, 1) for _ in range(l)]

lay = 5

qubits = [cirq.GridQubit(0, i) for i in range(q)]
num_param = lay * 2 * q
params = sympy.symbols('vqe0:%d'%num_param)

class VQE(tf.keras.layers.Layer):
    def __init__(self, num_weights, circuits, ops) -> None:
        super(VQE, self).__init__()
        self.w = tf.Variable(np.random.uniform(0, np.pi, (1, num_weights)), dtype=tf.float32)
        #self.layers = [tfq.layers.ControlledPQC(circuits[i], ops[i], repetitions=1000, differentiator=tfq.differentiators.ParameterShift()) for i in range(len(circuits))]
        self.layers = [tfq.layers.ControlledPQC(circuits[i], ops[i], differentiator=tfq.differentiators.Adjoint()) for i in range(len(circuits))]


    def call(self, input):
        return sum([self.layers[i]([input, self.w]) for i in range(len(self.layers))])

c_inputs = tfq.convert_to_tensor([cirq.Circuit()])
vqe_components = []
cs = []
op = []

for i in range(len(hamilton)):
    readout_ops = h_weights[i] * expcost(qubits, hamilton[i])
    op.append(readout_ops)
    cs.append(create_vqe(qubits, lay, params, hamilton[i]))

opt = tf.keras.optimizers.Adam(lr=0.01)

def loss_fn(x):
    return x.numpy()

es = 150

def encoding(size, num_q, num_d, struct, weights, error):
    state = np.zeros(shape=(size, 8))
    for i in range(len(weights)):
        q = i % num_q
        state[i] = [-error, weights[i], struct[i], q, i//num_q, num_q, num_d, 0]
    return state.flatten()

def cnn_enc(max_q, max_d, num_q, struct, weights, error):
    state = np.zeros(shape=(max_q, max_d, 5))
    for i in range(len(weights)):
        qubit_number = i % num_q
        depth_number = i // num_q
        state[qubit_number][depth_number][struct[i]] = weights[i]
        state[:,:,3] = 0
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
    v = VQE(num_param, cs, op)(ins)
    vqc = tf.keras.models.Model(inputs=ins, outputs=v)

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    layer1 = VQE(num_param, cs, op)(inputs)
    sac_vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    layer1 = VQE(num_param, cs, op)(inputs)
    sac_cnn_vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    layer1 = VQE(num_param, cs, op)(inputs)
    mixed = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    layer1 = VQE(num_param, cs, op)(inputs)
    mixed_test = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    sac_vqc.set_weights(vqc.get_weights())
    sac_cnn_vqc.set_weights(vqc.get_weights())
    mixed.set_weights(vqc.get_weights())
    
    history = []
    sac_loss = []
    sac_cnn_loss = []
    mixed_loss = []

    for i in range(es):
        print(i, es)

        # SAC
        sac_error = loss_fn(sac_vqc(c_inputs))
        sac_enc = encoding(400, q, lay * 2, ([0 for _ in range(q)] + [2 for _ in range(q)]) * lay, sac_vqc.trainable_variables[0].numpy()[0], sac_error)
        action, _ = sac_agent.predict(sac_enc)
        sac_vqc.set_weights([np.array([action[:num_param],])])

        # SAC_CNN
        sac_cnn_error = loss_fn(sac_cnn_vqc(c_inputs))
        sac_cnn_enc = cnn_enc(20, 20, q, ([0 for _ in range(q)] + [2 for _ in range(q)]) * lay, sac_cnn_vqc.trainable_variables[0].numpy()[0], sac_cnn_error)
        action, _ = sac_cnn_agent.predict(sac_cnn_enc)
        sac_cnn_vqc.set_weights([np.array([action[:num_param],])])


        with tf.GradientTape() as tape:
            loss = mixed(c_inputs)
        grads = tape.gradient(loss, mixed.trainable_variables)
        opter.apply_gradients(zip(grads, mixed.trainable_variables))

        sac_loss.append(loss_fn(sac_vqc(c_inputs)))
        sac_cnn_loss.append(loss_fn(sac_cnn_vqc(c_inputs)))

        sac_enc = encoding(400, q, lay * 2, ([0 for _ in range(q)] + [2 for _ in range(q)]) * lay, sac_vqc.trainable_variables[0].numpy()[0], loss.numpy())
        mlp_action, _ = sac_agent.predict(sac_enc)
        mixed_test.set_weights([np.array([mlp_action[:num_param],])])
        mlp_loss = loss_fn(mixed_test(c_inputs))
        sac_cnn_enc = cnn_enc(20, 20, q, ([0 for _ in range(q)] + [2 for _ in range(q)]) * lay, sac_cnn_vqc.trainable_variables[0].numpy()[0], loss.numpy())
        cnn_action, _ = sac_cnn_agent.predict(sac_cnn_enc)
        mixed_test.set_weights([np.array([cnn_action[:num_param],])])
        cnn_loss = loss_fn(mixed_test(c_inputs))

        losses = [mlp_loss, cnn_loss, loss_fn(mixed(c_inputs))]
        best = losses.index(min(losses))
        if best == 0:
            mixed.set_weights([np.array([mlp_action[:num_param],])])
        elif best == 1:
            mixed.set_weights([np.array([cnn_action[:num_param],])])
        mixed_loss.append(loss_fn(mixed(c_inputs)))
    
    for i in range(es):
        with tf.GradientTape() as tape:
            error = vqc(c_inputs)
        grads = tape.gradient(error, vqc.trainable_variables)
        opt.apply_gradients(zip(grads, vqc.trainable_variables))
        history.append(error.numpy())
        print(i, history[-1])

    cnn_mins_train.append(min(sac_cnn_loss))
    mlp_mins_train.append(min(sac_loss))
    mixed_mins_train.append(min(mixed_loss))
    grad_mins_train.append(min(history))

print("Training")
print("$", np.mean(mlp_mins_train), "\pm", np.std(mlp_mins_train), "$ & $", np.mean(cnn_mins_train), "\pm", np.std(cnn_mins_train), "$ & $",\
     np.mean(grad_mins_train), "\pm", np.std(grad_mins_train), "$ & $", np.mean(mixed_mins_train), "\pm", np.std(mixed_mins_train), "$")
