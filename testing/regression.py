import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import numpy as np
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import SAC

x, y = ds.load_boston(return_X_y=True)
x = MinMaxScaler().fit_transform(x)
y = (y - np.min(y)) / (np.max(y) - np.min(y))

# Quantum NN
def convert_data(data, qubits, test=False):
    cs = []
    for i in data:
        cir = cirq.Circuit()
        for j in range(len(qubits)):
            cir += cirq.rx(i[j] * np.pi).on(qubits[j])
            cir += cirq.ry(i[j] * np.pi).on(qubits[j])
        cs.append(cir)
    if test:
        return tfq.convert_to_tensor([cs])
    return tfq.convert_to_tensor(cs)

def encode(data, labels, qubits):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2, random_state=43)
    return convert_data(X_train, qubits), convert_data(X_test, qubits), y_train, y_test

def layer(circuit, qubits, params):
    for i in range(len(qubits)):
        if i + 1 < len(qubits):
            circuit += cirq.CNOT(qubits[i], qubits[i + 1])
        circuit += cirq.ry(params[i]).on(qubits[i])
    for i in range(len(qubits)):
        circuit += cirq.rz(params[i + len(qubits)]).on(qubits[i])
    circuit += cirq.CNOT(qubits[-1], qubits[0])
    return circuit

def model_circuit(qubits, depth):
    cir = cirq.Circuit()
    num_params = depth * 2 * len(qubits)
    params = sympy.symbols("q0:%d"%num_params)
    for i in range(depth):
        cir = layer(cir, qubits, params[i * 2 * len(qubits):i * 2 * len(qubits) + 2 * len(qubits)])
    return cir

qs = [cirq.GridQubit(0, i) for i in range(13)]
d = 3
num_params = d * 2 * len(qs)
X_train, X_test, y_train, y_test = encode(x, y, qs)
c = model_circuit(qs, d)

readout_operators = [cirq.Z(qs[0])]

loss_fn = tf.keras.losses.MeanSquaredError()

es = 150
bs = len(X_train)

def encoding(size, num_q, num_d, struct, weights, error):
    state = np.zeros(shape=(size, 8))
    for i in range(len(weights)):
        q = i % num_q
        #print(weights[i], self.struct[i], q, i//self.max_qubits, self.max_qubits, self.max_depth, 0 if self.ins is None else 1)
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
mlp_mins_val = []
cnn_mins_val = []
grad_mins_val = []
mixed_mins_val = []

rep = 3
for it in range(rep):
    print(it)
    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    #layer1 = tfq.layers.PQC(c, readout_operators, repetitions=1000, differentiator=tfq.differentiators.ParameterShift())(inputs)
    layer1 = tfq.layers.PQC(c, readout_operators, differentiator=tfq.differentiators.Adjoint())(inputs)
    vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)
    vqc.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=0.01))

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    #layer1 = tfq.layers.PQC(c, readout_operators, repetitions=1000, differentiator=tfq.differentiators.ParameterShift())(inputs)
    layer1 = tfq.layers.PQC(c, readout_operators, differentiator=tfq.differentiators.Adjoint())(inputs)
    sac_vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    #layer1 = tfq.layers.PQC(c, readout_operators, repetitions=1000, differentiator=tfq.differentiators.ParameterShift())(inputs)
    layer1 = tfq.layers.PQC(c, readout_operators, differentiator=tfq.differentiators.Adjoint())(inputs)
    sac_cnn_vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    #layer1 = tfq.layers.PQC(c, readout_operators, repetitions=1000, differentiator=tfq.differentiators.ParameterShift())(inputs)
    layer1 = tfq.layers.PQC(c, readout_operators, differentiator=tfq.differentiators.Adjoint())(inputs)
    mixed = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    #layer1 = tfq.layers.PQC(c, readout_operators, repetitions=1000, differentiator=tfq.differentiators.ParameterShift())(inputs)
    layer1 = tfq.layers.PQC(c, readout_operators, differentiator=tfq.differentiators.Adjoint())(inputs)
    mixed_test = tf.keras.models.Model(inputs=inputs, outputs=layer1)

    mixed_test.set_weights(vqc.get_weights())
    sac_vqc.set_weights(vqc.get_weights())
    sac_cnn_vqc.set_weights(vqc.get_weights())
    mixed.set_weights(vqc.get_weights())

    sac_loss = []
    sac_val_loss = []
    sac_cnn_loss = []
    sac_cnn_val_loss = []
    mixed_loss = []
    mixed_val_loss = []

    for i in range(es):
        print(i, es)
        indexes = np.random.choice(len(X_train), len(X_train))

        # SAC
        sac_error = loss_fn(y_train[indexes], sac_vqc(tf.gather(X_train, indexes))).numpy()
        sac_enc = encoding(400, 13, d * 2, ([1 for _ in range(13)] + [2 for _ in range(13)]) * d, sac_vqc.trainable_variables[0].numpy(), sac_error)
        action, _ = sac_agent.predict(sac_enc)
        sac_vqc.set_weights([action[:num_params]])

        # SAC_CNN
        sac_cnn_error = loss_fn(y_train[indexes], sac_cnn_vqc(tf.gather(X_train, indexes))).numpy()
        sac_cnn_enc = cnn_enc(20, 20, 13, ([1 for _ in range(13)] + [2 for _ in range(13)]) * d, sac_cnn_vqc.trainable_variables[0].numpy(), sac_cnn_error)
        action, _ = sac_cnn_agent.predict(sac_cnn_enc)
        sac_cnn_vqc.set_weights([action[:num_params]])

        with tf.GradientTape() as tape:
            loss = loss_fn(y_train[indexes], mixed(tf.gather(X_train, indexes)))
        grads = tape.gradient(loss, mixed.trainable_variables)
        opter.apply_gradients(zip(grads, mixed.trainable_variables))

        sac_loss.append(loss_fn(y_train, sac_vqc(X_train)).numpy())
        sac_val_loss.append(loss_fn(y_test, sac_vqc(X_test)).numpy())
        sac_cnn_loss.append(loss_fn(y_train, sac_cnn_vqc(X_train)).numpy())
        sac_cnn_val_loss.append(loss_fn(y_test, sac_cnn_vqc(X_test)).numpy())

        sac_enc = encoding(400, 13, d * 2, ([1 for _ in range(13)] + [2 for _ in range(13)]) * d, sac_vqc.trainable_variables[0].numpy(), loss.numpy())
        mlp_action, _ = sac_agent.predict(sac_enc)
        mixed_test.set_weights([mlp_action[:vqc.trainable_variables[0].shape[0]]])
        mlp_loss = loss_fn(y_train, mixed_test(X_train)).numpy()
        sac_cnn_enc = cnn_enc(20, 20, 13, ([1 for _ in range(13)] + [2 for _ in range(13)]) * d, sac_cnn_vqc.trainable_variables[0].numpy(), loss.numpy())
        cnn_action, _ = sac_cnn_agent.predict(sac_cnn_enc)
        mixed_test.set_weights([cnn_action[:vqc.trainable_variables[0].shape[0]]])
        cnn_loss = loss_fn(y_train, mixed_test(X_train)).numpy()

        losses = [mlp_loss, cnn_loss, loss_fn(y_train, mixed(X_train)).numpy()]
        best = losses.index(min(losses))
        if best == 0:
            mixed.set_weights([mlp_action[:vqc.trainable_variables[0].shape[0]]])
        elif best == 1:
            mixed.set_weights([cnn_action[:vqc.trainable_variables[0].shape[0]]])

        mixed_loss.append(loss_fn(y_train, mixed(X_train)).numpy())
        mixed_val_loss.append(loss_fn(y_test, mixed(X_test)).numpy())

    history = vqc.fit(X_train, y_train, epochs=es, batch_size=bs, validation_data=(X_test, y_test))
    cnn_mins_val.append(min(sac_cnn_val_loss))
    cnn_mins_train.append(min(sac_cnn_loss))
    mlp_mins_train.append(min(sac_loss))
    mlp_mins_val.append(min(sac_val_loss))
    mixed_mins_val.append(min(mixed_val_loss))
    mixed_mins_train.append(min(mixed_loss))
    grad_mins_train.append(min(history.history['loss']))
    grad_mins_val.append(min(history.history['val_loss']))

print("Training")
print("$", np.mean(mlp_mins_train), "\pm", np.std(mlp_mins_train), "$ & $", np.mean(cnn_mins_train), "\pm", np.std(cnn_mins_train), "$ & $",\
     np.mean(grad_mins_train), "\pm", np.std(grad_mins_train), "$ & $", np.mean(mixed_mins_train), "\pm", np.std(mixed_mins_train), "$")
print("Validation")
print("$", np.mean(mlp_mins_val), "\pm", np.std(mlp_mins_val), "$ & $", np.mean(cnn_mins_val), "\pm", np.std(cnn_mins_val), "$ & $",\
     np.mean(grad_mins_val), "\pm", np.std(grad_mins_val), "$ & $", np.mean(mixed_mins_val), "\pm", np.std(mixed_mins_val), "$")
