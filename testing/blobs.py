import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import numpy as np
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from stable_baselines3 import SAC


blob_data, blob_labels = ds.make_blobs(1000, centers=7, shuffle=True)
blob_data = MinMaxScaler().fit_transform(blob_data)

#plt.scatter(blob_data[blob_labels == 0][:,0], blob_data[blob_labels == 0][:,1], label='0', color='blue')
#plt.scatter(blob_data[blob_labels == 1][:,0], blob_data[blob_labels == 1][:,1], label='1', color='red')
#plt.scatter(blob_data[blob_labels == 2][:,0], blob_data[blob_labels == 2][:,1], label='2', color='green')
#plt.scatter(blob_data[blob_labels == 3][:,0], blob_data[blob_labels == 3][:,1], label='3', color='black')
#plt.scatter(blob_data[blob_labels == 4][:,0], blob_data[blob_labels == 4][:,1], label='4', color='yellow')
#plt.scatter(blob_data[blob_labels == 5][:,0], blob_data[blob_labels == 5][:,1], label='5', color='m')
#plt.scatter(blob_data[blob_labels == 6][:,0], blob_data[blob_labels == 6][:,1], label='6', color='c')
#plt.show()

# Quantum NN
def convert_data(data, qubits, test=False):
    cs = []
    for i in data:
        cir = cirq.Circuit()
        for j in qubits:
            cir += cirq.rx(i[0] * np.pi).on(j)
            cir += cirq.ry(i[1] * np.pi).on(j)
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
        circuit += cirq.rz(params[i * 2]).on(qubits[i])
    for i in range(len(qubits)):
        circuit += cirq.rx(params[i * 2 + 1]).on(qubits[i])
    return circuit

def model_circuit(qubits, depth):
    cir = cirq.Circuit()
    num_params = depth * 2 * len(qubits)
    params = sympy.symbols("q0:%d"%num_params)
    for i in range(depth):
        cir = layer(cir, qubits, params[i * 2 * len(qubits):i * 2 * len(qubits) + 2 * len(qubits)])
    return cir

q = 7
qs = [cirq.GridQubit(0, i) for i in range(q)]
d = 5
X_train, X_test, y_train, y_test = encode(blob_data, blob_labels, qs)
c = model_circuit(qs, d)

readout_operators = [cirq.Z(i) for i in qs]

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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
    vqc.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=['acc'])

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
        sac_enc = encoding(400, 7, d * 2, ([0 for _ in range(q)] + [2 for _ in range(q)]) * d, sac_vqc.trainable_variables[0].numpy(), sac_error)
        action, _ = sac_agent.predict(sac_enc)
        sac_vqc.set_weights([action[:vqc.trainable_variables[0].shape[0]]])

        # SAC_CNN
        sac_cnn_error = loss_fn(y_train[indexes], sac_cnn_vqc(tf.gather(X_train, indexes))).numpy()
        sac_cnn_enc = cnn_enc(20, 20, 5, ([0 for _ in range(q)] + [2 for _ in range(q)]) * d, sac_cnn_vqc.trainable_variables[0].numpy(), sac_cnn_error)
        action, _ = sac_cnn_agent.predict(sac_cnn_enc)
        sac_cnn_vqc.set_weights([action[:vqc.trainable_variables[0].shape[0]]])

        with tf.GradientTape() as tape:
            loss = loss_fn(y_train[indexes], mixed(tf.gather(X_train, indexes)))
        grads = tape.gradient(loss, mixed.trainable_variables)
        opter.apply_gradients(zip(grads, mixed.trainable_variables))

        sac_loss.append(loss_fn(y_train, sac_vqc(X_train)).numpy())
        sac_val_loss.append(loss_fn(y_test, sac_vqc(X_test)).numpy())
        sac_cnn_loss.append(loss_fn(y_train, sac_cnn_vqc(X_train)).numpy())
        sac_cnn_val_loss.append(loss_fn(y_test, sac_cnn_vqc(X_test)).numpy())

        sac_enc = encoding(400, 5, d * 2, ([0 for _ in range(q)] + [2 for _ in range(q)]) * d, sac_vqc.trainable_variables[0].numpy(), loss.numpy())
        mlp_action, _ = sac_agent.predict(sac_enc)
        mixed_test.set_weights([mlp_action[:vqc.trainable_variables[0].shape[0]]])
        mlp_loss = loss_fn(y_train, mixed_test(X_train)).numpy()
        sac_cnn_enc = cnn_enc(20, 20, 5, ([0 for _ in range(q)] + [2 for _ in range(q)]) * d, sac_cnn_vqc.trainable_variables[0].numpy(), loss.numpy())
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
