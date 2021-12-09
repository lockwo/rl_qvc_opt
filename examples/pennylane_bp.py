import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from augment import mixed

# Code from: https://pennylane.ai/qml/demos/tutorial_local_cost_functions.html

wires = 6
dev = qml.device("default.qubit", wires=wires, shots=10000)


def global_cost_simple(rotations):
    for i in range(wires):
        qml.RX(rotations[i], wires=i)
    for i in range(wires):
        qml.RY(rotations[wires + i], wires=i)
    return qml.probs(wires=range(wires))


global_circuit = qml.QNode(global_cost_simple, dev)

def cost_global(rotations):
    return 1 - global_circuit(rotations)[0]


rotations = np.array([[3.] * len(range(wires)), [0.] * len(range(wires))], requires_grad=True).flatten()
#rotations = np.random.uniform(-1, 1, 12)
params_aug = np.array([[3.] * len(range(wires)), [0.] * len(range(wires))], requires_grad=True).flatten()
params_aug1 = params_aug.copy()
opt = qml.GradientDescentOptimizer(stepsize=0.2)
steps = 50
params_global = rotations

gds = []
augs = []
# No augmentation
for i in range(steps):
    # update the circuit parameters
    current_loss = cost_global(params_global)
    param_old = params_aug
    current_aug = cost_global(param_old)
    params_global = opt.step(cost_global, params_global)
    params_aug1 = mixed(cost_global, [0] * 6 + [1] * 6, 6, params_aug1, current_loss, 0, True)
    temp_gd = np.array([i for i in params_aug1], requires_grad=True)
    temp_gd1 = np.array([i for i in param_old], requires_grad=True)
    temp_gd = opt.step(cost_global, temp_gd)
    temp_gd1 = opt.step(cost_global, temp_gd1)
    aug_l = cost_global(params_aug1)
    gd_l = cost_global(temp_gd)
    gd_l1 = cost_global(temp_gd1)
    x = min([aug_l, gd_l, gd_l1])
    y = np.argmin([aug_l, gd_l, gd_l1])
    if x < current_aug:
        if y == 0:
            params_aug = params_aug1
        elif y == 1:
            params_aug = temp_gd
        elif y == 2:
            params_aug = temp_gd1
    else:
        params_aug = param_old        

        
    gd_l = cost_global(params_global)
    aug_l = cost_global(params_aug)
    gds.append(gd_l)
    augs.append(aug_l)
    if (i + 1) % 5 == 0:
        print("GD Cost after step {:5d}: {: .7f}".format(i + 1, gd_l))
        print("Aug Cost after step {:5d}: {: .7f}".format(i + 1, aug_l))
    #if cost_global(params_global) < 0.1:
    #    break

print(qml.draw(global_circuit)(params_global))
print(qml.draw(global_circuit)(params_aug))


plt.plot(gds, label='Gradient Descent')
plt.plot(augs, label='Augmented (Rolling Average)')
plt.legend()
plt.ylabel('Loss')
plt.show()
