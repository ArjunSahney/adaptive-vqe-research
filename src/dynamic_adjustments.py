import os
import psutil
import pennylane as qml
from pennylane import numpy as np

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Return memory usage in MB

def adjust_threshold(energies, current_threshold):
    if len(energies) < 2:
        return current_threshold
    improvement = energies[-2] - energies[-1]
    if improvement < 0.0001:  # Stricter improvement condition
        return current_threshold * 0.95  # Reduce threshold by 5%
    return current_threshold

def optimize_basis_params(basis_params, grad, learning_rate=0.01):
    return basis_params - learning_rate * grad[:len(basis_params)]

def compute_gradient(params, basis_params):
    import pennylane as qml
    from vqe_initialization import initialize_vqe
    @qml.qnode(dev, diff_method='backprop')
    def energy_func(basis_params):
        BeH2, h_vanilla, n_qubits = initialize_vqe(basis_params)
        qml.templates.AllSinglesDoubles(params, wires=range(n_qubits), hf_state=hf_state, singles=singles, doubles=doubles)
        return qml.expval(h_vanilla)
    
    grad_fn = qml.grad(energy_func)
    return grad_fn(basis_params)
