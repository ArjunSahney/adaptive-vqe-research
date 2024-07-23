import pennylane as qml
from pennylane import numpy as np

def compute_gradient(params, basis_params, energy_expval_ASD, epsilon=1e-5):
    base_energy = energy_expval_ASD(params)
    gradients = []
    
    for i in range(len(basis_params)):
        basis_params_plus = basis_params.copy()
        basis_params_plus[i] += epsilon
        basis_params_minus = basis_params.copy()
        basis_params_minus[i] -= epsilon
        
        energy_plus = energy_expval_ASD(params)
        energy_minus = energy_expval_ASD(params)
        
        grad = (energy_plus - energy_minus) / (2 * epsilon)
        gradients.append(grad)
    
    return np.array(gradients)

def optimize_basis_params(params, basis_params, energy_expval_ASD, learning_rate=0.01):
    grad = compute_gradient(params, basis_params, energy_expval_ASD)
    return basis_params - learning_rate * grad