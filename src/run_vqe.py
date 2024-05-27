import time
import numpy as np
import pennylane as qml
from vqe_initialization import initialize_vqe, print_vqe_info
from dynamic_adjustments import get_memory_usage, adjust_threshold, optimize_basis_params, compute_gradient

def run_vqe_adaptive(params, basis_params, opt, iterations, initial_threshold, min_iterations_per_basis=10, learning_rate=0.01):
    ti = time.time()
    energies = []
    runtime = []
    energy_threshold = initial_threshold

    global energy_expval_ASD

    lowest_energy = float('inf')
    best_params = None
    best_basis_params = None

    for i in range(iterations):
        t1 = time.time()
        params, energy = opt.step_and_cost(energy_expval_ASD, params)
        t2 = time.time()
        runtime.append(t2 - ti)
        energies.append(energy)
        print(f"Iteration {i + 1}, Energy: {energy} Ha, Memory Usage: {get_memory_usage()} MB")
        
        if energy < lowest_energy:
            lowest_energy = energy
            best_params = params.copy()
            best_basis_params = basis_params.copy()
        
        energy_threshold = adjust_threshold(energies, energy_threshold)
        
        grad = compute_gradient(params, basis_params)
        basis_params = optimize_basis_params(basis_params, grad, learning_rate)
        
        if (i + 1) % min_iterations_per_basis == 0 or energy < energy_threshold:
            print(f"Adjusting basis set parameters due to insufficient improvement or reaching the threshold.")
            BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(basis_params)
            params = np.zeros(len(singles) + len(doubles), requires_grad=True)
        
        if len(energies) > 2 and abs(energies[-1] - energies[-2]) < 1e-6:
            print(f"Converged after {i + 1} iterations.")
            break

    print(f"Optimized energy: {lowest_energy} Ha")
    print(f"Best VQE parameters: {best_params}")
    print(f"Best basis set parameters: {best_basis_params}")
    return energies, runtime, best_basis_params

# Initial basis set parameters (example)
initial_basis_params = np.array([1.24, 0.5, 0.9, 0.3, 1.2, 0.6], requires_grad=True)

# Initialize with custom basis set parameters
BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(initial_basis_params)

# Print out basic information about the VQE setup
print_vqe_info(BeH2, h_vanilla, n_qubits)

# Initialize parameters for the VQE circuit
params_vanilla = np.zeros(len(doubles) + len(singles), requires_grad=True)

# Setup the optimizer (Adam) with specified hyperparameters
adam_opt = qml.AdamOptimizer(stepsize=0.02, beta1=0.9, beta2=0.99, eps=1e-08)

# Execute the VQE algorithm and capture energies and runtimes
initial_threshold = -30.0  # Higher initial energy threshold
min_iterations_per_basis = 20  # Minimum iterations to run before switching basis sets
energies_vanilla, runtime_vanilla, final_basis_params = run_vqe_adaptive(params_vanilla, initial_basis_params, adam_opt, 100, initial_threshold, min_iterations_per_basis)
