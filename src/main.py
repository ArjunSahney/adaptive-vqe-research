import numpy as np
import pennylane as qml
from vqe_initialization import initialize_vqe, print_vqe_info
from dynamic_adjustments import get_memory_usage, adjust_threshold, optimize_basis_params, compute_gradient
from run_vqe import run_vqe_adaptive

def main():
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
    energies_vanilla, runtime_vanilla, final_basis_params = run_vqe_adaptive(
        params_vanilla, initial_basis_params, adam_opt, 100, initial_threshold, min_iterations_per_basis)
    
    # Output final results
    print("Final energies:", energies_vanilla)
    print("Runtime per iteration:", runtime_vanilla)
    print("Optimized basis set parameters:", final_basis_params)

if __name__ == "__main__":
    main()
