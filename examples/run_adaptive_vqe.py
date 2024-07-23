import pennylane as qml
from pennylane import numpy as np
import time
import os
import psutil
from src.molecule import create_beh2_molecule
from src.vqe import initialize_vqe
from src.basis import optimize_basis_params

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Return memory usage in MB

def adjust_threshold(energies, current_threshold):
    if len(energies) < 2:
        return current_threshold
    improvement = energies[-2] - energies[-1]
    if improvement < 0.0001:
        return current_threshold * 0.95
    return current_threshold

def run_vqe_adaptive(params, basis_params, opt, iterations, initial_threshold, min_iterations_per_basis=10, learning_rate=0.01):
    symbols, coords, charge = create_beh2_molecule()
    molecule, hamiltonian, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(basis_params, symbols, coords, charge)
    
    ti = time.time()
    energies = []
    runtime = []
    energy_threshold = initial_threshold

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
        
        if (i + 1) % min_iterations_per_basis == 0:
            basis_params = optimize_basis_params(params, basis_params, energy_expval_ASD, learning_rate)
            molecule, hamiltonian, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(basis_params, symbols, coords, charge)
            print(f"Updated basis set parameters: {basis_params}")
        
        if len(energies) > 2 and abs(energies[-1] - energies[-2]) < 1e-6:
            print(f"Converged after {i + 1} iterations.")
            break

    print(f"Optimized energy: {lowest_energy} Ha")
    print(f"Best VQE parameters: {best_params}")
    print(f"Best basis set parameters: {best_basis_params}")
    return energies, runtime, best_basis_params