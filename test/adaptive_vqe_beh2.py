import pennylane as qml
from pennylane import numpy as np
import time
import math
import os
import psutil

# Define molecular geometry constants for Beryllium Hydride (BeH2)
angHBeH = math.pi  # Bond angle in radians (180 degrees for linear BeH2)
lenBeH = 1.3264  # Bond length in Angstroms
angToBr = 1 / 0.529177210903  # Conversion factor from Angstroms to Bohr radii

# Convert bond length from Angstroms to Bohr
lenInBr = lenBeH * angToBr
cx = lenInBr * math.sin(0.5 * angHBeH)  # x-coordinate for hydrogen atoms
cy = lenInBr * math.cos(0.5 * angHBeH)  # y-coordinate for Be atom

# Calculate the distance between the two hydrogen atoms
lenHH = 2 * cx

# Define the symbols and coordinates for the nuclei in BeH2
BeHHsymbols = ["Be", "H", "H"]
BeHHcoords = np.array([[0., cy, 0.], [-cx, 0., 0.], [cx, 0., 0.]])

# Define the net charge of the molecule
BeHHcharge = 0

# Function to create Molecule object and Hamiltonian in PennyLane with custom basis set parameters
def create_molecule_and_hamiltonian(basis_params):
    # Define custom basis set parameters
    custom_basis = {
        "sto-3g": {
            "H": [basis_params[:2]],
            "Be": [basis_params[2:]]
        }
    }
    
    # Create the Molecule object
    BeH2 = qml.qchem.Molecule(BeHHsymbols, BeHHcoords, charge=BeHHcharge, basis_name="sto-3g")
    
    # Create the Hamiltonian using the custom basis set
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        BeH2.symbols, BeH2.coordinates, charge=BeH2.charge, mult=1, basis="sto-3g", mapping="jordan_wigner"
    )
    
    return BeH2, hamiltonian, qubits

# Function to initialize VQE parameters and states with custom basis set parameters
def initialize_vqe(basis_params):
    BeH2, h_vanilla, n_qubits = create_molecule_and_hamiltonian(basis_params)
    n_electrons = BeH2.n_electrons
    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, diff_method='backprop')
    def energy_expval_ASD(params):
        qml.templates.AllSinglesDoubles(params, wires=range(n_qubits), hf_state=hf_state, singles=singles, doubles=doubles)
        return qml.expval(h_vanilla)
    
    return BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD

# Print out basic information about the VQE setup
def print_vqe_info(BeH2, h_vanilla, n_qubits):
    print("\n<Info of VQE with custom basis>")
    print("Number of qubits needed:", n_qubits)
    print('Number of Pauli strings:', len(h_vanilla.ops))

# Function to monitor memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Return memory usage in MB

# Function to dynamically adjust the energy threshold
def adjust_threshold(energies, current_threshold):
    if len(energies) < 2:
        return current_threshold
    improvement = energies[-2] - energies[-1]
    if improvement < 0.0001:  # Stricter improvement condition
        return current_threshold * 0.95  # Reduce threshold by 5%
    return current_threshold

# Gradient-based optimizer for basis set parameters
def optimize_basis_params(basis_params, grad, learning_rate=0.01):
    return basis_params - learning_rate * grad[:len(basis_params)]

# Function to compute the gradient of the energy with respect to basis set parameters
def compute_gradient(params, basis_params):
    @qml.qnode(dev, diff_method='backprop')
    def energy_func(basis_params):
        BeH2, h_vanilla, n_qubits = create_molecule_and_hamiltonian(basis_params)
        qml.templates.AllSinglesDoubles(params, wires=range(n_qubits), hf_state=hf_state, singles=singles, doubles=doubles)
        return qml.expval(h_vanilla)
    
    grad_fn = qml.grad(energy_func)
    return grad_fn(basis_params)

# Function to run the VQE algorithm with adaptive basis sets and dynamic threshold
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
        
        # Check if this is the lowest energy found
        if energy < lowest_energy:
            lowest_energy = energy
            best_params = params.copy()
            best_basis_params = basis_params.copy()
        
        # Adjust the threshold dynamically
        energy_threshold = adjust_threshold(energies, energy_threshold)
        
        # Compute gradient and update basis set parameters
        grad = compute_gradient(params, basis_params)
        basis_params = optimize_basis_params(basis_params, grad, learning_rate)
        
        if (i + 1) % min_iterations_per_basis == 0 or energy < energy_threshold:
            print(f"Adjusting basis set parameters due to insufficient improvement or reaching the threshold.")
            BeH2, h_vanilla, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(basis_params)
            params = np.zeros(len(singles) + len(doubles), requires_grad=True)
        
        # Convergence check
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


