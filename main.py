from src.molecule import create_beh2_molecule
from src.vqe import initialize_vqe
from examples.run_adaptive_vqe import run_vqe_adaptive
import pennylane as qml
from pennylane import numpy as np

def main():
    initial_basis_params = np.array([1.24, 0.5, 0.9, 0.3, 1.2, 0.6], requires_grad=True)
    symbols, coords, charge = create_beh2_molecule()
    molecule, hamiltonian, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(initial_basis_params, symbols, coords, charge)

    print("\n<Info of VQE with custom basis>")
    print("Number of qubits needed:", n_qubits)
    print('Number of Pauli strings:', len(hamiltonian.ops))

    params_vanilla = np.zeros(len(doubles) + len(singles), requires_grad=True)
    adam_opt = qml.AdamOptimizer(stepsize=0.02, beta1=0.9, beta2=0.99, eps=1e-08)

    initial_threshold = -30.0
    min_iterations_per_basis = 20
    energies_vanilla, runtime_vanilla, final_basis_params = run_vqe_adaptive(params_vanilla, initial_basis_params, adam_opt, 100, initial_threshold, min_iterations_per_basis)

    print(f"Final optimized basis set parameters: {final_basis_params}")

if __name__ == "__main__":
    main()