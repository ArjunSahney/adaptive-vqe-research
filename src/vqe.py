import pennylane as qml
from pennylane import numpy as np

def create_molecule_and_hamiltonian(symbols, coordinates, charge, basis_params):
    custom_basis = {
        "sto-3g": {
            "H": [basis_params[:2]],
            "Be": [basis_params[2:]]
        }
    }
    
    molecule = qml.qchem.Molecule(symbols, coordinates, charge=charge, basis_name="sto-3g")
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        molecule.symbols, molecule.coordinates, charge=molecule.charge, mult=1, basis="sto-3g", mapping="jordan_wigner"
    )
    
    return molecule, hamiltonian, qubits

def initialize_vqe(basis_params, symbols, coordinates, charge):
    molecule, hamiltonian, n_qubits = create_molecule_and_hamiltonian(symbols, coordinates, charge, basis_params)
    n_electrons = molecule.n_electrons
    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, diff_method='backprop')
    def energy_expval_ASD(params):
        qml.templates.AllSinglesDoubles(params, wires=range(n_qubits), hf_state=hf_state, singles=singles, doubles=doubles)
        return qml.expval(hamiltonian)
    
    return molecule, hamiltonian, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD
