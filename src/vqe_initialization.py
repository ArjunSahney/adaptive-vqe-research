import pennylane as qml

def initialize_vqe(basis_params):
    from molecular_geometry import create_molecule_and_hamiltonian
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

def print_vqe_info(BeH2, h_vanilla, n_qubits):
    print("\n<Info of VQE with custom basis>")
    print("Number of qubits needed:", n_qubits)
    print('Number of Pauli strings:', len(h_vanilla.ops))
