import unittest
from src.molecule import create_beh2_molecule
from src.vqe import create_molecule_and_hamiltonian, initialize_vqe

class TestVQE(unittest.TestCase):
    def test_beh2_molecule_creation(self):
        symbols, coords, charge = create_beh2_molecule()
        self.assertEqual(len(symbols), 3)
        self.assertEqual(charge, 0)
        self.assertEqual(coords.shape, (3, 3))

    def test_vqe_initialization(self):
        symbols, coords, charge = create_beh2_molecule()
        basis_params = np.array([1.24, 0.5, 0.9, 0.3, 1.2, 0.6])
        molecule, hamiltonian, n_qubits, hf_state, singles, doubles, dev, energy_expval_ASD = initialize_vqe(basis_params, symbols, coords, charge)
        self.assertIsNotNone(molecule)
        self.assertIsNotNone(hamiltonian)
        self.assertGreater(n_qubits, 0)