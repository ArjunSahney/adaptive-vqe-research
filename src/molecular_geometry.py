import math
import numpy as np

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

def create_molecule_and_hamiltonian(basis_params):
    import pennylane as qml
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
