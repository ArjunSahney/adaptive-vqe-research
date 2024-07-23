import math
import pennylane as qml
from pennylane import numpy as np

def create_beh2_molecule():
    angHBeH = math.pi
    lenBeH = 1.3264
    angToBr = 1 / 0.529177210903
    lenInBr = lenBeH * angToBr
    cx = lenInBr * math.sin(0.5 * angHBeH)
    cy = lenInBr * math.cos(0.5 * angHBeH)
    
    BeHHsymbols = ["Be", "H", "H"]
    BeHHcoords = np.array([[0., cy, 0.], [-cx, 0., 0.], [cx, 0., 0.]])
    BeHHcharge = 0
    
    return BeHHsymbols, BeHHcoords, BeHHcharge