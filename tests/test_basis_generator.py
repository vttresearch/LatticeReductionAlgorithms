import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bkz.basis_generator import basis_gen
from bkz.bkz_params import *

#RUN root: pytest tests/test_basis_generator.py
# Allow prints: pytest -s tests/test_basis_generator.py

def test_basis_gen():
    basis = basis_gen(LATTICE_DIMENSION, ENTRY_BOUND)
    rank = np.linalg.matrix_rank(basis)

    assert basis.shape == (LATTICE_DIMENSION, LATTICE_DIMENSION), "Generated basis has malformed shape."
    assert rank == LATTICE_DIMENSION, "Generated basis is not full-rank."
