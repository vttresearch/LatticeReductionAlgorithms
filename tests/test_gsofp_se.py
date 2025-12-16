import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from bkz.L3FP.gsofp_se import gso_step
from bkz.basis_generator import basis_gen
from tests.test_utils import verify_gso_structure

LATTICE_DIMENSION = 10
ENTRY_BOUND = 173
TEST_CASES = 10

#RUN root: pytest tests/test_gsofp_se.py
# Allow prints: pytest -s tests/test_gsofp_se.py

def test_case_gso_step(dim=LATTICE_DIMENSION, entry_bound=ENTRY_BOUND, test_cases=TEST_CASES):
	for _ in range(test_cases):
		basis = basis_gen(dim, entry_bound)
		width = len(basis[0])
		gs_squared_norms = np.zeros(width).astype(np.float64)  # Column-wise squared norms
		gscs = np.zeros((width, width)).astype(np.float64)
		for stage in range(len(basis[0])):
			gs_squared_norms[: stage + 1], gscs[:, : stage + 1] = gso_step(
				basis[:, : stage + 1], gscs, gs_squared_norms, stage
			)
		assert verify_gso_structure(basis, gscs, gs_squared_norms), "GSO structure is malformed."
