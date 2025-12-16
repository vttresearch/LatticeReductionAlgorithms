import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bkz.basis_generator import basis_gen
from bkz import BKZ_ALGORITHMS
from tests.test_utils import *

LATTICE_DIMENSION = 10
ENTRY_BOUND = 173
BLOCK_SIZE = LATTICE_DIMENSION // 2
BKZ_VERSION = "1"
ENUM_VERSION = "1"
TEST_CASES = 10

#RUN root: pytest tests/test_bkz.py
# Allow prints: pytest -s tests/test_bkz.py

def test_case_basic(dim=LATTICE_DIMENSION, entry_bound=ENTRY_BOUND, block_size=BLOCK_SIZE, test_cases=TEST_CASES):
	bkz_reduce = BKZ_ALGORITHMS[BKZ_VERSION]
	warning_amount = 0
	for _ in range(test_cases):
		basis = basis_gen(dim, entry_bound)
		bkz_reduced_basis, gsc, gs_squared_norms =bkz_reduce(basis.copy(), block_size, ENUM_VERSION)
		rhf_upper_bound = compute_estimated_root_hermite_factor(block_size)
		rhf_tolerance = get_rhf_tolerance(block_size)
		root_hermite_factor = get_root_hermite_factor(bkz_reduced_basis)

		if root_hermite_factor > rhf_upper_bound * (1 + rhf_tolerance):
			print(f"[WARNING] RHF {root_hermite_factor:.4f} exceeds bound {rhf_upper_bound + rhf_tolerance:.4f}.")
			warning_amount += 1
		else:
			print(f"[INFO] RHF {root_hermite_factor:.4f} within given bound {rhf_upper_bound + rhf_tolerance:.4f}.")

		assert verify_lattice_invariance(basis, bkz_reduced_basis), "Determinant mismatch."
		assert is_size_reduced(gsc), "Condition mu is not satisfied."
		assert verify_Lovasz_condition(gs_squared_norms, gsc), "Condition delta is not satisfied."

	print(f"Test passed with {warning_amount} warnings!")
