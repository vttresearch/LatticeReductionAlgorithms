import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bkz.basis_generator import basis_gen
from bkz.L3FP.L3fp import l3fp
from tests.test_utils import *

LATTICE_DIMENSION = 10
ENTRY_BOUND = 173
TEST_CASES = 10

#RUN root: pytest tests/test_L3fp.py
# Allow prints: pytest -s tests/test_L3fp.py

def test_case_square_many(dim=LATTICE_DIMENSION, entry_bound=ENTRY_BOUND, test_cases=TEST_CASES):
    for _ in range(test_cases):
        basis = basis_gen(dim, entry_bound)
        lll_basis, gsc, gs_squared_norms = l3fp(basis.copy())
        upper_bound = (4/3) ** ((dim-1)/4) #Reference?!
        assert verify_lattice_invariance(basis, lll_basis), "Determinant mismatch."
        assert verify_hermite_factor(lll_basis, dim, upper_bound), "Hermite factor out of bounds."
        assert verify_gso_structure(lll_basis, gsc, gs_squared_norms), "GSO structure is malformed."
        assert is_size_reduced(gsc), "Condition mu is not satisfied."
        assert verify_Lovasz_condition(gs_squared_norms, gsc), "Condition delta is not satisfied."
        