import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bkz.basis_generator import basis_gen
from bkz.L3FP.L3fp_deep_insertion import l3fp_deep_insert
from bkz.L3FP.L3fp import l3fp
from tests.test_utils import *

LATTICE_DIMENSION = 10
ENTRY_BOUND = 173
TEST_CASES = 10

#RUN root: pytest tests/test_L3fp_di_inclusive.py
# Allow prints: pytest -s tests/test_L3fp_di_inclusive.py

def test_case_square_many(dim=LATTICE_DIMENSION, entry_bound=ENTRY_BOUND, test_cases=TEST_CASES):
    for _ in range(test_cases):
        basis = basis_gen(dim, entry_bound)
        lll_basis, gs_coeffs, gs_squared_norms = l3fp(basis.copy())
        idx1, idx2 = np.random.choice(basis.shape[1], 2, replace=False)
        product_vector = basis[:, idx1] + basis[:, idx2]
        for insert_pos in range(0, dim + 1):
            injected_basis = np.insert(lll_basis.copy(), insert_pos, product_vector, axis=1)
            lll_basis_final, gs_coeffs_final, gs_squared_norms_final = l3fp_deep_insert(
                injected_basis_matrix=injected_basis.copy(),
                gs_coeff_matrix=gs_coeffs[:insert_pos, :insert_pos].copy(),
                gs_squared_norms=gs_squared_norms[:insert_pos].copy(),
                start_stage=insert_pos,
            )
            upper_bound = (4/3) ** ((dim-1)/4) #Reference?!
            assert verify_lattice_invariance(basis, lll_basis_final), "Determinant mismatch."
            assert verify_hermite_factor(lll_basis, dim, upper_bound), "Hermite factor out of bounds."
            assert verify_gso_structure(lll_basis_final, gs_coeffs_final, gs_squared_norms_final), "GSO structure is malformed."
            assert is_size_reduced(gs_coeffs_final), "Condition mu is not satisfied."
            assert verify_Lovasz_condition(gs_squared_norms_final, gs_coeffs_final), "Lovasz condition delta is not satisfied."
        
