
import sys
import os
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bkz.basis_generator import basis_gen
from bkz.L3FP.initializer import initialize
from bkz.L3FP.gsofp_se import gso_step
from tests.test_utils_minkowski import *
from bkz.SVPsolvers import ENUM_ALGORITHMS

# -------------------------
# Configurable parameters
# -------------------------
ENTRY_BOUND = 173
LATTICE_DIMENSION = 10
BLOCK_SIZE = LATTICE_DIMENSION // 2
SVP_SOLVER = ENUM_ALGORITHMS["1"]
TEST_CASES = 10

# Minkowski tiers: strict → relaxed
MINKOWSKI_TIERS = [1.00, 1.05, 1.10, 1.20, 1.30]
EPSILON = 1e-12

#RUN root: pytest tests/test_enum_minkowski.py
# Allow prints: pytest -s tests/test_enum_minkowski.py

def test_enum_against_minkowski(entry_bound=ENTRY_BOUND,
                                dim=LATTICE_DIMENSION,
                                block_size=BLOCK_SIZE,
                                svp_solver=SVP_SOLVER,
                                test_cases=TEST_CASES,
                                epsilon=EPSILON):

    for _ in range(test_cases):
        # Generate random basis and LLL-reduce
        basis_matrix = basis_gen(dim, entry_bound)
        start_stage = 0
        basis_matrix, gs_coeff_matrix, gs_squared_norms, stage, end_stage = initialize(
		basis_matrix, None, None, start_stage
	    )
        for i in range(stage, dim):
            gs_squared_norms[: i + 1], gs_coeff_matrix[:, : i + 1] = gso_step(
			basis_matrix[:, : i + 1],
			gs_coeff_matrix[:, : i + 1],
			gs_squared_norms[: i + 1],
			i,
		)

        # Run approx svp-solver
        candidate_squared_gs_norm, u = svp_solver(
            basis_matrix,
            gs_squared_norms,
            gs_coeff_matrix
        )
        candidate_vector = basis_matrix @ u
        candidate_norm = np.linalg.norm(candidate_vector)

        # Minkowski bound
        _, logdet_basis = np.linalg.slogdet(basis_matrix)
        bound = minkowski_bound(dim, logdet_basis)

        assert try_tiers(candidate_norm, bound, MINKOWSKI_TIERS, epsilon), (
            f"Minkowski violation: cand={candidate_norm:.6g}, "
            f"bound={bound:.6g}, tiers={MINKOWSKI_TIERS}"
        )
