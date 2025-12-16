import sys
import os
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bkz.basis_generator import basis_gen
from bkz.L3FP.L3fp import l3fp
from bkz.SVPsolvers import ENUM_ALGORITHMS
from tests.test_utils_fpylll import *

# Configurable parameters
ENTRY_BOUND = 173
LATTICE_DIMENSION = 10
BLOCK_SIZE = LATTICE_DIMENSION // 2
SVP_SOLVER = ENUM_ALGORITHMS["1"]
TEST_CASES = 10

# Tiered thresholds
APPROX_TIERS = [1.00, 1.05, 1.10, 1.20, 1.30]


# Comparison tolerance to avoid float-equality pitfalls
EPSILON = 1e-12  # tighten/loosen as needed

#RUN root: pytest tests/test_enum_fpylll.py
# Allow prints: pytest -s tests/test_enum_fpylll.py

def test_case_find_candidate(entry_bound = ENTRY_BOUND, dim = LATTICE_DIMENSION, block_size = BLOCK_SIZE, svp_solver=SVP_SOLVER, test_cases=TEST_CASES):
    for _ in range(test_cases):
        basis = basis_gen(dim, entry_bound)
        lll_basis, gsc, gs_sq_norms = l3fp(basis)
        gso_object = initialize_fpylll_gso_object(lll_basis)
        for i in range(0, len(lll_basis[0])-1):
            block_end = min(dim, i+block_size)
            # Our approx svp_solver
            candidate_norm, u = svp_solver(
                lll_basis[:, i:block_end],
                gs_sq_norms[i:i + block_end],
                gsc[i:i + block_end,
                i:i + block_end]
            )
            #Exact enumeration baseline
            comparison_vec = fpylll_enum(gso_object, i, block_end, gs_sq_norms[i])
            exact_norm = comparison_vec[0][0]

            alpha = candidate_norm / exact_norm

            # Try tiered thresholds
            passed = False
            for thr in APPROX_TIERS:
                if alpha <= thr + EPSILON:
                    passed = True
                    break
                else:
                    # Log a warning and escalate to next tier
                    warnings.warn(
                        f"Block [{i}:{block_end}] alpha={alpha:.6f} exceeded {thr:.2f}. "
                        f"our={candidate_norm:.6g}, exact={exact_norm:.6g}, u={u}"
                    )

            
            assert passed, (
                f"Final failure at block [{i}:{block_end}]: alpha={alpha:.6f}\n"
                f"candidate_norm={candidate_norm:.6g}, exact={exact_norm:.6g}, u={u}, tiers={APPROX_TIERS}"
            )

