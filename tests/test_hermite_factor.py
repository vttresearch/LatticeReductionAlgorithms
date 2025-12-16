import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from bkz.BasisQualityEvaluation.basis_quality_characteristics import compute_hermite_factor

# RUN from root: pytest tests/test_hermite_factor.py
# Allow prints: pytest -s tests/test_hermite_factor.py

def test_hermite_factor_known_case():
    shortest_vector_len = 2.0
    volume = 2.0
    dim = 2
    expected_hermite = 2.0 / (2.0 ** (1/2))
    hermite = compute_hermite_factor(shortest_vector_len, np.log(volume), dim)
    assert np.isclose(hermite, expected_hermite), f"Expected {expected_hermite}, got {hermite}"

def test_hermite_factor_identity_basis():
    shortest_vector_len = 1.0
    volume = 1.0
    dim = 3
    expected_hermite = 1.0 / (1.0 ** (1/3))
    hermite = compute_hermite_factor(shortest_vector_len, np.log(volume), dim)
    assert np.isclose(hermite, expected_hermite), f"Expected {expected_hermite}, got {hermite}"

def test_hermite_factor_large_dim():
    shortest_vector_len = 5.0
    volume = 1000.0
    dim = 5
    expected_hermite = 5.0 / (1000.0 ** (1/5))
    hermite = compute_hermite_factor(shortest_vector_len, np.log(volume), dim)
    assert np.isclose(hermite, expected_hermite), f"Expected {expected_hermite}, got {hermite}"
