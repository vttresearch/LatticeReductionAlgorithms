import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from bkz.BasisQualityEvaluation.basis_quality_characteristics import compute_lattice_volume_log

# RUN from root: pytest tests/test_lattice_volume.py

def test_lattice_volume_known_case():
    # Basis matrix with known determinant
    basis = np.array([[1, 0], [0, 2]])
    expected_volume = 2.0
    volume = np.exp(compute_lattice_volume_log(basis))
    assert np.isclose(volume, expected_volume), f"Expected {expected_volume}, got {volume}"

def test_lattice_volume_identity():
    basis = np.eye(3)
    expected_volume = 1.0
    volume = np.exp(compute_lattice_volume_log(basis))
    assert np.isclose(volume, expected_volume), f"Expected {expected_volume}, got {volume}"

def test_lattice_volume_negative_det():
    basis = np.array([[0, 1], [1, 0]])
    expected_volume = 1.0
    volume = np.exp(compute_lattice_volume_log(basis))
    assert np.isclose(volume, expected_volume), f"Expected {expected_volume}, got {volume}"


def test_lattice_volume_variable_matrix():
    basis = np.array([
        [3, 1, 4],
        [1, 5, 9],
        [2, 6, 5]
    ])
    expected_volume = 90.0
    volume = np.exp(compute_lattice_volume_log(basis))
    assert np.isclose(volume, expected_volume), f"Expected {expected_volume}, got {volume}"

