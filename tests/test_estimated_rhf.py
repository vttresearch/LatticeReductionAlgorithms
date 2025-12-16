import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_utils import compute_estimated_root_hermite_factor

# RUN from root: pytest tests/test_estimated_rhf.py

def test_estimated_rhf_block_20():
    rhf = compute_estimated_root_hermite_factor(20)
    assert rhf > 1.0, f"RHF should be > 1 for block size 20, got {rhf}"
    assert np.isclose(rhf, 1.0097, atol=0.001), f"Expected ~1.0097, got {rhf}"

def test_estimated_rhf_block_40():
    rhf = compute_estimated_root_hermite_factor(40)
    assert rhf > 1.0, f"RHF should be > 1 for block size 40, got {rhf}"
    assert np.isclose(rhf, 1.0125, atol=0.001), f"Expected ~1.0125, got {rhf}"

def test_estimated_rhf_block_60():
    rhf = compute_estimated_root_hermite_factor(60)
    assert np.isclose(rhf, 1.0107, atol=0.001), f"Expected ~1.0107, got {rhf}"
   
