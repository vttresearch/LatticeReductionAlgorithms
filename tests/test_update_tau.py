import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import update_tau
from bkz.L3FP.L3fp_params import TAU

#RUN root: pytest tests/test_update_tau.py
# Allow prints: pytest -s tests/test_update_tau.py

def test_prec_level_high():
	basis = np.random.rand(100, 10)
	update_tau(basis, precision_level="high")
	assert TAU <= 80
	assert TAU >= 30

def test_prec_level_low():
	basis = np.random.rand(100, 10)
	update_tau(basis, precision_level="low")
	assert TAU <= 40
	assert TAU >= 10

def test_prec_level_default():
	basis = np.random.rand(100, 10)
	update_tau(basis)
	assert TAU <= 60
	assert TAU >= 20
