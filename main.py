import argparse
import time

from tqdm import tqdm

import plotter
from bkz import BKZ_ALGORITHMS
from bkz.basis_generator import basis_gen
from bkz.BasisQualityEvaluation.basis_quality_evaluation import (
	compute_basis_quality_characteristics,
)
from bkz.bkz_params import *
from bkz.L3FP.L3fp import l3fp
from bkz.L3FP.L3fp_params import update_tau

# RUN: python3 main.py --lattice_dimension 10 --entry_bound 73 --bkz_version 1 --svp_solver 1 --block_size 5 --precision default --repetitions 5
# Simple RUN: # RUN: python3 main.py


def positive_integer(value):
	"""Helper function to check if given integer is positive.

	Args:
		value (int): Integer to check.

	Returns:
		(bool): True if value is positive, otherwise False.
	"""
	try:
		int_value = int(value)
		return int_value > 0
	except (ValueError, TypeError):
		return False


def compute_and_print_quality_metrics(args):
	results_original = []
	results_lll = []
	results_bkz = []

	for i in tqdm(
		range(0, args.repetitions), desc="Repetitions", position=0, ascii="-##", colour="green"
	):
		# Generate basis using either provided or default dimension
		original_basis = basis_gen(args.lattice_dimension, args.entry_bound)
		update_tau(original_basis, args.precision)
		characteristics_original = compute_basis_quality_characteristics(original_basis, False)
		characteristics_original.append(0.0)  # append Run time = 0.0
		results_original.append(characteristics_original)

		lll_start = time.time()
		lll_reduced_basis = run_lll(original_basis)
		lll_end = time.time()
		lll_time = lll_end - lll_start
		characteristics_lll = compute_basis_quality_characteristics(lll_reduced_basis, reduced=True)
		characteristics_lll.append(lll_time)
		results_lll.append(characteristics_lll)

		bkz_start = time.time()
		bkz_reduced_basis = run_bkz(
			original_basis, args.block_size, args.bkz_version, args.svp_solver
		)
		bkz_end = time.time()
		bkz_time = bkz_end - bkz_start
		characteristics_bkz = compute_basis_quality_characteristics(bkz_reduced_basis, reduced=True)
		characteristics_bkz.append(bkz_time)
		results_bkz.append(characteristics_bkz)

	plotter.print_results_data_in_tables(
		results_original, results_lll, results_bkz, lattice_dimension=args.lattice_dimension
	)


def run_lll(basis):
	"""Calls the LLL-reduction algorithm.

	Args:
		basis (np.ndarray):
			A 2D NumPy array of shape (n, n) representing a lattice basis,
			where each column is a basis vector.

	Returns:
		lll_reduced_basis (np.ndarray):
			A 2D NumPy array of shape (n, n) representing a LLL-reduced lattice basis,
			where each column is a basis vector.
	"""

	lll_reduced_basis, gs_coeff_matrix, gs_squared_norms = l3fp(basis)

	return lll_reduced_basis


def run_bkz(basis, block_size, bkz_version, svp_solver):
	"""Executes a BKZ (Block Korkine–Zolotarev) reduction on a given lattice basis. This function serves as a unified entry point for invoking one of the
	available BKZ variants registered in `BKZ_ALGORITHMS`. The selected BKZ
	routine will repeatedly call the provided SVP solver on local blocks,
	update the basis, and maintain Gram–Schmidt data as required by the
	algorithm.

	Args:
		basis (np.ndarray):
			A 2D NumPy array of shape (n, n) representing a lattice basis, where each column is a basis vector.
		block_size (int):
			An integer specifying the block size in the BKZ algorithm, which determines the dimension of sublattice for the SVP solver calls.
		bkz_version (str):
			A string key selecting the BKZ algorithm variant from `BKZ_ALGORITHMS`.
		svp_solver (str):
			A string key referring to an entry in `ENUM_ALGORITHMS`.

	Returns:
		bkz_reduced_basis (np.ndarray):
			A 2D NumPy array of shape (n, n) representing the BKZ-reduced lattice basis, where each column is a basis vector.
	"""
	bkz_reduce = BKZ_ALGORITHMS[bkz_version]
	bkz_reduced_basis, gs_coeff_matrix, gs_squared_norms = bkz_reduce(basis, block_size, svp_solver)

	return bkz_reduced_basis


def main():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
		description="Run lattice reduction algorithms.",
	)
	parser.add_argument(
		"--lattice_dimension",
		type=int,
		default=LATTICE_DIMENSION,
		help="Desired lattice dimension.",
	)
	parser.add_argument(
		"--entry_bound",
		type=int,
		default=ENTRY_BOUND,
		help="Bound for basis entry values",
	)
	parser.add_argument(
		"--bkz_version",
		choices=["1", "2"],
		default="1",
		help="Specify the version of bkz implementation: 1: bkz_se_pc, 2: bkz_se",
	)
	parser.add_argument(
		"--svp_solver",
		choices=["1", "2", "3"],
		default="1",
		help="Specify the svp_solver utilized during bkz execution: 1: enum_se_og_solver, 2: enum_se_solver, 3: enum_sh_solver",
	)
	parser.add_argument(
		"--block_size", type=int, default=BLOCK_SIZE, help="Desired block size for bkz."
	)
	parser.add_argument(
		"--precision",
		type=str,
		default="default",
		help="Precision of floating point arithmetic: high, default, low.",
	)
	parser.add_argument(
		"--repetitions", type=int, default=5, help="Number of random lattice bases to operate on."
	)
	args = parser.parse_args()

	if (
		not positive_integer(args.lattice_dimension)
		or not positive_integer(args.entry_bound)
		or not positive_integer(args.repetitions)
		or not positive_integer(args.block_size)
	):
		raise TypeError("All numerical command line arguments should be positive integers.")

	compute_and_print_quality_metrics(args)


if __name__ == "__main__":
	main()
