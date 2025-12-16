import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bkz.L3FP.L3fp_params import *
from bkz.L3FP.gsofp_se import gso_step
from bkz.BasisQualityEvaluation.basis_quality_characteristics import *


def pretty_print_matrix(matrix, title="Matrix"):
	print(f"{title}:")
	for row in matrix:
		print("  ".join(f"{val:8.3f}" for val in row))
	print()


def verify_gso_step_by_step(spanning_matrix, gs_coefficient_matrix, gs_squared_norms):
	gs_coefficient_matrix_sbs = np.zeros(
		(gs_coefficient_matrix.shape[0], gs_coefficient_matrix.shape[1])
	)
	gs_coefficient_matrix_sbs[0, 0] = 1.0
	gs_squared_norms_sbs = np.zeros(gs_squared_norms.shape[0])
	gs_squared_norms_sbs[0] = np.dot(spanning_matrix[:, 0], spanning_matrix[:, 0])
	if not np.all(gs_coefficient_matrix_sbs[:, 0] == gs_coefficient_matrix[:, 0]) or not np.all(
		gs_squared_norms_sbs[0] == gs_squared_norms[0]
	):
		print("GSO construction breaks at stage 0!")
		return False

	for i in range(1, len(spanning_matrix[0])):
		gs_squared_norms_sbs[: i + 1], gs_coefficient_matrix_sbs[:, : i + 1] = gso_step(
			spanning_matrix[:, : i + 1],
			gs_coefficient_matrix_sbs[:, : i + 1],
			gs_squared_norms_sbs[: i + 1],
			i,
		)
		if not np.all(
			gs_coefficient_matrix_sbs[:, : i + 1] == gs_coefficient_matrix[:, : i + 1]
		) or not np.all(gs_squared_norms_sbs[: i + 1] == gs_squared_norms[: i + 1]):
			print(f"GSO construction breaks at stage {i}!")
			return False
		return True


def verify_gso_structure(span_mat, gsc_mat, gs_squared_norms):
	for i in range(0, span_mat.shape[1]):
		span_vec_squared_norm = np.dot(span_mat[:, i], span_mat[:, i])
		comp_squared_norm = 0
		comp_squared_norm += gs_squared_norms[i]

		for j in range(0, i):
			comp_squared_norm += (gsc_mat[j, i] ** 2) * gs_squared_norms[j]

		if not np.isclose(span_vec_squared_norm, comp_squared_norm, atol=1e-8):
			print("GSO structure is malformed!")
			print(f"Span_vec_squared_norm: {span_vec_squared_norm}")
			print(f"Comparison values: {comp_squared_norm}")
			return False
	return True

def verify_lattice_invariance_log(original_basis, reduced_basis, rtol=1e-6):
    """
    Verifies det(original_basis) == det(reduced_basis) in a numerically stable way,
    using log-space computations. Works safely even for high dimensions and
    for negative determinants (volume = |det(B)|).
    """
    # Get log(|det(B)|) for both bases
    _, logdet_orig = np.linalg.slogdet(original_basis)
    _, logdet_red  = np.linalg.slogdet(reduced_basis)

    # Relative difference in log-space:
    # |log(det1) - log(det2)| <= rtol
    # This is equivalent to: det2/det1 ≈ 1
    return abs(logdet_orig - logdet_red) <= rtol

def verify_lattice_invariance(original_basis, reduced_basis):
    # Get log(|det(B)|) for both bases
    _, logdet_original = np.linalg.slogdet(original_basis)
    _, logdet_reduced = np.linalg.slogdet(reduced_basis)
    if np.isclose(logdet_original, logdet_reduced, rtol=1e-6):
        return True
    else:
        return False

#---------------------------------------------------------------------------------
# Based on the BKZ-δβ / GSA heuristic described in:
#   M. R. Albrecht, S. Bai, P.-A. Fouque, P. Kirchner, D. Stehlé, W. Wen,
#   "Faster Enumeration-Based Lattice Reduction: Root Hermite Factor Time,"
#   in Annual International Cryptology Conference (CRYPTO 2020), pp. 186–212.
	
def compute_estimated_root_hermite_factor(block_size):
    val = ((block_size / (2.0 * np.pi * np.e)) * (np.pi * block_size) ** (1.0 / block_size)) ** (1.0 / (2.0 * (block_size - 1.0)))
    return max(val, 1.0 + 1e-12)


def get_rhf_tolerance(block_size):
    if block_size < 30:
        return 0.005  # 0.5% for small beta (more variance)
    elif block_size < 50:
        return 0.003  # 0.3%
    else:
        return 0.002  # 0.2%


def get_root_hermite_factor(basis):
    lattice_dimension = basis.shape[1]
    log_lattice_volume = compute_lattice_volume_log(basis)
    norm = np.linalg.norm(basis[:, 0])
    return compute_root_hermite_factor(norm, log_lattice_volume, lattice_dimension)

#---------------------------------------------------------------------------------
	
def verify_hermite_factor(basis, lattice_dim, upper_bound):
	log_lattice_volume = compute_lattice_volume_log(basis)
	norm = np.linalg.norm(basis[:, 0])
	hermite_factor = compute_hermite_factor(norm, log_lattice_volume, lattice_dim)
	if hermite_factor <= upper_bound:
		return True
	else:
		return False


def is_size_reduced(gram_schmidt_coefficients):
	# Efficiently check upper triangular part of the matrix
	# np.triu(): This is a NumPy function that returns the upper triangular part of a matrix.
	upper_triangular = np.triu(gram_schmidt_coefficients, k=1)  # k=1 excludes diagonal
	# compare each element of the absolute value matrix with the constant
	return np.all(np.abs(upper_triangular) <= SIZE_REDUCTION_CONDITION_PARAM + 1e-10)


def verify_Lovasz_condition(gs_squared_norms, span_gs_coeffs, Lovasz_cond=LOVASZ_CONDITION_PARAM):
	# Get the number of stages (number of basis vectors)
	n = len(gs_squared_norms)
	# Loop through each stage from 1 to n-1 (since stage-1 is needed)
	for stage in range(1, n):
		# Compute the left-hand side of the condition
		lhs = Lovasz_cond * gs_squared_norms[stage - 1]

		# Compute the right-hand side of the condition
		rhs = (
			gs_squared_norms[stage]
			+ (span_gs_coeffs[stage - 1, stage] ** 2) * gs_squared_norms[stage - 1]
		)

		# Check if the condition holds for the current stage
		if lhs > rhs:
			return False  # If any condition fails, return False

	# If all conditions passed, return True
	return True

#TRASH?
def norm_shortest_in_block(candidate_vector, basis_block):
        squared_norms_block = (np.sum(basis_block**2, axis=0))
        squared_norm_candidate = np.dot(candidate_vector, candidate_vector)
        if np.any(squared_norms_block > squared_norm_candidate):
            return True
        else:
            return False