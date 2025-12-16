import numpy as np
from fpylll import IntegerMatrix, LLL, Enumeration, GSO, SVP, EvaluatorStrategy, BKZ

def initialize_fpylll_gso_object(basis):
        #Transpose basis and convert it to list of lists
        transposed_basis = np.transpose(basis)
        basis_list = transposed_basis.astype(int).tolist()
        # Convert basis to IntegerMatrix for fpylll
        mat = IntegerMatrix.from_matrix(basis_list)
        # Create a GSO object
        gso = GSO.Mat(mat)
        gso.update_gso()
        return gso

def fpylll_enum(gso_object, block_start, block_end, search_radius):
	enum_object = Enumeration(M=gso_object, nr_solutions=1, strategy=EvaluatorStrategy.BEST_N_SOLUTIONS, sub_solutions=False, callbackf=None)
	shortest_vector = enum_object.enumerate(block_start, block_end, search_radius, 1, target=None, subtree=None, pruning=None, dual=False, subtree_reset=False)
	return shortest_vector

def is_svp_reduced(basis, gs_squared_norms, block_size):
	n = basis.shape[0]
	# Transpose basis and convert it to list of lists
	transposed_basis = np.transpose(basis.copy())
	basis_list = transposed_basis.astype(int).tolist()
	# Convert basis to IntegerMatrix for fpylll
	mat = IntegerMatrix.from_matrix(basis_list)
	# Create a GSO object
	gso = GSO.Mat(mat)
	gso.update_gso()

	# Execute block-wise enumeration
	for i in range(0, n - 1):
		block_end = min(i + block_size, n - 1)
		enum_obj = Enumeration(
			M=gso,
			nr_solutions=3,
			strategy=EvaluatorStrategy.FIRST_N_SOLUTIONS,
			sub_solutions=False,
			callbackf=None,
		)
		solutions = enum_obj.enumerate(i, block_end, gs_squared_norms[i], 3)
		# Extract projected norms of the top 3 solutions
		top_proj_norms = [float(sol[0]) for sol in solutions]
		if not any(np.isclose(gs_squared_norms[i], norm) for norm in top_proj_norms):
			print(
				f"Block {i}: projected length {gs_squared_norms[i]} not in top 3 norms {top_proj_norms}"
			)
			return False
	return True