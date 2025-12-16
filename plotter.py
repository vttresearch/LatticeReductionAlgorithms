import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_results_data_in_tables(results_original, results_lll, results_bkz, lattice_dimension):
	"""Function converts given lists to Pandas DataFrames and prints the resulting data and plots figures using matplotlib.

	Args:
		results_original (list): Lattice quality metrics for a random basis.
		results_lll (list): Quality metrics for a lll-reduced lattice basis.
		results_bkz (list): Quality metrics for a bkz-reduced lattice basis.
		lattice_dimension (int): Number of dimensions in each lattice (received via command line argument).
	"""

	df_original = pd.DataFrame(results_original)
	lattice_dim_array = [lattice_dimension for _ in range(df_original.shape[0])]
	df_original.columns = [
		"Length of the first vector",
		"Root Hermite Factor",
		"Orthogonality Defect",
		"Run time",
	]
	df_original["Lattice dimension"] = lattice_dim_array
	df_original.index.name = "Lattice ID"

	df_lll = pd.DataFrame(results_lll)
	df_lll.columns = [
		"Length of the first vector",
		"Root Hermite Factor",
		"Orthogonality Defect",
		"Run time",
	]
	df_lll["Lattice dimension"] = lattice_dim_array
	df_lll.index.name = "Lattice ID"

	df_bkz = pd.DataFrame(results_bkz)
	df_bkz.columns = [
		"Length of the first vector",
		"Root Hermite Factor",
		"Orthogonality Defect",
		"Run time",
	]
	df_bkz["Lattice dimension"] = lattice_dim_array
	df_bkz.index.name = "Lattice ID"

	print("-----------------Original lattice-----------------")
	print(df_original)
	df_original.to_csv("docs/csv/original_lattice.csv")
	df_original.to_markdown("docs/original_lattice_table.md")
	print("-----------------lll reduced-----------------")
	print(df_lll)
	df_lll.to_csv("docs/csv/lll.csv")
	df_lll.to_markdown("docs/lll_table.md")
	print("-----------------bkz reduced-----------------")
	print(df_bkz)
	df_bkz.to_csv("docs/csv/bkz.csv")
	df_bkz.to_markdown("docs/bkz_table.md")

	# ------- Plotting the figures with matplotlib ------------
	x_axis = [x + 1 for x in range(df_bkz.shape[0])]
	plt.figure(1)
	ax = plt.subplot(111)
	ax.stem(
		x_axis,
		df_original["Length of the first vector"],
		"C0-",
		basefmt="k-",
		label="Original lattice",
	)
	ax.stem(x_axis, df_bkz["Length of the first vector"], "C1--", basefmt="k-", label="BKZ reduced")
	ax.stem(x_axis, df_lll["Length of the first vector"], "C2:", basefmt="k-", label="LLL reduced")
	plt.xticks([])
	plt.ylabel("Length of the shortest vector")
	plt.xlabel("Algorithm iteration")
	plt.title("Euclidean length of the shortest vector in current basis")
	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# Put a legend to the right of the current axis
	ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
	plt.savefig("docs/figures/figure1.png")

	plt.figure(2)
	ax = plt.subplot(111)
	ax.stem(
		x_axis,
		df_original["Root Hermite Factor"],
		"C0-",
		basefmt="k-",
		label="Original lattice",
	)
	ax.stem(x_axis, df_bkz["Root Hermite Factor"], "C1--", basefmt="k-", label="BKZ reduced")
	ax.stem(x_axis, df_lll["Root Hermite Factor"], "C2:", basefmt="k-", label="LLL reduced")
	ax.set_ylim(
		np.min(df_bkz["Root Hermite Factor"]) - 0.005,
		np.max(df_original["Root Hermite Factor"]) + 0.005,
	)
	plt.xticks([])
	plt.xlabel("Algorithm iteration")
	plt.ylabel("Root Hermite factor")
	plt.title("Root Hermite factor")
	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# Put a legend to the right of the current axis
	ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
	plt.savefig("docs/figures/figure2.png")

	plt.figure(3)
	ax = plt.subplot(111)
	ax.stem(
		x_axis,
		df_original["Orthogonality Defect"],
		"C0-",
		basefmt="k-",
		label="Original lattice",
	)
	ax.stem(x_axis, df_lll["Orthogonality Defect"], "C1--", basefmt="k-", label="LLL reduced")
	ax.stem(x_axis, df_bkz["Orthogonality Defect"], "C2:", basefmt="k-", label="BKZ reduced")
	plt.xticks([])
	plt.xlabel("Algorithm iteration")
	plt.ylabel("Orthogonality Defect")
	plt.title("Orthogonality Defect")
	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# Put a legend to the right of the current axis
	ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
	plt.savefig("docs/figures/figure3.png")

	plt.figure(4)
	ax = plt.subplot(111)
	ax.stem(x_axis, df_lll["Orthogonality Defect"], "C1--", basefmt="k-", label="LLL reduced")
	ax.stem(x_axis, df_bkz["Orthogonality Defect"], "C2-", basefmt="k-", label="BKZ reduced")
	plt.xlabel("Algorithm iteration")
	plt.ylabel("Orthogonality Defect")
	plt.title("Orthogonality Defect (lll & bkz reduced)")
	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# Put a legend to the right of the current axis
	ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
	plt.savefig("docs/figures/figure4.png")
