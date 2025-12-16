# basis_quality_characteristics

## Definitions

### Lattice volume

For a full-rank square basis matrix $B$, the lattice volume is defined as
the absolute value of the $det(B)$.

### Root Hermite Factor

The Root Hermite factor is defined as

$$
\bigg(\frac{||b_0||}{Vol(L)^{\frac{1}{n}}}\bigg)^\frac{1}{n}
$$ 

where $||b_0||$
is the length of the shortest vector, $Vol(L)$ is the lattice volume, and $n$ is the lattice dimension.

### Dimension-Normalized Orthogonality defect

The dimension-normalized orthogonality defect is defined as

$$
\bigg(\frac{\prod_{i=0}^{n-1}||b_i||}{Vol(L)}\bigg)^\frac{1}{n}
$$

where $\prod_{i=0}^{n-1}||b_i||$ is the product of the column norms, $Vol(L)$ is the lattice volume, and $n$ is the lattice dimension.

## Functions

::: BasisQualityEvaluation.basis_quality_characteristics