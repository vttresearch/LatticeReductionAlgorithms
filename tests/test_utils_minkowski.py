import math
import numpy as np
import warnings


# Based on Minkowski’s theorem description in:
#   D. Micciancio, “Minimum distance and SVP,” CSE 206A Lecture 3, UCSD, Spring 2007.
#   https://cseweb.ucsd.edu/classes/sp07/cse206a/lec3.pdf


def minkowski_bound(n, logdet):
    """
    Minkowski upper bound using log-space to avoid overflow.

    Args:
        n (int): Lattice dimension
        logdet (float): log(|det(B)|)

    Returns: float: Minkowski bound
    """
    log_bound = 0.5 * math.log(n) + logdet / n
    return math.exp(log_bound)


def try_tiers(value, base, tiers, epsilon):
    """Evaluate a numeric quantity against progressively relaxed Minkowski thresholds.

    This function verifies whether a given value (typically the norm of a
    candidate lattice vector) satisfies any of a sequence of scaled Minkowski
    bounds. The sequence of scaling factors, referred to as *tiers*, represents
    progressively weaker acceptance criteria. The function returns ``True`` as
    soon as one tier admits the candidate and otherwise returns ``False``.

    For diagnostic purposes, the function also issues a warning for each tier
    that is violated before termination.

    Args:
        value : float
            The numeric value to be tested. In the context of the tests, this is the
            Euclidean norm of the SVP candidate vector returned by an enumeration
            algorithm.

        base : float
            The reference bound used for the comparison. This is typically the
            Minkowski upper bound computed as:

            ``M = exp(0.5 * log(n) + logdet / n)``

            where ``logdet`` is the logarithm of the absolute determinant of the
            basis matrix.

        tiers : Sequence[float]
            An ordered sequence of multiplicative factors. Each tier represents an
            acceptance threshold of the form:

                ``value <= base * tier + epsilon``

            The list is expected to be sorted from strictest (closest to 1.0) to
            most relaxed.

        epsilon : float
            A small numerical tolerance used to mitigate floating-point inaccuracies
            in the comparisons. A typical value is ``1e-12``.

    Returns:
        bool:
        ``True`` if the value satisfies the bound under at least one tier.
        ``False`` otherwise.
    """
    ratio = value / base
    for t in tiers:
        if ratio <= t + epsilon:
            return True
        else:
            warnings.warn(f"ratio={ratio:.6f} exceeded {t:.2f}."
                          f"value={value:.6g}, base={base:.6g}.")
    return False