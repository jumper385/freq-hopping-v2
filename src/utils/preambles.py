import numpy as np


def zadoff_chu(N: int, n: int) -> np.ndarray:
    """
    Generate a Zadoff-Chu sequence.

    Parameters
    ----------
    N : int
        Sequence length (must be odd and coprime with n for ideal properties)
    n : int
        Root index (1 <= n < N, gcd(n, N) == 1)
    Returns
    -------
    np.ndarray, shape (N,), dtype complex128
    """
    k = np.arange(N)
    exponent = -1j * np.pi * n * k * (k + 1) / N
    return np.exp(exponent)
