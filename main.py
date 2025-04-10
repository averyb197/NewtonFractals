from numba import njit, prange
import numpy as np
from tqdm import tqdm

@njit
def f(z):
    return z - 0.5 * np.sin(z)

@njit
def fprime(z):
    return 1 - 0.5 * np.cos(z)

@njit
def newton(z0, max_iter=50, tol=1e-6, diverge=1e10):
    z = z0
    for _ in range(max_iter):
        dz = f(z) / fprime(z)
        z -= dz
        if abs(dz) < tol:
            break
        if abs(z) > diverge:
            return 1e20 + 1e20j
    return z

@njit(parallel=True)
def compute_fractal_parallel(xmin, xmax, ymin, ymax, n, max_roots=50):
    x_vals = np.linspace(xmin, xmax, n)
    y_vals = np.linspace(ymin, ymax, n)
    fractal = np.zeros((n, n), dtype=np.uint8)
    root_list = np.empty((max_roots,), dtype=np.complex128)
    root_count = 0

    for ix in prange(n):
        for iy in range(n):
            z0 = x_vals[ix] + 1j * y_vals[iy]
            z_final = newton(z0)
            matched = False

            for i in range(root_count):
                if abs(z_final - root_list[i]) < 1e-4:
                    fractal[iy, ix] = i
                    matched = True
                    break

            if not matched and root_count < max_roots:
                root_list[root_count] = z_final
                fractal[iy, ix] = root_count
                root_count += 1

    return fractal, root_list[:root_count]

# Call this outside njit to use tqdm (tqdm is not supported in njit functions)
def compute_and_save():
    n = 4096
    fractal, roots = compute_fractal_parallel(-2, 2, -2, 2, n)
    np.save("cooler_fractal.npy", fractal)
    return roots

roots = compute_and_save()

