from numba import njit
import numpy as np
from tqdm import tqdm

@njit
def f(z): return z**3 - 1

@njit
def fprime(z): return 3 * z**2

@njit
def newton(z0, max_iter=50, tol=1e-6):
    z = z0
    for _ in range(max_iter):
        dz = f(z) / fprime(z)
        z -= dz
        if abs(dz) < tol:
            break
    return z


def compute_fractal(xmin, xmax, ymin, ymax, n):
    x_vals = np.linspace(xmin, xmax, n)
    y_vals = np.linspace(ymin, ymax, n)
    fractal = np.zeros((n, n), dtype=np.uint8)
    root_list = []

    for ix in tqdm(range(n), desc="Computing fractal"):
        for iy in range(n):
            z0 = x_vals[ix] + 1j * y_vals[iy]
            z_final = newton(z0)
            matched = False
            for i, r in enumerate(root_list):
                if abs(z_final - r) < 1e-4:
                    fractal[iy, ix] = i
                    matched = True
                    break
            if not matched:
                root_list.append(z_final)
                fractal[iy, ix] = len(root_list) - 1

    return fractal, root_list

fractal, roots = compute_fractal(-2, 2, -2, 2, 1024)
np.save("newton_fractal.npy", fractal)

