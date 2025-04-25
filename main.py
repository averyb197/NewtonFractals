import torch
from numba import njit, prange
import numpy as np
import time
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")


def f(z):
    return z - 0.5 * torch.sin(z)

def df(z):
    return 1 - 0.5 * torch.cos(z)

# def f(z):
#     numerator = z**12 - 3 * z**3 - 1
#     denominator = z**9 - z + 5
#     return numerator / denominator

# def df(z):
#     numerator = z**12 - 3 * (z**3) - 1
#     denominator = z**9 - z + 5
#     return numerator / denominator


# Newton iteration of whole gird of ic
def newton(z, f, df, max_iter, tol):
    #stores if a given point has converged or not, init all to no, torch tensor
    converged = torch.zeros_like(z, dtype=torch.bool)
    #stores the root of a given initial guess, init with nan to clearly mark roots that dont converge
    #so they can be marked spacifically later
    roots = torch.full_like(z, complex('nan'))
    #looping through entire grid at once, need to be careful of doing extra work
    for i in range(max_iter):
        #step size
        dz = f(z) / df(z)
        #compute change
        z_next = z - dz
        #check convergence
        pt_converge = torch.abs(dz) < tol
        # ~ element wise not
        # updates roots if algorithm converged for a given root
        # ie pt_converge and not already marked as converged -> mark as converged to freeze that value of z
        roots[pt_converge & ~converged] = z[pt_converge & ~converged]
        #element wise or -> converged updated to mark any newly converged points
        converged |= pt_converge
        #update z
        z = z_next
    return roots

#numba will throw a fit if you dare to pass a tensor to it so need to convert to
#numpy array and specify to use cpu
#return vector of length n^2
def numba_friendly_roots(roots):
    return roots.flatten().cpu().numpy()

#switching to numba for cpu heavy task like this dumb iteration, must specifiy to
#torch that we are switching to numpy array
@njit(parallel=True)
def sort_roots(roots_NP, tol=1e-3):
    n = roots_NP.size
    #stores which root is converged to based on index of that root in unique root array, will leave diverged ic roots as -1
    indices = np.full(n, -1, dtype=np.int32)

    #Numba will also yell at you if you try to resize a list mid loop, set to n
    #since could never be more roots than this
    unique_roots = np.empty(n, dtype=roots_NP.dtype)

    #keep track of number of roots found to get rid of most of unique_roots later
    unique_count = 0

    #loop through all the found roots
    for i in prange(n):
        r = roots_NP[i]
        #since we let diverging roots stay nan need to ignore them, must check real and imaginary part seperately
        #for numba reasons otherwise you die
        if np.isnan(r.real) or np.isnan(r.imag):
            continue
        #init to not found yet
        found = False
        #check against all other roots found so far
        for j in range(unique_count):
            if abs(r - unique_roots[j]) < tol:
                indices[i] = j
                found = True
                break
        #add to unique if not found yet
        if not found:
            unique_roots[unique_count] = r
            #its index is the unique count since new
            indices[i] = unique_count
            unique_count += 1
    unique_roots = unique_roots[:unique_count]

    return unique_roots, indices

def fractal(f, df, max_iter=50, tol=1e-3, res=128, fov_size=4):
    # make complex plane
    xmin, ymin = -fov_size, -fov_size
    xmax, ymax = fov_size, fov_size
    x = torch.linspace(xmin, xmax, res, device=device)
    y = torch.linspace(ymin, ymax, res, device=device)
    X, Y = torch.meshgrid(x, y)
    Z = X + 1j * Y

    z0 = Z.clone()

    roots = newton(z0, f, df, max_iter, tol)
    unique_roots, root_indices = sort_roots(numba_friendly_roots(roots))
    print(root_indices)

    return root_indices.reshape(res, res)

parser = argparse.ArgumentParser()
parser.add_argument("-r","--res", help="Resolution", type=int)
parser.add_argument("-f", "--fov_size", help="Fov Size", type=float)
parser.add_argument("-m", "--max_itera", default=50, help="Max Number of Iterations", type=int)
parser.add_argument("-t", "--tol", default=1e-4, help="Tolerance", type=float)
parser.add_argument("-p", "--path", help="Filename to save to", type=str)
args = parser.parse_args()

res = args.res
fov_size = args.fov_size
max_iter = args.max_itera
tol = args.tol
path = args.path 

if res is not None and fov_size is not None and max_iter is not None and tol is not None and path is not None:
	pass
else:
	raise ValueError("need args")

start = time.time()
frac = fractal(f, df, max_iter=max_iter, tol=tol, res=res, fov_size=fov_size)


end = time.time()
print(f"Time: {end-start} \n Resolution: {res} \n Tolerance: {tol}")

from splitter import *

splitter(frac, path)

