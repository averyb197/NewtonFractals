import numpy as np
import os


def splitter(arr, filename):
	chunk_size = 100*1024*1024
	
	byte_pelem = arr.dtype.itemsize
	max_elems = chunk_size // byte_pelem

	epr = np.prod(arr.shape[1:]) if arr.ndim > 1 else 1
	rpc = max_elems // epr

	if rpc == 0:
		raise ValueError("Broken")
	dir_name = filename
	while True:
		ccc = 1		
		try:
			os.mkdir(dir_name)
			break
		except FileExistsError as e: 
			print(f"WARNING DIR ALREADY EXISTS, SAVING TO {filename}_{ccc}")
			dir_name = f"{filename}_{ccc}"
			ccc += 1
	total_rows = arr.shape[0]
	num_chunks = (total_rows + rpc - 1) // rpc
	for i in range(num_chunks):
		start = i * rpc
		end = min(start+rpc, total_rows)
		chunk = arr[start:end]
		fname = f"{dir_name}/{filename}_{i}.npy"
		np.save(fname, chunk)
		print(f"Saved {fname} with shape {chunk.shape} and size {chunk.nbytes / (1024 * 1024):.2f} MB")

	
