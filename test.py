import cupy as cp

print(cp.cuda.runtime.getDeviceCount())   # should be > 0
print(cp.cuda.Device(0).name)             # print GPU name

