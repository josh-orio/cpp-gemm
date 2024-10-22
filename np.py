import numpy as np
import multiproccesing as mp
import time

dim = 512
duration = 0
iterations = int(1e3)

for i in range(iterations):
    a = np.random.rand(dim,dim)
    b = np.random.rand(dim,dim)

    start = time.time()
    c = np.matmul(a, b)
    end = time.time()

    duration += end - start

gflops = ((dim ** 3) * 2) / 1e9
# print(gflops)

print("NP GFLOP/s:", gflops / (duration / iterations), "(All Core)")
print("NP GFLOP/s:", (gflops / (duration / iterations)) / mp.cpu_count(), "(Est. Single Core)")
