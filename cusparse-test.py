import cupy
import scipy.sparse
import time
from cupy import cusparse
from cupyx.scipy import sparse
from scipy.io import mmread
from numba import cuda

#a = scipy.sparse.random(2, 3, density=0.5)
#b = scipy.sparse.random(3, 4, density=0.5)

#a2 = sparse.csr_matrix(a)
#b2 = sparse.csr_matrix(b)

## load mtx file
G = mmread('./ca-HepPh_adj.mtx')
G = G+G.transpose()
Gnx = sparse.csr_matrix(G)
Gny = sparse.csr_matrix(G.transpose())
#print(Gnx.number_of_edges())
#print(type(Gnx))
#print(type(Gny))
#print(Gnx.shape)
#print(Gny.shape)

print("NNZ x:", Gnx.nnz)
print("NNZ y:", Gny.nnz)

t1 = time.time()
y = cusparse.csrgemm2(Gnx, Gny)
cuda.synchronize()
t2 = time.time()

print("total time is:", t2 - t1)
print("NNZ:", y.nnz)

#print("\n y is: \n",y)
