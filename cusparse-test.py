import cupy
import scipy
from cupy import cusparse
from cupyx.scipy import sparse
from scipy.io.mmio import mminfo, mmread, mmwrite

a = scipy.sparse.random(2, 3, density=0.5)
b = scipy.sparse.random(3, 4, density=0.5)

a2 = sparse.csr_matrix(a)
b2 = sparse.csr_matrix(b)

## load mtx file
# G = mmread(graph_file)
# Gnx = nx.from_scipy_sparse_matrix(G, create_using=nx.DiGraph)
# print(Gnx.number_of_edges())

y = cusparse.csrgemm2(a2, b2)

print("\n a is: \n",a2)
print("\n b is: \n",b2)
print("\n y is: \n",y)
