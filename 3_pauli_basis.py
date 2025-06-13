from itertools import product
import numpy as np
from qiskit.quantum_info import Operator
from qiskit.circuit import library
import matplotlib.pyplot as plt

def pad_matrix(matrix):
    # Pad with 0 to make square matrix
    max_shape = max(matrix.shape[0], matrix.shape[1])
    deficiency = int(np.power(2,np.ceil(np.log2(max_shape))) - max_shape)
    if matrix.shape[0] != matrix.shape[1]:
        if matrix.shape[0] > matrix.shape[1]:
            pad_width = [(0, 0), (0, matrix.shape[0] - matrix.shape[1])]
        else:
            pad_width = [(0, matrix.shape[1] - matrix.shape[0]), (0, 0)]
        matrix = np.pad(matrix, pad_width)
    matrix = np.pad(matrix,[(0,deficiency),(0,deficiency)])
    return matrix
    
def decompose_pauli(matrix):
    matrix = pad_matrix(matrix)
    matrix_len = matrix.shape[0]
    nqubits = int(np.log2(matrix_len))
    
    pauli = {
        'x': Operator(library.XGate().to_matrix()),
        'y': Operator(library.YGate().to_matrix()),
        'z': Operator(library.ZGate().to_matrix()),
        'i': Operator(library.IGate().to_matrix())
    }
    
    decomposition = {}
    for permutation in product(*[list(pauli.keys())]*nqubits):
        permutation = "".join(permutation)
        base_matrix = pauli[permutation[0]]
        for idx in range(1, len(permutation)):
            base_matrix = base_matrix.tensor(pauli[permutation[idx]])
        
        decomposition_component = np.trace(np.dot(base_matrix, matrix)) / matrix_len
        if 0!=decomposition_component:
            decomposition[permutation] = decomposition_component
    
    return decomposition
    
max_n = 8
N = [2**n for n in range(1,max_n)]
sparsity = [len(decompose_pauli(2*np.eye(2**n)
                                - np.diag(np.ones(2**n-1),-1)
                                - np.diag(np.ones(2**n-1),1))) for n in range(1,max_n)]
plt.plot(N,sparsity)
plt.xlabel('N')
plt.ylabel('Pauli Terms')
plt.title('Sparsity in Pauli Basis of NxN Laplacian')
plt.show()
