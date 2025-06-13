from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit import QuantumCircuit
import numpy as np
from scipy.optimize import minimize
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import StatevectorSampler as Sampler
from matplotlib import pyplot as plt

# Create a dictionary of all the weights for the vertices
weights = {(1,2):2, (1,4):2, (1,6):3, (2,4):1, (2,3):3, (3,6):2, (3,5):2, (4,6):2, (4,5):4}

# Create weight matrix W
n = max([max(key) for key in weights.keys()])
W = np.zeros((n,n))
for key in weights:
    i,j = key
    W[i-1,j-1] = weights[key]
    W[j-1,i-1] = weights[key]

# Create QUBO problem from weight matrix
Q = -W
b = np.zeros((n,1))
for i in range(n):
    b[i] = np.sum(W[i,:])

# Form Cost Hamiltonian from QUBO problem
# This will be an Ising Hamiltonian
string_list = []
coeff_list = []
for i in range(n):
    for j in range(n):
        if Q[i,j]!=0:
            string = 'I'*n
            string = string[:i] + 'Z' + string[i+1:]
            string = string[:j] + 'Z' + string[j+1:]
            string_list.append(string)
            coeff_list.append(Q[i,j]/4)
for i in range(n):
    coeff = -b[i]/2
    for j in range(n):
        coeff += -Q[i,j]/2
    string = 'I'*n
    string = string[:i] + 'Z' + string[i+1:]
    if coeff!=0:
        string_list.append(string)
        coeff_list.append(coeff)
H_c = SparsePauliOp(string_list,coeff_list)
# At this point using the following:
# circuit = QAOAAnsatz(cost_operator=H_c, reps=2)
# will be sufficient. The following lines prepare
# the mixer Hamiltonian and initial state for
# completeness of demonstration.

# Form mixer Hamiltonian
string_list = []
coeff_list = []
for i in range(n):
    string = 'I'*n
    string = string[:i] + 'X' + string[i+1:]
    string_list.append(string)
    coeff_list.append(1)
H_m = SparsePauliOp(string_list,coeff_list)

# Quantum circuit to prepare initial state
initial_state = QuantumCircuit(n)
initial_state.h(range(n))

# Create QAOA ansatz
circuit = QAOAAnsatz(cost_operator=H_c, mixer_operator=H_m, initial_state=initial_state, reps=5)

# Define QAOA cost function,
# return with -ve sign since we are maximizing using a scipy minimizer
def QAOA_cost(parameters, circuit, H_c, estimator):
    pub = (circuit, H_c, parameters)
    job = estimator.run([pub])
    result = job.result()[0]
    cost = result.data.evs
    cost_history.append(cost)
    return -cost

# Set up Estimator primitive for optimization
estimator = Estimator()
# Track optimization progress
cost_history = []
# Guess initial parameters
init_params = np.ones(len(circuit.parameters))

# Maximize the cost (cost returns -ve)
result = minimize(QAOA_cost,
                  init_params,
                  args=(circuit, H_c, estimator),
                  method='L-BFGS-B',
                  tol=1e-5,
                 )

# Set up sampler primitive to get optimized results
sampler = Sampler()

# Set up optimized circuit for sampling
optimized_circuit = circuit.assign_parameters(result.x)
optimized_circuit.measure_all()
pub = (optimized_circuit)
# Sample circuit
shots = 100
job = sampler.run([pub],shots=shots)
counts = job.result()[0].data.meas.get_counts()

# Sort by number of counts
sorted_counts = [(key, counts[key]) for key in sorted(counts,key=counts.get)]

# Compute costs of each counts bitstring
def get_cost_from_string(string, Q, b):
    x = [int(i) for i in string]
    cost = np.einsum('i,ij,j->',x,Q,x) + np.einsum('i,ij->',x,b)
    return cost
costs = [(key,get_cost_from_string(key,Q,b)) for (key,value) in sorted_counts]

# Plot results
fig, axes = plt.subplots(1,2, sharey=True, figsize=(10, 8))
axes[0].barh([i for (i,j) in sorted_counts],[j/shots for (i,j) in sorted_counts], align='center')
axes[0].invert_xaxis()
axes[0].set_xlabel('Quasiprobability')
axes[0].set_ylabel('Bitstrings')
axes[0].yaxis.tick_right()
axes[1].barh([i for (i,j) in costs],[j for (i,j) in costs], align='center')
axes[1].set_xlabel('Cost')
axes[1].set_xticks(list(range(0,20,1)))
plt.show()
