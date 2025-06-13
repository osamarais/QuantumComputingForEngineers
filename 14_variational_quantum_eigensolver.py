import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import random_pauli, SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

# Define number of qubits
n = 3

# Define Hamiltonian as a sum of Pauli strings
num_paulis = 10
mypaulis = [SparsePauliOp(random_pauli(n), np.random.rand()) for _i in range(num_paulis)]
hamiltonian = sum(mypaulis)

# Create VQE ansatz
ansatz = EfficientSU2(num_qubits=n,reps=3)

# Define cost function
def cost_function_generator(circuit,observables):
    def cost_function(params):
        estimator = StatevectorEstimator()
        pub = (circuit, observables, params)
        job = estimator.run([pub])
        cost = job.result()[0].data['evs']
        return cost
    return cost_function
    
mycostfunction = cost_function_generator(ansatz,hamiltonian)


# Define a callback function to track progress of optimization
cost_history = []
def callback_function_generator(cost_history, cost_function):
    def callback(theta):
        cost = cost_function(theta)
        cost_history.append(cost)
        return None
    return callback
        
mycallback = callback_function_generator(cost_history, mycostfunction)    


# Create initial guess
theta = np.zeros(ansatz.num_parameters)

# Optimize ansatz
result = minimize(mycostfunction,theta,method='COBYLA',callback=mycallback)

# Get exact minimum eignevalue
exact_min_eig = min(np.linalg.eig(hamiltonian.to_matrix())[0]).real

# Compute errors
error = cost_history - exact_min_eig

# Plot results
fig, axs = plt.subplots(2,sharex=True)
fig.suptitle('VQE Convergence')
axs[0].plot(cost_history)
axs[0].hlines(exact_min_eig,xmin=0,xmax=len(cost_history),colors='r',linestyles='dashed')
axs[0].legend(['VQE', 'Minimum Eigenvalue'])
axs[0].set(ylabel='Cost')
axs[1].plot(error)
axs[1].set_yscale('log')
axs[1].set(xlabel='Iteration',ylabel='Error')
plt.show()
