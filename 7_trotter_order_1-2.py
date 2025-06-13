#!/usr/bin/python3

import qiskit
from qiskit_aer import UnitarySimulator
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt

min_t = 1
max_t = 5
# Number of Trotter Steps
m = [10,20,40,80,100, 200, 400, 800, 1000, 2000, 4000, 8000, 10000]

# 1st order Trotter
for t in range(min_t,max_t+1):
    # Calculate exact solution classically
    exact_solution = expm( -t * 1j * ( np.array([[0, 1], [1, 0]])
      + np.array([[1, 0], [0, -1]]) ) )

    errors = []
    for r in m:
        # Create a register of 1 qubit
        myQRegister = qiskit.QuantumRegister(1, '\psi')

        # Create a quantum circuit with using myRegister
        myCircuit = qiskit.QuantumCircuit(myQRegister)
        for _r in range(r):
            myCircuit.rx(t*2/r,0)
            myCircuit.rz(t*2/r,0)

        # Simulate the circuit to obtain overall unitary of Trotterization
        mySimulator = UnitarySimulator()
        result = mySimulator.run(myCircuit).result()
        finalUnitary = result.get_unitary()

        # Compare circuit unitary with exact unitary
        errors.append( np.linalg.norm(finalUnitary - exact_solution,2) )

    plt.loglog(m,errors)
    
# 2nd order Trotter
for t in range(min_t,max_t+1):
    # Calculate exact solution classically
    exact_solution = expm( -t * 1j * ( np.array([[0, 1], [1, 0]])
      + np.array([[1, 0], [0, -1]]) ) )

    errors = []
    for r in m:
        # Create a register of 1 qubit
        myQRegister = qiskit.QuantumRegister(1, '\psi')

        # Create a quantum circuit with using myRegister
        myCircuit = qiskit.QuantumCircuit(myQRegister)
        for _r in range(r):
            myCircuit.rx(t/r,0)
            myCircuit.rz(t*2/r,0)
            myCircuit.rx(t/r,0)
            
        # Simulate the circuit to obtain overall unitary of Trotterization
        mySimulator = UnitarySimulator()
        result = mySimulator.run(myCircuit).result()
        finalUnitary = result.get_unitary()

        # Compare circuit unitary with exact unitary
        errors.append( np.linalg.norm(finalUnitary - exact_solution,2) )

    plt.loglog(m,errors,linestyle='dashed')
plt.xlabel('# of Trotter steps')
plt.ylabel(r'$|error|_2$')
plt.legend(['t = {}, 1st Order'.format(t) for t in range(min_t,max_t+1)]
	+['t = {}, 2nd Order'.format(t) for t in range(min_t,max_t+1)])
plt.show()