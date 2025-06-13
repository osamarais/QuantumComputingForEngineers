#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import StatevectorSimulator

# Create parameter lambda
Lambda = Parameter("lambda")

# Create Circuits for the two cases
circ1 = QuantumCircuit(2)
circ1.h(0)
circ1.crz(Lambda,0,1)

circ2 = QuantumCircuit(2)
circ2.x(1) # Initialize second qubit in state |1>
circ2.h(0)
circ2.crz(Lambda,0,1)

# Obtain statevectors of the end results for various parameter values
simulator = StatevectorSimulator()
lambdas = np.linspace(0,np.pi*2,100)
case_1_statevectors = np.array([simulator.run(circ1.reverse_bits().assign_parameters({"lambda":_lambda})).result().get_statevector() for _lambda in lambdas])
case_2_statevectors = np.array([simulator.run(circ2.reverse_bits().assign_parameters({"lambda":_lambda})).result().get_statevector() for _lambda in lambdas])

# Plot real and imaginary parts of basis states
plt.rcParams['text.usetex'] = True
fig, axs = plt.subplots(4,2,sharex=True, sharey=True)
for _i in range(4):
    axs[_i,0].plot(lambdas, case_1_statevectors[:,_i].real, 'k-')
    axs[_i,0].plot(lambdas, case_1_statevectors[:,_i].imag, 'r--')
    axs[_i,1].plot(lambdas, case_2_statevectors[:,_i].real, 'k-')
    axs[_i,1].plot(lambdas, case_2_statevectors[:,_i].imag, 'r--')
axs[0,0].set_xlim(0,np.pi*2)
axs[0,0].set_ylim(-1,1)
axs[3,0].set_xlabel(r"$ \lambda $")
axs[3,1].set_xlabel(r"$ \lambda $")
axs[0,1].legend(["Re", "Im"])
axs[0,0].set_ylabel(r"$| 00 \rangle$")
axs[1,0].set_ylabel(r"$| 01 \rangle$")
axs[2,0].set_ylabel(r"$| 10 \rangle$")
axs[3,0].set_ylabel(r"$| 11 \rangle$")
axs[0,0].set_title(r"$ \frac{1}{\sqrt{2}} ( | 0 \rangle + | 1 \rangle ) | 0 \rangle $")
axs[0,1].set_title(r"$ \frac{1}{\sqrt{2}} ( | 0 \rangle + | 1 \rangle ) | 1 \rangle $")
plt.show()
