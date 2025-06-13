#!/usr/bin/python3

import qiskit
from qiskit_aer import UnitarySimulator
from qiskit.circuit.library import XGate, ZGate
import numpy as np

# LCU register
lcuRegister = qiskit.QuantumRegister(2, 'LCU')
# Work register
workRegister = qiskit.QuantumRegister(2, '\psi')

myCircuit = qiskit.QuantumCircuit(workRegister,lcuRegister)

# PREP+ operation
myCircuit.h(lcuRegister)

# SELECT operation
myCircuit.append(XGate().control(num_ctrl_qubits=2,ctrl_state='11'),[*lcuRegister, workRegister[0]])
myCircuit.append(XGate().control(num_ctrl_qubits=2,ctrl_state='01'),[*lcuRegister, workRegister[1]])
myCircuit.append(ZGate().control(num_ctrl_qubits=2,ctrl_state='10'),[*lcuRegister, workRegister[0]])

# PREP operation
myCircuit.h(lcuRegister)

# Simulate circuit
mySimulator = UnitarySimulator()
result = mySimulator.run(myCircuit.decompose(reps=2)).result()
# Extract subspace of successfully measuring LCU qubits as 0
# and multiply by submormalization factor 4
print(np.array(result.get_unitary().data[0:4,0:4]).round(10)*4)
