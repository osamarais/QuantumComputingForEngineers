import numpy as np
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
# from qiskit_aer import UnitarySimulator, StatevectorSimulator
# from qiskit.primitives import Estimator, Sampler
from qiskit_aer.primitives import SamplerV2 as Sampler, EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import SGate
from qiskit.circuit.random import random_circuit

# Create a random quantum circuit
seed = 12345
n = 4
circ = random_circuit(n, n*2, seed=seed)

# Define some random observables as Pauli operators
num_paulis = 10
ops = []
pauli_pool = ['I','X','Y','Z']
for _i in range(num_paulis):
    pauli_string = ''
    for _j in range(n):
        pauli_string = pauli_string + pauli_pool[random.randint(0,3)]
    ops.append(SparsePauliOp(pauli_string,random.random()))
# Combine into one operator
H = sum(ops)

# Create circuits corresponding to Pauli string diagonalizations:
# Applying the appropriate eigenvectors to each qubit for the corresponding Pauli gate in the Pauli string
obs_circs = []
for pauli_string in H.to_list():
    pauli_circ = QuantumCircuit(n)
    for index, pauli in enumerate(pauli_string[0]):
        if pauli=='X':
            pauli_circ.h(n-1-index)
        elif pauli=='Y':
            pauli_circ.append(SGate().inverse(),[n-1-index])
            pauli_circ.h(n-1-index)
    obs_circs.append((pauli_circ,pauli_string))

# Sample these circuits and package into a tuple with 
# quasiprobabilities, Pauli strings, and coefficients of the Pauli strings
sampler = Sampler()
qp_ps_coeff = []
for obs_circ in obs_circs:
    newcirc = circ.compose(obs_circ[0])
    newcirc.measure_all()
    job = sampler.run([newcirc.decompose()],shots=10_000_000)
    result = job.result()[0].data.meas.get_counts()
    # Get the quasi-probabilities from the result
    # The result is a dictionary with bitstrings as keys and their counts as values
    total_counts = sum(result.values())
    for key in result:
        result[key] = result[key] / total_counts
    q_p = result
    qp_ps_coeff.append((q_p,obs_circ[1][0],obs_circ[1][1]))

# Estimate the expectation values
evs = []
for _i in qp_ps_coeff:
    # Unpack tuple
    qp = _i[0]
    ps = _i[1]
    coeff = _i[2].real

    # Create bitstring from Pauli string
    # Any qubit with a Pauli applied will be marked as 1 in the bitstring
    pauli_bs = ''
    for _j in ps:
        if (_j == 'I'):
            pauli_bs = pauli_bs + '0'            
        else:
            pauli_bs = pauli_bs + '1'            
    print(f'Pauli string: {ps}   Bitstring: {pauli_bs}\n')
    print(f'Quasi-probs: {qp}  \n')
    
    # Expectation value is computed as a sum over all the 
    # quasiprobabilities of the sampled bitstrings
    ev = 0
    for index_bs in qp:
        # Get quasiprobability
        _k = qp[index_bs]
        print('Index BS   : ' + index_bs)
        print('Pauli BS   : ' + pauli_bs)
        # Perform bitwise AND with these two
        # This will tell us how many -ve signs are being picked up
        bs_mask_ps = f'{{0:0{n}b}}'.format(int(index_bs,2) & int(pauli_bs,2))
        print('Bitwise AND: ' + bs_mask_ps)
        # Compute parity of this final bitstring
        # This will give the overall number of -ve signs
        parity = 1
        for _l in bs_mask_ps:
            if _l=='1':
                parity = parity*-1
        print(f'Parity : {parity}')
        print()
        # Compute the contribution of the sampled bitstring + its quasiprobability
        # towards the expectation value
        ev = ev + parity * _k
    evs.append(coeff * ev)
    print()
    print()
    
# Print out expectation values of each Pauli string
print(np.array(evs))

# Estimate observables using Qiskit's Estimator method and compare
estimator = Estimator()
job = estimator.run([(circ.decompose(),o) for o in H])
qiskit_estimator_result = [_result.data.evs for _result in job.result()]
print(qiskit_estimator_result)

# Compute total error for each expectation value
print(f'Total error for each expectation value: {np.abs(np.sum(evs) - np.sum(qiskit_estimator_result))}')