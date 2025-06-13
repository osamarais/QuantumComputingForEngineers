#!/usr/bin/python3

import numpy as np
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library import RZGate
from qiskit.primitives import StatevectorSampler
from matplotlib import pyplot as plt

min_ancillae = 8
max_ancillae = 17
step = 4
ancillae = range(min_ancillae,max_ancillae,step)

phi = np.arange(-1.5,1.,0.01).round(3)

phi_measured = np.zeros((len(ancillae),len(phi)))
errors = np.zeros((len(ancillae),len(phi)))

mysampler = StatevectorSampler()

for i, m in enumerate(ancillae):
    for j in range(len(phi)):
        myunitary = RZGate(-4*np.pi*phi[j])
        myqpe = PhaseEstimation(m,myunitary)
        myqpe.measure_all()
        pub = (myqpe)
        job = mysampler.run([(pub)],shots=1_00_000)
        result = job.result()[0]
        raw = result.data['meas']
        counts = raw.get_counts()
        maxkey = max(counts, key=counts.get)
        # Compute and store the measured value of QPE
        phi_measured[i,j] = 0
        for _i, bit in enumerate(reversed(maxkey)):
            if bit=='1':
                phi_measured[i,j] += 1/(2**(_i+1))
        # Store the errors
        # Phi has a period of 1, adjust for it when computing errors
        errors[i,j] = np.abs((phi[j] % 1) - phi_measured[i,j])

fig, axs = plt.subplots(2)
for i in range(len(ancillae)):
    axs[0].scatter(phi, np.log10(errors[i,:]))
axs[0].set(xlabel='phi', ylabel='log10( phi_measured - (phi mod 1) )')
axs[0].legend([str(_i) + ' ancillae' for _i in list(ancillae)])
for i in range(len(ancillae)):
    axs[1].scatter(phi % 1, np.log10(errors[i,:]))
axs[1].set(xlabel='phi mod 1', ylabel='log10( phi_measured - (phi mod 1) )')
axs[1].legend([str(_i) + ' ancillae' for _i in list(ancillae)])
fig.set_size_inches(10, 20)
plt.show()
