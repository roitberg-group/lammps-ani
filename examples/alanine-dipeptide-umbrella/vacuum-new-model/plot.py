## import required packages
import os, math
import numpy as np
import matplotlib.pyplot as plt
import simtk.unit as unit
import mdtraj
from FastMBAR import *
from pathlib import Path

log_dir = "logs-umbrella"
plot_file = f"PMF_fast_mbar.png"
# pdb = "alanine-dipeptide.vacuum.pdb"
# topology = mdtraj.load_pdb(pdb)

# psi_indices = [6, 8, 14, 16]
# phi_indices = [4, 6, 8, 14]

m = 25  # changed from 25 to 10
M = m*m

psis = []
phis = []
phi_values = np.linspace(-np.pi, np.pi, 25, endpoint=False)
psi_values = np.linspace(-np.pi, np.pi, 25, endpoint=False)

# Loop over all combinations of phi and psi
for i, phi in enumerate(phi_values):
    for j, psi in enumerate(psi_values):
        label = f'{i:02d}-{j:02d}'
        # traj = mdtraj.load_dcd(f"./output/{traj_dir}/traj_psi_{psi_index}_phi_{phi_index}.dcd", topology)
        data = np.loadtxt(f'{log_dir}/{label}.colvar.dat')
        phis.append(data[:, 1])
        psis.append(data[:, 2])

psi_array = np.squeeze(np.stack(psis))
phi_array = np.squeeze(np.stack(phis))

## compute energy matrix A
K = 100
Temp = 300.00 * unit.kelvin
kbT = unit.BOLTZMANN_CONSTANT_kB * Temp * unit.AVOGADRO_CONSTANT_NA
kbT = kbT.value_in_unit(unit.kilojoule_per_mole)

n = psi_array.shape[1]
A = np.zeros((M, n*M))

psi_array = np.reshape(psi_array, (-1,))
phi_array = np.reshape(phi_array, (-1,))

for index in range(M):
    psi_index = index // m
    phi_index = index % m

    psi_c = psi_values[psi_index]
    phi_c = phi_values[phi_index]

    psi_diff = np.abs(psi_array - psi_c)
    psi_diff = np.minimum(psi_diff, 2*math.pi-psi_diff)

    phi_diff = np.abs(phi_array - phi_c)
    phi_diff = np.minimum(phi_diff, 2*math.pi-phi_diff)

    A[index, :] = 0.5*K*(psi_diff**2 + phi_diff**2)/kbT

## solve MBAR equations
num_conf_all = np.array([n for i in range(M)])
fastmbar = FastMBAR(energy = A, num_conf = num_conf_all, cuda = False, verbose = True)

## compute the reduced energy matrix B
l_PMF = 25
L_PMF = l_PMF * l_PMF
psi_PMF = np.linspace(-math.pi, math.pi, l_PMF, endpoint = False)
phi_PMF = np.linspace(-math.pi, math.pi, l_PMF, endpoint = False)
width = 2*math.pi / l_PMF

B = np.zeros((L_PMF, A.shape[1]))

for index in range(L_PMF):
    psi_index = index // l_PMF
    phi_index = index % l_PMF
    psi_c_PMF = psi_PMF[psi_index]
    phi_c_PMF = phi_PMF[phi_index]

    psi_low = psi_c_PMF - 0.5*width
    psi_high = psi_c_PMF + 0.5*width

    phi_low = phi_c_PMF - 0.5*width
    phi_high = phi_c_PMF + 0.5*width

    psi_indicator = ((psi_array > psi_low) & (psi_array <= psi_high)) | \
                     ((psi_array + 2*math.pi > psi_low) & (psi_array + 2*math.pi <= psi_high)) | \
                     ((psi_array - 2*math.pi > psi_low) & (psi_array - 2*math.pi <= psi_high))

    phi_indicator = ((phi_array > phi_low) & (phi_array <= phi_high)) | \
                     ((phi_array + 2*math.pi > phi_low) & (phi_array + 2*math.pi <= phi_high)) | \
                     ((phi_array - 2*math.pi > phi_low) & (phi_array - 2*math.pi <= phi_high))

    indicator = psi_indicator & phi_indicator
    B[index, ~indicator] = np.inf

## compute PMF using the energy matrix B
results = fastmbar.calculate_free_energies_of_perturbed_states(B)
PMF = results['F']


# Define phi and psi grids
phi_values = np.linspace(-180, 180, l_PMF)
psi_values = np.linspace(180, -180, l_PMF)  # Flipped because of np.flipud
phi_grid, psi_grid = np.meshgrid(phi_values, psi_values)

# Plotting
fig = plt.figure(dpi=150)
fig.clf()

# Define data for imshow and contour
data_PMF = PMF.reshape((l_PMF, l_PMF))
flipped_data_PMF = np.flipud(data_PMF)

# imshow
flipped_data_PMF = flipped_data_PMF - flipped_data_PMF.min()
im = plt.imshow(flipped_data_PMF, extent = (-180, 180, -180, 180), cmap='RdYlBu_r')

# contour
# contour_levels = [-12, -10, -7, -5, 0, 5, 10, 15, 20, 25]
cont = plt.contour(phi_grid, psi_grid, flipped_data_PMF, levels=15, colors='k', linewidths=1.5)

plt.xlabel(r"$\phi$")
plt.ylabel(r"$\psi$")

# Create a colorbar using the data from imshow
cbar = plt.colorbar(im)

plt.savefig(plot_file)
