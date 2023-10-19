import os
import numpy as np

# Define the range and spacing for phi and psi
phi_values = np.linspace(-np.pi, np.pi, 25, endpoint=False)
psi_values = np.linspace(-np.pi, np.pi, 25, endpoint=False)

log_dir = 'logs-umbrella'
os.makedirs(log_dir, exist_ok=True)

# Loop over all combinations of phi and psi
for i, phi in enumerate(phi_values):
    for j, psi in enumerate(psi_values):
        label = f'{i:02d}-{j:02d}'
        # Create a new PLUMED input file for this window
        with open(f'{log_dir}/{label}.plumed.dat', 'w') as file:
            file.write(f'MOLINFO STRUCTURE=alanine-dipeptide.npt.pdb\n')
            file.write(f'phi: TORSION ATOMS=@phi-2\n')
            file.write(f'psi: TORSION ATOMS=@psi-2\n')
            file.write(f'fphi: RESTRAINT ARG=phi KAPPA=100.0 AT={phi}\n')
            file.write(f'fpsi: RESTRAINT ARG=psi KAPPA=100.0 AT={psi}\n')
            file.write(f'PRINT ARG=phi,psi,fphi.bias,fpsi.bias FILE={log_dir}/{label}.colvar.dat STRIDE=100\n')

        command = f"python run_one.py alanine-dipeptide.npt.data --kokkos --num_gpus=1 --input_file=in.lammps --log_dir={log_dir} --ani_model_file='ani2x_solvated_alanine_dipeptide.pt' --run_name=umbrella --ani_num_models=1 --timestep=0.5 --label={label} --run"
        # read slurm template file
        with open('submit.lmp.expanse.template.sh', 'r') as file:
            slurm_template = file.read()
        # add command to template
        slurm_template += '\n' + command + '\n'
        # write slurm file to submit.{label}.sh
        with open(f'tmp_submit_{label}.sh', 'w') as file:
            file.write(slurm_template)
        # submit job
        os.system(f'sbatch tmp_submit_{label}.sh')

