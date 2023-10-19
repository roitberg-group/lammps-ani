import math
import os
import numpy as np
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Divide and submit jobs in groups.')
parser.add_argument('--num_groups', type=int, default=12, help='Number of groups to split the total jobs into.')
args = parser.parse_args()

# Define the range and spacing for phi and psi
phi_values = np.linspace(-np.pi, np.pi, 25, endpoint=False)
psi_values = np.linspace(-np.pi, np.pi, 25, endpoint=False)

log_dir = 'logs-umbrella'
os.makedirs(log_dir, exist_ok=True)

# Total number of jobs
total_jobs = len(phi_values) * len(psi_values)
# Jobs per group, ensuring even distribution of jobs across groups
jobs_per_group = math.ceil(total_jobs / args.num_groups)

group_count = 0
job_count = 0

# Function to read and get slurm template
def get_slurm_template():
    with open('submit.lmp.expanse.template.sh', 'r') as file:
        slurm_template = file.read()
    return slurm_template

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

        command = f"python run_one.py alanine-dipeptide.npt.data --kokkos --num_gpus=1 --input_file=in.lammps --log_dir={log_dir} --ani_model_file='ani2x_solvated_alanine_dipeptide.pt' --run_name=umbrella --ani_num_models=1 --timestep=0.5 --label={label} --run\n"

        # If it's the first job in the group, create a new slurm file with slurm template
        if job_count % jobs_per_group == 0:
            with open(f'tmp_submit_group_{group_count:02d}.sh', 'w') as file:
                file.write(get_slurm_template() + '\n')

        # Append command to the current group's slurm file
        with open(f'tmp_submit_group_{group_count:02d}.sh', 'a') as file:
            file.write(command)

        # If it's the last job in the group, increment the group_count
        if job_count % jobs_per_group == jobs_per_group - 1:
            group_count += 1
            if group_count >= args.num_groups:
                group_count = 0  # Reset group count if it reaches the maximum number of groups

        job_count += 1


# After all jobs are divided among groups, submit the slurm files
for i in range(args.num_groups):
    os.system(f'sbatch tmp_submit_group_{i:02d}.sh')
