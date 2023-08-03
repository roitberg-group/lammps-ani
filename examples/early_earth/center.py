import argparse
import os
import pytraj as pt

# Set up argument parser
parser = argparse.ArgumentParser(description='Map atoms into the PBC box.')
parser.add_argument('trajectory', help='Path to the trajectory file')
parser.add_argument('topology', help='Path to the topology file (if necessary)')
args = parser.parse_args()

# Load the trajectory
traj = pt.load(args.trajectory, args.topology)

# Apply autoimage to put everything back into the box
traj = pt.autoimage(traj)

# Create the output file name
base, extension = os.path.splitext(args.trajectory)
output_file = base + '.centered' + extension

# Save the trajectory
pt.write_traj(output_file, traj, overwrite=True)

