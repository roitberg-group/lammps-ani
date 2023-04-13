import argparse
import mdtraj as md
import os

def main(xyz_file, pdb_file):
    # Load the XYZ trajectory with the specified PDB topology
    trajectory = md.load(xyz_file, top=pdb_file)

    # Derive the output NetCDF file name from the XYZ file name
    output_file = os.path.splitext(xyz_file)[0] + '.nc'

    # Save the trajectory in NetCDF format
    trajectory.save_netcdf(output_file)

    print(f"Exported trajectory to: {output_file}")

if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Convert XYZ trajectory to NetCDF format.')
    parser.add_argument('xyz_file', type=str, help='Path to the XYZ trajectory file.')
    parser.add_argument('pdb_file', type=str, help='Path to the corresponding PDB topology file.')

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.xyz_file, args.pdb_file)

