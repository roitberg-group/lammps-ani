import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_ramachandran(traj_file, top_file, timestep, dump_interval):
    # Load the trajectory
    traj = md.load(traj_file, top=top_file)

    # Specify indices for psi and phi dihedral angles
    psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]

    # Compute the dihedral angles, and convert to degrees
    angles = md.compute_dihedrals(traj, [phi_indices, psi_indices])
    angles = angles * (180 / np.pi)

    # Compute time in nanoseconds, assuming input timestep was in femtoseconds
    traj.time = traj.time * timestep * 1e-6 * dump_interval

    # Set up the plot
    plt.figure()

    # Add the histogram using a more pleasing color map and the 2D histogram of angles
    # TBH, I think "RdYlBu_r" is a better color map...
    hist, xedges, yedges, img = plt.hist2d(
        angles[:, 0], angles[:, 1], bins=100, density=True, cmap="OrRd"
    )

    # Add a color bar to the right of the plot
    plt.colorbar()

    # Compute centers of bins
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2

    # Create meshgrid of bin centers
    X, Y = np.meshgrid(x_centers, y_centers)

    # Draw contour lines
    plt.contour(X, Y, hist.T, colors="k", linewidths=0.5)

    # Set the plot limits, labels and title
    plt.xlim([-180, 180])
    plt.ylim([-180, 180])
    plt.xlabel(r"$\Phi$ [degree]")
    plt.ylabel(r"$\Psi$ [degree]")
    plt.title(
        f"Dihedral Map: Alanine dipeptide\n Timestep: {timestep} fs, Total time: {traj.time[-1]} ns"
    )

    # Show the plot
    plt.show()

    # Generate the filename including the timestep value (converted to ns)
    filename = traj_file.rsplit(".", 1)[0] + ".newcmap" + ".png"
    print("Saving figure to {}".format(filename))

    # Save the figure
    plt.savefig(filename)


if __name__ == "__main__":
    # Define command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_file", type=str, help="Path to the trajectory file.")
    parser.add_argument("top_file", type=str, help="Path to the topology file.")
    parser.add_argument(
        "timestep", type=float, help="Timestep used in the simulation (in fs)."
    )
    parser.add_argument("-d", "--dump_interval", type=int, default=100)

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the plotting function with the provided log file and timestep
    plot_ramachandran(
        args.traj_file,
        args.top_file,
        args.timestep,
        args.dump_interval,
    )
