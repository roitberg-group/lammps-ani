import re
import argparse
import matplotlib.pyplot as plt
from lammps_logfile import File

# Set the global font size for labels in the figure
plt.rcParams['axes.labelsize'] = 14
# Set the global font size for titles in the figure
plt.rcParams['axes.titlesize'] = 16
# Set the global font size for the tick parameters in the figure
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14


def plot_properties_vs_time(
    log_file, output_filename
):
    # Read the LAMMPS log file using lammps_logfile library
    lammps_log = File(log_file)

    # Regular expression pattern for finding timestep
    with open(log_file, 'r') as file:
        contents = file.read()
    timestep_pattern = re.compile(r"timestep\s+(\d+\.\d+|\d+)")
    matches = timestep_pattern.findall(contents)
    # The final timestep value is usually the last one if multiple timestep settings are present
    timestep = None
    for match in matches:
        timestep = float(match)
    print(f"Final timestep: {timestep}")
    assert timestep is not None, "timestep not found in log file"

    # Extract the "Step" data from the log file
    steps = lammps_log.get("Step")
    # Convert steps to time using the provided timestep (in fs)
    # and convert time from fs to ps by multiplying by 1e-3
    time = [timestep * step * 1e-3 for step in steps]

    # Extract the properties data (volume, density, temperature) from the log file
    volumes = lammps_log.get("Volume")
    densities = lammps_log.get("Density")
    temperatures = lammps_log.get("Temp")
    pressures = lammps_log.get("Press")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(8, 12))

    # Create a plot of Volume vs. Time
    ax1.plot(time, volumes, label="Volume", color="blue")
    ax1.set_ylabel("Volume (A^3)")
    ax1.legend(loc="upper right")
    ax1.set_title("Volume vs. Time")

    # Create a plot of Density vs. Time
    ax2.plot(time, densities, label="Density", color="green")
    ax2.set_ylabel("Density (g/cm^3)")
    ax2.legend(loc="upper right")
    ax2.set_title("Density vs. Time")

    # Create a plot of Temperature vs. Time
    ax3.plot(time, temperatures, label="Temperature", color="red")
    ax3.set_xlabel("Time (ps)")
    ax3.set_ylabel("Temperature (K)")
    ax3.legend(loc="upper right")
    ax3.set_title("Temperature vs. Time")

    # Create a plot of Pressure vs. Time
    ax4.plot(time, pressures, label="Pressure", color="purple")
    ax4.set_xlabel("Time (ps)")
    ax4.set_ylabel("Pressure (atm)")
    ax4.legend(loc="upper right")
    ax4.set_title("Pressure vs. Time")

    fig.suptitle('NPT Simulation of Water using LAMMPS', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()


if __name__ == "__main__":
    # Define command-line argument parser
    parser = argparse.ArgumentParser(
        description="Plot Volume, Density and Temperature vs. Time from LAMMPS log file."
    )
    parser.add_argument("log_file", type=str, help="Path to the LAMMPS log file.")

    # Parse command-line arguments
    args = parser.parse_args()

    output_filename = args.log_file.rsplit(".", 1)[0] + ".png"

    # Call the plotting function with the provided log file and timestep
    plot_properties_vs_time(args.log_file, output_filename)

    print("saved plot to", output_filename)
