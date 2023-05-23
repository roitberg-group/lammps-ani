import argparse
import matplotlib.pyplot as plt
from ase.io import read


def read_log_file(log_file):
    data = {"time": [], "temperature": [], "volume": [], "density": [], "pressure": []}
    with open(log_file, "r") as f:
        for line in f:
            if line.startswith("Time"):
                continue
            cols = line.split()
            if len(cols) == 8:
                time, etot, epot, ekin, temp, volume, density, pressure = map(
                    float, cols
                )
                data["time"].append(time)  # store time in ps
                data["temperature"].append(temp)
                data["volume"].append(volume)
                data["density"].append(density)
                data["pressure"].append(pressure)
    return data


def plot_data(data, output_filename):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(8, 16))

    # Plot volume
    ax1.plot(data["time"], data["volume"], label="Volume", color="blue")
    ax1.set_ylabel("Volume (A^3)")
    ax1.legend(loc="upper right")
    ax1.set_title("Volume over Time")

    # Plot density
    ax2.plot(data["time"], data["density"], label="Density", color="green")
    ax2.set_ylabel("Density (g/cm^3)")
    ax2.legend(loc="upper right")
    ax2.set_title("Density over Time")

    # Plot temperature
    ax3.plot(data["time"], data["temperature"], label="Temperature", color="red")
    ax3.set_xlabel("Time (ps)")
    ax3.set_ylabel("Temperature (K)")
    ax3.legend(loc="upper right")
    ax3.set_title("Temperature over Time")

    # Plot pressure
    ax4.plot(data["time"], data["pressure"], label="Pressure", color="purple")
    ax4.set_xlabel("Time (ps)")
    ax4.set_ylabel("Pressure (atm)")
    ax4.legend(loc="upper right")
    ax4.set_title("Pressure over Time")

    fig.suptitle('NPT Simulation of Water using ASE', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)  # Save plot as an image
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=str, help="Path to the log file")
    args = parser.parse_args()

    data = read_log_file(args.log_file)

    output_filename = args.log_file.rsplit(".", 1)[0] + ".png"
    plot_data(data, output_filename)

    print("saved plot to", output_filename)


if __name__ == "__main__":
    main()
