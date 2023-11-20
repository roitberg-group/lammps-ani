import pandas as pd
import matplotlib.pyplot as plt


def plot(df, save_to_file=None):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    df[df["formula"] == "O2"].plot.line(x="time", y="counts", ax=ax, label="$O_2$", color="C3")
    df[df["formula"] == "CH4"].plot.line(x="time", y="counts", ax=ax, label="$CH_4$", color="C1")
    df[df["formula"] == "H2O"].plot.line(x="time", y="counts", ax=ax, label="$H_2O$")
    df[df["formula"] == "CO2"].plot.line(x="time", y="counts", ax=ax, label="$CO_2$", color="C2")
    df[df["formula"] == "CO"].plot.line(x="time", y="counts", ax=ax, label="$CO$", color="C9")

    # df[df["formula"] == "H"].plot.line(x="time", y="counts", ax=ax, label="H")
    # df[df["formula"] == "O"].plot.line(x="time", y="counts", ax=ax, label="O")
    plt.legend(loc="center right")
    plt.ylabel("molecule counts")
    plt.xlabel("time (ns)")
    if save_to_file is not None:
        plt.savefig(fname=save_to_file)
    else:
        plt.show()

df = pd.read_csv("analyze/2023-09-25-125628.988569.csv")
plot(df, "analyze/2023-09-25-125628.988569.png")
