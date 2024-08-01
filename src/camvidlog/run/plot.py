import argparse

import pandas as pd
import seaborn as sns

# figure size in inches at 100 dpi
sns.set_theme(rc={"figure.figsize": (19.20, 10.80)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()
    filenames: tuple[str] = tuple(args.filename)
    for filename in filenames:
        filename_out = filename + ".png"
        data = pd.read_csv(filename)

        plot = sns.lineplot(data, x="frame_no", y="mask.mean", hue="res")
        fig = plot.figure
        fig.axes[0].set_ylim(0, 1)
        fig.savefig(filename_out)
        fig.clf()
