import argparse

import pandas as pd
import seaborn as sns

sns.set_theme()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()
    filenames: tuple[str] = tuple(args.filename)
    for filename in filenames:
        filename_out = filename[:-4] + ".png"
        data = pd.read_csv(filename)
        plot = sns.relplot(data=data, x="frame_no", y="hits.0.score", kind="line")
        fig = plot.figure
        fig.axes[0].set_ylim(0, 1)
        fig.savefig(filename_out)
