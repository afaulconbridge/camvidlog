import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# figure size in inches at 100 dpi
sns.set_theme(rc={"figure.figsize": (19.20, 10.80)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()
    filenames_videos: tuple[str] = tuple(args.filename)
    for filename_video in filenames_videos:
        filename_out = filename_video + ".png"
        data = pd.read_csv(filename_video+".csv")
        data_ai = pd.read_csv(filename_video+".ai.csv")

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

        sns.lineplot(data_ai, x="frame_no", y="hits.0.score", ax=ax1)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, 900)

        sns.lineplot(data, x="frame_no", y="mask.mean", hue="res", ax=ax2)
        ax2.set_ylim(0, 0.25)
        ax2.set_xlim(0, 900)

        fig.savefig(filename_out)
        fig.clf()
