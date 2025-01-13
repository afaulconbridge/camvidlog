import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer


def plot(filename_in: str | Path, filename_out: str | None = None) -> None:
    data = pd.read_csv(filename_in)
    if filename_out is None:
        filename_out = Path(filename_in).with_suffix(".png")

    # data = data[data.label.isin(("deer", "cat"))]

    # model_id,frame_no,label,score
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
    fig.set_size_inches(18.5, 10.5)
    sns.lineplot(data, x="frame_no", y="score", style="label", hue="label", ax=ax1)
    sns.lineplot(data, x="frame_no", y="score_bg", style="label", hue="label", ax=ax2)
    sns.lineplot(data, x="frame_no", y="score_bg_sub", style="label", hue="label", ax=ax3)

    fig.savefig(filename_out)
    fig.clf()

    # cleanup
    plt.close()


app = typer.Typer()


@app.command()
def setup(filenames: list[str]) -> None:
    logging.basicConfig(level=logging.INFO)

    for filename in filenames:
        plot(filename)


if __name__ == "__main__":
    app()
