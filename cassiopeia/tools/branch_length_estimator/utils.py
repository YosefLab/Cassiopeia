import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional


def plot_grid(
    grid,
    yticklabels,
    xticklabels,
    ylabel,
    xlabel,
    figure_file: Optional[str],
    show_plot: str = True,
    title: str = ""
) -> None:
    sns.heatmap(
        grid,
        yticklabels=yticklabels,
        xticklabels=xticklabels,
        mask=np.isneginf(grid),
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if figure_file:
        plt.savefig(fname=figure_file)
    if show_plot:
        plt.show()
