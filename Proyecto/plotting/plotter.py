import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(
    results: dict[str, np.ndarray],
    title: str = 'Rendimiento promedio por episodio',
    xlabel: str = 'Episodio',
    ylabel: str = 'Retorno promedio',
    y_min: float = None,
    y_max: float = None,
    figsize=(16, 9),
    legend_loc='upper left',
    save_path: str = None,
    vertical_line_x: float = None,
):
    """Plot learning curves for multiple algorithms.

    Args:
        results (dict[str, np.ndarray]): Dictionary where keys are algorithm names
        and values are lists of experiments (seeds), where each experiment is a list
        of returns per episode.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        y_min (float, optional): Minimum value for the Y-axis.
        y_max (float, optional): Maximum value for the Y-axis.
        figsize (tuple, optional): Size of the figure.
        legend_loc (str, optional): Location of the legend.
        vertical_line_x (float, optional): X position for a vertical dashed red line.

    """
    plt.figure(figsize=figsize)
    for label, runs in results.items():
        mean = runs.mean(axis=0)
        std = runs.std(axis=0)

        x = np.arange(len(mean))
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)

    if y_min is not None:
        plt.ylim(bottom=y_min)
    if y_max is not None:
        plt.ylim(top=y_max)

    if vertical_line_x is not None:
        plt.axvline(x=vertical_line_x, color='red', linestyle='--', alpha=0.7)

    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
        )

    plt.show()
