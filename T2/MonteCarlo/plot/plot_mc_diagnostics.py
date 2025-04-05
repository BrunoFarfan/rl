import numpy as np
import matplotlib.pyplot as plt


def plot_blackjack_visits(state_visits, title="Frecuencia de visitas (Blackjack)", save=False):
    usable = np.zeros((10, 10))
    no_usable = np.zeros((10, 10))

    for (player_total, usable_ace, dealer_card), count in state_visits.items():
        if 12 <= player_total <= 21 and 1 <= dealer_card <= 10:
            row = player_total - 12
            col = dealer_card - 1
            if usable_ace:
                usable[row, col] += count
            else:
                no_usable[row, col] += count

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mat, label in zip(axs, [no_usable, usable], ["Sin As usable", "Con As usable"]):
        im = ax.imshow(mat, cmap='viridis', origin='lower')
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(1, 11))
        ax.set_yticks(range(10))
        ax.set_yticklabels(range(12, 22))
        ax.set_xlabel("Carta visible del dealer")
        ax.set_ylabel("Puntaje del jugador")
        ax.set_title(label)
        fig.colorbar(im, ax=ax)
    fig.suptitle(title)

    plt.tight_layout()
    if save:
        plt.savefig("blackjack_state_visits.png")

    plt.show()


def plot_cliff_visits(state_visits, width=6, height=4, title="Frecuencia de visitas (Cliff)", save=False):
    grid = np.zeros((height, width))
    for (r, c), count in state_visits.items():
        if 0 <= r < height and 0 <= c < width:
            grid[r, c] += count

    plt.figure(figsize=(8, 4))
    plt.imshow(grid, cmap='viridis', origin='lower')
    plt.colorbar(label="N° de visitas")
    plt.title(title)
    plt.xlabel("Columna")
    plt.ylabel("Fila")
    plt.xticks(range(width))
    plt.yticks(range(height))

    if save:
        plt.savefig("cliff_state_visits.png")

    plt.show()
