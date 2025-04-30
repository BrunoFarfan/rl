import matplotlib.pyplot as plt
import numpy as np


def plot_blackjack_policy(policy, title='Política aprendida (stick/hit)', save=False):
    usable = np.zeros((10, 10))  # filas: player 12-21, columnas: dealer 1-10
    no_usable = np.zeros((10, 10))

    for player in range(12, 22):
        for dealer in range(1, 11):
            for usable_ace in [True, False]:
                state = (player, usable_ace, dealer)
                if state not in policy:
                    continue
                greedy_action = max(policy[state], key=policy[state].get)
                val = 1 if greedy_action == 'stick' else 0
                row = player - 12
                col = dealer - 1
                if usable_ace:
                    usable[row, col] = val
                else:
                    no_usable[row, col] = val

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mat, label in zip(axs, [no_usable, usable], ['Sin As usable', 'Con As usable']):
        im = ax.imshow(mat, cmap='Greys', origin='lower', vmin=0, vmax=1)
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(1, 11))
        ax.set_yticks(range(10))
        ax.set_yticklabels(range(12, 22))
        ax.set_xlabel('Carta visible del dealer')
        ax.set_ylabel('Puntaje del jugador')
        ax.set_title(label)
    fig.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig('blackjack_policy_greedy.png')
    plt.show()


def plot_cliff_policy(policy, width, height=4):
    arrow = {(1, 0): '↑', (-1, 0): '↓', (0, 1): '→', (0, -1): '←'}

    grid = np.full((height, width), ' ')
    for r in range(height):
        for c in range(width):
            s = (r, c)
            if s not in policy:
                continue
            best_a = max(policy[s], key=policy[s].get)
            grid[r, c] = arrow.get(best_a, '?')

    grid[0, width - 1] = 'G'

    # cliff
    for c in range(1, width - 1):
        grid[0, c] = 'X'

    print('\nPolítica aprendida (Cliff):\n')
    for row in reversed(grid):
        print(' '.join(row))
