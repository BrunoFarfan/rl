import matplotlib.pyplot as plt


def plot_blackjack_results(all_returns, eval_points, save=False):
    plt.figure(figsize=(10, 6))
    for idx, run_returns in enumerate(all_returns):
        plt.plot(eval_points, run_returns, label=f'Run {idx + 1}')
    plt.xlabel('Episodios de entrenamiento')
    plt.ylabel('Retorno promedio')
    plt.title('Rendimiento de la política greedy durante el entrenamiento')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig('blackjack_results.png')

    plt.show()


def plot_cliff_results(all_returns, eval_points, save=False):
    plt.figure(figsize=(10, 6))
    for idx, run_returns in enumerate(all_returns):
        plt.plot(eval_points, run_returns, label=f'Run {idx + 1}')
    plt.xlabel('Episodios de entrenamiento')
    plt.ylabel('Retorno de una ejecución')
    plt.title('Rendimiento de la política greedy durante el entrenamiento')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig('cliff_results.png')

    plt.show()
