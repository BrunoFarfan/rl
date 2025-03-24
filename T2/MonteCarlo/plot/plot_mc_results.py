import matplotlib.pyplot as plt


def plot_blackjack_results(
        results,
        eval_every: int,
        total_episodes: int,
        filename="blackjack_results.png",
        save: bool = False,
    ):
    x = [1] + list(range(eval_every, total_episodes + 1, eval_every))
    for i, r in enumerate(results):
        plt.plot(x, r, label=f"Run {i+1}")
    plt.xlabel("Episodios")
    plt.ylabel("Retorno promedio")
    plt.title("Blackjack - Monte Carlo epsilon-soft control")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(filename)

    plt.show()

def plot_cliff_results(
        results,
        eval_every: int,
        total_episodes: int,
        filename="cliff_results.png",
        save: bool = False,
    ):
    x = [1] + list(range(eval_every, total_episodes + 1, eval_every))
    for i, r in enumerate(results):
        plt.plot(x, r, label=f"Run {i+1}")
    plt.xlabel("Episodios")
    plt.ylabel("Retorno desde estado inicial")
    plt.title("Cliff - Monte Carlo epsilon-soft control")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(filename)

    plt.show()
