import matplotlib.pyplot as plt


def plot_optimal_policies(states, policies, title='Políticas óptimas', save_path=None):
    plt.figure(figsize=(10, 6))

    for pi in policies:
        plt.plot(states, pi, marker='o', linewidth=1, alpha=0.7)

    plt.xlabel('Estado')
    plt.ylabel('Apuesta óptima pi(s)')
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
