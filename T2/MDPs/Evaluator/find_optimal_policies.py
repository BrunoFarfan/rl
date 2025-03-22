from itertools import product
import random
from functools import reduce
from operator import mul


def find_all_optimal_policies(problem, V, gamma):
    states = [s for s in problem.states if not problem.is_terminal(s)]
    optimal_actions_per_state = []

    for s in states:
        best_value = float('-inf')
        best_actions = []
        for a in problem.get_available_actions(s):
            q = 0.0
            for prob, s_next, reward in problem.get_transitions(s, a):
                q += prob * (reward + gamma * V[s_next])

            q = round(q, 5)
            
            if q > best_value:
                best_value = q
                best_actions = [a]
            elif q == best_value:
                best_actions.append(a)
        optimal_actions_per_state.append(best_actions)

    for s in states[:5]:
        print(f"Estado {s}: acciones = {problem.get_available_actions(s)}")

    
    for i, actions in enumerate(optimal_actions_per_state):
        print(f"Estado {i + 1}: {len(actions)} acción(es) óptima(s) → {actions}")

    from functools import reduce
    from operator import mul

    num_policies = reduce(mul, [len(a) for a in optimal_actions_per_state], 1)
    print(f"Total de combinaciones posibles: {num_policies}")

    # todas las combinaciones de politicas optimas
    all_policies = list(product(*optimal_actions_per_state))

    return states, all_policies  # states define el orden de las acciones


def find_random_subset_of_optimal_policies(problem, V, gamma, num_samples=3):
    states = [s for s in problem.states if not problem.is_terminal(s)]
    optimal_actions_per_state = []

    for s in states:
        best_value = float('-inf')
        best_actions = []
        for a in problem.get_available_actions(s):
            q = 0.0
            for prob, s_next, reward in problem.get_transitions(s, a):
                q += prob * (reward + gamma * V[s_next])
            q = round(q, 5)
            if q > best_value:
                best_value = q
                best_actions = [a]
            elif q == best_value:
                best_actions.append(a)
        optimal_actions_per_state.append(best_actions)

    total_policies = reduce(mul, [len(actions) for actions in optimal_actions_per_state], 1)
    print(f"Total de políticas óptimas posibles: {total_policies:,}".replace(",", "."))

    print(f"Generando {num_samples} políticas aleatorias para visualización...")
    policies = []
    for _ in range(num_samples):
        policy = [random.choice(acts) for acts in optimal_actions_per_state]
        policies.append(policy)

    return states, policies
