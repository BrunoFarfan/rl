import random

from Evaluator.find_optimal_policies import find_random_subset_of_optimal_policies
from Evaluator.ipe import iterative_policy_evaluation
from Evaluator.value_iteration import value_iteration
from plot.plot_optimal_policies import plot_optimal_policies
from Policies.greedy_policy import greedy_policy
from Policies.uniform_policy import uniform_policy
from Problems.CookieProblem import CookieProblem
from Problems.GamblerProblem import GamblerProblem
from Problems.GridProblem import GridProblem


def get_action_from_user(actions):
    print('Valid actions:')
    for i in range(len(actions)):
        print(f'{i}. {actions[i]}')
    print('Please select an action:')
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def sample_transition(transitions):
    probs = [prob for prob, _, _ in transitions]
    transition = random.choices(population=transitions, weights=probs)[0]
    prob, s_next, reward = transition
    return s_next, reward


def play(problem):
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        actions = problem.get_available_actions(state)
        action = get_action_from_user(actions)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print('Done.')
    print(f'Total reward: {total_reward}')


def play_problem(problem_type: str, uniform: bool = False, greedy: bool = False):
    assert uniform or greedy, 'At least one of uniform or greedy must be True'

    if problem_type == 'gambler':
        gamma = 1.0
        param_list = [0.25, 0.4, 0.55]
    elif problem_type == 'grid':
        gamma = 1.0
        param_list = range(3, 11)
    elif problem_type == 'cookie':
        gamma = 0.99
        param_list = range(3, 11)
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')

    for param in param_list:
        if problem_type == 'gambler':
            problem = GamblerProblem(param)
            label = f'Gambler p: {param}'
        elif problem_type == 'grid':
            problem = GridProblem(param)
            label = f'Grid size: {param}'
        elif problem_type == 'cookie':
            problem = CookieProblem(param)
            label = f'Cookie param: {param}'

        policy_uniform = {
            s: uniform_policy(problem, s) for s in problem.states if not problem.is_terminal(s)
        }
        V_uniform, t_uniform = iterative_policy_evaluation(problem, gamma, policy_uniform)

        initial_state = problem.get_initial_state()

        if uniform:
            print(
                f'{label}, Uniform V(s0) = {V_uniform[initial_state]:.3f}, Time = {t_uniform:.3f}s'
            )

        policy_greedy = greedy_policy(problem, V_uniform, gamma)
        V_greedy, t_greedy = iterative_policy_evaluation(problem, gamma, policy_greedy)

        if greedy:
            print(f'{label}, Greedy V(s0) = {V_greedy[initial_state]:.3f}, Time = {t_greedy:.3f}s')


def play_value_iteration():
    problems = [
        ('grid', 1.0, range(3, 11)),
        ('cookie', 0.99, range(3, 11)),
        ('gambler', 1.0, [0.25, 0.4, 0.55]),
    ]

    for problem_type, gamma, param_list in problems:
        for param in param_list:
            if problem_type == 'grid':
                problem = GridProblem(param)
                label = f'Grid size: {param}'
            elif problem_type == 'cookie':
                problem = CookieProblem(param)
                label = f'Cookie param: {param}'
            elif problem_type == 'gambler':
                problem = GamblerProblem(param)
                label = f'Gambler p: {param}'
            else:
                raise ValueError('Invalid problem type.')

            V_opt, pi_opt, t = value_iteration(problem, gamma)
            initial_state = problem.get_initial_state()
            print(f'{label}, Optimal V(s0) = {V_opt[initial_state]:.3f}, Time = {t:.3f}s')


def analyze_gambler_multiple_optimal_policies(
    p: float = 0.25, gamma: float = 1.0, save_path: str = None
):
    problem = GamblerProblem(p)

    # obtener V*
    V_opt, pi_opt, _ = value_iteration(problem, gamma)

    # encontrar todas las politicas optimas
    states, all_policies = find_random_subset_of_optimal_policies(problem, V_opt, gamma)

    # graficar
    plot_optimal_policies(
        states,
        all_policies,
        title=f'Políticas óptimas en GamblerProblem (p={p})',
        save_path=save_path,
    )


if __name__ == '__main__':
    # play_problem('grid', uniform=False, greedy=True)
    # play_problem('cookie', uniform=False, greedy=True)
    # play_problem('gambler', uniform=False, greedy=True)
    # play_value_iteration()
    analyze_gambler_multiple_optimal_policies(p=0.55)
