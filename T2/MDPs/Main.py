import random

from Evaluator.ipe import iterative_policy_evaluation

from Policies.uniform_policy import uniform_policy
from Policies.greedy_policy import greedy_policy

from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem


def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
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
    print("Done.")
    print(f"Total reward: {total_reward}")


def play_gambler_problem(uniform: bool = False, greedy: bool = False):
    for p in [0.25, 0.4, 0.55]:
        problem = GamblerProblem(p)
        gamma = 1.0

        # Política uniforme
        policy_uniform = {
            s: uniform_policy(problem, s) for s in problem.states if not problem.is_terminal(s)
        }
        V_uniform, t = iterative_policy_evaluation(problem, gamma, policy_uniform)
        if uniform:
            initial_state = problem.get_initial_state()
            print(
                f"Gambler p: {p}, Uniform V(s0) = " +
                f"{V_uniform[initial_state]:.3f}, Time = {t:.3f}s"
            )
        # Política greedy
        policy_greedy = greedy_policy(problem, V_uniform, gamma)
        V_greedy, t = iterative_policy_evaluation(problem, gamma, policy_greedy)
        if greedy:
            initial_state = problem.get_initial_state()
            print(
                f"Gambler p: {p}, Greedy V(s0) = " +
                f"{V_greedy[problem.get_initial_state()]:.3f}, Time = {t:.3f}s"
            )


def play_grid_problem(uniform: bool = False, greedy: bool = False):
    for size in range(3, 11):
        problem = GridProblem(size)
        gamma = 1.0

        # Política uniforme
        policy_uniform = {
            s: uniform_policy(problem, s) for s in problem.states if not problem.is_terminal(s)
        }
        V_uniform, t = iterative_policy_evaluation(problem, gamma, policy_uniform)
        if uniform:
            initial_state = problem.get_initial_state()
            print(
                f"Grid size: {size}, Uniform V(s0) = " + 
                f"{V_uniform[initial_state]:.3f}, Time = {t:.3f}s"
            )

        # Política greedy
        policy_greedy = greedy_policy(problem, V_uniform, gamma)
        V_greedy, t = iterative_policy_evaluation(problem, gamma, policy_greedy)
        if greedy:
            initial_state = problem.get_initial_state()
            print(
                f"Grid size: {size}, Greedy V(s0) = " +
                f"{V_greedy[problem.get_initial_state()]:.3f}, Time = {t:.3f}s"
            )



def play_problem(problem_type: str, uniform: bool = False, greedy: bool = False):
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
        raise ValueError(f"Unknown problem type: {problem_type}")

    for param in param_list:
        if problem_type == 'gambler':
            problem = GamblerProblem(param)
            label = f"Gambler p: {param}"
        elif problem_type == 'grid':
            problem = GridProblem(param)
            label = f"Grid size: {param}"
        elif problem_type == 'cookie':
            problem = CookieProblem(param)
            label = f"Cookie param: {param}"

        policy_uniform = {
            s: uniform_policy(problem, s)
            for s in problem.states
            if not problem.is_terminal(s)
        }
        V_uniform, t_uniform = iterative_policy_evaluation(problem, gamma, policy_uniform)

        if uniform:
            initial_state = problem.get_initial_state()
            print(
                f"{label}, Uniform V(s0) = "
                f"{V_uniform[initial_state]:.3f}, Time = {t_uniform:.3f}s"
            )

        policy_greedy = greedy_policy(problem, V_uniform, gamma)
        V_greedy, t_greedy = iterative_policy_evaluation(problem, gamma, policy_greedy)

        if greedy:
            initial_state = problem.get_initial_state()
            print(
                f"{label}, Greedy V(s0) = "
                f"{V_greedy[initial_state]:.3f}, Time = {t_greedy:.3f}s"
            )


if __name__ == '__main__':
    play_problem('grid', uniform=False, greedy=True)
    play_problem('cookie', uniform=False, greedy=True)
    play_problem('gambler', uniform=False, greedy=True)

