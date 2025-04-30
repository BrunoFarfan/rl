from Problems.AbstractProblem import AbstractProblem


def greedy_policy(problem: AbstractProblem, V: dict, gamma: float) -> dict:
    policy = {}
    for s in problem.states:
        if problem.is_terminal(s):
            continue
        best_value = float('-inf')
        best_actions = []
        for a in problem.get_available_actions(s):
            q = 0.0
            for prob, s_next, reward in problem.get_transitions(s, a):
                q += prob * (reward + gamma * V[s_next])
            if q > best_value:
                best_value = q
                best_actions = [a]
            elif q == best_value:
                best_actions.append(a)

        best_action = best_actions[0]
        policy[s] = {best_action: 1.0}
    return policy
