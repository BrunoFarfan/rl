from Problems.AbstractProblem import AbstractProblem


def uniform_policy(problem: AbstractProblem, state) -> dict:
    actions = problem.get_available_actions(state)
    if not actions:
        return {}
    prob = 1.0 / len(actions)
    return {a: prob for a in actions}
