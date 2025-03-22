import time
from Problems.AbstractProblem import AbstractProblem


def iterative_policy_evaluation(
        problem: AbstractProblem,
        gamma: float,
        policy: callable=None,
        theta: float=1e-10
    ):
    V = {s: 0.0 for s in problem.states}

    start_time = time.time()

    while True:
        delta = 0.0
        for s in problem.states:
            if problem.is_terminal(s):
                continue
            v = V[s]
            new_v = 0.0
            for a, prob_a in policy.get(s, {}).items():
                for prob, s_next, reward in problem.get_transitions(s, a):
                    new_v += prob_a * prob * (reward + gamma * V[s_next])
            delta = max(delta, abs(v - new_v))
            V[s] = new_v
        if delta < theta:
            break

    elapsed_time = time.time() - start_time
    return V, round(elapsed_time, 3)
