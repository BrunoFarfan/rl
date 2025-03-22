import time

def value_iteration(problem, gamma, theta=1e-10):
    V = {s: 0.0 for s in problem.states}
    policy = {}

    start = time.time()

    while True:
        delta = 0.0
        for s in problem.states:
            if problem.is_terminal(s):
                continue
            v = V[s]
            max_q = float('-inf')
            for a in problem.get_available_actions(s):
                q = 0.0
                for prob, s_next, reward in problem.get_transitions(s, a):
                    q += prob * (reward + gamma * V[s_next])
                max_q = max(max_q, q)
            V[s] = max_q
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    # derivar la politica optima una vez que V* converge
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
        policy[s] = {best_actions[0]: 1.0}

    elapsed = time.time() - start
    return V, policy, round(elapsed, 3)
