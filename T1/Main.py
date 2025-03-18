import argparse
from BanditEnv import BanditEnv
from agents.ConstantAgent import ConstantAgent
from agents.SimpleAgent import SimpleAgent
from agents.RandomAgent import RandomAgent
from agents.GradientAgent import GradientBanditAgent
from PlotResults import plot_optimal_action_percentage, plot_average_reward
from BanditResults import BanditResults


def testbed(agent_class, agent_name: str, env_mean: int = 0, **agent_kwargs):
    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000

    results = BanditResults()

    for run_id in range(NUM_OF_RUNS):
        bandit = BanditEnv(seed=run_id, mean=env_mean)
        num_of_arms = bandit.action_space
        agent = agent_class(num_of_arms, **agent_kwargs)
        best_action = bandit.best_action

        for step in range(NUM_OF_STEPS):
            action = agent.get_action()
            reward = bandit.step(action)
            agent.learn(action, reward)
            is_best_action = action == best_action
            results.add_result(reward, is_best_action)

        results.save_current_run()

    avg_rewards = results.get_average_rewards()
    avg_optimal_actions = results.get_optimal_action_percentage()

    return agent_name, avg_rewards, avg_optimal_actions



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run bandit experiments with optional parts."
    )
    parser.add_argument(
        "--no-parte-a",
        action="store_false",
        dest="parte_a",
        help="Disable SimpleAgent & RandomAgent experiments."
    )
    parser.add_argument(
        "--no-parte-c",
        action="store_false",
        dest="parte_c",
        help="Disable Optimistic vs. Realistic experiments."
    )
    parser.add_argument(
        "--no-parte-f",
        action="store_false",
        dest="parte_f",
        help="Disable GradientBanditAgent experiments."
    )

    args = parser.parse_args()

    # ------------------------------
    # Run SimpleAgent & RandomAgent experiments
    # ------------------------------
    if args.parte_a:
        results_dict = {}
        agents_to_run = [
            (SimpleAgent, "epsilon = 0 (greedy)", {"epsilon": 0}),
            (SimpleAgent, "epsilon = 0.01", {"epsilon": 0.01}),
            (SimpleAgent, "epsilon = 0.1", {"epsilon": 0.1}),
            (RandomAgent, "Random Agent", {})
        ]

        print("Running SimpleAgent and RandomAgent experiments...")
        for agent_class, agent_name, agent_kwargs in agents_to_run:
            print(f"Running {agent_name}...")
            name, avg_rewards, avg_optimal_actions = testbed(agent_class, agent_name, **agent_kwargs)
            results_dict[name] = (avg_rewards, avg_optimal_actions)

        plot_average_reward(results_dict)
        plot_optimal_action_percentage(results_dict)

    # ------------------------------
    # Run Optimistic vs. Realistic ConstantAgent experiments
    # ------------------------------
    if args.parte_c:
        results_dict = {}

        print("Running Optimistic vs. Realistic experiments...")
        
        print("Optimistic, epsilon-greedy (Q1=0, epsilon=0.1)...")
        name, avg_rewards, avg_optimal_actions = testbed(
            ConstantAgent, "Realistic, epsilon-greedy (Q1=0, epsilon=0.1)", epsilon=0.1, alpha=0.1, initial_Q=0
        )
        results_dict[name] = (avg_rewards, avg_optimal_actions)

        print("Optimistic, epsilon-greedy (Q1=5, epsilon=0)...")
        name, avg_rewards, avg_optimal_actions = testbed(
            ConstantAgent, "Optimistic, greedy (Q1=5, epsilon=0)", epsilon=0.0, alpha=0.1, initial_Q=5
        )
        results_dict[name] = (avg_rewards, avg_optimal_actions)

        plot_optimal_action_percentage(results_dict)

    # ------------------------------
    # Run GradientBanditAgent experiments
    # ------------------------------
    if args.parte_f:
        results_dict = {}

        print("Running Gradient Bandit experiments...")

        agents_to_run = [
            (GradientBanditAgent, "alpha = 0.1$, with baseline", {"alpha": 0.1, "use_baseline": True}),
            (GradientBanditAgent, "alpha = 0.4$, with baseline", {"alpha": 0.4, "use_baseline": True}),
            (GradientBanditAgent, "alpha = 0.1$, without baseline", {"alpha": 0.1, "use_baseline": False}),
            (GradientBanditAgent, "alpha = 0.4$, without baseline", {"alpha": 0.4, "use_baseline": False}),
        ]

        for agent_class, agent_name, agent_kwargs in agents_to_run:
            print(f"Running {agent_name}...")
            name, avg_rewards, avg_optimal_actions = testbed(
                agent_class,
                agent_name,
                **agent_kwargs,
                env_mean=4
            )
            results_dict[name] = (avg_rewards, avg_optimal_actions)

        plot_optimal_action_percentage(results_dict)