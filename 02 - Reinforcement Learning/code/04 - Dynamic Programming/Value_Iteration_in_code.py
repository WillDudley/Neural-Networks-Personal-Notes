import numpy as np
from Gridworld_in_code import standard_environment, negative_environment
from Iterative_Policy_Evaluation_in_code import print_values, print_policy

GAMMA = 0.9
ALL_ACTIONS = ('U', 'D', 'L', 'R')

# Remember, this is deterministic, ie. all p(s',r|s,a) = 1 or 0

if __name__ == '__main__':

    env = negative_environment()

    print('Rewards:')
    print_values(env.rewards, env)

    policy = {}
    for state in env.actions.keys():
        policy[state] = np.random.choice(ALL_ACTIONS)

    print('Initial Policy:')
    print_policy(policy, env)

    V = {}
    states = env.all_states()
    for state in states:
        if state in env.actions:
            V[state] = np.random.random()
        else:
            V[state] = 0

    while True:

        biggest_change = 0

        for state in states:
            old_V = V[state]

            if state in policy:
                highest_V = float('-inf')

                for action in ALL_ACTIONS:

                    env.set_pos(state)

                    reward = env.move(action)

                    current_V = reward + GAMMA * V[env.return_pos()]

                    if current_V > highest_V:
                        highest_V = current_V

                V[state] = highest_V

                biggest_change = max(biggest_change, np.abs(old_V - V[state]))

        if biggest_change < 0.001:
            break

    for state in policy.keys():

        best_action = None
        highest_V = float('-inf')

        for action in ALL_ACTIONS:

            env.set_pos(state)

            reward = env.move(action)

            current_V = reward + GAMMA * V[env.return_pos()]

            if current_V > highest_V:
                highest_V = current_V
                best_action = action

        policy[state] = best_action

    print('Values:')
    print_values(V, env)

    print('Policy:')
    print_policy(policy, env)
