import numpy as np
from Gridworld_in_code import standard_environment, negative_environment
from Iterative_Policy_Evaluation_in_code import print_values, print_policy

GAMMA = 0.9
ALL_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == '__main__':
    # init env
    env = negative_environment(step_cost=-1.0)

    # print rewards
    print('Rewards: ')
    print_values(env.rewards, env)

    # init random policy (Step 1)
    policy = {}
    for s in env.actions.keys():
        policy[s] = np.random.choice(ALL_ACTIONS)
    print('Initial Policy: ')
    print_policy(policy, env)

    #init V (Step 1)
    V = {}
    states = env.all_states()
    for s in states:
        if s in env.actions:
            V[s] = np.random.random()
        else:  # term state
            V[s] = 0

    #repeat until policy doesn't change
    while True:

        # Copied from Iter_Pol_Eval (Step 2)
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]  # save current V[s]

                # V(s) only nonzero if not in terminal state
                new_v = 0
                if s in policy:
                    for a in ALL_ACTIONS:  # sum over all a
                        if a == policy[s]:
                            p = 0.5
                        else:
                            p = 0.5 / 3

                        env.set_pos(s)

                        r = env.move(a)

                        new_v += p * (r + GAMMA * V[env.return_pos()])

                    V[s] = new_v
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))

            if biggest_change < 1e-3:  # threshold for convergence
                break

        # Pol Iteration (Step 3)
        is_policy_converged = True

        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None

                best_value = float('-inf')
                # Loop through all possible actions to find best current action
                for a in ALL_ACTIONS:
                    v = 0
                    for a2 in ALL_ACTIONS:  # sum over all actions
                        if a == a2:
                            p = 0.5
                        else:
                            p = 0.5 / 3
                        env.set_pos(s)
                        r = env.move(a2)
                        v += p * (r + GAMMA * V[env.return_pos()])
                    if v > best_value:
                        best_value = v
                        new_a = a

                policy[s] = new_a

                if old_a != new_a:
                    is_policy_converged = False

        if is_policy_converged:
            break

    print('Values: ')
    print_values(V, env)

    print('Policy: ')
    print_policy(policy, env)

print('test')