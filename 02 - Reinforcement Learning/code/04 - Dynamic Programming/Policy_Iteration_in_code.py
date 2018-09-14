import numpy as np
from Gridworld_in_code import standard_environment, negative_environment
from Iterative_Policy_Evaluation_in_code import print_values, print_policy

GAMMA = 0.9

if __name__ == '__main__':
    # init env
    env = negative_environment()

    # print rewards
    print('Rewards: ')
    print_values(env.rewards, env)

    # init random policy (Step 1)
    policy = {}
    for s in env.actions.keys():
        policy[s] = np.random.choice(env.actions[s])
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
                if s in policy:
                    # <major difference> (simpler because only one potential action per state/position)
                    a = policy[s]
                    env.set_pos(s)
                    r = env.move(a)
                    V[s] = r + GAMMA * V[env.return_pos()]
                    # </major difference>
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))

            if biggest_change < 0.001:  # threshold for convergence
                break

        # Pol Iteration (Step 3)
        is_policy_converged = True

        for s in env.actions.keys():
            if s in policy:
                old_policy = policy[s]
                new_policy = None

                best_value = float('-inf')
                # Loop through all possible actions to find best current action
                for a in env.actions[s]:
                    env.set_pos(s)
                    r = env.move(a)
                    v = r + GAMMA * V[env.return_pos()]
                    if v > best_value:
                        best_value = v
                        new_policy = a

                policy[s] = new_policy

                if old_policy != policy[s]:
                    is_policy_converged = False

        if is_policy_converged:
            break

    print('Values: ')
    print_values(V, env)

    print('Policy: ')
    print_policy(policy, env)