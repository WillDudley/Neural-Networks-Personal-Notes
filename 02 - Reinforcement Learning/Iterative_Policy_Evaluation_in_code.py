import numpy as np
from Gridworld_in_code import standard_environment


def print_values(V, g):
    for i in range(g.height):
        print('\n------------------------')
        for j in range(g.width):
            v = V.get((i, j), 0)
            print('{:+.2f}|'.format(v), end='')
            # if v >= 0:
            #     print(' {}|'.format(v), end='')
            # else:
            #     print('{}|'.format(v), end='')
    print('\n------------------------')


def print_policy(P, g):
    for i in range(g.height):
        print('\n----------------')
        for j in range(g.width):
            a = P.get((i, j), 0)
            print(' {} |'.format(a), end='')
    print('\n----------------')


if __name__ == '__main__':
    # init env, get all positions
    env = standard_environment()
    states = env.all_states()

    ### Random Uniform Policy ###
    # init V(s) = 0
    V = {}
    for s in states:
        V[s] = 0

    # set discount factor
    gamma = 1.0

    # repeat until convergence
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]  # save current V[s]

            # V(s) only nonzero if not in terminal state
            if s in env.actions:

                new_v = 0  # init new v
                p_a = 1 / len(env.actions[s])  # probability of taking action a (each action has equal probability)

                for a in env.actions[s]:  # for every possible action in state s
                    env.set_pos(s)
                    r = env.move(a)  # remember env.move() returns a reward
                    new_v += p_a * (r + gamma * V[env.return_pos()])  # See chapter 03 "continued investigation of V(s)" - as environment is deterministic, p(r,s'|s,a)=1
                V[s] = new_v  # essentially performs the Bellman equation
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < 0.001:  # threshold for convergence
            break
    print('\nValues for uniformly random actions:', end='')
    print_values(V, env)
    print('\n')


    ### Completely Deterministic (Fixed) Policy ###
    # Define the fixed policy
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    print_policy(policy, env)

    # init V(s) = 0
    V = {}
    for s in states:
        V[s] = 0

    # set discount factor
    gamma = 0.9

    # repeat until convergence
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
                V[s] = r + gamma * V[env.return_pos()]
                # </major difference>
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < 0.001:  # threshold for convergence
            break
    print('\nValues for deterministic actions:', end='')
    print_values(V, env)
    print('\n')
