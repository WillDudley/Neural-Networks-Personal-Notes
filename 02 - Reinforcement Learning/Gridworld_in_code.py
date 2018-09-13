

class Environment:
    def __init__(self, height, width, start_pos):
        self.height = height
        self.width = width
        self.i = start_pos[0]
        self.j = start_pos[1]
        self.rewards = None
        self.actions = None

    def set_rewards_and_actions(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    def set_pos(self, pos_tuple):
        self.i = pos_tuple[0]
        self.j = pos_tuple[1]

    def return_pos(self):
        return self.i, self.j

    def in_terminal(self):
        return True if self.return_pos() not in self.actions else False

    def move(self, action):
        if action in self.actions[self.return_pos()]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1

        return self.rewards.get(self.return_pos(), 0)

    def game_over(self):
        return self.return_pos() if self.in_terminal() else False

    def all_states(self):
        return set(list(self.actions.keys()) + list(self.rewards.keys()))


def standard_environment():
    # @ @ @ 1
    # @ X @ -1
    # S @ @ @
    g = Environment(3, 4, (0, 2))
    rewards = {
        (0, 3): 1,
        (1, 3): -1
    }
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('D', 'L', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('U', 'L', 'R'),
        (2, 3): ('U', 'L')
    }
    g.set_rewards_and_actions(rewards, actions)
    return g


def negative_environment(step_cost=-0.1):
    g = standard_environment()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost
    })
    return g
