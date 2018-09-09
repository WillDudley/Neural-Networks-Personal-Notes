

class Agent:
    def __init__(self):
        pass

    def update_state(self):
        pass

    def update_value_function(self):
        pass


class Environment:
    def __init__(self):
        Environment.self = self

    def is_empty(self, i, j):
        pass

    def reward(self, sym):
        pass

    def get_state(self):
        pass

    def game_over(self):
        pass

    def draw_board(self):
        pass


def play_game(p1, p2, env):

    current_player = p2

    while not game_end():
        #switch players
        if current_player == p1:
            current_player = p2
        elif current_player == p2:
            current_player = p1

        #draw board


        #action
        current_player.action(env)

        #update state history storage
        state = env.get_state()
        p1.update_state(state)
        p2.update_state(state)

    #draw board


    #value function update/backprop
    p1.update_value_function()
    p2.update_value_function()

