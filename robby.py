# AIDAN GRIEVE -- CS441 ARTIFICIAL INTELLIGENCE -- PROGRAMMING ASSIGNMENT 3
#
# Robby the Robot
## Robby lives in a wonderful 10x10 world filled with cans! He wants to clean up the world in the best way possible, but every day new cans respawn!
## Using reinforcement learning, Robby will learn the best way to clean up the world from any position and any orientation of cans!

import numpy as np
import random as ran

# QMatrix class
#################################################################################


class QMatrix:
    def __init__(self, map_rows, map_cols, discount, eta):
        self.matrix = np.zeros((100, 6))
        self.dict = self.fill_dictionary(map_rows, map_cols)
        self.discount = discount
        self.eta = eta

    def fill_dictionary(self, map_rows, map_cols):
        dict = {}
        list = []
        state_num = map_rows * map_cols
        # def a better way to combine these two loops
        for i in range(0, map_rows):
            for j in range(0, map_cols):
                list.append("[{}, {}]".format(i, j))
        for count in range(0, state_num):
            dict[list[count]] = count
        return dict

    def update_matrix(self, state, action, reward):
        # find state row in matrix using
        # row = dict(state)
        row = self.dict[str(state)]
        match (action):
            case "Pickup-Can":
                col = 0
            case "Move-North":
                col = 1
            case "Move-South":
                col = 2
            case "Move-East":
                col = 3
            case "Move-West":
                col = 4

        # look through next state actions and find the best value
        max_action = 0

        self.matrix[row, col] = self.matrix[row, col] + self.eta * (
            reward + self.discount * max_action - self.matrix[row, col]
        )

    def get_Q_Value(self, state, action):
        match (action):
            case "Pickup-Can":
                col = 0
            case "Move-North":
                col = 1
            case "Move-South":
                col = 2
            case "Move-East":
                col = 3
            case "Move-West":
                col = 4
        row = self.dict[str(state)]
        return self.matrix[row, col]

    def get_State_Row(self, state):
        return self.dict[str(state)]


#################################################################################


# Robby class
#################################################################################


class Robby:
    def __init__(self):
        self.location = place_robby()
        self.reward_total = 0

    def take_Step(self, map, epsilon):
        state = self.location
        # observe
        vals = self.observe_Current_State(map)
        # choose action
        action = self.choose_Action(map, vals, epsilon)
        # perform action
        reward = self.take_Action(action, vals, map)
        # return state-action pair and reward for qmatrix updates
        return [state, action, reward]

    # returns action chosen
    def choose_Action(self, map, values, epsilon):
        # determine exploration vs exploitation
        greedy = True
        roll = ran.random()
        if roll < epsilon:
            greedy = False

        # exploitation
        if greedy:
            cans = []
            empty = []
            best_action = [None, None]
            size = map.shape[0] - 1

            # INSTEAD OF ALL THIS, WE JUST NEED TO COMPARE Q VALUES OF STATES
            # still needs work here, what exactly is the greedy decision to make?
            if values[0][1] == "Can":
                return "Pickup-Can"
            # search for cans in neighbors, the 2nd best option
            for i in range(0, len(values)):
                if values[i][1] == "Can":
                    cans.append(i)
                if values[i][1] == "Empty":
                    empty.append(i)

            if cans != []:
                # choose a random neighbor
                pass
            # possible is empty, check if any neighbors are walls and move to a random empty neighbor
            else:
                pass

        # exploration
        else:
            roll = ran.uniform(0, 4)
            match (roll):
                case 0:
                    return "Move-North"
                case 1:
                    return "Move-South"
                case 2:
                    return "Move-East"
                case 3:
                    return "Move-West"
                case 4:
                    return "Pickup-Can"

    # takes action passed in and returns the reward from the action
    def take_Action(self, action, values, map):
        match action:
            # reward of -5 if crashing into wall
            # each reward total change needs to update q-matrix
            case "Move-North":
                if values[1][1] == "Wall":
                    self.reward_total -= 5
                    return -5
                else:
                    self.location[1] -= 1
            case "Move-South":
                if values[2][1] == "Wall":
                    self.reward_total -= 5
                    return -5
                else:
                    self.location[1] += 1
            case "Move-East":
                if values[3][1] == "Wall":
                    self.reward_total -= 5
                    return -5
                else:
                    self.location[1] += 1
            case "Move-West":
                if values[4][1] == "Wall":
                    self.reward_total -= 5
                else:
                    self.location[1] -= 1
            case "Pickup-Can":
                # reward of 10 if successful
                # reward of -1 if not
                if values[0] == "Can":
                    map[self.location[0], self.location[1]] = 0
                    self.reward_total += 10
                    return 10
                else:
                    self.reward_total -= 1
                    return -1

        return 0

    # looks at current state and returns a list of values of the results of each action, either "Can", "Empty", or "Wall"
    # this may need to change so that we are observing the qmatrix
    # could get the qvalue of each action and take the highest
    def observe_Current_State(self, map):
        values = []

        # current (values[0])
        values.append(
            [
                [self.location[0], self.location[1]],
                self.run_Sensors([self.location[0], self.location[1]], map),
            ]
        )
        # north   (values[1])
        values.append(
            [
                [self.location[0], self.location[1] - 1],
                self.run_Sensors([self.location[0], self.location[1] - 1], map),
            ]
        )
        # south   (values[2])
        values.append(
            [
                [self.location[0], self.location[1] + 1],
                self.run_Sensors([self.location[0], self.location[1] + 1], map),
            ]
        )
        # east    (values[3])
        values.append(
            [
                [self.location[0] + 1, self.location[1]],
                self.run_Sensors([self.location[0] + 1, self.location[1]], map),
            ]
        )
        # west    (values[4])
        values.append(
            [
                [self.location[0] - 1, self.location[1]],
                self.run_Sensors([self.location[0] - 1, self.location[1]], map),
            ]
        )

        return values

    def run_Sensors(self, square_location, map):
        size = map.shape[0] - 1
        if (
            (square_location[0] > size)
            or (square_location[0] < 0)
            or (square_location[1] > size)
            or (square_location[1] < 0)
        ):
            return "Wall"
        elif map[square_location[0], square_location[1]] == 1:
            return "Can"
        else:
            return "Empty"

    def choose_Action(self):
        pass


#################################################################################


# setup environment
#################################################################################


def generate_map(rows, cols):
    new_map = np.zeros((rows, cols))
    for i in range(0, 10):
        for j in range(0, 10):
            roll = ran.uniform(0, 1)
            if roll > 0.5:
                new_map[i, j] = 1

    return new_map


def place_robby():
    return [ran.uniform(0, 9), ran.uniform(0, 9)]


#################################################################################


# main
#################################################################################


def main():
    # values for experiments
    episodes = 5000
    steps = 200
    discount = 0.9
    eta = 0.2
    epsilon = 0.1

    # setup
    grid = 10
    # not right, rows should be states and columns to actions
    q_matrix = QMatrix(grid, grid, discount, eta)

    # learning
    for episode in range(0, episodes):
        map = generate_map(grid, grid)
        curr_Robby = Robby()
        for step in range(0, steps):
            # take step (returns [state, action, reward])
            state_action_reward = curr_Robby.take_Step(map, epsilon)
            # observe new state s_t+1
            new_state_vals = curr_Robby.observe_Current_State(map)
            # get max of new_state vals

            # update Q(s_t, a_t)
            q_matrix.update_matrix(
                state_action_reward[0], state_action_reward[1], state_action_reward[2]
            )
            pass

    # test episodes using QMatrix
    # plot point every 100 episodes
    # calc everage sum of rewards per episode
    for episode in range(0, episodes):
        map = generate_map(grid, grid)
        curr_Robby = Robby()
        for step in range(0, steps):
            pass


if __name__ == "__main__":
    main()

#################################################################################
