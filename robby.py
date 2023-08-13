# AIDAN GRIEVE -- CS441 ARTIFICIAL INTELLIGENCE -- PROGRAMMING ASSIGNMENT 3
#
# Robby the Robot
## Robby lives in a wonderful 10x10 world filled with cans! He wants to clean up the world in the best way possible, but every day new cans respawn!
## Using reinforcement learning, Robby will learn the best way to clean up the world from any position and any orientation of cans!

import numpy as np
import random as ran
import matplotlib.pyplot as plt

# QMatrix class
#################################################################################


class QMatrix:
    def __init__(self, map_rows, map_cols, discount, eta):
        self.matrix = np.zeros((100, 5))
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

    def update_matrix(self, state, action, reward, next_states):
        # find state row in matrix using
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
        best = None
        best_arr = []
        for i in range(0, len(next_states)):
            if best == None:
                best = next_states[i]
                best_arr = [i]
            else:
                if best < next_states[i]:
                    best = next_states[i]
                    best_arr = [i]
                elif best == next_states[i]:
                    best_arr.append(i)

        # choose between best vals
        if len(best_arr) > 1:
            roll = ran.randrange(0, len(best_arr))
            max_q = next_states[best_arr[roll]]
        else:
            max_q = best

        self.matrix[row, col] += self.eta * (
            reward + (self.discount * max_q) - self.matrix[row, col]
        )
        # print(self.matrix[row, col])

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
        return np.copy(self.matrix[self.dict[str(state)], :])


#################################################################################


# Robby class
#################################################################################


class Robby:
    def __init__(self):
        self.location = place_robby()
        self.reward_total = 0

    def take_Step(self, map, epsilon, qmatrix):
        state = self.location
        # observe
        cur_state_action_vals = self.observe_Current_State(qmatrix)
        # choose action
        action = self.choose_Action(cur_state_action_vals, epsilon)
        # perform action
        reward = self.take_Action(action, map)
        # return state-action pair and reward for qmatrix updates
        return [state, action, reward]

    # returns action chosen
    def choose_Action(self, values, epsilon):
        # determine exploration vs exploitation
        greedy = True
        action_index = 0
        roll = ran.random()
        if roll < epsilon:
            greedy = False

        # exploitation
        if greedy:
            best_val = None
            best_arr = []
            for i in range(0, 5):
                if best_val == None:
                    best_val = values[i]
                    best_arr = [i]
                else:
                    if best_val < values[i]:
                        best_val = values[i]
                        best_arr = [i]
                    elif best_val == values[i]:
                        best_arr.append(i)

            # choose between best vals
            if len(best_arr) > 1:
                roll = ran.randrange(0, len(best_arr))
                action_index = best_arr[roll]
            else:
                action_index = best_arr[0]

            match (action_index):
                case 0:
                    return "Pickup-Can"
                case 1:
                    return "Move-North"
                case 2:
                    return "Move-South"
                case 3:
                    return "Move-East"
                case 4:
                    return "Move-West"

        # exploration
        else:
            roll = ran.randrange(0, 5)
            match (roll):
                case 0:
                    return "Pickup-Can"
                case 1:
                    return "Move-North"
                case 2:
                    return "Move-South"
                case 3:
                    return "Move-East"
                case 4:
                    return "Move-West"

    # takes action passed in and returns the reward from the action
    def take_Action(self, action, map):
        match action:
            # reward of -5 if crashing into wall
            # each reward total change needs to update q-matrix
            case "Move-North":
                if (
                    self.run_Sensors([self.location[0], self.location[1] - 1], map)
                    == "Wall"
                ):
                    self.reward_total -= 5
                    return -5
                else:
                    self.location[1] -= 1
            case "Move-South":
                if (
                    self.run_Sensors([self.location[0], self.location[1] + 1], map)
                    == "Wall"
                ):
                    self.reward_total -= 5
                    return -5
                else:
                    self.location[1] += 1
            case "Move-East":
                if (
                    self.run_Sensors([self.location[0] + 1, self.location[1]], map)
                    == "Wall"
                ):
                    self.reward_total -= 5
                    return -5
                else:
                    self.location[0] += 1
            case "Move-West":
                if (
                    self.run_Sensors([self.location[0] - 1, self.location[1]], map)
                    == "Wall"
                ):
                    self.reward_total -= 5
                else:
                    self.location[0] -= 1
            case "Pickup-Can":
                # reward of 10 if successful
                # reward of -1 if not
                if self.run_Sensors([self.location[0], self.location[1]], map) == "Can":
                    map[self.location[0], self.location[1]] = 0
                    self.reward_total += 10
                    return 10
                else:
                    self.reward_total -= 1
                    return -1

        return 0

    # this should really be in qmatrix class
    def observe_Current_State(self, qmatrix):
        return qmatrix.get_State_Row(self.location)

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
    return [ran.randrange(0, 10), ran.randrange(0, 10)]


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
    first_map = None
    last_map = None

    # learning
    for episode in range(0, episodes):
        # print("\nEPISODE {}".format(episode))
        map = generate_map(grid, grid)
        curr_Robby = Robby()
        for step in range(0, steps):
            print(curr_Robby.reward_total)
            # print(q_matrix.matrix)
            # if step == 0:
            #     first_map = np.copy(map)
            # print("step {}".format(step))
            # take step (returns [state, action, reward])
            state_action_reward = curr_Robby.take_Step(map, epsilon, q_matrix)
            # observe new state s_t+1
            new_states = curr_Robby.observe_Current_State(q_matrix)
            # get max of new_state vals

            # update Q(s_t, a_t)
            q_matrix.update_matrix(
                state_action_reward[0],
                state_action_reward[1],
                state_action_reward[2],
                new_states,
            )

            # if step == steps - 1:
            #     last_map = np.copy(map)

        # update epsilon
        if epsilon != 0:
            epsilon -= 0.000025

    # print("First Map")
    # print(first_map)

    # print("Last Map")
    # print(last_map)
    # print(np.array_equal(first_map, last_map))
    print(q_matrix.matrix)
    # print(map)

    # test episodes using QMatrix
    # plot point every 100 episodes
    # calc everage sum of rewards per episode
    # epsilon = 0.1
    # fig, ax = plt.subplots()
    # for episode in range(0, episodes):
    #     map = generate_map(grid, grid)
    #     curr_Robby = Robby()
    #     reward_total = 0
    #     for step in range(0, steps):
    #         # take step (returns [state, action, reward])
    #         state_action_reward = curr_Robby.take_Step(map, epsilon, q_matrix)
    #         reward_total += state_action_reward[2]

    #     if episode % 100 == 0:
    #         ax.plot(episode, reward_total, linewidth=100.0)
    #     print(reward_total)

    # ax.set(
    #     xlim=(0, 5000),
    #     xticks=np.arange(0, 5000, 500),
    #     ylim=(-50, 50),
    #     yticks=np.arange(-50, 50, 5.0),
    # )

    # plt.show()


if __name__ == "__main__":
    main()

#################################################################################
