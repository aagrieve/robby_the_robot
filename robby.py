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
    def __init__(self, discount, eta):
        self.dict = self.fill_dictionary()
        self.matrix = np.zeros((len(self.dict), 5))
        self.discount = discount
        self.eta = eta

    def fill_dictionary(self):
        dict = {}
        state_list = []

        for one in range(0, 3):
            for two in range(0, 3):
                for three in range(0, 3):
                    for four in range(0, 3):
                        for five in range(0, 3):
                            state_list.append(
                                str(one) + str(two) + str(three) + str(four) + str(five)
                            )

        for count in range(0, len(state_list)):
            dict[state_list[count]] = count

        return dict

    def update_Qmatrix(self, state, action, reward, next_state):
        current_q_value = self.get_Q_Value(state, action)
        max_next = max(self.get_State_Row(next_state))

        new_val = (
            current_q_value
            + self.eta * (reward + self.discount * max_next)
            - current_q_value
        )

        self.assign_Q_Value(state, action, new_val)

    def assign_Q_Value(self, state, action, new_val):
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
        row = self.dict[state]
        self.matrix[row, col] = new_val

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
        row = self.dict[state]
        return self.matrix[row, col]

    def get_State_Row(self, state):
        return np.copy(self.matrix[self.dict[state], :])


#################################################################################


# Robby class
#################################################################################


class Robby:
    def __init__(self, map):
        self.location = place_robby()
        self.sensor_string = self.run_Sensors(map)
        self.reward_total = 0

    def take_Step(self, map, epsilon, qmatrix):
        state = self.sensor_string
        # observe
        cur_state_action_vals = self.observe_Current_State(qmatrix)
        # choose action
        action = self.choose_Action(cur_state_action_vals, epsilon)
        # perform action, updates location and sensor_string if need be
        reward = self.take_Action(action, map)
        # return state-action pair and reward for qmatrix updates
        next_state = self.sensor_string

        return [state, action, reward, next_state]

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

    def take_Action(self, action, map):
        reward = 0
        match action:
            case "Pickup-Can":
                if self.sensor_string[0] == "0":
                    # pickup can
                    map[self.location[0], self.location[1]] = 0
                    # receive reward
                    self.reward_total += 10
                    reward = 10
                else:
                    self.reward_total -= 1
                    reward = -1

            case "Move-North":
                if self.sensor_string[1] == "2":
                    self.reward_total -= 5
                    reward = -5
                else:
                    self.location[0] -= 1

            case "Move-South":
                if self.sensor_string[2] == "2":
                    self.reward_total -= 5
                    reward = -5
                else:
                    self.location[0] += 1

            case "Move-East":
                if self.sensor_string[3] == "2":
                    self.reward_total -= 5
                    reward = -5
                else:
                    self.location[1] += 1

            case "Move-West":
                if self.sensor_string[4] == "2":
                    self.reward_total -= 5
                    reward = -5
                else:
                    self.location[1] -= 1

        self.update_Sensor_String(map)
        return reward

    # this should really be in qmatrix class
    def observe_Current_State(self, qmatrix):
        return qmatrix.get_State_Row(self.sensor_string)

    def update_Sensor_String(self, map):
        self.sensor_string = self.run_Sensors(map)

    # creates sensor string for qmatrix -- 0: can, 1: empty, 2: wall
    def run_Sensors(self, map):
        sensor_string = ""
        size = map.shape[0] - 1
        # CURR (sensor_string[0])
        if map[self.location[0], self.location[1]] == 1:
            sensor_string += "0"
        else:
            sensor_string += "1"

        # NORTH (sensor_string[1])
        if self.location[0] - 1 < 0:
            # WALL
            sensor_string += "2"
        elif map[self.location[0] - 1, self.location[1]] == 1:
            # CAN
            sensor_string += "0"
        else:
            # EMPTY
            sensor_string += "1"

        # SOUTH (sensor_string[2])
        if self.location[0] + 1 > size:
            # WALL
            sensor_string += "2"
        elif map[self.location[0] + 1, self.location[1]] == 1:
            # CAN
            sensor_string += "0"
        else:
            # EMPTY
            sensor_string += "1"

        # EAST (sensor_string[3])
        if self.location[1] + 1 > size:
            # WALL
            sensor_string += "2"
        elif map[self.location[0], self.location[1] + 1] == 1:
            # CAN
            sensor_string += "0"
        else:
            # EMPTY
            sensor_string += "1"

        # WEST (sensor_string[4])
        if self.location[1] - 1 < 0:
            # WALL
            sensor_string += "2"
        elif map[self.location[0], self.location[1] - 1] == 1:
            # CAN
            sensor_string += "0"
        else:
            # EMPTY
            sensor_string += "1"

        return sensor_string


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
    q_matrix = QMatrix(discount, eta)
    first_map = None
    last_map = None

    fig1, trx = plt.subplots()
    tr_ep_list = []
    tr_rew_list = []

    # learning
    for episode in range(0, episodes):
        # print("\nEPISODE {}".format(episode))
        sum_of_rewards = 0
        map = generate_map(grid, grid)
        curr_Robby = Robby(map)
        for step in range(0, steps):
            # take step (returns [state, action, reward, next_state_qmatrix_vals])
            state_action_reward = curr_Robby.take_Step(map, epsilon, q_matrix)

            q_matrix.update_Qmatrix(
                state_action_reward[0],
                state_action_reward[1],
                state_action_reward[2],
                state_action_reward[3],
            )

            # NEED A PLOT FOR TRAINING
            sum_of_rewards += state_action_reward[2]
        if episode % 50 == 0:
            epsilon -= 0.005
        if episode % 50 == 0:
            tr_ep_list.append(episode)
            tr_rew_list.append(sum_of_rewards)

    trx.plot(tr_ep_list, tr_rew_list)
    trx.set(
        xlim=(0, 5000),
        xticks=np.arange(0, 5000, 500),
        ylim=(-10, 650),
        yticks=np.arange(0, 600, 100),
    )
    plt.xlabel("Episode")
    plt.ylabel("Sum of Rewards")
    plt.show()

    # test episodes using QMatrix
    # plot point every 100 episodes
    # calc everage sum of rewards per episode
    epsilon = 0.1
    sums = []
    sum_total = 0
    for episode in range(0, episodes):
        map = generate_map(grid, grid)
        curr_Robby = Robby(map)
        reward_total = 0
        for step in range(0, steps):
            # take step (returns [state, action, reward])
            state_action_reward = curr_Robby.take_Step(map, epsilon, q_matrix)
            reward_total += state_action_reward[2]
        sums.append(reward_total)
        sum_total += reward_total

    rewards_average = sum_total / len(sums)
    standard_dev = np.std(sums)

    print("Average")
    print(rewards_average)
    print("Standard Deviation")
    print(standard_dev)


if __name__ == "__main__":
    main()

#################################################################################
