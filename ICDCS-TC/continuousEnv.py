import numpy as np
from collections import defaultdict


class ContinuousEnv(object):
    state_map = None  # state pool
    his_len = 5

    def __init__(self, user_num, his_len):
        self.user_num = user_num
        self.his_len = his_len
        self.state_map = np.zeros(
            (self.user_num, self.user_num * self.his_len), 'float32')
        self.S = np.zeros((self.user_num, self.user_num))  # user's state

    def reset(self):
        # transmission power of user i
        self.p = np.ones(self.user_num, dtype=np.float32)
        # Data for detection of user i
        self.C = np.ones(self.user_num, dtype=np.float32)
        action = np.random.random(self.user_num - 1)  # random number 0~1
        # self.b = [1.0 / 6, 1.0 / 4]  # transmission bandwidth of user i
        # transmission bandwidth of user i
        # self.b = [1.0 / 11.5, 1.0 /
        #           10.5, 1.0 / 9.5, 1.0 / 8.5]
        self.b = [1 / 8.5, 1 / 8.5, 1 / 8.5, 1 /
                  8.5, 1 / 8.5, 1 / 8.5, 1 / 8.5, 1 / 8.5, 1 / 8.5, 1 / 8.5]
        # self.b = [7 / 77, 7 / 75, 7 / 73, 7 /
        #           71, 7 / 69, 7 / 67, 7 / 65, 7 / 63]
        # self.b = [9 / 99, 9 / 97, 9 / 95, 9 / 93, 9 /
        #           91, 9 / 89, 9 / 87, 9 / 85, 9 / 83, 9 / 81]
        # init state
        # self.S[0] = [action, self.b[0]]
        # action = np.random.random()  # random number 0~1
        # self.S[1] = [action, self.b[1]]
        for i in range(self.user_num):
            flag = 0
            action = np.random.random(self.user_num - 1)
            for j in range(self.user_num):
                if j != i:
                    self.S[i, flag] = action[flag]
                    flag += 1
            if flag != self.user_num - 1:
                print("Index error!")
            else:
                self.S[i, flag] = self.b[i]

        # instant utility of user i
        self.user_profit = np.zeros(self.user_num)
        return self.state_map

    def step(self, x):  # continuous action
        sum = 0.0
        R = 24.0
        for i in range(self.user_num):
            sum += x[i] * self.C[i]
        for i in range(self.user_num):
            self.user_profit[i] = x[i] * self.C[i] * R / \
                sum - self.p[i] * x[i] * self.C[i] * \
                (1 / self.b[i])  # - 0.25 + 0.5 * np.random.random())

        # update S and S_map based on the last action
        for i in range(self.user_num):
            flag = 0
            for j in range(self.user_num):
                if j != i:
                    self.S[i, flag] = x[j]
                    flag += 1
            if flag != self.user_num - 1:
                print("Index error!")
            else:
                self.S[i, flag] = self.b[i]

        # self.S[0] = [x[1], self.b[0]]
        # self.S[1] = [x[0], self.b[1]]
        self.state_map[:, :-self.user_num] = self.state_map[:, self.user_num:]
        self.state_map[:, -self.user_num:] = self.S

        self.user_profit = np.clip(self.user_profit, 0, R)
        return self.state_map, self.user_profit
