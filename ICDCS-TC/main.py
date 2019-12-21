import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from continuousEnv import ContinuousEnv
import numpy as np
import random
import matplotlib.pyplot as plt
from DNC_PPO import PPO
import csv


def main():
    train_ppo()


def train_ppo():
    user_num = 10
    his_len = 5
    A_DIM, S_DIM = 1, user_num * his_len
    BATCH = 20
    A_UPDATE_STEPS = 5
    C_UPDATE_STEPS = 5
    A_LR = 0.00003
    C_LR = 0.00003
    v_s = np.zeros(user_num)
    ppo = []
    env = ContinuousEnv(user_num, his_len)
    GAMMA = 0.95
    EP_MAX = 1000
    EP_LEN = 400
    dec = 0.5
    action = np.zeros(user_num)
    Algs = "dnc"

    max_r = np.zeros(user_num)
    max_a = np.random.random(user_num)

    for i in range(user_num):
        ppo.append(PPO(S_DIM, 1, BATCH, A_UPDATE_STEPS,
                       C_UPDATE_STEPS, False, i))
    csvFile1 = open("./0_Rewards/static_result_" + Algs +
                    "_" + str(user_num) + ".csv", 'w', newline='')
    writer1 = csv.writer(csvFile1)
    csvFile2 = open("./1_Actions/static_result_" + Algs +
                    "_" + str(user_num) + ".csv", 'w', newline='')
    writer2 = csv.writer(csvFile2)
    csvFile3 = open("./3_loss_pi/static_result_" + Algs +
                    "_" + str(user_num) + ".csv", 'w', newline='')
    writer3 = csv.writer(csvFile3)
    csvFile4 = open("./4_loss_v/static_result_" + Algs +
                    "_" + str(user_num) + ".csv", 'w', newline='')
    writer4 = csv.writer(csvFile4)

    rewards = []
    actions = []
    closs = []
    aloss = []
    cur_state = env.reset()
    for ep in range(EP_MAX):
        if ep % 50 == 0:
            dec = dec * 1
            A_LR = A_LR * 0.8
            C_LR = C_LR * 0.8
        buffer_s = [[] for _ in range(user_num)]
        buffer_a = [[] for _ in range(user_num)]
        buffer_r = [[] for _ in range(user_num)]
        sum_reward = np.zeros(user_num)
        sum_action = np.zeros(user_num)
        sum_closs = np.zeros(user_num)
        sum_aloss = np.zeros(user_num)
        for t in range(EP_LEN):
            for i in range(user_num):
                action[i] = ppo[i].choose_action(cur_state[i], dec)
                # Greedy algorithm
                # if np.random.random() < 0.1:
                #     action[i] = np.random.random()
                # else:
                #     action[i] = max_a[i]
                # action[i] = np.random.random()

            next_state, reward = env.step(action)
            sum_reward += reward
            sum_action += action

            # Greedy algorithm
            # for i in range(user_num):
            #     if reward[i] > max_r[i]:
            #         max_r[i] = reward[i]
            #         max_a[i] = action[i]
            #     if max_a[i] == action[i]:
            #         max_r[i] = reward[i]

            for i in range(user_num):
                v_s[i] = ppo[i].get_v(next_state[i])

            for i in range(user_num):
                buffer_a[i].append(action[i])
                buffer_s[i].append(cur_state[i])
                buffer_r[i].append(reward[i])

            cur_state = next_state
            # update ppo
            if (t + 1) % BATCH == 0:
                for i in range(user_num):
                    discounted_r = np.zeros(len(buffer_r[i]), 'float32')
                    v_s[i] = ppo[i].get_v(next_state[i])
                    running_add = v_s[i]

                    for rd in reversed(range(len(buffer_r[i]))):
                        running_add = running_add * GAMMA + buffer_r[i][rd]
                        discounted_r[rd] = running_add

                    discounted_r = discounted_r[np.newaxis, :]
                    discounted_r = np.transpose(discounted_r)
                    ppo[i].update(np.vstack(buffer_s[i]), np.vstack(
                        buffer_a[i]), discounted_r, dec, A_LR, C_LR, ep)

        if ep % 10 == 0:
            print('ep:', ep)
            print("reward:", reward)
            print("action:", action)
            rewards.append(sum_reward / EP_LEN)
            actions.append(sum_action / EP_LEN)
            closs.append(sum_closs / EP_LEN)
            aloss.append(sum_aloss / EP_LEN)
            print("average reward:", sum_reward / EP_LEN)
            print("average action:", sum_action / EP_LEN)
            print("average closs:", sum_closs / EP_LEN)
            print("average aloss:", sum_aloss / EP_LEN)

    for i in range(user_num):
        usr_reward = [data[i] for data in rewards]
        usr_action = [data[i] for data in actions]
        usr_closs = [data[i] for data in closs]
        usr_aloss = [data[i] for data in aloss]
        plt.plot(usr_reward)
        writer1.writerow(usr_reward)
        writer2.writerow(usr_action)
        writer3.writerow(usr_closs)
        writer4.writerow(usr_aloss)
    plt.show()
    csvFile1.close()
    csvFile2.close()
    csvFile3.close()
    csvFile4.close()


if __name__ == '__main__':
    main()
