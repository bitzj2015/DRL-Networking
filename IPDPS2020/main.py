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
    user_num = 5
    his_len = 5
    info_num = 2
    A_DIM, S_DIM = user_num, user_num * his_len * info_num
    BATCH = 20
    A_UPDATE_STEPS = 5
    C_UPDATE_STEPS = 5
    A_LR = 0.00003
    C_LR = 0.00003
    v_s = np.zeros(user_num)
    env = ContinuousEnv(user_num,his_len,info_num,bandwidth)
    GAMMA = 0.95
    EP_MAX = 1000
    EP_LEN = 400
    dec = 0.5
    action = np.zeros(user_num)
    Algs = "dnc"

#     max_r = 0
#     max_a = np.random.random(user_num)

    ppo = PPO(S_DIM,A_DIM,BATCH,A_UPDATE_STEPS,C_UPDATE_STEPS,False,0)
    csvFile1 = open("Rewards_" + Algs +
                    "_" + str(user_num) + ".csv", 'w', newline='')
    writer1 = csv.writer(csvFile1)
    csvFile2 = open("Actions_" + Algs +
                    "_" + str(user_num) + ".csv", 'w', newline='')
    writer2 = csv.writer(csvFile2)
    csvFile3 = open("Aloss_" + Algs +
                    "_" + str(user_num) + ".csv", 'w', newline='')
    writer3 = csv.writer(csvFile3)
    csvFile4 = open("Closs_" + Algs +
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
        buffer_s = []
        buffer_a = []
        buffer_r = []
        sum_reward = np.zeros(user_num)
        sum_action = np.zeros(user_num)
        sum_closs = np.zeros(user_num)
        sum_aloss = np.zeros(user_num)
        for t in range(EP_LEN):
            action = ppo.choose_action(cur_state, dec)
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

            v_s = ppo.get_v(next_state)
            
            buffer_a.append(action)
            buffer_s.append(cur_state)
            buffer_r.append(reward)

            cur_state = next_state
            # update ppo
            if (t + 1) % BATCH == 0:
                discounted_r = np.zeros(len(buffer_r), 'float32')
                v_s = ppo.get_v(next_state)
                running_add = v_s

                for rd in reversed(range(len(buffer_r))):
                    running_add = running_add * GAMMA + buffer_r[rd]
                    discounted_r[rd] = running_add

                discounted_r = discounted_r[np.newaxis, :]
                discounted_r = np.transpose(discounted_r)
                ppo.update(np.vstack(buffer_s), np.vstack(buffer_a), discounted_r, dec, A_LR, C_LR, ep)

        if ep % 10 == 0:
            print('instant ep:', ep)
            print("instant reward:", reward)
            print("instant action:", action)
            rewards.append(sum_reward / EP_LEN)
            actions.append(sum_action / EP_LEN)
            closs.append(sum_closs / EP_LEN)
            aloss.append(sum_aloss / EP_LEN)
            print("average reward:", sum_reward / EP_LEN)
            print("average action:", sum_action / EP_LEN)
            print("average closs:", sum_closs / EP_LEN)
            print("average aloss:", sum_aloss / EP_LEN)

    plt.plot(rewards)
    plt.show()
    writer1.writerow(rewards)
    for i in range(len(actions)):
        writer2.writerow(actions[i])
    writer3.writerow(closs)
    writer4.writerow(aloss)
    csvFile1.close()
    csvFile2.close()
    csvFile3.close()
    csvFile4.close()


if __name__ == '__main__':
    main()
