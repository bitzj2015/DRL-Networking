import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from continuousEnv import ContinuousEnv, EnvArgs
import numpy as np
import random
from DNC_PPO import PPO
import csv
import matplotlib.pyplot as plt#matplotlib inline
import math


def get_bandwidth(main_path):
    file_list = os.listdir(main_path)
    bandwidth = {}
    count = 0
    bandwidth[0] = []
    for f in file_list:
        if (f.startswith("report_foot_") == True and count < 5):
            with open(main_path + '/' + f, 'r') as file_to_read:
                while True:
                    lines = file_to_read.readline()
                    if not lines:
                        break
                    item = [i for i in lines.split()]
                    bandwidth[count].append(float(item[-2])/1000/1000)
            count += 1
            bandwidth[count] = []
    return bandwidth

# def main():
#     train_ppo()

# set the experiment environment
user_num = 200
his_len = 5
info_num = 1
main_path = "./Dataset"
bandwidth = get_bandwidth(main_path)
C = np.array([18,20,22,24,26]).astype("float")
D = np.array([0.08, 0.06, 0.07, 0.06, 0.09]).astype("float")
alpha = np.array([1,1,1,1,1]) / 50
tau = 2
epsilon = 5
env_args = EnvArgs(user_num, his_len, info_num, bandwidth, C, D, alpha, tau, epsilon)
env = ContinuousEnv(env_args)

# set the DRL agent
A_DIM, S_DIM = user_num, user_num * his_len * info_num
BATCH = 20
A_UPDATE_STEPS = 5
C_UPDATE_STEPS = 5
HAVE_TRAIN = False
A_LR = 0.00003
C_LR = 0.00003
v_s = np.zeros(user_num)
GAMMA = 0.95
EP_MAX = 500
EP_LEN = 400
dec = 0.3
action = np.zeros(user_num)
ppo = PPO(S_DIM, A_DIM, BATCH, A_UPDATE_STEPS, C_UPDATE_STEPS, HAVE_TRAIN, 0)

# define csvfiles for writing results
Algs = "dnc"
csvFile1 = open("test-lambda=0.5-Rewards_" + Algs + "_" + str(user_num) + ".csv", 'w', newline='')
writer1 = csv.writer(csvFile1)
csvFile2 = open("test-lambda=0.5-Actions_" + Algs + "_" + str(user_num) + ".csv", 'w', newline='')
writer2 = csv.writer(csvFile2)
csvFile3 = open("test-lambda=0.5-Aloss_" + Algs + "_" + str(user_num) + ".csv", 'w', newline='')
writer3 = csv.writer(csvFile3)
csvFile4 = open("test-lambda=0.5-Closs_" + Algs + "_" + str(user_num) + ".csv", 'w', newline='')
writer4 = csv.writer(csvFile4)

rewards = []
actions = []
closses = []
alosses = []
Ts = []
Es = []
cur_state = env.reset()
for ep in range(EP_MAX):
#     cur_state = env.reset()
    if ep % 50 == 0:
        dec = dec * 0.95
        A_LR = A_LR * 0.85
        C_LR = C_LR * 0.85
    buffer_s = []
    buffer_a = []
    buffer_r = []
    sum_reward = 0
    sum_action = 0
    sum_closs = 0
    sum_aloss = 0
    sum_T = 0
    sum_E = 0
    for t in range(EP_LEN):
        action = ppo.choose_action(cur_state.reshape(-1,S_DIM), dec)
#         action = np.random.random(np.shape(action))
        next_state, reward, T, E= env.step(1 + action * 1)
#         print(action,T,E)

        sum_reward += reward
        sum_action += action
        sum_T += T
        sum_E += E
        buffer_a.append(action.copy())
        buffer_s.append(cur_state.reshape(-1,S_DIM).copy())
        buffer_r.append(reward)

        cur_state = next_state
        # update ppo
        if (t + 1) % BATCH == 0:
            discounted_r = np.zeros(len(buffer_r), 'float32')
            v_s = ppo.get_v(next_state.reshape(-1, S_DIM))
            running_add = v_s

            for rd in reversed(range(len(buffer_r))):
                running_add = running_add * GAMMA + buffer_r[rd]
                discounted_r[rd] = running_add

            discounted_r = discounted_r[np.newaxis, :]
            discounted_r = np.transpose(discounted_r)
            if HAVE_TRAIN == False:
                closs, aloss = ppo.update(np.vstack(buffer_s), np.vstack(buffer_a), discounted_r, dec, A_LR, C_LR, ep)
                sum_closs += closs
                sum_aloss += aloss
    if ep % 10 == 0:
        print('instant ep:', ep)
        print("instant reward:", reward)
        print("instant action:", action)
        rewards.append(sum_reward / EP_LEN)
        actions.append(sum_action / EP_LEN)
        closses.append(sum_closs / EP_LEN)
        alosses.append(sum_aloss / EP_LEN)
        Ts.append(sum_T / EP_LEN)
        Es.append(sum_E / EP_LEN)
        print("average reward:", sum_reward / EP_LEN)
        print("average T:", sum_T / EP_LEN)
        print("average E:", sum_E / EP_LEN)
        print("average action:", sum_action / EP_LEN)
        print("average closs:", sum_closs / EP_LEN)
        print("average aloss:", sum_aloss / EP_LEN)

plt.plot(rewards)
plt.show()
writer1.writerow(-rewards * 10)
writer1.writerow(Ts)
writer1.writerow(Es)
for i in range(len(actions)):
    writer2.writerow(actions[i])
writer3.writerow(closses)
writer4.writerow(alosses)
csvFile1.close()
csvFile2.close()
csvFile3.close()
csvFile4.close()


# if __name__ == '__main__':
#     main()
writer1.writerow(rewards * (-10))
writer1.writerow(Ts)
writer1.writerow(Es)
for i in range(len(actions)):
    writer2.writerow(actions[i])
writer3.writerow(closses)
writer4.writerow(alosses)
csvFile1.close()
csvFile2.close()
csvFile3.close()
csvFile4.close()
tmp = []
fig = plt.figure()
for i in range(len(alosses)):
    tmp.append(-sum(alosses[0:0+i+1])/len(alosses[0:0+i+1]))
plt.plot(tmp)
plt.show()   
fig.savefig("aloss.png")
print(tmp)