"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import random
import math
from collections import deque
import matplotlib.pyplot as plt
import csv

#Define MCS System
class MCS(object):
    def __init__(self,user_num, user_cost, user_max_resource, user_gain, lam, his_len):
        self.user_num = user_num # Number of uers
        self.user_cost = user_cost # Cost
        self.user_max_resource = user_max_resource # Resource constraint of users
        self.user_gain = user_gain # Profit gain for allocating resource
        self.user_state = np.zeros(self.user_num, 'float32') # resource allocation of users
        self.user_profit = np.zeros(self.user_num, 'float32')
        self.state_map = np.zeros((self.user_num, 2 * his_len), 'float32')
        self.reward = 0.0
        self.lam = lam
        self.his_len = his_len

    def reset(self):
        self.user_state = np.zeros(self.user_num, 'float32')
        self.state_map = np.zeros((self.user_num, 2 * self.his_len), 'float32')
        self.reward = 0.0
        return np.reshape(self.state_map, [-1])

    def step(self, platform_price):
        self.reward = 0.0
        former_state = self.state_map
        tmp = 0.0
        self.user_profit = np.zeros(self.user_num, 'float32')
        # for i in range(self.user_num):
        #     #print("hh", platform_price[i], self.user_cost[i], self.user_gain[i])
        #     if platform_price[i] >=  0.8*self.user_cost[i]+0.2*self.user_gain[i] and platform_price[i] <= self.user_gain[i]:
        #         self.user_state[i] = self.user_max_resource[i] - self.user_max_resource[i] *\
        #                              (self.user_gain[i] - platform_price[i]) / (self.user_gain[i] - self.user_cost[i])
        #     elif platform_price[i] < 0.8*self.user_cost[i]+0.2*self.user_gain[i]:
        #         self.user_state[i] = 0
        #     elif platform_price[i] > self.user_gain[i]:
        #         self.user_state[i] = self.user_max_resource[i]
        #     self.reward += np.log(1 + self.user_state[i])
        #     tmp += platform_price[i] * self.user_state[i]
        #     self.user_profit[i] = (platform_price[i] - self.user_cost[i]) * self.user_state[i]
        
        self.user_state[0] = 6 * platform_price[0] / 5.5 * (1 - 1.5 / 5.5)
        self.user_state[1] = 6 * platform_price[0] / 5.5 * (1 - 4 / 5.5)
        self.user_profit[0] = 0
        self.user_profit[1] = 0
        self.reward = self.lam * np.log(1 + self.user_state[0] + self.user_state[1]) - 6 * platform_price[0]
        self.state_map[:,:-2] = self.state_map[:,2:]
        self.state_map[0,-2] = platform_price[0]
        self.state_map[1,-2] = platform_price[0]
        self.state_map[:,-1] = self.user_state
        return np.reshape(former_state, [-1]), platform_price, np.reshape(self.state_map, [-1]), self.reward, self.user_state, self.user_profit

user_num = 2
his_len = 5
user_gain = np.array([0.8678, 0.6776, 0.7918, 0.5903, 0.6074])
user_cost = np.array([0.1204, 0.0402, 0.4033, 0.3163, 0.2473])
#user_gain = np.array([0.8678, 0.6776, 0.8312, 0.8537, 0.7112, 0.7338, 0.7918, 0.5903, 0.6074, 0.7640])
#user_cost = np.array([0.0204, 0.2402, 0.2167, 0.2073, 0.1801, 0.2859, 0.1033, 0.4163, 0.3473, 0.4979])
#user_cost = 0.5 * np.random.random(user_num) # C
user_max_resource = 20.0 * np.ones(user_num, 'float32') # Phi
#user_gain = 0.5 * np.random.random(user_num) + 0.5 # Omega
# print(user_cost, user_max_resource, user_gain)
price_bound = 1.0
lam = 10
env = MCS(user_num, user_cost, user_max_resource, user_gain, lam, his_len)

EP_MAX =500

EP_LEN = 400
GAMMA = 0.95
A_LR = 0.00005
C_LR = 0.00005
BATCH = 20
A_UPDATE_STEPS = 5
C_UPDATE_STEPS = 5
S_DIM, A_DIM = user_num * his_len * 2, 1
BELTA = 0.0003
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.1),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization



class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.count = 0
        self.decay = tf.placeholder(tf.float32, (), 'decay')
        self.a_lr = tf.placeholder(tf.float32, (), 'a_lr')
        self.c_lr = tf.placeholder(tf.float32, (), 'c_lr')


        # critic
        with tf.variable_scope('critic'):
            w1 = tf.Variable(tf.truncated_normal([S_DIM, 100], stddev=0.01), name='w1')
            bias1 = tf.Variable(tf.constant(0.0,shape=[100],dtype=tf.float32), name='b1')
            l1 = tf.nn.relu(tf.matmul(self.tfs, w1)+bias1)
            w2 = tf.Variable(tf.truncated_normal([100, 50], stddev=0.01), name='w2')
            bias2 = tf.Variable(tf.constant(0.0,shape=[50],dtype=tf.float32), name='b2')
            l2 = tf.nn.relu(tf.matmul(l1, w2)+bias2)
            w3 = tf.Variable(tf.truncated_normal([50, 1], stddev=0.01), name='w3')
            bias3 = tf.Variable(tf.constant(0.0,shape=[1],dtype=tf.float32), name='b3')
            self.v = tf.nn.relu(tf.matmul(l2, w3) + bias3)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))+\
                         BELTA * (tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(w3)) 
            optimizer = tf.train.AdamOptimizer(learning_rate=self.c_lr)
            vars_ = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.closs, vars_), 5.0)
            self.ctrain_op = optimizer.apply_gradients(zip(grads, vars_))

        # actor
        pi, pi_params, l2_loss_a = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params, _ = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))+\
                    BELTA * l2_loss_a

        with tf.variable_scope('atrain'):
            #self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.a_lr)
            vars_ = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.aloss, vars_), 5.0)
            self.atrain_op = optimizer.apply_gradients(zip(grads, vars_))

        tf.summary.FileWriter("log/", self.sess.graph)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('tmp/vintf/', self.sess.graph)
        self.sess.run(init)

    def update(self, s, a, r, dec, alr, clr):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r, self.decay: dec})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv, self.decay: dec, self.a_lr:alr, self.c_lr:clr}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r, self.decay: dec, self.a_lr:alr, self.c_lr:clr}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            w4 = tf.Variable(tf.truncated_normal([S_DIM,100], stddev=0.01), name='w4')
            bias4 = tf.Variable(tf.constant(0.0,shape=[100],dtype=tf.float32), name='b4')
            l3 = tf.nn.sigmoid(tf.matmul(self.tfs, w4)+bias4)
            
            w5 = tf.Variable(tf.truncated_normal([100, 50], stddev=0.01), name='w5')
            bias5 = tf.Variable(tf.constant(0.0,shape=[50],dtype=tf.float32), name='b5')
            l4 = tf.nn.sigmoid(tf.matmul(l3, w5)+bias5)
            
            w6 = tf.Variable(tf.truncated_normal([50, A_DIM], stddev=0.01), name='w6')
            bias6 = tf.Variable(tf.constant(0.0,shape=[A_DIM],dtype=tf.float32), name='b6')
            mu = tf.nn.sigmoid(tf.matmul(l4, w6)+bias6)
            
            w7 = tf.Variable(tf.truncated_normal([50, A_DIM], stddev=0.01), name='w7')
            bias7 = tf.Variable(tf.constant(0.0,shape=[A_DIM],dtype=tf.float32), name='b7')
            sigma = self.decay * tf.nn.sigmoid(tf.matmul(l4, w7)+bias7) + 0.00001
            #mu = tf.layers.dense(l2, A_DIM, tf.nn.sigmoid, trainable=trainable) 
            #sigma = tf.layers.dense(l2, A_DIM, tf.nn.sigmoid, trainable=trainable) + 0.0001
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        l2_loss_a = tf.nn.l2_loss(w4)+tf.nn.l2_loss(w5)+tf.nn.l2_loss(w6)+tf.nn.l2_loss(w7)
        return norm_dist, params, l2_loss_a

    def choose_action(self, s, dec):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s, self.decay: dec})[0]
        return np.clip(a, 0, 1)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


#env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
all_ep_r = []
state = np.zeros((EP_MAX, user_num), "float32")
user_profit = np.zeros((EP_MAX, user_num), "float32")
action_profile = np.zeros((EP_MAX, user_num), "float32")
dec = 0.3
s = env.reset()
for ep in range(EP_MAX):
    if ep % 50 == 0:
        dec = dec * 0.9
        A_LR = A_LR * 0.9
        C_LR = C_LR * 0.9
    #s = np.reshape(s, [-1])
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    ep_s = np.zeros(user_num)
    ep_up = np.zeros(user_num)
    ep_a = 0
    for t in range(EP_LEN):    # in one episode
        if ep == EP_MAX:
            t = EP_LEN-1
        a = ppo.choose_action(s, dec)
        s, a, s_, r, sta, upro = env.step(a)
        r = r 
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append(r)    # normalize reward, find to be useful
        s = s_
        ep_r += r
        ep_s += sta
        ep_up += upro
        ep_a += a[0]
        #print(ep_r)

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            
            ppo.update(bs, ba, br, dec, A_LR, C_LR)


    if ep == 0:
        all_ep_r.append(ep_r * 1 / EP_LEN)
    else:
        all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1* 1/ EP_LEN)
    print(
        'Ep: %i' % ep,
        "|Ep_r: " ,all_ep_r[-1],
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )
    print(a)
    state[ep] = ep_s / EP_LEN
    user_profit[ep] = ep_up / EP_LEN
    action_profile[ep] = ep_a / EP_LEN *6
    #print(adv)
print("######------Final results------######")
print("User_num:", user_num)
print("User_max_resource:", user_max_resource)
print("User_gain:", user_gain)
print("User_cost:", user_cost)
print("Platform_price:", a)
print("User_resource_allocation:", s[90:100])
print("User_payoff_profile:", upro)
print("Platform_utility:", r )
s = env.reset()
a = np.random.rand(1)
s, a, s_, r, sta, upro = env.step(a)
print("random", a)
print("randon_results:",s_)
print(r)

csvFile1 = open('Platform_utility.csv','w', newline='')
writer1 = csv.writer(csvFile1)
csvFile2 = open('User_utility.csv','w', newline='')
writer2 = csv.writer(csvFile2)
csvFile3 = open('User_resource_allocation.csv','w', newline='')
writer3 = csv.writer(csvFile3)
csvFile4 = open('Platform_price.csv','w', newline='')
writer4 = csv.writer(csvFile4)
csvFile5 = open('User_omega.csv','w', newline='')
writer5 = csv.writer(csvFile5)
csvFile6 = open('Final_result.csv','w', newline='')
writer6 = csv.writer(csvFile6)

plt.figure()
plt.plot(np.arange(len(all_ep_r)), all_ep_r)
writer1.writerow(all_ep_r)
writer5.writerow(user_gain)
writer5.writerow(user_cost)
plt.xlabel('Episode')
plt.ylabel('Platform Utility')
plt.show()
state_T = np.transpose(state, [1,0])
user_profit_T = np.transpose(user_profit, [1,0])
writer4.writerow(action_profile)

plt.figure()
plt.xlabel('Episode')
plt.ylabel('User Profile')
for i in range(user_num):
    plt.plot(np.arange(EP_MAX), state_T[i])
    writer2.writerow(user_profit_T[i])
    writer3.writerow(state_T[i])
plt.show()
csvFile1.close()
csvFile2.close()
csvFile3.close()
csvFile4.close()
csvFile5.close()
