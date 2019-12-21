import numpy as np
from collections import defaultdict

class EnvArgs(object):
    def __init__(self, user_num, his_len, info_num, bandwidth, C, D, alpha, tau, epsilon):
        self.user_num = user_num
        self.his_len = his_len
        self.info_num = 1 #info_num
        self.bandwidth = bandwidth
        self.C = C
        self.D = D
        self.alpha = alpha
        self.tau = tau
        self.epsilon = epsilon
        
        
class ContinuousEnv(object):
    state_map = None  # state pool
    his_len = 5

    def __init__(self, env_args):
        self.user_num = env_args.user_num
        self.his_len = env_args.his_len
        self.info_num = env_args.info_num
        self.bandwidth = env_args.bandwidth
        self.state = np.zeros(
            (self.user_num, self.his_len, self.info_num), 'float32')
        self.reward = 0
        self.global_step = 0
        self.global_time = 0
        self.C = env_args.C
        self.D = env_args.D
        self.alpha = env_args.alpha
        self.tau = env_args.tau
        self.epsilon = env_args.epsilon
        self.cur_delta = np.zeros(self.user_num, "float32")
        self.cur_B = np.zeros(self.user_num, "float32")
        self.cur_user_T = np.zeros(self.user_num, "float32")
        self.cur_T = 0

    def reset(self):
        self.global_time = 0
        self.global_step = int(self.global_time / 4)
        for i in range(self.user_num):
            idx = i % 5
            for j in range(self.his_len):
                self.state[i,j,0] = 0
                for k in range(4):
                    self.state[i,j,0] += np.clip(self.bandwidth[idx][
                        ((self.global_step+j-self.his_len) * 4 - k) % len(self.bandwidth[idx])],
                                                0.2,10)
                self.state[i,j,0] = self.state[i,j,0] / 4
#                 self.state[i,j,1] = np.random.rand()
            self.cur_B[i] = self.state[i,-1,0] 
            self.cur_delta[i] = np.random.rand() # self.state[i,-1,1]       
        self.reward = 0
        self.cur_T, self.cur_user_T = self.count_T(self.cur_delta, self.cur_B)
        return self.state
    
    def count_T(self, delta, B):
        user_T = np.zeros(self.user_num, "float32")
        for i in range(self.user_num):
            idx = i % 5
            user_T[i] = self.C[idx] * self.D[idx] / delta[i] * self.tau + self.epsilon / B[i]
         
        return np.max(user_T), user_T
    
    def step(self, delta):  # continuous action
        self.global_time += self.cur_T
        self.global_step = int(self.global_time / 4)
        self.state[:,:-1,:] = self.state[:,1:,:]
        self.reward = 0
        for i in range(self.user_num):
#             self.state[i,-1,0] = 0
            idx = i % 5
            tmp_step = int((self.C[idx] * self.D[idx] / delta[idx] * self.tau + self.global_time)/4)
            for his in range(self.his_len):
                tmp_step = tmp_step - 4
                for k in range(4):
                    self.state[i,-his-1,0] += np.clip(self.bandwidth[idx][
                        (tmp_step * 4 - k) % len(self.bandwidth[idx])],0.2,10)
#             self.state[i,-1,0] = self.state[i,-1,0] / 4
#             self.state[i,-1,1] = delta[i]
            self.state[i,:,0] = self.state[i,:,0] / 4
#             self.state[i,:,1] = self.state[i,:,0]
            self.cur_B[i] = self.state[i,-1,0]
            self.cur_delta[i] = delta[i] # self.state[i,-1,1]
            self.reward += self.alpha[idx] / 2 * self.C[idx] * self.D[idx] * delta[i] * delta[i]
        self.cur_T, self.cur_user_T = self.count_T(self.cur_delta, self.cur_B)
        self.reward += 1 * self.cur_T
        
        return self.state, -self.reward / 10, self.cur_T , self.reward - 1 * self.cur_T 
