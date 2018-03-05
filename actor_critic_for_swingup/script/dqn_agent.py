#!/usr/bin/env python
#coding:utf-8

import argparse

import copy, sys
import time

import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, optimizers, FunctionSet
from random import random, randint

import math

import pickle

#  parser = argparse.ArgumentParser(description='agent')
#  parser.add_argument('--gpu', '-g', default=0, type=int,
                    #  help='GPU ID (negative value indicates CPU)')
#  args = parser.parse_args()
args_gpu = 0

if args_gpu >=0:
    cuda.check_cuda_available()
xp = cuda.cupy if args_gpu >=0 else np

class Network(FunctionSet):
    def __init__(self, n_in, n_out):
        super(Network, self).__init__(
                l1 = F.Linear(n_in, 400),
                l2 = F.Linear(400, 200),
                l3 = F.Linear(200, 100),
                l4 = F.Linear(100, n_out, initialW=xp.zeros((n_out, 100), dtype=xp.float32)),
                )


    def Q_func(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        output = self.l4(h3)
        return output

class Agent():
    ALPHA = 1e-6
    GAMMA = 0.9

    #  data_size = 10
    #  replay_size = 2
    #  initial_exploration = 10
    data_size = 1000
    replay_size = 100
    initial_exploration = 1000
    target_update_freq = 20

    def __init__(self, n_st, n_act, seed):
        np.random.seed(seed)
        sys.setrecursionlimit(10000)
        self.n_act = n_act
        
        self.model = Network(n_st, n_act)
        self.target_model = copy.deepcopy(self.model)
        if args_gpu>=0:
            self.model.to_gpu()
            self.target_model.to_gpu()

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        
        self.loss = 0
        self.step = 0
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.00001
        self.epsilon_min = 0.1

        self.D = [xp.zeros((self.data_size, 1, 3), dtype=np.float32),
                  xp.zeros(self.data_size, dtype=np.float32),
                  xp.zeros(self.data_size, dtype=np.float32),
                  xp.zeros((self.data_size, 1, 3), dtype=np.float32),
                  xp.zeros(self.data_size, dtype=np.bool)]
        #  print "self.D : ", self.D


    def stock_experience(self, t, st, act, r, st_dash, ep_end):
        data_index = t % self.data_size
        st_gpu = cuda.to_gpu(st)
        act_gpu = cuda.to_gpu(act)
        r_gpu = cuda.to_gpu(r)
        st_dash_gpu = cuda.to_gpu(st_dash)
        ep_end_gpu = cuda.to_gpu(ep_end)
        #  print "st : ", st
        #  print "act : ", act
        #  print "r : ", r
        #  print "st_dash : ", st_dash
        #  print "ep_end : ", ep_end

        self.D[0][data_index] = st_gpu
        self.D[1][data_index] = act_gpu
        self.D[2][data_index] = r_gpu
        self.D[3][data_index] = st_dash_gpu
        self.D[4][data_index] = ep_end_gpu
        #  print "self.D : ", self.D

    def forward(self, st, act, r, st_dash, ep_end):
        num_of_batch = st.shape[0]
        #  print "num_of_batch : ", num_of_batch

        s = chainer.Variable(st)
        s_dash = chainer.Variable(st_dash)

        Q = self.model.Q_func(s)
        #  print "Q.data : ", Q.data
        tmp = self.target_model.Q_func(s_dash)
        #  print "tmp.data : ", tmp.data

        tmp = list(map(np.max, tmp.data.get()))
        max_Q_dash = xp.asanyarray(tmp, dtype=xp.float32)
        #  print "max_Q_dash : ", max_Q_dash
        target = xp.asanyarray(Q.data.get(), dtype=xp.float32)
        #  print "target : ", target

        for i in xrange(num_of_batch):
            if not ep_end[i]:
                tmp_ = r[i] + self.GAMMA * max_Q_dash[i]
                #  print "tmp_(not ep_end) : ", tmp_
            else:
                tmp_ = r[i]
                #  print "tmp_(ep_end) : ", tmp_

            action_index = int(act[i][0])
            #  print "action_index : ", action_index
            target[i, action_index] = tmp_
            #  print "target(after) : ", target

        td = chainer.Variable(target) - Q
        #  print "td.data : ", td.data

        zero_val = chainer.Variable(xp.zeros((self.replay_size, self.n_act), dtype=xp.float32))

        loss = F.mean_squared_error(td, zero_val)
        self.loss = loss.data

        return loss, Q

    def experience_replay(self, t):
        if self.initial_exploration < t:
            if t < self.data_size:
                replay_index = np.random.randint(0, t, (self.replay_size, 1))
                #  print "replay_index : ", replay_index
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))
                #  print "replay_index : ", replay_index
            
            st_replay = xp.ndarray(shape=(self.replay_size, 1, 3), dtype=xp.float32)
            act_replay = xp.ndarray(shape=(self.replay_size, 1), dtype=xp.float32)
            r_replay = xp.ndarray(shape=(self.replay_size, 1), dtype=xp.float32)
            st_dash_replay = xp.ndarray(shape=(self.replay_size, 1, 3), dtype=xp.float32)
            ep_end_replay = xp.ndarray(shape=(self.replay_size, 1), dtype=np.bool)

            for i in xrange(self.replay_size):
                st_replay[i] = xp.asarray(self.D[0][replay_index[i][0]], dtype=xp.float32)
                act_replay[i] = self.D[1][replay_index[i][0]]
                r_replay[i] = self.D[2][replay_index[i][0]]
                st_dash_replay[i] = xp.asarray(self.D[3][replay_index[i][0]], dtype=xp.float32)
                ep_end_replay[i] = self.D[4][replay_index[i][0]]
            #  print "st_replay : ", st_replay
            #  print "act_replay : ", act_replay
            #  print "r_replay : ", r_replay
            #  print "st_dash_replay : ", st_dash_replay
            #  print "ep_end_replay : ", ep_end_replay
            
            self.optimizer.zero_grads()
            #  print "self.model.parameters : ", self.model.parameters
            #  print "self.model.gradients : ", self.model.gradients
            loss, _ = self.forward(st_replay, act_replay, r_replay, st_dash_replay, ep_end_replay)
            loss.backward()
            self.optimizer.update()
            #  print "self.model.gradients(after) : ", self.model.gradients
            #  print "self.model.parameters(after) : ", self.model.parameters

    def get_action(self, st, evaluation):
        if not evaluation:
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.n_act), 0
            else:
                st_gpu = cuda.to_gpu(st)
                s = chainer.Variable(st_gpu)
                Q = self.model.Q_func(s)
                Q = Q.data[0]
                a = np.argmax(Q)
                #  print "a : ", int(a)
                #  print "type(a) : ", type(int(a))
                return int(a), cuda.to_cpu(max(Q))
        else:
            st_gpu = cuda.to_gpu(st)
            s = chainer.Variable(st_gpu)
            Q = self.model.Q_func(s)
            Q = Q.data[0]
            a = np.argmax(Q)
            #  print "a : ", int(a)
            #  print "type(a) : ", type(int(a))
            return int(a), cuda.to_cpu(max(Q))

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            #  print "self.epsilon : ", self.epsilon
            self.epsilon -= self.epsilon_decay
        else:
            #  print "self.epsilon___ : ", self.epsilon
            self.epsilon = self.epsilon_min

    def train(self, t):
        if self.initial_exploration < t:
            self.experience_replay(t)
            self.reduce_epsilon()
        if t % self.target_update_freq == 0:
            self.target_model = copy.deepcopy(self.model)
        self.step += 1

    def save_model(self, model_dir):
        #  print "Now Saving!!!!!"
        f_model = open(model_dir+"model.dat", 'w')
        pickle.dump(self.model, f_model)
        f_model.close()

    def load_model(self, model_dir):
        #  print "Now Loading!!!!!"
        f_model = open(model_dir+"model.dat", 'rb')
        self.model = pickle.load(f_model)


if __name__=="__main__":
    my_agent = Agent(1, 1, 0)

