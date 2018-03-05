#!/usr/bin/env python
#coding:utf-8
import argparse

import rospy
import copy

import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, optimizers

from std_msgs.msg import Float64, Int64, Float64MultiArray, String
from gazebo_msgs.msg import LinkState
from gazebo_msgs.srv import*
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3

import tf

import math
from random import random, randint, uniform

import pickle
 
parser = argparse.ArgumentParser(description='deep_actor_critic_for_swingup')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

if args.gpu >=0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >=0 else np

class agent:
    ALPHA = 1e-6
    #  ALPHA = 0.001
    GAMMA = 0.9

    wait_flag = True

    data_size = 500
    #  data_size = 10
    replay_size = 16
    #  replay_size = 3
    initial_exploration = 500
    #  initial_exploration = 10

    target_update_freq = 1000
    
    filename_critic_model = "/home/amsl/ros_catkin_ws/src/deep_actor_critic/actor_critic_for_swingup/test_results/models/myrrbot5_critic_model.dat"
    filename_actor_model = "/home/amsl/ros_catkin_ws/src/deep_actor_critic/actor_critic_for_swingup/test_results/models/myrrbot5_actor_model.dat"

    filename_result = "/home/amsl/ros_catkin_ws/src/deep_actor_critic/actor_critic_for_swingup/test_results/results/myrrbot5_result.txt"

    def __init__(self):
        self.init_theta = 0.0
        self.init_omega = 0.0
        
        self.episode_time = 10.0
        self.hz = 20.0

        self.evaluation_freq = 100

        self.critic_model = chainer.FunctionSet(
                l1 = F.Linear(5, 100),
                l2 = F.Linear(100, 100),
                l3 = F.Linear(100, 1, initialW=np.zeros((1, 100), dtype=np.float32)),
                )
        self.target_critic_model = copy.deepcopy(self.critic_model)
        self.actor_model = chainer.FunctionSet(
                l1 = F.Linear(5, 100),
                l2 = F.Linear(100, 100),
                l3 = F.Linear(100, 1, initialW=np.zeros((1, 100), dtype=np.float32)),
                )
        if args.gpu >= 0:
            self.critic_model.to_gpu()
            self.target_critic_model.to_gpu()
            self.actor_model.to_gpu()
        #  self.critic_optimizer = optimizers.SGD(self.ALPHA)
        self.critic_optimizer = optimizers.Adam(self.ALPHA)
        self.critic_optimizer.setup(self.critic_model)
        #  self.actor_optimizer = optimizers.SGD(self.ALPHA)
        self.actor_optimizer = optimizers.Adam(self.ALPHA)
        self.actor_optimizer.setup(self.actor_model)
        
        #history data : D = [state(theta, omega), action, old_action, reward, next_state(theta, omega), episode_end_flag]
        self.D = [xp.zeros((self.data_size, 1, 5), dtype=xp.float32),
                  xp.zeros(self.data_size, dtype=xp.float32),
                  xp.zeros(self.data_size, dtype=xp.float32),
                  xp.zeros((self.data_size, 1), dtype=xp.float32),
                  xp.zeros((self.data_size, 1, 5), dtype=xp.float32),
                  xp.zeros((self.data_size, 1), dtype=np.bool)]
        #  print "D : ", self.D
        
        self.D_actor = [xp.zeros((self.data_size, 1, 5), dtype=xp.float32),
                        xp.zeros(self.data_size, dtype=xp.float32),
                        xp.zeros(self.data_size, dtype=xp.float32),
                        xp.zeros((self.data_size, 1), dtype=xp.float32),
                        xp.zeros((self.data_size, 1, 5), dtype=xp.float32),
                        xp.zeros((self.data_size, 1), dtype=np.bool)]
        #  print "D_actor : ", self.D_actor

        self.sigma = 10.0

        self.limit_action = 5.0
        self.min_action = -5.0
        self.max_action = 5.0

        self.data_index_actor = 0
        
    ######################callback or client function for ROS #######################################
    def effort_callback(self, msg):
        pass
        #  print "effort : ", msg.data
    
    def get_link_state(self, link_name, reference_frame_name):
        try:
            getlinkstate = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
            resp = getlinkstate(link_name, reference_frame_name)
            #  print "resp.position : ", resp.position
            #  print "resp.rate : ", resp.rate
            quaternion = (
                    resp.link_state.pose.orientation.x,
                    resp.link_state.pose.orientation.y,
                    resp.link_state.pose.orientation.z,
                    resp.link_state.pose.orientation.w)
            return quaternion, resp.link_state.twist.angular
        except rospy.ServiceException, e:
            print "Service call faild : %s" % e

    def get_joint_properties(self, joint_name):
        try:
            getjointproperties = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
            resp = getjointproperties(joint_name)
            print "resp.position : ", resp.position
            #  print "resp.rate : ", resp.rate
            #  return resp.position[0], resp.rate[0]
            return resp.position[0]
        except rospy.ServiceException, e:
            print "Service call faild : %s" % e

    def reset_client(self, angle):
        q = tf.transformations.quaternion_from_euler(0.0, angle, 0.0)
        request = LinkState()
        request.link_name = "myrrbot5::myrrbot5_link2"
        request.pose.position.x = 1.0
        request.pose.position.y = -0.9
        request.pose.position.z = 1.45
        request.pose.orientation.x = q[0]
        request.pose.orientation.y = q[1]
        request.pose.orientation.z = q[2]
        request.pose.orientation.w = q[3]
        #  print "request.pose : "
        #  print request.pose
        try:
            reset = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
            resp = reset(request)
            #  print "resp : ", resp
        except rospy.ServiceException, e:
            print "Service call faild : %s" % e
    ###############################################################################################

    #################################for Deep Reinforcement Learning####################################
    def episode_end_flag(self, s):
        if math.fabs(s[0][0]) > 3*math.pi:
            return True
        else:
            return False
    
    def reward_func(self, angle):
        r = math.cos(angle)
        #  print "r : ", r
        if r >= 0.0:
            r = 1.0
        else:
            r = -1.0
        return r

    def stock_experience(self, t, s, a, o_a, r, n_s, episode_end_flag):
        data_index = t % self.data_size
        #  print "s(stock) : ", s
        #  print "a(stock) : ", a
        #  print "o_a(stock) : ", o_a
        #  print "r(stock) : ", r
        #  print "n_s(stock) : ", n_s
        #  print "episode_end_flag(stock) : ", episode_end_flag

        self.D[0][data_index] = s
        self.D[1][data_index] = a
        self.D[2][data_index] = o_a
        self.D[3][data_index] = r
        self.D[4][data_index] = n_s
        self.D[5][data_index] = episode_end_flag
        #  print "self.D : ", self.D

    def experience_replay(self, t):
        if t > self.initial_exploration:
            if t < self.data_size:
                replay_index = np.random.randint(0, t, (self.replay_size, 1))
                #  print "replay_index(small) : ", replay_index
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))
                #  print "replay_index(large) : ", replay_index

            s_replay = xp.ndarray(shape=(self.replay_size, 1, 5), dtype=xp.float32)
            a_replay = xp.ndarray(shape=(self.replay_size, 1), dtype=xp.float32)
            o_a_replay = xp.ndarray(shape=(self.replay_size, 1), dtype=xp.float32)
            r_replay = xp.ndarray(shape=(self.replay_size, 1), dtype=xp.float32)
            n_s_replay = xp.ndarray(shape=(self.replay_size, 1, 5), dtype=xp.float32)
            episode_end_replay = xp.ndarray(shape=(self.replay_size, 1), dtype=np.bool)

            for i in xrange(self.replay_size):
                s_replay[i] = xp.asarray(self.D[0][replay_index[i][0]], dtype=xp.float32)
                a_replay[i] = self.D[1][replay_index[i][0]]
                o_a_replay[i] = self.D[2][replay_index[i][0]]
                r_replay[i] = self.D[2][replay_index[i][0]]
                n_s_replay[i] = xp.asarray(self.D[4][replay_index[i][0]], dtype=xp.float32)
                episode_end_replay[i] = self.D[5][replay_index[i][0]]
            
            #  print "s_replay : ", s_replay
            #  print "s_replay : ", a_replay
            #  print "o_a_replay : ", o_a_replay
            #  print "r_replay : ", r_replay
            #  print "n_s_replay : ", n_s_replay
            #  print "episode_end_replay : ", episode_end_replay

            self.critic_optimizer.zero_grads()
            loss_critic, _, td = self.critic_forward(s_replay, r_replay, n_s_replay, episode_end_replay)

            loss_critic.backward()
            self.critic_optimizer.update()

    def V_func(self, s):
        #  input_data = chainer.Variable(s)
        #  print "input_data.data : ", input_data.data
        #  h1 = F.relu(self.critic_model.l1(input_data))
        #  print "h1 : ", h1.data
        #  output_data = self.critic_model.l2(h1)
        input_data = chainer.Variable(s)
        h1 = F.relu(self.critic_model.l1(input_data))
        h2 = F.relu(self.critic_model.l2(h1))
        output_data = self.critic_model.l3(h2)
        return output_data
    
    def target_V_func(self, s):
        #  input_data = chainer.Variable(s)
        #  print "input_data.data : ", input_data.data
        #  h1 = F.relu(self.critic_model.l1(input_data))
        #  print "h1 : ", h1.data
        #  output_data = self.critic_model.l2(h1)
        input_data = chainer.Variable(s)
        h1 = F.relu(self.critic_model.l1(input_data))
        h2 = F.relu(self.critic_model.l2(h1))
        output_data = self.critic_model.l3(h2)
        return output_data

    def critic_forward(self, s, r, next_s, episode_end):
        num_of_batch = s.shape[0]
        #  print "num_of_batch : ", num_of_batch
        V = self.V_func(s)
        #  print "V(critic_forward) : ", V.data
        tmp = self.target_V_func(next_s)
        #  print "tmp(critic_forward) : ", tmp.data
        target = tmp
        #  print "target(critic_forward) : ", target.data
        
        for i in xrange(num_of_batch):
            if not episode_end[i][0]:
                tmp_ = r[i] + self.GAMMA * tmp.data[i]
                #  print "tmp_ : ", tmp_
                #  print "r : ", r
                #  print "self.GAMMA : ", self.GAMMA
                #  print "tmp.data : ", tmp.data
                #  print "tmp.data[%d] : %f" % (i, tmp.data[i])
            else:
                tmp_ = r[i]
            #  print "target : ", target.data
            #  print "target.data : ", target.data[0] 
            target.data[i][0] = tmp_ 

        td = target - V
        #  print "target : ", target.data
        #  print "V : ", V.data
        #  print "td : ", td.data

        zero_val = chainer.Variable(xp.zeros((self.replay_size, 1), dtype=xp.float32))
        #  print "zero_val : ", zero_val.data

       #  self.critic_optimizer.zero_grads()
        loss = F.mean_squared_error(td, zero_val)
        #  print "loss : ", loss.data

        #  loss.backward()
        #  self.critic_optimizer.update()

        return loss, V, td
    
    def calculate_td_error(self, s, r, next_s):
        V = self.V_func(s)
        #  print "V : ", V.data
        tmp = self.V_func(next_s)
        #  print "tmp : ", tmp.data
        target = r + self.GAMMA * tmp
        #  print "target : ", target.data
        td = target - V
        #  print "td : ", td.data

        return td

    def stock_experience_for_actor(self, s, a, old_a, r, next_s, episode_end_flag):
        if self.calculate_td_error(s, r, next_s).data[0][0] > 0.0:
        #  if self.calculate_td_error(s, r, next_s).data[0][0]+100 > 0.0:
            data_index = self.data_index_actor % self.data_size
            #  print "s(stock_actor) : ", s
            #  print "a(stock_actor) : ", a
            #  print "old_a(stock_actor) : ", old_a
            #  print "r(stock_actor) : ", r
            #  print "next_s(stock_actor) : ", next_s
            #  print "episode_end_flag(stock_actor) : ", episode_end_flag

            self.D_actor[0][data_index] = s
            self.D_actor[1][data_index] = a
            self.D_actor[2][data_index] = old_a
            self.D_actor[3][data_index] = r
            self.D_actor[4][data_index] = next_s
            self.D_actor[5][data_index] = episode_end_flag
            #  print "self.D_actor : ", self.D_actor
            self.data_index_actor += 1
            


    def experience_replay_actor(self):
        if self.data_index_actor > self.initial_exploration:
            if self.data_index_actor < self.data_size:
                replay_index = np.random.randint(0, self.data_index_actor, (self.replay_size, 1))
                #  print "replay_index_actor(small) : ", replay_index
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))
                #  print "replay_index_actor(large) : ", replay_index

            s_replay = xp.ndarray(shape=(self.replay_size, 1, 2), dtype=xp.float32)
            a_replay = xp.ndarray(shape=(self.replay_size, 1), dtype=xp.float32)
            old_a_replay = xp.ndarray(shape=(self.replay_size, 1), dtype=xp.float32)
            r_replay = xp.ndarray(shape=(self.replay_size, 1), dtype=xp.float32)
            next_s_replay = xp.ndarray(shape=(self.replay_size, 1, 2), dtype=xp.float32)
            episode_end_replay = xp.ndarray(shape=(self.replay_size, 1), dtype=np.bool)

            for i in xrange(self.replay_size):
                s_replay[i] = xp.asarray(self.D[0][replay_index[i][0]], dtype=xp.float32)
                a_replay[i] = self.D[1][replay_index[i][0]]
                old_a_replay[i] = self.D[2][replay_index[i][0]]
                r_replay[i] = self.D[2][replay_index[i][0]]
                next_s_replay[i] = xp.asarray(self.D[4][replay_index[i][0]], dtype=xp.float32)
                episode_end_replay[i] = self.D[5][replay_index[i][0]]
            
            #  print "s_replay : ", s_replay
            #  print "a_replay : ", a_replay
            #  print "old_a_replay : ", old_a_replay
            #  print "r_replay : ", r_replay
            #  print "next_s_replay : ", next_s_replay
            #  print "episode_end_replay : ", episode_end_replay

            self.actor_optimizer.zero_grads()
            loss_actor, _ = self.actor_forward(s_replay, a_replay, old_a_replay, episode_end_replay)

            loss_actor.backward()
            self.actor_optimizer.update()

    def actor_func(self, s):
        input_data = chainer.Variable(s)
        h1 = F.relu(self.actor_model.l1(input_data))
        h2 = F.relu(self.actor_model.l2(h1))
        output_data = self.actor_model.l3(h2)
        return output_data

    def actor_forward(self, s, a, old_a, episode_end_flag):
        num_of_batch = s.shape[0]
        #  print "s(act) : ", s
        act = self.actor_func(s)
        #  print "act(actor) : ", act.data
        target = act.data.copy()
        #  print "target(actor) : ", target
        
        #  print "old_a : ", old_a
        for i in xrange(num_of_batch):
            tmp = (act.data[i][0] + old_a[i][0]) / 2.0
            #  print "tmp(actor) : ", tmp
            #  self.sigma = (self.sigma + (old_action - act.data[0][0])) / 2.0
            
            target[i][0] = tmp

        #  print "target(actor) : ", target

        error = chainer.Variable(target) - act

        zero_val = chainer.Variable(xp.zeros((self.replay_size, 1), dtype=xp.float32))
        self.actor_optimizer.zero_grads()
        loss = F.mean_squared_error(error, zero_val)

        loss.backward()
        self.actor_optimizer.update()
        
        return loss, act

    def BoxMuller(self, mean, var):
        r1 = random()
        r2 = random()

        z1 = math.sqrt(-2.0 * math.log(r1))
        z2 = math.sin(2.0 * math.pi * r2)
        return var * z1 * z2 + mean

    def actor(self, s, train_flag=True):
        mu = self.actor_func(s)
        print "mu(actor) : ", mu.data        
        if train_flag:
            act = self.BoxMuller(mu.data[0][0], self.sigma)
            
            if act < self.min_action:
                while 1:
                    act += self.limit_action
                    if act > self.min_action:
                        break

            if act >= self.max_action:
                while 1:
                    act -= self.limit_action
                    if act <= self.max_action:
                        break
        else:
            act = mu.data[0][0]


        return act
    
    def save_result(self, result):
        f_critic_model = open(self.filename_critic_model, 'w')
        f_actor_model = open(self.filename_actor_model, 'w')
        f_result = open(self.filename_result, 'w')
        np.savetxt(self.filename_result, result, fmt="%.6f", delimiter=",")
        pickle.dump(self.critic_model, f_critic_model)
        pickle.dump(self.actor_model, f_actor_model)
        
        f_critic_model.close()
        f_actor_model.close()



    def main(self, limit_episode, limit_step):
        rospy.init_node('deep_actor_critic_for_swingup_myrrbot5')

        rospy.Subscriber("/joint_effort", Float64, self.effort_callback)

        pub = rospy.Publisher("/myrrbot5/joint1_controller/command", Float64, queue_size = 1)

        loop_rate = rospy.Rate(20)

        count = 0
        wait_count = 0
        episode_count = 0
        time = 0.0

        index_for_experience = 0

        #  init_angle = uniform(-1*self.pi, self.pi)
        init_angle = math.pi / 2.0

        #  state = xp.array([[self.init_theta, self.init_omega]], dtype=np.float32)
        #  next_state = xp.array([[self.init_theta, self.init_omega]], dtype=np.float32)
        state = xp.zeros((1, 5), dtype=xp.float32)
        next_state = xp.zeros((1, 5), dtype=xp.float32)
        reward = 0.0
        action = 0.0
        old_action = 0.0

        V_list = np.array([])
        max_V = 0.0
        ave_V = 0.0
        reward_list = np.array([])
        ave_reward = 0.0
        total_reward = 0.0

        temp_result = np.array([[]])
        test_result = np.array([[]])
        
        evaluation_flag = False

        
        while not rospy.is_shutdown():
            if(self.wait_flag):
                wait_count += 1
                if wait_count%self.hz==0:
                    print "Please Wait %d seconds" % (1-wait_count/self.hz)
                    print "state : ", state
                    print "state(main) : [%f %f]" % (math.degrees(state[0][0]), math.degrees(state[0][1]))
                    print "next_state(main) : ", next_state
                    print "next_state(main) : [%f %f]" % (math.degrees(next_state[0][0]), math.degrees(next_state[0][1]))

                #  print "Reset!!!!"
                #  print "init_angle : ", init_angle
                self.reset_client(self.init_theta)
                #  self.reset_client(init_angle)
                self.init_omega = 0.0
                action = 0.0
                old_action = 0.0

                #  print "state : ", state
                #  print "state(main) : [%f %f]" % (math.degrees(state[0][0]), math.degrees(state[0][1]))
                #  print "next_state(main) : ", next_state
                #  print "next_state(main) : [%f %f]" % (math.degrees(next_state[0][0]), math.degrees(next_state[0][1]))
                pub.publish(action)
                
                if wait_count == self.hz*0.5:
                    self.init_theta = uniform(-1*math.pi, math.pi)
                    
                    q = tf.transformations.quaternion_from_euler(0.0, self.init_theta, 0.0)
                    state[0][0] = q[0]
                    state[0][1] = q[1]
                    state[0][2] = q[2]
                    state[0][3] = q[3]
                    state[0][4] = self.init_omega
                    next_state[0][0] = q[0]
                    next_state[0][1] = q[1]
                    next_state[0][2] = q[2]
                    next_state[0][3] = q[3]
                    next_state[0][4] = self.init_omega
                    print "state : ", state
                    #  print "state : ", state
                    #  print "state(main) : [%f %f]" % (math.degrees(state[0][0]), math.degrees(state[0][1]))
                    #  print "next_state(main) : ", next_state
                    #  print "next_state(main) : [%f %f]" % (math.degrees(next_state[0][0]), math.degrees(next_state[0][1]))
                if wait_count == self.hz*1:
                    wait_count = 0
                    self.wait_flag = False
            else:
                print ""
                if not evaluation_flag:
                    if index_for_experience < self.initial_exploration:
                        print "Initial Exploration (for critic) : %d/%d steps" % (index_for_experience, self.initial_exploration)
                    #  print "self.critic_model.parameters : ", self.critic_model.parameters
                    #  print "self.actor_model.parameters : ", self.actor_model.parameters
                    if self.data_index_actor < self.initial_exploration:
                        print "Initial Exploration (for actor) : %d/%d steps" % (self.data_index_actor, self.initial_exploration)

                    print "Episode : %d / Time : %f" % (episode_count, time)
                    count += 1
                    time = float(count) / self.hz
                    print "state(main) : ", state
                    print "state(main) : [%f %f]" % (math.degrees(state[0][0]), math.degrees(state[0][1]))
                    #  print "state=%s, next_state=%s" % (state, next_state)
                    action = self.actor(state)
                    print "action(self.sigma) : %f(%f)" % (action, self.sigma)
                    pub.publish(action)
                    
                    #  print "state=%s, next_state=%s" % (state, next_state)
                    #  next_state[0][0], next_state[0][1] = self.get_joint_properties('myrrbot5_joint1')
                    #  print "next_state(main) : ", next_state
                    #  print "next_state(main) : [%f %f]" % (math.degrees(next_state[0][0]), math.degrees(next_state[0][1]))
                    
                    quaternion, angular = self.get_link_state('myrrbot5::myrrbot5_link2', '')
                    print "quaternion : ", quaternion
                    print "angular : ", angular
                    next_state[0][0] = quaternion[0] 
                    next_state[0][1] = quaternion[1]
                    next_state[0][2] = quaternion[2]
                    next_state[0][3] = quaternion[3]
                    next_state[0][4] = angular.y
                    print "next_state : ", next_state
                    #  print "state=%s, next_state=%s" % (state, next_state)
                    #  self.reward = self.reward_func(next_state[0][0])
                    self.reward = self.reward_func(self.get_joint_properties('myrrbot5_joint1'))
                    print "self.reward : ", self.reward
                    print "V.data : ", float(self.V_func(state).data[0][0])

                    #  print "state=%s, next_state=%s" % (state, next_state)
                    self.stock_experience(index_for_experience, state, action, old_action, self.reward, next_state, False)
                    self.stock_experience_for_actor(state, action, old_action, self.reward, next_state, False)
                    self.experience_replay(index_for_experience)
                    self.experience_replay_actor()
                    index_for_experience += 1

                    if self.episode_end_flag(state):
                        print "Episode End!!!!(rolling over 4*PI)"
                        if episode_count%self.evaluation_freq==0:
                            evaluation_flag =True

                        count = 0
                        episode_count += 1
                        self.wait_flag = True
                        action = 0.0
                    else:
                        #  print "state update!!!!!!!!!!!!!!!!!!!!!!"
                        #  print "state : ", state
                        #  print "next_state : ", next_state
                        state = next_state.copy()
                        #  print "state : ", state
                        #  print "next_state : ", next_state
                        #  print "action : ", action
                        #  print "old_action : ", old_action
                        old_action = action.copy()
                        #  print "action : ", action
                        #  print "old_action : ", old_action
                    
                        if count >= self.episode_time*self.hz:
                            print "Episode End!!!!(over %d seconds)" % self.episode_time
                            if episode_count%self.evaluation_freq==0:
                                evaluation_flag =True

                            count = 0
                            episode_count += 1
                            self.wait_flag = True
                            action = 0.0

                    if self.sigma > 0.1:
                        self.sigma -= 0.0001
                    else:
                        self.sigma = 0.1
                    
                    if episode_count % self.target_update_freq == 0:
                        self.target_critic_model = copy.deepcopy(self.critic_model)

                    if episode_count >= limit_episode:
                        print "Finish!!!!!"
                        break
                else:
                    print "Evaluation now!!!!!!!!!"
                    print "Episode : %d / Time : %f" % (episode_count-1, time)
                    count += 1
                    time = float(count) / self.hz
                    print "state(main) : ", state
                    print "state(main) : [%f %f]" % (math.degrees(state[0][0]), math.degrees(state[0][1]))
                    #  print "state=%s, next_state=%s" % (state, next_state)
                    action = self.actor(state, False)
                    print "action(self.sigma) : %f(%f)" % (action, self.sigma)
                    pub.publish(action)
                    
                    #  print "state=%s, next_state=%s" % (state, next_state)
                    #  next_state[0][0], next_state[0][1] = self.get_joint_properties('myrrbot5_joint1')
                    #  print "next_state(main) : ", next_state
                    #  print "next_state(main) : [%f %f]" % (math.degrees(next_state[0][0]), math.degrees(next_state[0][1]))
                    quaternion, angular = self.get_link_state('myrrbot5::myrrbot5_link2', '')
                    print "quaternion : ", quaternion
                    print "angular : ", angular
                    next_state[0][0] = quaternion[0] 
                    next_state[0][1] = quaternion[1]
                    next_state[0][2] = quaternion[2]
                    next_state[0][3] = quaternion[3]
                    next_state[0][4] = angular.y
                    print "next_state : ", next_state
                    
                    #  print "state=%s, next_state=%s" % (state, next_state)
                    #  self.reward = self.reward_func(next_state[0][0])
                    self.reward = self.reward_func(self.get_joint_properties('myrrbot5_joint1'))
                    print "self.reward : ", self.reward
                    reward_list = np.append(reward_list, [self.reward])
                    V_list = np.append(V_list, [float(self.V_func(state).data[0][0])])

                    if self.episode_end_flag(state):
                        print "Evaluation End!!!!(rolling over 4*PI)"
                        #  print "V_list : ", V_list
                        max_V = np.max(V_list)
                        ave_V = np.average(V_list)
                        print "max_V : ", max_V
                        print "ave_V : ", ave_V
                        V_list = np.array([])
                        #  print "reward_lsit : ", reward_list
                        ave_reward = np.average(reward_list)
                        total_reward = np.sum(reward_list)
                        print "ave_reward : ", ave_reward
                        print "total_reward : ", total_reward
                        eward_list = np.array([])

                        temp_result = np.array(([[episode_count-1, max_V, ave_V, ave_reward, total_reward]]), dtype=np.float32)
                        #  print "temp_result : ", temp_result
                        if episode_count-1 == 0:
                            #  print "test_result : ", test_result
                            test_result = temp_result
                            #  print "test_result : ", test_result
                        else:
                            test_result = np.r_[test_result, temp_result]
                        #  print "test_result : ", test_result
                        self.save_result(test_result)


                        count = 0
                        episode_count += 1
                        self.wait_flag = True
                        action = 0.0
                        evaluation_flag = False
                    else:
                        #  print "state update!!!!!!!!!!!!!!!!!!!!!!"
                        #  print "state : ", state
                        #  print "next_state : ", next_state
                        state = next_state.copy()
                        #  print "state : ", state
                        #  print "next_state : ", next_state
                        #  print "action : ", action
                        #  print "old_action : ", old_action
                        old_action = action.copy()
                        #  print "action : ", action
                        #  print "old_action : ", old_action
                    
                        if count >= self.episode_time*self.hz:
                            print "Evaluation End!!!!(over %d seconds)" % self.episode_time
                            #  print "V_list : ", V_list
                            max_V = np.max(V_list)
                            ave_V = np.average(V_list)
                            print "max_V : ", max_V
                            print "ave_V : ", ave_V
                            V_list = np.array([])
                            #  print "reward_lsit : ", reward_list
                            ave_reward = np.average(reward_list)
                            total_reward = np.sum(reward_list)
                            print "ave_reward : ", ave_reward
                            print "total_reward : ", total_reward
                            reward_list = np.array([])

                            temp_result = np.array(([[episode_count-1, max_V, ave_V, ave_reward, total_reward]]), dtype=np.float32)
                            #  print "temp_result : ", temp_result
                            if episode_count-1 == 0:
                                #  print "test_result : ", test_result
                                test_result = temp_result
                                #  print "test_result : ", test_result
                            else:
                                test_result = np.r_[test_result, temp_result]
                            #  print "test_result : ", test_result
                            
                            self.save_result(test_result)
                            
                            
                            count = 0
                            episode_count += 1
                            self.wait_flag = True
                            action = 0.0
                            evaluation_flag =False

            loop_rate.sleep()

if __name__=="__main__":
    deep_actor_critic_agent = agent()
    deep_actor_critic_agent.main(100000000, 100)

